import os
from pystac import Item
import numpy as np 
from numpy.typing import NDArray
import geopandas as gpd 
import rasterio as rio 
from rasterio.transform import Affine, xy
from pyproj import Transformer
from rasterstats import zonal_stats 
import logging

logging.basicConfig(level=logging.INFO)

class ISDADataExtract:
    def __init__(
            self,
            region: str, 
            start_lat_lon: tuple[float, float], 
            end_lat_lon: tuple[float, float],
            assets: dict[str, Item],
            qty_type: str
    ) -> None:
        self.region = region
        self.start_lat_lon = start_lat_lon
        self.end_lat_lon = end_lat_lon
        self.assets = assets
        self.qty_type = qty_type

    @staticmethod
    def write_raster(region: str, data: NDArray, x: NDArray, y: NDArray, qty_type: str) -> None:
        """
        This function converts multiband image arrays (from ISDA) to GeoTiff raster files.

        Args: (i) data - the multiband data array; for iSDA data, they are (4, w, h)
              (ii) x - the longitude coordinate array
              (iii) y - the latitude coordinate array 
              (iv) qty_type - type of soil quantity

        Returns: None
        """
        # x and y resolutions
        xres = (x[-1] - x[0])/len(x) 
        yres = (y[-1] - y[0])/len(y) 

        # Define affine transformation and raster dimensions
        transform = Affine.translation(x[0] - xres / 2, y[0] - yres / 2) * Affine.scale(xres, yres)
        bands = data.shape[0]
        height = data.shape[1] 
        width = data.shape[2]

        params = {
            "mode": "w",
            "driver": "GTiff", 
            "height": height,
            "width": width, 
            "count": bands, 
            "dtype": data.dtype, 
            "crs": "epsg:4326", 
            "transform": transform
        }

        output_dir = os.path.join("rasters", region)
        os.makedirs(output_dir, exist_ok=True)
        raster_path = os.path.join(output_dir, f"{qty_type}_{region}.tif")

        with rio.open(raster_path, **params) as src: 
            for i in range(bands):
                # Write all bands; rasterio uses 1-indexing
                src.write(data[i], i + 1)
                logging.info(f"Raster for {region} written.")

    def get_data_subset(self, write_to_raster: bool=True) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
        """
        This function accesses rasters within the specified bounding box from the 
        iSDA sources.
        
        Args: (i) start_lat_lon - the upper left corner of bounding box 
            (ii) end_lat_long - the lower right corner of bounding box 
            (iii) asset_type - the kind of soil quantity required

        Returns: (i) arr_transformed - the raster as a numpy array (accounting for back-transforms)
                (ii) new_profile - metadata associated with raster
                (iii) lon_grid - grid of longitude coordinates
                (iv) lat_grid - grid of latitude coordinates
        """
        asset_type = self.assets[self.qty_type]
        file_location = asset_type.assets["image"].href
        
        with rio.open(file_location) as file:
            # The assets are in Mercator projection. The lat/lon bounding box corners
            # are appropriately converted.
            transformer_to_crs = Transformer.from_crs("epsg:4326", file.crs)

            # converting lat/lon to x and y for upper left and lower right corners
            x_0, y_0 = transformer_to_crs.transform(self.start_lat_lon[0], self.start_lat_lon[1])
            x_1, y_1 = transformer_to_crs.transform(self.end_lat_lon[0], self.end_lat_lon[1])

            # obtain pixel values associated with bounding box coordinates
            row_start, col_start = file.index(x_0, y_0)
            row_end, col_end = file.index(x_1, y_1)

            # ensure that row, col are sorted as upper left to lower right
            row_min, row_max = sorted([row_start, row_end])
            col_min, col_max = sorted([col_start, col_end])

            height = row_max - row_min
            width = col_max - col_min

            window = rio.windows.Window(col_min, row_min, height, width)

            arr = file.read(window=window)

            new_profile = file.profile.copy()

            # Generate all row and column indices
            rows = np.arange(height)
            cols = np.arange(width)

            # Get 1D arrays of x and y coordinates for rows and cols
            transform = file.window_transform(window)
            x_coords, _ = xy(transform, rows=0, cols=cols)  # all x for row 0
            _, y_coords = xy(transform, rows=rows, cols=0)  # all y for col 0

            # Broadcast into full grids
            xs, ys = np.meshgrid(x_coords, y_coords)

            # Convert pixel CRS coordinates back to lat/lon
            transformer_to_latlon = Transformer.from_crs(file.crs, "epsg:4326", always_xy=True)
            lon_grid, lat_grid = transformer_to_latlon.transform(xs, ys)

        new_profile.update({
                'height': window.height,
                'width': window.width,
                'count': file.count,
                'transform': file.window_transform(window)
        })
        """
        Given how the data are stored, certain back-transformations are required to scale
        them properly.
        """
        conversion_funcs = {
            "x": np.vectorize(lambda x: x),
            "x/10": np.vectorize(lambda x: x/10, otypes=["float32"]),
            "x/100": np.vectorize(lambda x: x/100, otypes=["float32"]),
            "expm1(x/10)": np.vectorize(lambda x: np.expm1(x / 10), otypes=["float32"]),
            "%3000": np.vectorize(lambda x: int(x%3000), otypes=["int16"])
        }
        
        if asset_type.extra_fields["back-transformation"] is not None:
            # Obtain the kind of back-transformation associated with type
            conversion = asset_type.extra_fields["back-transformation"]
            arr_transformed = conversion_funcs[conversion](arr)

            if write_to_raster:
                self.write_raster(self.region, arr_transformed, lon_grid[0,:], lat_grid[:,0], self.qty_type)

            return arr_transformed, new_profile, lon_grid, lat_grid
        else:
            if write_to_raster:
                self.write_raster(self.region, arr, lon_grid[0,:], lat_grid[:,0], self.qty_type)
            return arr, new_profile, lon_grid, lat_grid

def data_within_polygons(gdf: gpd.GeoDataFrame, raster_file_paths: list[str], region: str="Trans_Nzoia_1") -> gpd.GeoDataFrame:
    # Making a copy so that the original is not changed
    gdf_copied = (
        gdf.copy(deep=True)
        .reset_index(drop=True)
    )

    for file_path in sorted(raster_file_paths):
        qty_name = file_path.split("/")[-1].split(f"_{region}.tif")[0] # get name of soil quantity
        logging.info(f"Working on {qty_name}")

        """ 
        We now calculate the zonal statistics using rasterstats. Looking through
        the rows of the gdf is inefficient.
        """
        with rio.open(file_path, mode="r") as src:
            data = src.read(1) 
            transform = src.transform
            
            stats = zonal_stats(
                gdf_copied,
                data,
                affine=transform,
                stats=["mean"]
            )
        means = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
        gdf_copied[qty_name] = means

    return gdf_copied