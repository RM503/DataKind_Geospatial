''' 
Functions for processing segmentation results.
'''
import os
import glob
import pandas as pd 
import geopandas as gpd 
import uuid
import shapely
from shutil import make_archive
import logging 

logging.basicConfig(level=logging.INFO)

def vectors_list(base_path: str, file_extension, region: str='All') -> list[str]:
    ''' 
    This function returns a list of paths to polygon files given base path and extension.

    Arguments: (i) base_path - base directory containing all polygon files by region
               (ii) file_extension - type of file being searched
               (iii) region - region being searched; defaults to all regions

    Returns: list of file paths
    '''
    if base_path.endswith("/"):
        base_path = base_path[:-1]
    if region == 'All':
        return glob.glob(f"{base_path}/*/*.{file_extension}")
    else:
        return glob.glob(f"{base_path}/{region}/*.{file_extension}")
    
def zip_csv_from_gdf(file_paths: list[str], archive: bool=False) -> None:
    if not isinstance(file_paths, list):
        raise ValueError(f"{file_paths} should be a list of file paths")
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} cannot be found")
        
        gdf = gpd.read_file(file_path)
        # Group the dataframe by tile_name
        gdf_grouped = gdf.groupby("tile_name")

        FOLDER_NAME = file_path.split('/')[-1].split('.')[0]
        os.makedirs(f"vectors/csv_files/{FOLDER_NAME}", exist_ok=True)

        logging.info(f"Processing {FOLDER_NAME}")

        for tile_name, group in gdf_grouped:
            file_name = f"{tile_name}.csv"
            group.to_csv(f"vectors/csv_files/{FOLDER_NAME}/{file_name}", index=False)

    if archive:
        make_archive(
            base_name="csv_files_archived",
            format="zip",
            base_dir="csv_files"
        )

def make_gdf(file_paths: list[str], export_gdf: bool=True) -> gpd.GeoDataFrame:
    '''
    This function converts the concatenated pandas dataframe into a
    geopandas one by converting the WKT column into geometry. Additionally,
    computes centroids and areas for each polygon. This is only for tile geometries
    that were stored as csv files.

    Arguments: (i) file_paths - a list of file paths to the csv files
               (ii) export_gdf - a boolean tag for exporting geodataframe; defaults to True

    Returns: returns a concated geodataframe containing all the tiles in the given region
    '''
    if not isinstance(file_paths, list):
        raise ValueError(f"{file_paths} should be a list of file paths")

    df_list = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} cannot be found")
            
        df = (
            pd.read_csv(file_path)
            .drop(columns=['value'])
        )
        if 'uuid' not in df.columns:
            df['uuid'] = df.apply(lambda row: uuid.uuid4(), axis=1)

        df = df[['uuid', 'geometry']]

        # check if `region` and `tile_name` columns exist or not
        if not pd.Series(['region', 'tile_name']).isin(df.columns).all():
            region, file_name = (file_path.split('/')[1], file_path.split('/')[-1])
            tile_name = file_name.split('_delineation')[0]

            df.insert(loc=1, column='region', value=region)
            df.insert(loc=2, column='tile_name', value=tile_name)

        df_list.append(df)

    df_joined = pd.concat(df_list).reset_index(drop=True)

    geometry = gpd.GeoSeries.from_wkt(df_joined.geometry) 
    gdf = gpd.GeoDataFrame(df_joined, geometry=geometry, crs='EPSG:4326')

    #gdf['centroid'] = gdf.geometry.centroid 

    geometry_pseudo_mercator = gdf.geometry.to_crs('EPSG:3857') # convert the polygons to physical distance units
    gdf['area (sq. meters)'] = geometry_pseudo_mercator.area 
    gdf['area (acres)'] = gdf['area (sq. meters)'] * 0.000247

    ''' 
    Each tile has one (or two) large polygons that are composed of background and
    other field boundaries that were not identified. These are removed from each
    tile.
    '''
    tile_names = gdf['tile_name'].unique()
    idx_list = []
    for tile in tile_names:
        gdf_tile = gdf[gdf['tile_name']==tile]

        max_area = gdf_tile['area (acres)'].max()
        max_area_idx = gdf_tile[gdf_tile['area (acres)']==max_area].index[0]
        idx_list.append(max_area_idx)

    gdf = (
        gdf.drop(index=idx_list)
           .reset_index(drop=True)
    )

    if export_gdf:
        gdf.to_file(f"vectors/{region}.gpkg", driver="GPKG")

    return gdf

def aggregrate_polygons_between_gdfs(
        gdf_1: gpd.GeoDataFrame, 
        gdf_2: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
    """ 
    This function takes in two processed geopandas dataframes and finds non-overlapping
    polygons between them. This allows polygons, that have been generated using different
    parameters, to be intergrated into the final polygon data.
    """

    if gdf_1.crs is None and gdf_2.crs is None:
        gdf_1 = gdf_1.set_crs("EPSG:4326")
        gdf_2 = gdf_2.set_crs("EPSG:4326")

    # Perform spatial joins to search for intersections and extract unique uuid from left_df
    gdf_sjoined = gpd.sjoin(
        left_df=gdf_1,
        right_gdf=gdf_2,
        how="inner"
    )

    uuid_intersect = gdf_sjoined['uuid_left'].unique().tolist()

    # Polygons in gdf_2 without intersections with polygons in gdf_1
    gdf_nonintersect=gdf_2[~gdf_2["uuid"].isin(uuid_intersect)] 

    gdf_agg = pd.concat([gdf_1, gdf_nonintersect])

    return gdf_agg

def polygon_overlaps(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    This function calculates the overlapping regions between the predefined polygons and
    the ones generated using SamGeo. This can be used for ground-truth validation.

    Args: (i) gdf1 - a geopandas dataframe containing the predefined polygons
          (ii) gdf2 - a geopandas dataframe containing the generated polygons

    Returns: a geopandas dataframe containing the overlapping regions sorted by IoU

    The returned geopandas dataframe contains `left_index` and `right_index` indicating
    which polygons in the two original dataframes were overlapped. It also contains an
    extra geometry column (geom_right) for the purposes of IoU calculation.
    '''

    #Since image raster covers a smaller area than predefined polygon coverage area, we create a bounding box.
    xmin, ymin, xmax, ymax = gdf2.total_bounds

    # bbox = shapely.Polygon(
    #     [
    #         (xmin, ymin),
    #         (xmin, ymax),
    #         (xmax, ymax),
    #         (xmax, ymin)
    #     ]
    # )
    # gdf_bbox = gpd.GeoDataFrame([{'geometry': bbox}]) # bounding box dataframe

    gdf1_filtered = gdf1.cx[xmin:xmax, ymin:ymax]   # predefined polygons are now filtered within bbox

    '''
    To get a sense of how many of the generated mask polygons intersect with the predefined ones,
    we perform a `spatial join` with an `intersects` predicate. This will return a dataframe containing
    information on which mask polygons intersect with which predefined polygons.
    '''

    gdf_joined = gpd.sjoin(
        left_df=gdf1_filtered,
        right_df=gdf2,
        predicate='intersects',
        how='inner'
    )

    gdf_joined['index_right'] = gdf_joined['index_right'].astype('Int32')
    gdf_joined.reset_index(inplace=True)
    gdf_joined.rename(columns={'index': 'index_left'}, inplace=True)

    # Attach the mashuru_polygons geometry as an `geom_right` for further calculations
    geom_right = gdf2.loc[gdf_joined['index_right']]['geometry'].reset_index(drop=True)
    gdf_joined.insert(6, 'geom_right', geom_right)

    gdf_joined.drop(columns=['elongation ratio'], inplace=True)

    def IoU(poly1: shapely.Polygon, poly2: shapely.Polygon) -> float:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        iou = intersection / union
        return iou

    gdf_joined['IoU'] = gdf_joined.apply(
        lambda x: IoU(x['geometry'], x['geom_right']),
        axis=1
    )

    '''
    There will be instances where a predefined polygon has intersections with multiple
    mask polygons. We group the data such that we keep only intersections with highest
    IoU and arrange it in descending order of IoU
    '''

    idx = gdf_joined.groupby('index_left')['IoU'].idxmax()
    gdf_joined = gdf_joined.loc[idx]

    gdf_joined = gdf_joined.sort_values(by='IoU', ascending=False).reset_index(drop=True)

    return gdf_joined

if __name__ == '__main__':
    
    gpkg_file_paths = glob.glob("vectors/*.gpkg")
    zip_csv_from_gdf(gpkg_file_paths)