# **SentinelHub raster generation utilities**

The raster generation utilities (`src/generate_rasters/`) contain helper modules
for creating Sentinel-2 GeoTIFF tiles with SentinelHub, optionally saving them
locally or uploading them to S3. These modules are intended to be called from
notebooks, scripts, or future job entry points; this package does not currently
include a command-line entry point like the segmentation workflow.

## **Directory Structure**

The directory structure for raster generation is defined as follows -

```text
datakind_geospatial/
|__ src/
|   |__ configs/
|   |   |__ raster_generation.py
|   |__ generate_rasters/
|       |__ evalscripts/
|       |   |__ highlight_optimized.js
|       |   |__ true_color_optimized.js
|       |__ geometry.py
|       |__ io.py
|       |__ naming.py
|       |__ request_builder.py
```

### **Main Scripts**

The `src/generate_rasters` directory contains utilities for generating and
persisting Sentinel-2 raster tiles:

- `geometry.py` initializes Earth Engine, generates covering grids around
  geometry points, builds longitude and latitude axes, and provides an image
  edge-enhancement helper.
- `request_builder.py` builds SentinelHub requests from tile polygons,
  evalscripts, and date ranges, then converts SentinelHub responses into
  `RasterTile` objects.
- `io.py` defines `RasterTile` and `GeoTiffWriter` for writing GeoTIFFs to disk
  or uploading them to S3.
- `naming.py` contains helpers for stable region and file naming.
- `evalscripts/` contains SentinelHub evalscripts for optimized true-color
  Sentinel-2 imagery.

The `src/configs/raster_generation.py` module defines the
`RasterGenerationConfig` dataclass used to group raster parameters and export
settings.

### **Configuration**

Raster generation parameters are represented by `RasterGenerationConfig`:

```python
from configs.raster_generation import RasterGenerationConfig

config = RasterGenerationConfig(
    start_date="2024-01-01",
    end_date="2024-12-31",
    resolution=5,
    evalscript_type="highlight_optimized",
    output_dir="data/01_raw/rasters",
)
```

Important fields include:

- `start_date` and `end_date`: SentinelHub request date interval.
- `resolution`: raster resolution in meters.
- `evalscript_type`: evalscript filename without the `.js` suffix.
- `buffer_distance`: point buffer size in meters when creating covering grids.
- `grid_scale`: Earth Engine `coveringGrid` scale.
- `output_dir`: local output directory.
- `bucket_name` and `s3_prefix`: S3 export destination.
- `evalscript_dir`: optional custom evalscript directory.

### **Credentials**

SentinelHub requests use the default SentinelHub SDK configuration unless a
profile name is provided. The current helper reads the profile name from:

```bash
SETINELHUB_USER=your-sentinelhub-profile
```

If `generate_covering_grid` is used, Google Earth Engine credentials are also
required. To initialize Earth Engine with a specific project, set:

```bash
GEE_PROJECT=your-google-cloud-project
```

S3 exports require AWS credentials available to `boto3`, for example through
environment variables, an AWS profile, or an attached IAM role.

### **Typical Usage**

The utilities can be used directly from a Python script or notebook. A minimal
local export flow looks like this:

```python
from pathlib import Path

from generate_rasters.io import GeoTiffWriter
from generate_rasters.naming import slugify
from generate_rasters.request_builder import fetch_tile, iter_requests

writer = GeoTiffWriter(output_dir="data/01_raw/rasters")

for idx, (request, bbox, size) in enumerate(
    iter_requests(
        tiles=tile_grids,
        start_date="2024-01-01",
        end_date="2024-12-31",
        evalscript_dir=Path("src/generate_rasters/evalscripts"),
        evalscript_type="highlight_optimized",
        resolution=5,
    )
):
    raster_tile = fetch_tile(request, bbox, size, resolution=5)
    writer.export(raster_tile, f"{slugify(region_name)}_tile_{idx}.tif")
```

To upload to S3 instead of, or in addition to, saving locally:

```python
from generate_rasters.io import GeoTiffWriter

writer = GeoTiffWriter(
    bucket_name="output-bucket-name",
    s3_prefix="rasters/region-name",
)
```

### **Evalscript Options**

The included evalscripts are:

- `highlight_optimized`
- `true_color_optimized`

Pass the evalscript name without `.js` as `evalscript_type`.

### **Outputs**

`GeoTiffWriter` writes GeoTIFF files with:

- CRS `EPSG:4326` by default.
- One band per image channel returned by the SentinelHub evalscript.
- A geotransform derived from the request bounding box and raster dimensions.

When S3 export is configured, outputs are uploaded to:

```text
s3://<bucket_name>/<s3_prefix>/<filename>
```

### **Operational Notes**

- `request_builder.py` uses Sentinel-2 L2A data from the Copernicus Data Space
  SentinelHub endpoint.
- Requests use `mosaickingOrder: leastCC` to prefer scenes with lower cloud
  cover.
- `generate_covering_grid` depends on Earth Engine and expects longitude and
  latitude columns named `Longitude` and `Latitude` by default.
- Import submodules directly, as shown above. The package-level
  `generate_rasters` import references a runner module that is not currently
  present in the repository.
