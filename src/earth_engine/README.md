# **Earth Engine VI time-series pipeline**

The Earth Engine pipeline (`src/earth_engine/`) contains scripts for generating
vegetation-index time series from Sentinel-2 surface reflectance imagery. It
loads configured geometry assets, computes index means per geometry and image
date, pivots the results to wide CSV tables, and starts Google Earth Engine
table export tasks to Google Drive.

## **Directory Structure**

The directory structure for Earth Engine time-series generation is defined as
follows:

```text
datakind_geospatial/
|__ configs/
|   |__ earth_engine/
|       |__ assets.yml
|__ src/
|   |__ common/
|   |   |__ ee_initialize.py
|   |   |__ logging_config.py
|   |__ earth_engine/
|       |__ cli.py
|       |__ geometry.py
|       |__ io.py
|       |__ vegetation_indices.py
|       |__ vi_timeseries.py
```

### **Main Scripts**

The `src/earth_engine` directory contains the processing code required to build
vegetation-index exports:

- `cli.py` parses command-line arguments, loads configured geometry paths, and
  starts one Earth Engine export task per tile.
- `geometry.py` loads GeoPackage geometries, reprojects them to EPSG:4326, and
  converts them to Earth Engine feature collections.
- `io.py` loads `configs/earth_engine/assets.yml` and resolves geometry asset
  paths.
- `vegetation_indices.py` registers supported vegetation indices and their
  Earth Engine band-generation functions.
- `vi_timeseries.py` filters Sentinel-2 imagery, masks clouds and shadows,
  reduces imagery over each geometry, and formats export tables.

### **Configuration**

Geometry inputs are configured in:

```text
configs/earth_engine/assets.yml
```

The `geometry_assets.selection` block controls whether all configured
GeoPackages are processed or only one named asset. The `gpkg_files` block maps
asset names to local GeoPackage paths, typically under `data/geometry/`.

Expected geometry columns include:

- `uuid`
- `region`
- `tile_name`
- `geometry`

### **Credentials**

The workflow requires Google Earth Engine access. If Earth Engine is not already
initialized locally, the code attempts interactive authentication. To initialize
with a specific Google Cloud project, set:

```bash
GEE_PROJECT=your-google-cloud-project
```

### **Running the CLI**

Run the module from the repository root so Python package imports and configured
relative paths resolve correctly:

```bash
cd /path/to/DataKind_Geospatial

uv run python -m earth_engine.cli \
  --vi_index_name ndvi \
  --vi_index_band ndvi \
  --scale 10 \
  --start_date 2024-01-01 \
  --end_date 2024-12-31
```

Do not run `src/earth_engine/cli.py` directly by path. The module uses relative
imports, so direct path execution will fail with an import error.

### **Supported Vegetation Indices**

The registry currently supports:

- `ndvi`
- `ndmi`
- `evi`
- `ndre`
- `savi`
- `bsi`

The CLI currently uses `--vi_index_name` to look up the registered index.
`--vi_index_band` is required by the parser but is not used by the current
implementation.

### **Outputs**

For each tile in each configured geometry file, the CLI starts a Google Drive
CSV export task with names like:

```text
ndvi_series_<tile_name>
```

Exports are written to a Google Drive folder named after the index:

```text
ndvi_series/
```

Each output table contains one row per geometry UUID and one column per observed
image date. Missing index values are filled with `-9999`.

### **Operational Notes**

- The Sentinel-2 collection used by the workflow is
  `COPERNICUS/S2_SR_HARMONIZED`.
- Dates should be passed as `YYYY-MM-DD`.
- The configured start date must be on or after `2016-06-13`.
- Earth Engine tasks are submitted asynchronously; monitor task completion in
  the Earth Engine task manager or Google Drive.
