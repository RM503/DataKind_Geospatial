import pandas as pd
import geopandas as gpd
import holoviews as hv
from holoviews import streams
from holoviews.plotting.bokeh import PolygonPlot
import geoviews as gv
import panel as pn
import cartopy.crs as ccrs
from shapely.geometry import Point

hv.extension("bokeh")
gv.extension("bokeh")
pn.extension() 

# Define widgets
data_group = pn.widgets.RadioButtonGroup(
    description="Select type",
    name="Type",
    options=["raw", "processed"]
)

region = pn.widgets.Select(
    description="Select region",
    name="Region",
    options=["Laikipia_1", "Trans_Nzoia_1"]
)

def read_file(data_group: str, region: str) -> gpd.GeoDataFrame:
    # Read feather files for efficient handling
    if data_group == "raw":
        filepath = f"assets/{data_group}/cached/{region}.feather"
        return gpd.read_feather(filepath)
    
    elif data_group == "processed":
        filepath = f"assets/{data_group}/cached/{region}_results_aggregated.feather"
        return gpd.read_feather(filepath)
    
def label_counts(region: str) -> hv.Bars:
    # Bar chart for displaying Farm/Field counts
    gdf = read_file("processed", region)
    counts = gdf["prediction_decoded"].value_counts()

    counts_df = pd.DataFrame(counts).reset_index()
    bar = hv.Bars(counts_df, kdims="prediction_decoded", vdims="count").opts(
        color="prediction_decoded",
        cmap='Category10', 
        line_color="black",
        line_width=2,
        width=600,
        height=400,
        title="Farm/Field counts",
        tools=["hover"]
    )

    return bar
    
def plot_polygons(data_group: str, region: str) -> gv.Polygons:
    # Plot delineated polygons in regions for 'raw' and 'processed' cases
    gdf = read_file(data_group, region)

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    if data_group == "raw":
        polygons = gv.Polygons(gdf, crs=ccrs.PlateCarree()).opts(
            tools=["hover", "wheel_zoom", "pan"], 
            width=800, 
            height=800, 
            color="lightblue",
            alpha=0.5,
            line_color='black'
        )
    elif data_group == "processed":
        # 'Farm' and 'Field designated polygons are given 'green' and 'brown' colors
        palette = ["green", "brown"]

        PolygonPlot.color_index = "prediction_decoded"

        polygons = gv.Polygons(
            gdf, 
            vdims=["uuid", "area (acres)", "prediction_decoded"], 
            crs=ccrs.PlateCarree()
        ).opts(
            tools=["hover", "wheel_zoom", "pan", "tap"],
            active_tools=["tap"],
            color_index = "prediction_decoded",
            hover_tooltips=[
                ("uuid", "@uuid"),
                ("area (acres)", "@area (acres)"),
                ("prediction_decoded", "@prediction_decoded")
            ],
            line_color="black",
            alpha=0.6,
            width=800,
            height=800,
            legend_position="right",
            cmap=palette
        )
    else:
        raise ValueError("Invalid data group")

    tiles = gv.tile_sources.EsriImagery()

    return tiles * polygons

tap_stream = streams.Tap(x=None, y=None)

@pn.depends(data_group=data_group, region=region)
def map_view(data_group: str, region: str) -> gv.Polygons:
    # Enable tap stream on processed polygon map
    plot = plot_polygons(data_group, region)

    if data_group == "processed":
        def tapped_plot(x, y):
            return plot
        return hv.DynamicMap(tapped_plot, streams=[tap_stream])
    else:
        return plot
    
def get_uuid_from_tap(region: str, x: float, y: float) -> str:
    # Get uuid from polygon tap to be used for displaying associated NDVI time-series
    if x is None or y is None:
        return None

    # Load GeoDataFrame for the current region
    gdf = read_file("processed", region)

    """ 
    By default, coordinates extracted from clicks will be in UTM coordinates. Each `tap_point`
    needs to be reprojected to EPSG:4326 for spatial queries.
    """
    tap_point = gpd.GeoSeries([Point(x, y)], crs="EPSG:3857")

    # Handle empty or missing CRS safely
    if gdf.crs is None:
        raise ValueError(f"The GeoDataFrame for region '{region}' has no CRS set.")

    tap_point_reprojected = tap_point.to_crs(gdf.crs)

    # Spatial query: includes edge points
    match = gdf[gdf.geometry.covers(tap_point_reprojected.iloc[0])]

    if not match.empty:
        return match["uuid"].values[0]
    else:
        print(f"No polygon found for tap at ({x:.2f}, {y:.2f}) in region '{region}'")
        return None

def plot_ndvi_time_series(region: str, uuid: str) -> hv.Overlay:
    # Plot NDVI time-series curve for uuid corresponding ot tap
    df = pd.read_csv(f"assets/processed/ndvi_series_{region}_aggregated.csv")

    df_uuid = df[df["uuid"] == uuid]
    if df_uuid["date"].dtype != "datetime64[ns]":
        df_uuid["date"] = pd.to_datetime(df_uuid["date"])

    s = hv.Scatter(df_uuid, kdims="date", vdims="ndvi").opts(tools=["hover"], size=8, ylim=(-0.1, 1.1), line_color="black")
    c = hv.Curve(df_uuid, kdims="date", vdims="ndvi")

    overlay = (s*c).opts(
        width=600,
        height=300,
        show_grid=True,
        title=f"NDVI time-series: {region}, {uuid}"
    )
    return overlay

@pn.depends(tap_stream.param.x, tap_stream.param.y, data_group, region)
def show_ndvi_time_series(x: float, y: float, data_group: str, region: str) -> hv.Overlay:
    if not data_group == "processed":
        return pn.pane.Markdown("NDVI time-series only available for processed polygons")
    
    uuid = get_uuid_from_tap(region, tap_stream.x, tap_stream.y)

    if uuid:
        return plot_ndvi_time_series(region, uuid)
    return pn.pane.Markdown("No polygon selected")

@pn.depends(tap_stream.param.x, tap_stream.param.y, data_group)
def show_ndvi_legend(x: float, y: float, data_group: str) -> pn.pane.Markdown:
    if data_group == "processed":
        return pn.pane.Markdown(
            """ 
            ## NDVI levels interpretation:

            * 0.6 < NDVI < 1.0: healthy, dense vegetation
            * 0.4 < NDVI < 0.6: moderate vegetation 
            * 0.2 < NDVI < 0.4: sparse vegetation
            * -0.1 < NDVI < 0.1: barren areas (rock, sand etc.)
            * -1.0 < NDVI < -0.1: water bodies/ice
            """
        )

@pn.depends(data_group, region)
def show_bars(data_group: str, region: str) -> hv.Bars:
    if not data_group == "processed":
        return pn.pane.Markdown("Farm/Field counts only available for processed polygons")

    return label_counts(region)

controls = pn.Row(region, data_group)
layout = pn.Row(
    pn.Column(controls, map_view),
    pn.Column(show_ndvi_time_series, show_ndvi_legend, show_bars)
)

logo_title = pn.Row(
    pn.pane.PNG("assets/datakind_logo.png", width=50),
    pn.pane.Markdown("# DataKind Geospatial")
)

pn.template.FastListTemplate(
    title="",
    header=logo_title,
    main=[layout]
).servable()