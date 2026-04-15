from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
TIME_SERIES_DIR = REPO_ROOT / "crop_classification" / "time_series_analyses"
NDVI_RAW_DIR = TIME_SERIES_DIR / "ndvi_series_raw"
NDMI_RAW_DIR = TIME_SERIES_DIR / "ndmi_series_raw"
LABELS_DIR = APP_DIR / "labels"
DEFAULT_REGION = "Kajiado_1"
DEFAULT_TILE = "tile_10"
LABEL_OPTIONS = ["Farm", "Field", "Tree", "Other"]


@dataclass(frozen=True)
class Dataset:
    region: str
    tile: str
    ndvi: pd.DataFrame
    ndmi: pd.DataFrame
    geometry_by_uuid: dict[str, dict]
    uuids: list[str]


def discover_region_tiles() -> dict[str, list[str]]:
    region_tiles: dict[str, set[str]] = {}
    for path in sorted(NDVI_RAW_DIR.glob("ndvi_series_*_tile_*.csv")):
        stem = path.stem.removeprefix("ndvi_series_")
        region, tile_number = stem.rsplit("_tile_", 1)
        ndmi_path = NDMI_RAW_DIR / f"ndmi_series_{region}_tile_{tile_number}.csv"
        if ndmi_path.exists():
            region_tiles.setdefault(region, set()).add(f"tile_{tile_number}")
    return {region: sorted(tiles, key=tile_sort_key) for region, tiles in sorted(region_tiles.items())}


def tile_sort_key(tile: str) -> tuple[int, str]:
    try:
        return (int(tile.removeprefix("tile_")), tile)
    except ValueError:
        return (10_000, tile)


REGION_TILES = discover_region_tiles()


def clean_vi_series(df: pd.DataFrame, index_type: str) -> pd.DataFrame:
    drop_columns = [col for col in ("system:index", ".geo") if col in df.columns]
    df = df.drop(columns=drop_columns)

    if "uuid" not in df.columns:
        uuid_col = df.columns[-1]
        df = df.rename(columns={uuid_col: "uuid"})

    new_cols = ["uuid"] + [col for col in df.columns if col != "uuid"]
    df = df.reindex(columns=new_cols)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    df.iloc[:, 1:] = df.iloc[:, 1:].interpolate(method="linear", axis=1, limit_direction="both")

    df_melted = df.melt(id_vars="uuid", var_name="date", value_name=index_type)
    df_melted["date"] = pd.to_datetime(df_melted["date"], errors="coerce")
    df_melted = df_melted.dropna(subset=["date"])
    df_melted = df_melted.drop_duplicates(subset=["uuid", "date"], keep="first")
    df_melted = df_melted.sort_values(["uuid", "date"]).reset_index(drop=True)
    return df_melted


def smooth_vi_series(df: pd.DataFrame, index_type: str) -> pd.DataFrame:
    smoothed_groups = []
    smooth_col = f"{index_type}_smoothed"

    for _, group in df.groupby("uuid", sort=False):
        group = group.sort_values("date").copy()
        values = group[index_type].astype(float)
        if len(group) >= 5:
            group[smooth_col] = savgol_filter(values, window_length=5, polyorder=2)
        else:
            group[smooth_col] = values
        smoothed_groups.append(group)

    return pd.concat(smoothed_groups, ignore_index=True)


def raw_csv_path(index_type: str, region: str, tile: str) -> Path:
    tile_number = tile.removeprefix("tile_")
    raw_dir = NDVI_RAW_DIR if index_type == "ndvi" else NDMI_RAW_DIR
    return raw_dir / f"{index_type}_series_{region}_tile_{tile_number}.csv"


def load_dataset(region: str, tile: str) -> Dataset:
    ndvi_path = raw_csv_path("ndvi", region, tile)
    ndmi_path = raw_csv_path("ndmi", region, tile)
    if not ndvi_path.exists():
        raise FileNotFoundError(f"Missing NDVI file: {ndvi_path.relative_to(REPO_ROOT)}")
    if not ndmi_path.exists():
        raise FileNotFoundError(f"Missing NDMI file: {ndmi_path.relative_to(REPO_ROOT)}")

    ndvi_raw = pd.read_csv(ndvi_path)
    ndmi_raw = pd.read_csv(ndmi_path)
    geometry_by_uuid = geometry_lookup(ndvi_raw)
    ndvi = smooth_vi_series(clean_vi_series(ndvi_raw, "ndvi"), "ndvi")
    ndmi = smooth_vi_series(clean_vi_series(ndmi_raw, "ndmi"), "ndmi")
    uuids = sorted(set(ndvi["uuid"]).intersection(set(ndmi["uuid"])))
    if not uuids:
        raise ValueError(f"No matching UUIDs found for {region} {tile}.")
    return Dataset(region=region, tile=tile, ndvi=ndvi, ndmi=ndmi, geometry_by_uuid=geometry_by_uuid, uuids=uuids)


def geometry_lookup(df: pd.DataFrame) -> dict[str, dict]:
    if "uuid" not in df.columns or ".geo" not in df.columns:
        return {}

    geometries: dict[str, dict] = {}
    for uuid, geometry_text in df[["uuid", ".geo"]].dropna().itertuples(index=False):
        try:
            geometries[str(uuid)] = json.loads(geometry_text)
        except json.JSONDecodeError:
            continue
    return geometries


def plot_timeseries(dataset: Dataset, uuid: str):
    ndvi_subset = dataset.ndvi[dataset.ndvi["uuid"] == uuid].set_index("date")
    ndmi_subset = dataset.ndmi[dataset.ndmi["uuid"] == uuid].set_index("date")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(ndvi_subset.index, ndvi_subset["ndvi_smoothed"], s=16, color="#0b5d1e", label="NDVI")
    ax.plot(ndvi_subset.index, ndvi_subset["ndvi_smoothed"], color="#1f9d55", linewidth=1.8)
    ax.scatter(ndmi_subset.index, ndmi_subset["ndmi_smoothed"], s=16, color="#195b8f", label="NDMI")
    ax.plot(ndmi_subset.index, ndmi_subset["ndmi_smoothed"], color="#2f80c1", linewidth=1.8)
    ax.set_title("VI Time Series")
    ax.set_ylabel("VI index")
    ax.set_ylim(-0.5, 1.1)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def polygon_bounds(geometry: dict) -> tuple[float, float, float, float]:
    coordinates = []

    def collect(items):
        if not items:
            return
        if isinstance(items[0], (int, float)) and len(items) >= 2:
            coordinates.append((float(items[0]), float(items[1])))
            return
        for item in items:
            collect(item)

    collect(geometry.get("coordinates", []))
    lons = [coord[0] for coord in coordinates]
    lats = [coord[1] for coord in coordinates]
    return min(lons), min(lats), max(lons), max(lats)


def map_html(dataset: Dataset, uuid: str) -> str:
    geometry = dataset.geometry_by_uuid.get(uuid)
    if not geometry:
        return "<p>No polygon geometry found for this UUID.</p>"

    min_lon, min_lat, max_lon, max_lat = polygon_bounds(geometry)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    feature = {
        "type": "Feature",
        "properties": {"uuid": uuid},
        "geometry": geometry,
    }

    srcdoc = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
        <style>
          html, body, #map {{ height: 100%; width: 100%; margin: 0; }}
        </style>
      </head>
      <body>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
          const map = L.map("map").setView([{center_lat}, {center_lon}], 16);
          L.tileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
            {{ attribution: "Tiles &copy; Esri" }}
          ).addTo(map);
          const layer = L.geoJSON({json.dumps(feature)}, {{
            style: {{ color: "#e31a1c", weight: 3, fillOpacity: 0.18 }}
          }}).addTo(map);
          map.fitBounds(layer.getBounds(), {{ padding: [30, 30], maxZoom: 18 }});
        </script>
      </body>
    </html>
    """
    return (
        '<iframe title="Polygon Map" '
        'style="height: 520px; width: 100%; border: 1px solid #d0d0d0;" '
        f'srcdoc="{html.escape(srcdoc, quote=True)}"></iframe>'
    )


def labels_path(region: str, tile: str) -> Path:
    return LABELS_DIR / f"{region}_{tile}_labels.csv"


def read_labels(region: str, tile: str) -> dict[str, str]:
    path = labels_path(region, tile)
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open(newline="") as label_file:
        reader = csv.DictReader(label_file)
        for row in reader:
            if row.get("uuid") and row.get("class"):
                labels[row["uuid"]] = row["class"]
    return labels


def write_labels(region: str, tile: str, labels: dict[str, str]) -> Path:
    LABELS_DIR.mkdir(exist_ok=True)
    path = labels_path(region, tile)
    with path.open("w", newline="") as label_file:
        writer = csv.DictWriter(label_file, fieldnames=["uuid", "class", "updated_at"])
        writer.writeheader()
        for uuid, label in sorted(labels.items()):
            writer.writerow({"uuid": uuid, "class": label, "updated_at": datetime.now().isoformat(timespec="seconds")})
    return path


def labels_table(labels: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"uuid": uuid, "class": label} for uuid, label in sorted(labels.items())],
        columns=["uuid", "class"],
    )


def load_selection(region: str, tile: str):
    try:
        dataset = load_dataset(region, tile)
        labels = read_labels(region, tile)
        uuid = dataset.uuids[0]
        label = labels.get(uuid, LABEL_OPTIONS[0])
        status = f"Loaded {region} {tile}: {len(dataset.uuids)} polygons. Labels save to {labels_path(region, tile).relative_to(REPO_ROOT)}."
        return (
            dataset,
            labels,
            gr.update(choices=dataset.uuids, value=uuid),
            label,
            plot_timeseries(dataset, uuid),
            map_html(dataset, uuid),
            labels_table(labels),
            status,
        )
    except Exception as exc:
        empty_fig = plt.figure(figsize=(11, 4))
        return None, {}, gr.update(choices=[], value=None), LABEL_OPTIONS[0], empty_fig, "", pd.DataFrame(), str(exc)


def update_tiles(region: str):
    tiles = REGION_TILES.get(region, [])
    value = tiles[0] if tiles else None
    return gr.update(choices=tiles, value=value)


def select_uuid(dataset: Dataset | None, labels: dict[str, str], uuid: str):
    if dataset is None or not uuid:
        return None, "", LABEL_OPTIONS[0]
    return plot_timeseries(dataset, uuid), map_html(dataset, uuid), labels.get(uuid, LABEL_OPTIONS[0])


def save_label(dataset: Dataset | None, labels: dict[str, str], uuid: str, label: str):
    if dataset is None or not uuid:
        return labels, labels_table(labels), "Load a dataset and choose a UUID before saving."
    if not label:
        return labels, labels_table(labels), "Choose a label before saving."

    labels = dict(labels)
    labels[uuid] = label
    path = write_labels(dataset.region, dataset.tile, labels)
    return labels, labels_table(labels), f"Saved {label} for {uuid} to {path.relative_to(REPO_ROOT)}."


def build_app() -> gr.Blocks:
    regions = list(REGION_TILES)
    default_region = DEFAULT_REGION if DEFAULT_REGION in REGION_TILES else regions[0]
    default_tiles = REGION_TILES[default_region]
    default_tile = DEFAULT_TILE if DEFAULT_TILE in default_tiles else default_tiles[0]

    with gr.Blocks(title="Polygon Labeling Widget") as app:
        dataset_state = gr.State()
        labels_state = gr.State({})

        gr.Markdown("# Polygon Labeling Widget")
        gr.Markdown("Inspect each polygon with its NDVI and NDMI time series, then save a class label.")

        with gr.Row():
            region = gr.Dropdown(regions, value=default_region, label="Region")
            tile = gr.Dropdown(default_tiles, value=default_tile, label="Tile")
            load_button = gr.Button("Load Dataset", variant="primary")

        with gr.Row():
            uuid = gr.Dropdown([], label="UUID", scale=3)
            label = gr.Radio(LABEL_OPTIONS, value=LABEL_OPTIONS[0], label="Label", scale=1)
            save_button = gr.Button("Save Label", variant="primary", scale=1)

        status = gr.Markdown()
        plot = gr.Plot(label="VI Time Series")
        polygon_map = gr.HTML(label="Polygon Map")
        labels = gr.Dataframe(headers=["uuid", "class"], label="Saved Labels", interactive=False)

        region.change(update_tiles, inputs=region, outputs=tile)
        load_button.click(
            load_selection,
            inputs=[region, tile],
            outputs=[dataset_state, labels_state, uuid, label, plot, polygon_map, labels, status],
        )
        uuid.change(select_uuid, inputs=[dataset_state, labels_state, uuid], outputs=[plot, polygon_map, label])
        save_button.click(
            save_label,
            inputs=[dataset_state, labels_state, uuid, label],
            outputs=[labels_state, labels, status],
        )
        app.load(
            load_selection,
            inputs=[region, tile],
            outputs=[dataset_state, labels_state, uuid, label, plot, polygon_map, labels, status],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
