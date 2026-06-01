import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Farmland Characteristics**

    Here, we further analyze the NDVI (also the NDMI) index of the polygons identified as farms. The aim now is to answer the following question

    * How many planting cycles are there per year? This can be determined by finding the number of NDVI peaks, which should translate to growing seasons of crops. This, however, is a nontrivial process given that the time-series data can still have spurious peaks even after smoothing.
    * How many of these farms are water via irrigation vs. natural rain water? Answering this can be tricky. Nevertheless, we can study the NDMI index to make inferences. This naturally leads to the question of classifying the various farms by their moisture contents as inferred from the maximum NDMI.
    """)
    return


@app.cell
def _():
    import warnings
    from pathlib import Path

    import dask.dataframe as dd
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")
    return Path, dd, gpd, np, pd


@app.cell
def _():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{siunitx}"
    })
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Read data**
    """)
    return


@app.cell
def _(Path):
    POLYGON_PATH = Path("../../data/05_inference/Trans_Nzoia_1_results_aggregated.gpkg")
    NDVI_PATH = Path("../../data/03_aggregated/ndvi_series_Trans_Nzoia_1_aggregated.csv")
    NDMI_PATH = Path("../../data/03_aggregated/ndmi_series_Trans_Nzoia_1_aggregated.csv")
    return NDMI_PATH, NDVI_PATH, POLYGON_PATH


@app.cell
def _(POLYGON_PATH, gpd):
    gdf = gpd.read_file(POLYGON_PATH)

    gdf.head()
    return (gdf,)


@app.cell
def _(gdf):
    uuid_list = gdf["uuid"].unique().tolist() # list of polygon uuid from inference file
    polygon_types = gdf["prediction_decoded"].tolist()
    return polygon_types, uuid_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the NDVI and NDMI time-series data using `Dask` for memory safety. `Pandas` will try to read the data into memory by default, which will slow down processing. The `extract_data()` function loads the NDVI and NDMI long-form time-series and performing the necessary computations before merging and returning them.
    """)
    return


@app.cell
def _(NDMI_PATH, NDVI_PATH, dd):
    ddf_ndvi = dd.read_csv(NDVI_PATH, parse_dates=["date"])
    ddf_ndmi = dd.read_csv(NDMI_PATH, parse_dates=["date"])
    return ddf_ndmi, ddf_ndvi


@app.cell
def _(dd, pd, uuid_list):
    def extract_data(
            ddf_1: dd.DataFrame,
            ddf_2: dd.DataFrame,
            uuid_list: list[str]=uuid_list
    ) -> pd.DataFrame:
        """
        Extracts NDVI/NDMI data frame Dask dataframes and merges
        them together.
        """
        df_1 = ddf_1[ddf_1["uuid"].isin(uuid_list)].compute()
        df_1 = df_1.reset_index(drop=True)

        df_2 = ddf_2[ddf_2["uuid"].isin(uuid_list)].compute()
        df_2 = df_2.reset_index(drop=True)

        df_1 = df_1.loc[:, ~df_1.columns.str.startswith("Unnamed:")]
        df_2 = df_2.loc[:, ~df_2.columns.str.startswith("Unnamed:")]

        df_merged = df_1.merge(df_2, on=["uuid", "date"], how="inner")

        return df_merged

    return (extract_data,)


@app.cell
def _(ddf_ndmi, ddf_ndvi, extract_data):
    df = extract_data(ddf_ndvi, ddf_ndmi)
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Polygon labels are provided to the aggregated NDVI-NDMI time-series dataset.
    """)
    return


@app.cell
def _(pd):
    def map_polygon_types(
        df: pd.DataFrame,
        uuid_list: list[str],
        polygon_labels: list[str]
    ) -> pd.DataFrame:
        if not isinstance(uuid_list, list) and not isinstance(polygon_labels, list):
            raise TypeError(
                "One of 'uuid_list' or 'polygon_labels' is not of type list"
            )

        if len(uuid_list) != len(polygon_labels):
            raise ValueError("Lists 'uuid_list' and 'polygon_labels' must be of same length.")

        label_map = dict(zip(uuid_list, polygon_labels))
        df["polygon_type"] = (
            df.groupby("uuid")["uuid"]
              .transform(lambda x: label_map[x.iloc[0]])
        )

        return df

    return (map_polygon_types,)


@app.cell
def _(df, map_polygon_types, polygon_types, uuid_list):
    df_w_labels = map_polygon_types(df, uuid_list, polygon_types)

    df_w_labels.head()
    return (df_w_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Smoothing**

    Peak  detection can be made more difficult if the signal is noisy. Even though the raw signals were passed through Savitzky-Golay filter, we pass them through one again such that only essential peak and periodic features are retained.
    """)
    return


@app.cell
def _():
    from scipy.signal import find_peaks, savgol_filter

    return find_peaks, savgol_filter


@app.cell
def _(pd, savgol_filter):
    # Savitzky-Golay filter parameters

    WINDOW_SIZE = 7
    POLYGON_ORDER = 3

    def vi_smoothing(
        df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        polygon_order: int = POLYGON_ORDER
    ) -> pd.DataFrame:
        if "uuid" not in df.columns:
            raise KeyError("Column 'uuid' must be present in dataframe.")

        if not isinstance(window_size, int) and not isinstance(polygon_order, int):
            raise TypeError(
                "Both 'window_size' and 'polygon_order' must be of type int."
            )

        df = df.copy()

        # Apply transformations to 'ndvi' and 'ndmi' columns
        df["ndvi"] = df.groupby("uuid")["ndvi"].transform(
            lambda x: savgol_filter(
                x,
                window_size,
                polygon_order
            ) if len(x) >= window_size else x
        )
        df["ndmi"] = df.groupby("uuid")["ndmi"].transform(
            lambda x: savgol_filter(
                x,
                window_size,
                polygon_order
            ) if len(x) >= window_size else x
        )

        return df

    return (vi_smoothing,)


@app.cell
def _(df_w_labels, vi_smoothing):
    df_smoothed = vi_smoothing(df_w_labels)
    return (df_smoothed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **NDVI-NDMI relationship**

    The two vegetation indices - NDVI and NDMI - obey a strongly linear relationship as can be seen in the following scatter plot. Although unknown at this point, an NDVI of 0.4 to 0.6 during growing season can indicate irrigation. We also observe that polygons labeled as "Farm" (shown as green) tend to traverse through a larger range of NDVI values. Even though the NDVI charactertics and farmland phenology are expected to vary by region, we should expect this linear relationship to hold.
    """)
    return


@app.cell
def _():
    import datashader as ds
    import holoviews as hv
    import holoviews.operation.datashader as hd
    import hvplot.pandas
    from holoviews.operation.datashader import dynspread

    hv.extension("bokeh")
    return dynspread, hd, hv


@app.cell
def _(dynspread, hd, hv, pd):
    def scatter_ndvi_ndmi(
        df: pd.DataFrame,
        ndvi_col: str = "ndvi",
        ndmi_col: str = "ndmi"
    ):
        points = hv.Points(
            df,
            kdims=[ndvi_col, ndmi_col],
            vdims=["polygon_type"]
        )
        ds_points = dynspread(
            hd.datashade(
                points,
                aggregator="count_cat",
                color_key={"Farm": "green", "Field": "red"}
            ).opts(tools=["hover"])
        )
        hline_1 = hv.HLine(0.4).opts(color="white", line_dash="dashed")
        hline_2 = hv.HLine(0.2).opts(color="white", line_dash="dashed")
        text_1 = hv.Text(0.4, 0.45, "Higher moisture content; possible irrigation").opts(color="white")
        text_2 = hv.Text(0.4, 0.25, "Moderate moisture content").opts(color="white")

        scatter_plot = (
            ds_points*hline_1*hline_2*text_1*text_2
        ).opts(
            width=600,
            height=400,
            xlabel="NDVI",
            ylabel="NDMI"
        )

        return scatter_plot

    return (scatter_ndvi_ndmi,)


@app.cell
def _(df_smoothed, scatter_ndvi_ndmi):
    scatter_ndvi_ndmi(df_smoothed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **NDVI peaks**

    From the figure below, we see how the NDVI varies as a function of time. The NDVI peaks are also identified as orange dots where see the presence of two annual peaks for a first three years. These relate to the summer and winter planting cycles. During the last two years, we also observe a loss of periodic behavior in the time-series. This can indicate that the farm has been out of use.
    """)
    return


@app.cell
def _(df_smoothed):
    df_subset = df_smoothed[
        df_smoothed["uuid"]=="024ac1c2-25eb-43c5-8d9d-6f9c3b3cf727"
    ]
    return (df_subset,)


@app.cell
def _():
    import seaborn as sns

    return (sns,)


@app.cell
def _(find_peaks, np, pd, plt, sns):
    def ndvi_peaks(df: pd.DataFrame):
        df = df.reset_index(drop=True)
        ndvi_values = df["ndvi"].values 

        peaks, _ = find_peaks(ndvi_values, height=(0.5, 1.0), prominence=0.10, distance=10)

        df["peak"] = np.isin(df.index, peaks).astype(int)

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.scatterplot(df, x="date", y="ndvi", hue="peak", ax=ax)
        sns.lineplot(df, x="date", y="ndvi", ax=ax)
        ax.set_xlabel("Date", fontsize=15)
        ax.set_ylabel("NDVI", fontsize=15)
        ax.grid(True, alpha=0.25)

        plt.show()

    return (ndvi_peaks,)


@app.cell
def _(df_subset, ndvi_peaks):
    ndvi_peaks(df_subset)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Monthly distribution of NDVI-NDMI values**

    From the boxplot below, we see that the NDVI and NDMI values show two different peaks – one during summer and the other during winter. Hence, this is statistically showing us that, as a whole, there are two dominant planting cycles. Furthermore, we observe that NDVI and NDMI peaks are correlated with one another.
    """)
    return


@app.cell
def _():
    MONTH_ORDER = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    return (MONTH_ORDER,)


@app.cell
def _(MONTH_ORDER, hv, pd):
    def plot_vi_distribution(
        df: pd.DataFrame,
        polygon_type: str = "Farm"
    ):
        df = df.copy()
        df = df[df["polygon_type"] == polygon_type]
        df["month"] = df["date"].dt.strftime("%b") # Creates a month column

        # Ensure months are in chronological order
        df["month"] = pd.Categorical(
            df["month"],
            categories=MONTH_ORDER,
            ordered=True
        )
        df_melted = pd.melt(
            df,
            id_vars=["date", "month", "uuid", "polygon_type"],
            var_name="vi_type",
            value_name="vi_value"
        )
        df_melted["month_num"] = df_melted["month"].cat.codes

        boxplot = hv.BoxWhisker(
            df_melted,
            kdims=["month_num", "vi_type"],
            vdims='vi_value'
        ).opts(
            whisker_color="white",
            outlier_color="white",
            tools=["hover"],
            width=1000,
            height=500
        )

        return boxplot

    return (plot_vi_distribution,)


@app.cell
def _(df_smoothed, plot_vi_distribution):
    plot_vi_distribution(df_smoothed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Estimating the number of annual planting cycles**

    Since crop growth is supposed to manifest in NDVI data in the form of peaks, we can use such information to estimate the number of annual planting cycles and comment on agricultural practices in the region. Apart from noisy data giving rise to spurious peaks, we should also be aware of the fact that any peak finding algorithm will contain parameters that need to be tuned. In Scipy's `find_peaks` method, we have the following –
    - `height`: Required height of the peak. This can be passed as a tuple to constrain it to an interval.
    - `prominence`: Prominence of the peak.
    - `distance`: Distance required between identified peaks.
    """)
    return


@app.cell
def _(find_peaks, np, pd):
    from typing import Any

    def pad_dict(max_len: int, dict_unpadded: dict[str, Any]) -> dict[str, Any]:
        """
        Pads dictionaries where keys correspond to list of unequal lengths with
        None values.
        """
        dict_padded = {
            k: v + [None]*(max_len - len(v)) for k, v in dict_unpadded.items()
        }
        return dict_padded


    def calculate_peak_position(df: pd.DataFrame) -> pd.DataFrame:
        df["year"] = df["ndvi_peak_date"].dt.year
        df["peak_position"] = (
            df.groupby(["uuid", "year"]).cumcount() + 1
        ).astype("int")
        df = df.drop(columns=["year"])
        return df


    def find_ndvi_peaks(
        df: pd.DataFrame,
        height: tuple[float, float],
        prominence: float,
        distance: int,
        vi_col: str = "ndvi",
        groupby_col: str = "uuid",
        polygon_type: str = "Farm",
    ) -> pd.DataFrame:
        """
        Calculates NDVI peaks 
        """
        df = df.copy()
        df = df[df["polygon_type"] == polygon_type]

        peaks_date_dict = {}
        peaks_val_dict = {}

        for uuid, group in df.groupby(groupby_col):
            group = group.reset_index(drop=True)
            peaks, _ = find_peaks(
                group[vi_col].values,
                height=height,
                prominence=prominence,
                distance=distance
            )

            group["peak"] = np.isin(group.index, peaks).astype(int)

            # Extract dates where NDVI peaks occur
            ndvi_peaks_dates = group[group["peak"] == 1]["date"].tolist()
            ndvi_peaks_values = group[group["peak"] == 1]["ndvi"].tolist()

            peaks_date_dict[uuid] = ndvi_peaks_dates
            peaks_val_dict[uuid] = ndvi_peaks_values

        # Pad elements in dictionaries
        max_len = max(len(v) for v in peaks_date_dict.values())
        peaks_date_dict_padded = pad_dict(max_len, peaks_date_dict)
        peaks_val_dict_padded = pad_dict(max_len, peaks_val_dict)

        # Create dataframes
        df_peaks_date = pd.DataFrame(peaks_date_dict_padded)
        df_peaks_val = pd.DataFrame(peaks_val_dict_padded)

        # Convert wide form df to long form
        df_peaks_date["index"] = range(max_len)
        df_peaks_val["index"] = range(max_len)

        df_peaks_date_melted = (
            pd.melt(
                df_peaks_date,
                id_vars="index",
                var_name="uuid",
                value_name="ndvi_peak_date"
            ).dropna()
        )
        df_peaks_val_melted = (
            pd.melt(
                df_peaks_date,
                id_vars="index",
                var_name="uuid",
                value_name="ndvi_peak_val"
            ).dropna()
        )

        # Merge dataframes
        df_merged = (
            df_peaks_date_melted.merge(
                df_peaks_val_melted,
                on=["uuid", "index"],
                how="inner"
            )
        ).drop(columns="index")

        df_merged = calculate_peak_position(df_merged)
        return df_merged

    return (find_ndvi_peaks,)


@app.cell
def _(df_smoothed, find_ndvi_peaks):
    df_ndvi_peaks = find_ndvi_peaks(
        df_smoothed,
        height=(0.4, 1.0),
        prominence=0.10,
        distance=10
    )

    df_ndvi_peaks.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
