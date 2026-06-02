import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # **Spatial Correlations**
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
    return Path, dd, gpd, pd


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


@app.cell
def _(Path):
    POLYGON_PATH = Path("../../data/05_inference/Trans_Nzoia_1_results_aggregated.gpkg")
    NDVI_PATH = Path("../../data/03_aggregated/ndvi_series_Trans_Nzoia_1_aggregated.csv")
    NDMI_PATH = Path("../../data/03_aggregated/ndmi_series_Trans_Nzoia_1_aggregated.csv")
    return NDMI_PATH, NDVI_PATH, POLYGON_PATH


@app.cell
def _(POLYGON_PATH, gpd):
    gdf = gpd.read_file(POLYGON_PATH)
    return (gdf,)


@app.cell
def _(gdf):
    gdf_farm = gdf[gdf["prediction"] == 0]
    return (gdf_farm,)


@app.cell
def _(gdf_farm):
    gdf_farm.plot(
        figsize=(10, 10)
    )
    return


@app.cell
def _(gdf):
    uuid_list = gdf["uuid"].unique().tolist() # list of polygon uuid from inference file
    polygon_types = gdf["prediction_decoded"].tolist()
    return polygon_types, uuid_list


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


@app.cell
def _(pd):
    from scipy.signal import find_peaks, savgol_filter

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


@app.cell
def _(gpd, pd):
    def calculate_max_ndvi(
        df: pd.DataFrame,
        gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        if "uuid" not in df.columns:
            raise KeyError("Column 'uuid' must be present in dataframe")

        df = df.copy()
        df["year"] = df["date"].dt.year

        df_max_vi = (
            df.groupby(["uuid", "year"])[["ndvi", "ndmi"]]
              .agg(
                    ndvi_max=pd.NamedAgg(column="ndvi", aggfunc="max"),
                    ndmi_max=pd.NamedAgg(column="ndmi", aggfunc="max")
              )
        ).reset_index()

        gdf_w_vi_max = gdf.merge(df_max_vi, on="uuid", how="left")

        return gdf_w_vi_max

    return (calculate_max_ndvi,)


@app.cell
def _(calculate_max_ndvi, df_smoothed, gdf_farm):
    gdf_vi_max = calculate_max_ndvi(df_smoothed, gdf_farm)

    gdf_vi_max.head()
    return (gdf_vi_max,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Moran's I**

    ### **Global**

    It might be of interest to test whether or not farms with high and low peak NDVI/NDMI cluster together. This can be done using Moran's I statistic. There are two variants of Moran's I – global and local. The global I measures whether or not there is any spatial correlation globally in the polygons (*i.e.* whether or not high NDVI/NDMI peaked farms are clustered togther). It is given by

    $$ I = \frac{N}{S_0}\frac{\sum_{i,j=1}^{N}w_{ij}\left( x_i - \bar{x} \right)\left( x_j - \bar{x} \right)}{\sum_{i=1}^{N}\left( x_i - \bar{x} \right)^2} $$
    where
    - $N$ is the number of spatial units indexed by $i$ and $j$
    - $x$ is the variable of interest
    - $\bar{x}$ is the mean of $x$
    - $w_{ij}$ are the elements of a matrix of spatial weights such that $w_{ii}=0$
    - $S_0$ is the sum of all $w_{ij}$

    In the following we calculate the global Moran's I for max NDVI and NDMI to check for spatial correlation. The choice of $w_{ij}$ will depend on whether or not polygons are connected. Since we cannot guarantee that it always is the case, we use KNN based spatial weights.
    """)
    return


@app.cell
def _():
    from typing import Any

    from esda.moran import Moran
    from libpysal.weights import KNN

    return Any, KNN, Moran


@app.cell
def _(Any, KNN, Moran, gpd):
    def calculate_global_moran(
        gdf: gpd.GeoDataFrame,
        var_col: str = "ndvi_max",
        year_col: str = "year",
        k: int = 5
    ) -> dict[int, dict[str, Any]]:
        """
        Calculates the global Moran's I statistic for a given variable in a GeoDataFrame
        as a function of time (year).
        """
        if var_col not in gdf.columns:
            raise KeyError(f"Column '{var_col}' must be present in GeoDataFrame.")

        if year_col not in gdf.columns:
            raise KeyError(f"Column '{year_col}' must be present in GeoDataFrame.")

        results = {}
        for year, group in gdf.groupby(year_col):
            group = group.reset_index(drop=True)

            weights = KNN.from_dataframe(group, k=k)
            weights.transform = "r"
            moran = Moran(group[var_col].values, weights)

            results[year] = {
                "moran": moran,
                "moran_i": moran.I,
                "moran_ei": moran.EI,
                "p_value": moran.p_sim,
                "z_score": moran.z_sim
            }
        return results

    return (calculate_global_moran,)


@app.cell
def _(calculate_global_moran, gdf_vi_max):
    moran_global_ndvi_max = calculate_global_moran(gdf_vi_max, var_col="ndvi_max")

    moran_global_ndvi_max
    return (moran_global_ndvi_max,)


@app.cell
def _(Any, plt):
    from splot.esda import plot_moran

    def plot_global_moran_by_year(
        results: dict[int, dict[str, Any]],
        var: str = "ndvi_max",
        figsize: tuple[int, int] = (12, 4)
    ) -> None:
        """
        Plots splot's plot_moran for each year in the results dict.
        """
        for year, res in sorted(results.items()):
            fig, axes = plot_moran(
                res["moran"],
                zstandard=True,
                figsize=figsize
            )

            scatter_ax = axes[1]
            for line in scatter_ax.lines:
                line.set_color("white")

            fig.suptitle(
                f"Global Moran's I — {var.upper()} ({year})  |  "
                f"I={res['moran_i']:.3f}, p={res['p_value']:.3f}",
                fontsize=13,
                y=1.02
            )
            plt.tight_layout()
            plt.show()

    return (plot_global_moran_by_year,)


@app.cell
def _(moran_global_ndvi_max, plot_global_moran_by_year):
    plot_global_moran_by_year(moran_global_ndvi_max, var="ndvi_max")
    return


@app.cell
def _(calculate_global_moran, gdf_vi_max):
    moran_global_ndmi_max = calculate_global_moran(gdf_vi_max, var_col="ndmi_max")

    moran_global_ndmi_max
    return (moran_global_ndmi_max,)


@app.cell
def _(moran_global_ndmi_max, plot_global_moran_by_year):
    plot_global_moran_by_year(moran_global_ndmi_max, var="ndmi_max")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### **Local**
    """)
    return


@app.cell
def _(Any, KNN, gpd):
    from esda.moran import Moran_Local

    def calculate_local_moran(
        gdf: gpd.GeoDataFrame,
        var: str = "ndvi_max",
        year_col: str = "year",
        k: int = 5
    ) -> dict[int, dict[str, Any]]:
        """
        Calculates Local Moran's I (LISA) per year.
        """
        if var not in gdf.columns:
            raise KeyError(f"Column '{var}' must be present in GeoDataFrame.")
        if year_col not in gdf.columns:
            raise KeyError(f"Column '{year_col}' must be present in GeoDataFrame.")

        results = {}

        for year, group in gdf.groupby(year_col):
            group = group.reset_index(drop=True)

            weights = KNN.from_dataframe(group, k=k)
            weights.transform = "r"

            y = group[var].values
            local_moran = Moran_Local(y, weights)

            results[year] = {
                "local_moran": local_moran,
                "Is": local_moran.Is,        # local I statistic per observation
                "q": local_moran.q,          # quadrant label per observation (1=HH,2=LH,3=LL,4=HL)
                "p_sim": local_moran.p_sim,  # permutation p-value per observation
            }

        return results

    return (calculate_local_moran,)


@app.cell
def _(Any, gpd, plt):
    from splot.esda import plot_local_autocorrelation

    def plot_local_moran_by_year(
        gdf: gpd.GeoDataFrame,
        results: dict[int, dict[str, Any]],
        var: str = "ndvi_max",
        figsize: tuple[int, int] = (15, 5)
    ) -> None:
        for year, res in sorted(results.items()):
            gdf_year = gdf[gdf["year"] == year].reset_index(drop=True)
            local_moran = res["local_moran"]

            fig, ax = plot_local_autocorrelation(
                local_moran,
                gdf_year,
                var,
                p=0.05,
                figsize=figsize
            )

            scatter_ax = ax[0]
            for line in scatter_ax.lines:
                line.set_color("white")

            fig.suptitle(
                f"Local Moran's I (LISA) — {var.upper()} ({year})",
                fontsize=13,
                y=1.02
            )
            fig.tight_layout()
            plt.show()

    return (plot_local_moran_by_year,)


@app.cell
def _(calculate_local_moran, gdf_vi_max):
    moran_local_ndvi_max = calculate_local_moran(gdf_vi_max, var="ndvi_max")
    return (moran_local_ndvi_max,)


@app.cell
def _(gdf_vi_max, moran_local_ndvi_max, plot_local_moran_by_year):
    plot_local_moran_by_year(gdf_vi_max, moran_local_ndvi_max, var="ndvi_max")
    return


@app.cell
def _(calculate_local_moran, gdf_vi_max):
    moran_local_ndmi_max = calculate_local_moran(gdf_vi_max, var="ndmi_max")
    return (moran_local_ndmi_max,)


@app.cell
def _(gdf_vi_max, moran_local_ndmi_max, plot_local_moran_by_year):
    plot_local_moran_by_year(gdf_vi_max, moran_local_ndmi_max, var="ndmi_max")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
