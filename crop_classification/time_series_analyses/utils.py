import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

def plot_timeseries(df: pd.DataFrame, uuid: str) -> None:
    df_subset = df[df["uuid"] == uuid]

    fig, ax = plt.subplots(figsize=(15, 5))

    sns.scatterplot(df_subset, x="date", y="ndvi", ax=ax)
    sns.lineplot(df_subset, x="date", y="ndvi", alpha=0.5, ax=ax)
    ax.set_xlabel("Date", fontsize=15)
    ax.set_ylabel("Mean NDVI", fontsize=15)
    ax.set_ylim(-0.1, 1.1)

    plt.show()