import numpy as np
import pandas as pd
from sklearn.decomposition import PCA 
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def umap_embeddings(X: pd.DataFrame, n_components: int=10) -> np.ndarray:
    return umap.UMAP(n_components=n_components, n_neighbors=15, random_state=1).fit_transform(X)

def pca_embeddings(X: pd.DataFrame, n_components: int=10) -> np.ndarray:
    return PCA(n_components=n_components).fit_transform(X)

def generate_silhoutte_plots(
        X_features: pd.DataFrame,
        dim_reduction_type: str,
        n_cluster_list: list[int]
    ) -> None:
    # Dimensionality reduction
    if dim_reduction_type == "PCA":
        X_embeddings = pca_embeddings(X_features)
    elif dim_reduction_type == "UMAP":
        X_embeddings = umap_embeddings(X_features)

    for n_cluster in n_cluster_list:
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)

        ax1.set_xlim(-0.1, 1)
        ax1.set_ylim([0, len(X_features) + (n_cluster + 1) * 10])

        # Initialize K-Means
        kmeans = KMeans(n_clusters=n_cluster, n_init=15, max_iter=1000, random_state=4)
        cluster_labels = kmeans.fit_predict(X_features)

        silhouette_avg = silhouette_score(X_features, cluster_labels)
        print(f"Silhouette score for {n_cluster}-clusters is: {silhouette_avg}")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        for i in range(n_cluster):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
             )
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("Silhouette scores", fontsize=15)
        ax1.set_ylabel("Cluster labels", fontsize=15)

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_cluster)
        
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(
            X_embeddings[:, 0], X_embeddings[:, 1], X_embeddings[:, 2], lw=0, alpha=0.7, c=colors, edgecolor="k"
        )
    plt.show()