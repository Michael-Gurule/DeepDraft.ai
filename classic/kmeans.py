"""
K-Means Clustering
==================
An unsupervised learning algorithm that partitions data into K clusters by
iteratively assigning each point to its nearest centroid and updating centroids
until convergence. Minimizes within-cluster sum of squared distances (inertia).

Use Cases:
    - Customer segmentation
    - Document clustering and topic discovery
    - Image compression (color quantization)
    - Anomaly detection (distance from nearest centroid)
    - Feature engineering (cluster membership as a feature)
    - Data exploration and structure discovery

Algorithm (Lloyd's algorithm):
    1. Initialize K centroids (randomly or with K-Means++)
    2. Assign each point to the nearest centroid
    3. Recompute centroids as cluster means
    4. Repeat steps 2–3 until convergence (centroid movement < tol)

Limitations:
    - Requires K to be specified in advance
    - Assumes spherical, equal-variance clusters (struggles with non-convex shapes)
    - Sensitive to initialization and outliers
    - Not suitable for categorical data (use K-Modes or K-Prototypes)

Fine-Tuning Recommendations:
    - K selection: use the elbow method (inertia vs K) and silhouette score.
      The silhouette score provides a more objective measure than the elbow.
    - init='k-means++' (default): smarter initialization that speeds convergence
      and avoids poor local minima; always prefer over 'random'.
    - n_init=10 (default): run with 10 different random seeds, keep best.
      Increase to 20–50 for important or noisy clustering tasks.
    - max_iter=300 (default): usually sufficient; increase if not converging.
    - Feature scaling: CRITICAL — K-Means uses Euclidean distance.
      Always StandardScale or MinMaxScale before clustering.
    - For non-spherical clusters: consider DBSCAN (density-based) or
      Gaussian Mixture Models (probabilistic, soft assignments).
    - For high-dimensional data: apply PCA first to reduce noise and
      alleviate the curse of dimensionality.
    - MiniBatchKMeans: use for large datasets (n_samples > 10k) — faster
      convergence with slight quality trade-off.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, calinski_harabasz_score
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_blobs, make_moons
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_blob_data(n_samples: int = 500, n_clusters: int = 4, n_features: int = 6):
    """
    Generate synthetic isotropic Gaussian clusters.

    K-Means is well-suited to this type of data (spherical, similar variance).

    Replace with your own data loading:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["id"]).values   # remove non-feature columns
        # No y needed — this is unsupervised

    Args:
        n_samples:  total number of data points
        n_clusters: number of ground truth clusters (for evaluation only)
        n_features: number of features

    Returns:
        X:          feature matrix
        y_true:     ground truth cluster labels (only used for evaluation,
                    not available in real unsupervised settings)
    """
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )
    return X, y_true


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: adjust n_clusters based on elbow/silhouette analysis
KMEANS_CONFIG = {
    "n_clusters": 4,         # K — number of clusters to find
    "init": "k-means++",     # initialization strategy; always use 'k-means++' over 'random'
    "n_init": 10,            # number of random starts; increase for noisy data
    "max_iter": 300,         # maximum EM iterations per run
    "tol": 1e-4,             # convergence threshold (centroid movement)
    "algorithm": "lloyd",    # 'lloyd' (standard) or 'elkan' (faster for low-dim dense data)
    "random_state": 42,
}


def build_pipeline(n_clusters: int = 4) -> Pipeline:
    """
    Build StandardScaler → KMeans pipeline.

    Feature scaling is essential: unscaled features with large ranges dominate
    the Euclidean distance calculation. StandardScaler brings all features to
    zero mean and unit variance before clustering.

    Args:
        n_clusters: number of clusters K

    Returns:
        sklearn Pipeline
    """
    config = {**KMEANS_CONFIG, "n_clusters": n_clusters}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KMeans(**config)),
    ])


# ─────────────────────────────────────────────
# K SELECTION
# ─────────────────────────────────────────────

def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11)) -> int:
    """
    Find the optimal K using the elbow method and silhouette score.

    Elbow method:
        Plot inertia (sum of squared distances to nearest centroid) vs K.
        The "elbow" — where the rate of decrease sharply slows — suggests
        the optimal K. Note: the elbow is not always obvious.

    Silhouette score:
        Measures how similar a sample is to its own cluster vs. other clusters.
        Range: [-1, 1]. Higher is better. Objective and does not require
        knowing the true number of clusters.
        Interpretation:
            > 0.7  — strong structure
            0.5–0.7 — reasonable structure
            0.25–0.5 — weak structure
            < 0.25 — no substantial structure

    Calinski-Harabasz score:
        Ratio of between-cluster dispersion to within-cluster dispersion.
        Higher is better. Tends to favor compact, well-separated clusters.

    Fine-tuning tip: if elbow and silhouette disagree, prefer silhouette.
    Also consider domain knowledge — the "right" K often comes from
    business requirements, not just statistics.

    Args:
        X:       feature matrix (raw, unscaled — scaling handled in pipeline)
        k_range: range of K values to evaluate

    Returns:
        recommended K from silhouette score
    """
    inertias, silhouettes, calinski = [], [], []

    for k in k_range:
        pipeline = build_pipeline(n_clusters=k)
        labels = pipeline.fit_predict(X)

        inertias.append(pipeline.named_steps["model"].inertia_)
        silhouettes.append(silhouette_score(pipeline.named_steps["scaler"].transform(X), labels))
        calinski.append(calinski_harabasz_score(pipeline.named_steps["scaler"].transform(X), labels))

    best_k = list(k_range)[np.argmax(silhouettes)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
    axes[0].set_title("Elbow Method — Inertia vs K")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, marker="o", color="green")
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    axes[1].set_title("Silhouette Score vs K")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    axes[2].plot(list(k_range), calinski, marker="o", color="orange")
    axes[2].set_title("Calinski-Harabász Score vs K")
    axes[2].set_xlabel("K"); axes[2].set_ylabel("Score")

    plt.suptitle("K Selection Analysis")
    plt.tight_layout()
    plt.savefig("kmeans_k_selection.png", dpi=150)
    plt.show()

    print(f"\nOptimal K by silhouette: {best_k}  (score: {max(silhouettes):.4f})")
    return best_k


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_clustering(pipeline: Pipeline, X: np.ndarray,
                          y_true: np.ndarray | None = None) -> dict:
    """
    Compute clustering quality metrics.

    Internal metrics (no labels needed):
        Silhouette score — measures cluster cohesion and separation
        Inertia          — within-cluster sum of squares (lower = more compact)

    External metrics (require ground truth labels):
        Adjusted Rand Index (ARI) — measures similarity to ground truth.
            0 = random labeling; 1 = perfect match.
            Adjusted for chance (unlike raw Rand Index).

    Note: in real-world unsupervised settings, y_true is not available —
    rely on internal metrics for evaluation.

    Args:
        pipeline: fitted pipeline
        X:        raw feature matrix
        y_true:   ground truth cluster labels (optional, for external metrics)

    Returns:
        dict of metrics
    """
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    labels   = pipeline.named_steps["model"].labels_
    inertia  = pipeline.named_steps["model"].inertia_
    sil      = silhouette_score(X_scaled, labels)

    print(f"  Inertia:         {inertia:.2f}")
    print(f"  Silhouette:      {sil:.4f}")

    metrics = {"inertia": inertia, "silhouette": sil}

    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        print(f"  Adjusted Rand:   {ari:.4f}")
        metrics["ari"] = ari

    return metrics


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_clusters_2d(pipeline: Pipeline, X: np.ndarray,
                     y_true: np.ndarray | None = None) -> None:
    """
    2D cluster visualization using PCA projection.

    Projects high-dimensional data to 2D via PCA for visualization.
    Shows predicted cluster labels; optionally overlays true labels for comparison.

    Args:
        pipeline: fitted KMeans pipeline
        X:        raw feature matrix
        y_true:   ground truth labels (optional, shown as markers)
    """
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    labels   = pipeline.named_steps["model"].labels_
    centroids = pipeline.named_steps["model"].cluster_centers_

    # PCA projection to 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    centroids_2d = pca.transform(centroids)

    fig, axes = plt.subplots(1, 2 if y_true is not None else 1, figsize=(14, 5))
    axes = [axes] if y_true is None else axes

    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10",
                    alpha=0.6, edgecolors="k", linewidths=0.2, s=20)
    axes[0].scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                    c="red", marker="X", s=200, edgecolors="k", zorder=5, label="Centroids")
    axes[0].set_title("K-Means Predicted Clusters (PCA)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend()

    if y_true is not None:
        axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="tab10",
                        alpha=0.6, edgecolors="k", linewidths=0.2, s=20)
        axes[1].set_title("Ground Truth Labels (PCA)")
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    plt.suptitle("K-Means Cluster Visualization")
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=150)
    plt.show()


def plot_silhouette_analysis(pipeline: Pipeline, X: np.ndarray) -> None:
    """
    Silhouette plot for per-sample and per-cluster quality inspection.

    Each cluster is represented as a horizontal bar. Width = silhouette score
    of samples in that cluster. All bars should extend well past the mean
    silhouette (red dashed line). Clusters with many negative-score samples
    may need to be split or merged.

    Fine-tuning tip: if one cluster has mostly negative silhouettes while
    others are high, that cluster likely contains noise or should be split.
    """
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    labels   = pipeline.named_steps["model"].labels_
    k        = pipeline.named_steps["model"].n_clusters
    sil_avg  = silhouette_score(X_scaled, labels)
    sil_vals = silhouette_samples(X_scaled, labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10

    for i in range(k):
        ith_silhouettes = np.sort(sil_vals[labels == i])
        size_cluster_i = ith_silhouettes.shape[0]

        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_silhouettes,
                          alpha=0.7, label=f"Cluster {i}")
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(sil_avg, color="red", linestyle="--", label=f"Mean: {sil_avg:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Analysis (K={k})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("kmeans_silhouette.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== K-Means Clustering ===")

    # 1. Load data
    TRUE_K = 4
    X, y_true = load_blob_data(n_clusters=TRUE_K)

    # 2. Find optimal K
    print("\n=== K Selection Analysis ===")
    optimal_k = find_optimal_k(X, k_range=range(2, 11))

    # 3. Fit with optimal K
    print(f"\n=== Fitting K-Means with K={optimal_k} ===")
    pipeline = build_pipeline(n_clusters=optimal_k)
    pipeline.fit(X)

    # 4. Evaluate
    print("\nClustering Metrics:")
    evaluate_clustering(pipeline, X, y_true)

    # 5. Visualize
    plot_clusters_2d(pipeline, X, y_true)
    plot_silhouette_analysis(pipeline, X)

    # 6. MiniBatchKMeans demo (for large datasets)
    print("\n=== MiniBatchKMeans (for large datasets) ===")
    mb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MiniBatchKMeans(
            n_clusters=optimal_k,
            batch_size=100,      # Placeholder: increase for larger datasets
            n_init=10,
            random_state=42,
        )),
    ])
    mb_pipeline.fit(X)
    print("MiniBatchKMeans Metrics:")
    evaluate_clustering(mb_pipeline, X, y_true)


if __name__ == "__main__":
    main()
