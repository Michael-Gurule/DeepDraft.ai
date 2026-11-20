"""
DBSCAN Anomaly Detection
=========================
Density-Based Spatial Clustering of Applications with Noise.
Discovers arbitrarily-shaped clusters and labels points that don't belong to
any cluster as noise — those noise points are the anomalies.

Algorithm (Ester et al. 1996):
    1. For each point p, find all points within distance ε (eps-neighborhood).
    2. If |N_ε(p)| ≥ min_samples → p is a core point (dense region).
    3. Density-reachable points form a cluster.
    4. Points not reachable from any core point → label = -1 (noise/anomaly).

Key insight: DBSCAN does not require specifying the number of clusters. It
naturally separates dense normal regions from sparse outlier points. Noise
label (-1) serves directly as the anomaly signal.

Parameters:
    eps:         neighborhood radius. Smaller → more anomalies flagged.
    min_samples: minimum points for a core point. Higher → stricter density.
                 Rule of thumb: 2 × n_features (for low dim), or use k-distance plot.

Comparison to other anomaly detectors:
    - Isolation Forest:  globally isolates; no cluster structure.
    - One-Class SVM:     global boundary; single "normal" manifold.
    - DBSCAN:            local density; handles multi-modal normal distributions
                         and non-convex clusters natively.
    - Autoencoder:       learns compressed representation; scales to high-dim.

Use Cases:
    - Geospatial anomaly detection (GPS traces, location patterns)
    - Network traffic analysis (dense normal traffic flows)
    - Log clustering (normal logs cluster; rare anomalous logs are noise)
    - Customer behavior (dense purchase patterns; outlier = unusual session)

Limitations:
    - Does not scale well to very high dimensions (curse of dimensionality)
    - Sensitive to eps and min_samples — use k-distance plot for eps selection
    - Cannot predict on new unseen samples (no .predict(); must refit or use
      DBSCAN + nearest-neighbor assignment for incremental scoring)

Fine-Tuning Recommendations:
    - eps: use the k-distance elbow plot (k = min_samples). Find the "knee"
      in the sorted k-distance curve — that knee is a natural eps choice.
    - min_samples: start at 2 × n_features. Increase for noisier data.
    - metric: 'euclidean' (default). Use 'cosine' for text embeddings,
      'mahalanobis' via sklearn metric_params for correlated features.
    - algorithm: 'ball_tree' or 'kd_tree' for low-dim; 'brute' for high-dim.
    - For high-dimensional data: reduce with PCA (retain 95% variance) before DBSCAN.
    - HDBSCAN (hierarchical variant): more robust to varying density clusters.
      Available in scikit-learn >= 1.3. Often preferred over DBSCAN.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)

from deepdraft.core import save_figure
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_anomaly_data(
    n_normal: int = 1000,
    n_anomalies: int = 50,
    n_features: int = 10,
):
    """
    Generate synthetic normal clusters with injected anomalies.

    Replace with your own dataset:
        import pandas as pd
        df = pd.read_csv("events.csv")
        X = df.drop(columns=["timestamp", "label"]).values
        y = df["label"].values  # 1=anomaly, 0=normal (if available)

    Returns:
        X:  all samples (float32)
        y:  labels (0=normal, 1=anomaly)
    """
    X_normal, _ = make_blobs(n_samples=n_normal, n_features=n_features,
                              centers=3, cluster_std=0.8, random_state=42)
    rng = np.random.RandomState(7)
    X_anomalies = rng.uniform(
        low=X_normal.min(axis=0) * 3,
        high=X_normal.max(axis=0) * 3,
        size=(n_anomalies, n_features),
    )
    X = np.vstack([X_normal, X_anomalies]).astype(np.float32)
    y = np.array([0] * n_normal + [1] * n_anomalies)
    return X, y


# ─────────────────────────────────────────────
# PARAMETER SELECTION
# ─────────────────────────────────────────────

def kdistance_plot(X_scaled: np.ndarray, k: int = None) -> float:
    """
    k-distance plot for eps selection.

    Algorithm:
        1. For each point, find its k-th nearest neighbor distance.
        2. Sort distances in descending order and plot.
        3. The "elbow" (point of maximum curvature) suggests a good eps.

    Args:
        X_scaled: scaled feature matrix
        k:        number of neighbors (default: 2 * n_features)

    Returns:
        Suggested eps (elbow point distance)
    """
    if k is None:
        k = max(2, 2 * X_scaled.shape[1])

    nbrs  = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    dists, _ = nbrs.kneighbors(X_scaled)
    kth_dists = np.sort(dists[:, -1])[::-1]  # sort descending

    # Find elbow via maximum second derivative
    d2 = np.gradient(np.gradient(kth_dists))
    elbow_idx = np.argmax(np.abs(d2))
    suggested_eps = kth_dists[elbow_idx]

    plt.figure(figsize=(9, 4))
    plt.plot(kth_dists, linewidth=1.5, color="steelblue")
    plt.axvline(elbow_idx, color="red", linestyle="--", alpha=0.7,
                label=f"Elbow at index {elbow_idx}")
    plt.axhline(suggested_eps, color="orange", linestyle="--", alpha=0.7,
                label=f"Suggested eps = {suggested_eps:.3f}")
    plt.xlabel("Points sorted by distance (desc)")
    plt.ylabel(f"{k}-th Nearest Neighbor Distance")
    plt.title(f"DBSCAN — k-Distance Plot (k={k})")
    plt.legend()
    plt.tight_layout()
    save_figure("dbscan_kdistance.png", dpi=150)

    print(f"Suggested eps: {suggested_eps:.4f}  (k={k})")
    return float(suggested_eps)


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune eps (neighborhood radius) and min_samples (density threshold)
DBSCAN_CONFIG = {
    "eps":         0.5,   # Placeholder: use kdistance_plot() to find optimal eps
    "min_samples": 10,    # Placeholder: 2 * n_features is a good starting point
    "metric":      "euclidean",  # Placeholder: 'cosine' for embeddings, 'manhattan' for L1
    "algorithm":   "auto",       # 'ball_tree', 'kd_tree', 'brute'
    "n_jobs":      -1,
}


def fit_dbscan(X: np.ndarray, config: dict = None) -> tuple:
    """
    Fit DBSCAN after scaling. Returns scaler, model, and labels.

    DBSCAN labels:
        >= 0: cluster ID (normal)
        -1:   noise (anomaly)

    Args:
        X:      raw feature matrix
        config: DBSCAN hyperparameters; defaults to DBSCAN_CONFIG

    Returns:
        scaler, model, labels (int array, -1=noise/anomaly)
    """
    if config is None:
        config = DBSCAN_CONFIG
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model   = DBSCAN(**config)
    labels  = model.fit_predict(X_scaled)
    return scaler, model, labels


def anomaly_scores_from_labels(labels: np.ndarray, X_scaled: np.ndarray,
                                model: DBSCAN) -> np.ndarray:
    """
    Convert DBSCAN labels to continuous anomaly scores.

    For binary classification the label -1 is sufficient. For ROC-AUC we need
    a continuous score. We use distance to the nearest core point as the score:
        - Core points: 0 (completely normal)
        - Non-core inliers: small positive distance
        - Noise: large positive distance (most anomalous)

    Args:
        labels:   DBSCAN cluster labels
        X_scaled: scaled feature matrix
        model:    fitted DBSCAN

    Returns:
        anomaly_scores: higher = more anomalous
    """
    scores = np.full(len(X_scaled), np.inf)

    if len(model.core_sample_indices_) == 0:
        # No core points found — reduce eps or min_samples
        return (labels == -1).astype(float)

    core_points = X_scaled[model.core_sample_indices_]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nbrs.fit(core_points)
    dists, _ = nbrs.kneighbors(X_scaled)
    scores = dists[:, 0]

    # Normalize to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(labels: np.ndarray, scores: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate DBSCAN anomaly detection performance.

    Args:
        labels: DBSCAN output labels (-1=noise/anomaly, >=0=cluster)
        scores: continuous anomaly scores (for ROC-AUC)
        y:      true binary labels (0=normal, 1=anomaly)

    Returns:
        dict of metrics
    """
    y_pred = (labels == -1).astype(int)  # noise = anomaly

    try:
        auc = roc_auc_score(y, scores)
        ap  = average_precision_score(y, scores)
    except Exception:
        auc, ap = 0.5, 0.0

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    print(f"  Clusters found: {n_clusters}  |  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {"roc_auc": auc, "avg_precision": ap,
            "precision": precision, "recall": recall,
            "n_clusters": n_clusters, "n_noise": int(n_noise)}


# ─────────────────────────────────────────────
# EPS / MIN_SAMPLES SWEEP
# ─────────────────────────────────────────────

def eps_sweep(X_scaled: np.ndarray, y: np.ndarray,
              eps_range: np.ndarray = None, min_samples: int = 10) -> float:
    """
    Sweep eps values and plot F1, ROC-AUC, and noise fraction.

    Use this to select eps when kd-distance plot is ambiguous. It needs labels,
    so run it on a labeled TUNING sample, not on the data you report results
    for: the eps that scores best on a sample always looks best on that sample.

    eps is chosen by the F1 of DBSCAN's own output, the noise label. ROC-AUC
    only rates the continuous distance score, which can rank anomalies
    perfectly even when eps is so small that almost every point is noise.

    Returns:
        the eps with the highest noise-label F1 on the tuning sample
    """
    if eps_range is None:
        eps_range = np.linspace(0.1, 2.0, 30)

    aucs        = []
    f1s         = []
    noise_fracs = []

    for eps in eps_range:
        db     = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_scaled)
        scores = anomaly_scores_from_labels(labels, X_scaled, db)
        try:
            auc = roc_auc_score(y, scores)
        except Exception:
            auc = 0.5
        aucs.append(auc)
        f1s.append(f1_score(y, (labels == -1).astype(int), zero_division=0))
        noise_fracs.append((labels == -1).mean())

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(eps_range, f1s, color="seagreen", label="F1 (noise = anomaly)")
    ax1.plot(eps_range, aucs, color="steelblue", label="ROC-AUC")
    ax1.set_xlabel("eps")
    ax1.set_ylabel("Score")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(eps_range, noise_fracs, color="tomato", linestyle="--", label="Noise Fraction")
    ax2.set_ylabel("Noise Fraction", color="tomato")

    ax1.set_title(f"DBSCAN — eps Sweep (min_samples={min_samples})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.tight_layout()
    save_figure("dbscan_eps_sweep.png", dpi=150)

    best_eps = float(eps_range[np.argmax(f1s)])
    print(f"Best eps by tuning F1: {best_eps:.3f}  (F1={max(f1s):.4f})")
    return best_eps


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, y: np.ndarray) -> None:
    """
    2D cluster visualization using PCA.
    Points colored by cluster; noise points (label=-1) shown as red X.
    """
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    pca     = PCA(n_components=2, random_state=42)
    X_2d    = pca.fit_transform(X_sc) if X.shape[1] > 2 else X_sc

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

    plt.figure(figsize=(9, 7))
    for k, col in zip(unique_labels, colors):
        mask = labels == k
        if k == -1:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c="red", s=40, alpha=0.9, marker="x", label="Noise (Anomaly)")
        else:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[col], s=8, alpha=0.5, label=f"Cluster {k}")

    plt.title("DBSCAN — Cluster Assignments (PCA 2D)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    save_figure("dbscan_clusters.png", dpi=150)


def plot_score_distribution(scores: np.ndarray, y: np.ndarray) -> None:
    """Histogram of continuous anomaly scores (distance to nearest core point)."""
    plt.figure(figsize=(9, 4))
    plt.hist(scores[y == 0], bins=50, alpha=0.6, label="Normal",  color="steelblue")
    plt.hist(scores[y == 1], bins=50, alpha=0.6, label="Anomaly", color="tomato")
    plt.xlabel("Anomaly Score (distance to nearest core point)")
    plt.ylabel("Count")
    plt.title("DBSCAN — Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    save_figure("dbscan_scores.png", dpi=150)


def plot_confusion_matrix(labels: np.ndarray, y: np.ndarray) -> None:
    """Confusion matrix: noise = predicted anomaly."""
    y_pred = (labels == -1).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred),
                           display_labels=["Normal", "Anomaly"]).plot(cmap="Oranges")
    plt.title("DBSCAN — Confusion Matrix")
    plt.tight_layout()
    save_figure("dbscan_confusion.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== DBSCAN Anomaly Detection ===")

    # Placeholder: replace with your dataset
    X, y = generate_anomaly_data(n_normal=1000, n_anomalies=50)
    print(f"Dataset: {X.shape[0]} samples ({y.sum()} anomalies) | {X.shape[1]} features")

    # Scale. The scaler uses no labels, so fitting it on all rows leaks nothing.
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN labels the points it is fit on and has no predict() for new
    # points. To tune eps with labels and still report an honest score, split
    # the data into two equal halves: tune on one, refit and report on the
    # other. Equal sizes keep the point density, and so the meaning of eps
    # and min_samples, comparable between the halves.
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_scaled, y, test_size=0.5, random_state=42, stratify=y
    )

    # Unsupervised choice: the k-distance curve needs no labels
    suggested_eps = kdistance_plot(X_test)

    # Placeholder: update eps after reading k-distance plot
    config = {**DBSCAN_CONFIG, "eps": suggested_eps}
    model  = DBSCAN(**config)
    labels = model.fit_predict(X_test)
    scores = anomaly_scores_from_labels(labels, X_test, model)

    print("\nTest half, eps from the k-distance plot:")
    evaluate(labels, scores, y_test)

    # Supervised choice: sweep eps on the labeled tuning half, then refit on
    # the test half with that eps
    best_eps = eps_sweep(X_tune, y_tune, min_samples=config["min_samples"])
    tuned = DBSCAN(**{**config, "eps": best_eps})
    tuned_labels = tuned.fit_predict(X_test)
    print("\nTest half, eps tuned on the tuning half:")
    evaluate(tuned_labels, anomaly_scores_from_labels(tuned_labels, X_test, tuned), y_test)

    # Visualizations (test half, k-distance eps)
    plot_clusters_2d(X_test, labels, y_test)
    plot_score_distribution(scores, y_test)
    plot_confusion_matrix(labels, y_test)


if __name__ == "__main__":
    main()
