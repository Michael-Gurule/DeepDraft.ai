"""
Isolation Forest
=================
An ensemble anomaly detector that isolates anomalies rather than modeling
normal data. Builds an ensemble of random isolation trees — anomalies are
isolated closer to the root because they are rare and have extreme values.

Key insight: anomalies are few and different. A random tree can isolate an
anomaly in very few splits; normal points require many splits. The anomaly
score is the average path length across trees, normalized to [0, 1].

Anomaly score:
    s(x, n) = 2^(-E[h(x)] / c(n))

    Where:
        h(x)  = average path length of x across all trees
        c(n)  = average path length of unsuccessful BST search = 2*H(n-1) - 2(n-1)/n
        E[h(x)] close to c(n) → normal; much shorter → anomaly

Use Cases:
    - Network intrusion detection
    - Financial fraud detection
    - Manufacturing defect detection
    - IoT sensor fault detection (e.g., Hyperion sensor health monitoring)
    - Log anomaly detection
    - Any high-dimensional, unlabeled dataset

Advantages:
    - No assumptions about data distribution
    - Linear time complexity O(n log n)
    - Handles high-dimensional data well
    - Robust to irrelevant features
    - No distance/density computation needed

Fine-Tuning Recommendations:
    - n_estimators: 100 (default). More → more stable scores, diminishing returns past 200.
    - max_samples: 'auto' (min(256, n_samples)).
        Smaller → faster, captures local anomalies better.
        Larger → more representative, smoother scores.
        Rarely beneficial to exceed 256 (original paper finding).
    - contamination: estimated fraction of anomalies.
        'auto' → uses the theoretical 0.5 offset threshold.
        Set explicitly (e.g., 0.01–0.1) when you have domain knowledge.
        Critical for proper thresholding — overestimating → too many false positives.
    - max_features: 1.0 (all features). Reducing adds diversity like Random Forest.
    - bootstrap: False (default). True → sampling with replacement.
    - Always scale features before use — extreme ranges can bias path lengths.
    - For streaming data: use partial_fit (sklearn does not support this;
      use PyOD's INNE or rolling window refit).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_anomaly_data(
    n_normal: int = 1000,
    n_anomalies: int = 50,
    n_features: int = 10,
    anomaly_type: str = "random",
):
    """
    Generate synthetic normal data with injected anomalies.

    Replace with your own data loading. For real anomaly detection:
        import pandas as pd
        df = pd.read_csv("sensor_data.csv")
        X = df.drop(columns=["timestamp", "label"]).values
        y = df["label"].values  # 1=anomaly, 0=normal (if labels exist)

    Args:
        n_normal:     number of normal samples
        n_anomalies:  number of anomalous samples to inject
        n_features:   feature dimensionality
        anomaly_type: 'random' (uniform outliers), 'cluster' (tight outlier cluster)

    Returns:
        X:     feature matrix (all samples, normal first then anomalies)
        y:     labels: 0=normal, 1=anomaly
        X_train: unlabeled training set (normal + some anomalies, as in real use)
    """
    # Normal data: tight Gaussian clusters
    X_normal, _ = make_blobs(n_samples=n_normal, n_features=n_features,
                              centers=3, cluster_std=0.8, random_state=42)

    # Anomalies
    if anomaly_type == "random":
        # Uniform outliers — far from any cluster
        rng = np.random.RandomState(42)
        X_anomalies = rng.uniform(
            low=X_normal.min(axis=0) * 3,
            high=X_normal.max(axis=0) * 3,
            size=(n_anomalies, n_features)
        ).astype(np.float32)
    else:
        # Tight anomaly cluster in an unusual region
        X_anomalies, _ = make_blobs(n_samples=n_anomalies, n_features=n_features,
                                     centers=[[10] * n_features], cluster_std=0.3,
                                     random_state=99)

    X = np.vstack([X_normal, X_anomalies]).astype(np.float32)
    y = np.array([0] * n_normal + [1] * n_anomalies)  # 1 = anomaly

    # Training set: unsupervised — no labels used during fitting
    X_train = X_normal.astype(np.float32)
    return X, y, X_train


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: key parameters to tune
IFOREST_CONFIG = {
    "n_estimators":  200,           # number of isolation trees
    "max_samples":   "auto",        # 'auto' = min(256, n_samples); try 64, 128, 256
    "contamination": 0.047,         # Placeholder: set to expected anomaly rate (50/1050 ≈ 0.047)
    "max_features":  1.0,           # fraction of features per tree
    "bootstrap":     False,         # True enables OOB estimation
    "n_jobs":        -1,
    "random_state":  42,
}


def build_pipeline(contamination: float = 0.05) -> tuple:
    """
    Build a StandardScaler + IsolationForest pipeline.

    Note: IsolationForest is not a standard sklearn pipeline estimator
    (no transform step), so we return scaler + model separately.

    Args:
        contamination: expected fraction of anomalies

    Returns:
        scaler, model
    """
    config = {**IFOREST_CONFIG, "contamination": contamination}
    scaler = StandardScaler()
    model  = IsolationForest(**config)
    return scaler, model


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, scaler, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate anomaly detection performance.

    sklearn IsolationForest conventions:
        predict():   +1 = normal (inlier), -1 = anomaly (outlier)
        score_samples(): raw anomaly score (more negative = more anomalous)
        decision_function(): centered anomaly score (negative = anomaly)

    We convert to standard convention: 1=anomaly, 0=normal.

    Metrics:
        ROC-AUC: threshold-independent discrimination ability
        AP (Avg Precision): area under precision-recall curve (better for imbalanced)
        Precision/Recall at predicted contamination threshold

    Args:
        model:  fitted IsolationForest
        scaler: fitted StandardScaler
        X:      all samples (normal + anomalies)
        y:      true labels (0=normal, 1=anomaly)

    Returns:
        dict of metrics
    """
    X_scaled = scaler.transform(X)

    # Raw anomaly scores: more negative = more anomalous
    scores = -model.score_samples(X_scaled)   # negate: higher = more anomalous

    # Predicted labels (at contamination threshold)
    pred_raw  = model.predict(X_scaled)       # +1 or -1
    y_pred    = (pred_raw == -1).astype(int)  # 1 = anomaly

    auc = roc_auc_score(y, scores)
    ap  = average_precision_score(y, scores)
    cm  = confusion_matrix(y, y_pred)

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {"roc_auc": auc, "avg_precision": ap,
            "precision": precision, "recall": recall}


# ─────────────────────────────────────────────
# THRESHOLD ANALYSIS
# ─────────────────────────────────────────────

def find_optimal_threshold(model, scaler, X: np.ndarray, y: np.ndarray) -> float:
    """
    Find the score threshold that maximizes F1 score.

    In production, the contamination parameter sets the threshold automatically.
    Use this function when:
        - The exact anomaly rate is unknown
        - You want to tune precision/recall trade-off explicitly
        - Validating that the contamination estimate is correct

    Fine-tuning tip: plot the precision-recall curve and pick the operating
    point that matches your business tolerance for false positives vs. false negatives.
    High-stakes systems (fraud, security) → optimize recall; others → balance F1.

    Args:
        model:  fitted IsolationForest
        scaler: fitted StandardScaler
        X:      all samples
        y:      true binary labels

    Returns:
        optimal score threshold
    """
    X_scaled = scaler.transform(X)
    scores   = -model.score_samples(X_scaled)

    precisions, recalls, thresholds = precision_recall_curve(y, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx   = np.argmax(f1s[:-1])
    best_thresh = thresholds[best_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1],    label="Recall")
    plt.plot(thresholds, f1s[:-1],        label="F1")
    plt.axvline(best_thresh, color="red", linestyle="--",
                label=f"Best threshold: {best_thresh:.4f}")
    plt.xlabel("Anomaly Score Threshold")
    plt.title("Isolation Forest — Threshold Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig("iforest_threshold.png", dpi=150)
    plt.show()

    print(f"Optimal threshold: {best_thresh:.4f}  (F1: {f1s[best_idx]:.4f})")
    return best_thresh


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_anomaly_scores(model, scaler, X: np.ndarray, y: np.ndarray) -> None:
    """
    Histogram of anomaly scores for normal vs. anomalous samples.

    Good separation between distributions → model is discriminating well.
    Overlapping distributions → model is struggling; try more features or tuning.
    """
    X_scaled = scaler.transform(X)
    scores   = -model.score_samples(X_scaled)

    plt.figure(figsize=(9, 4))
    plt.hist(scores[y == 0], bins=50, alpha=0.6, label="Normal",   color="steelblue")
    plt.hist(scores[y == 1], bins=50, alpha=0.6, label="Anomaly",  color="tomato")
    threshold = np.percentile(scores, 100 * (1 - IFOREST_CONFIG["contamination"]))
    plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.3f})")
    plt.xlabel("Anomaly Score (higher = more anomalous)")
    plt.ylabel("Count")
    plt.title("Isolation Forest — Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("iforest_scores.png", dpi=150)
    plt.show()


def plot_2d_decision_boundary(model, scaler, X: np.ndarray, y: np.ndarray) -> None:
    """
    2D decision boundary using first 2 PCA components (or raw features if 2D).

    Shading indicates anomaly score; red dots = predicted anomalies.
    """
    if X.shape[1] > 2:
        pca   = PCA(n_components=2, random_state=42)
        X_2d  = pca.fit_transform(scaler.transform(X))
        title = "Isolation Forest — Decision Boundary (PCA)"
    else:
        X_2d  = scaler.transform(X)
        title = "Isolation Forest — Decision Boundary"

    # Fit a 2D model for the boundary plot
    model_2d = IsolationForest(n_estimators=100, contamination=IFOREST_CONFIG["contamination"],
                                random_state=42)
    scaler_2d = StandardScaler()
    X_2d_scaled = scaler_2d.fit_transform(X_2d)
    model_2d.fit(X_2d_scaled)

    xx, yy = np.meshgrid(np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 100),
                          np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 100))
    Z = -model_2d.score_samples(scaler_2d.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z, levels=20, cmap="RdYlBu_r", alpha=0.7)
    plt.colorbar(label="Anomaly Score")
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c="steelblue", s=8,  alpha=0.5, label="Normal")
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c="red",       s=40, alpha=0.9, label="Anomaly", marker="x")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("iforest_boundary.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model, scaler, X: np.ndarray, y: np.ndarray) -> None:
    """Display confusion matrix."""
    X_scaled = scaler.transform(X)
    y_pred   = (model.predict(X_scaled) == -1).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred),
                           display_labels=["Normal", "Anomaly"]).plot(cmap="Oranges")
    plt.title("Isolation Forest — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("iforest_confusion.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Isolation Forest Anomaly Detection ===")

    X, y, X_train = generate_anomaly_data(n_normal=1000, n_anomalies=50)
    print(f"Dataset: {X.shape[0]} samples ({y.sum()} anomalies) | {X.shape[1]} features")

    # Fit on normal-only training data (unsupervised)
    scaler, model = build_pipeline(contamination=50 / len(X))
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled)

    print("\nEvaluation (on all samples including anomalies):")
    evaluate(model, scaler, X, y)

    # Threshold analysis using labeled data (if available)
    find_optimal_threshold(model, scaler, X, y)

    # Visualizations
    plot_anomaly_scores(model, scaler, X, y)
    plot_2d_decision_boundary(model, scaler, X, y)
    plot_confusion_matrix(model, scaler, X, y)


if __name__ == "__main__":
    main()
