"""
One-Class SVM Anomaly Detection
=================================
Learns a tight boundary around normal data in a kernel-induced feature space.
Only trained on normal samples — anything outside the boundary is an anomaly.

Formulation (ν-SVM, Schölkopf et al. 2001):
    Maximize the margin from the origin to the data in feature space, with
    at most ν fraction of training points outside the boundary:

        min  (1/2)||w||² - ρ + (1/νn) Σ ξᵢ
        s.t. w·Φ(xᵢ) ≥ ρ - ξᵢ,  ξᵢ ≥ 0

    Decision function: f(x) = sign(w·Φ(x) - ρ)
        +1 = inlier (normal), -1 = outlier (anomaly)

Kernel choices:
    - RBF (default): works well for most datasets; bandwidth γ controls
      tightness. Small γ → large smooth boundary; large γ → tight, bumpy.
    - Polynomial: useful for structured data.
    - Linear: fast but less flexible; equivalent to novelty detection in PCA space.

Alternative: SGDOneClassSVM
    Linear kernel variant supporting incremental learning (partial_fit).
    Use for streaming data or very large datasets.

Use Cases:
    - Network anomaly detection (known-good traffic profiling)
    - Fraud detection (model legitimate transactions)
    - Fault isolation (learn normal operational envelope)
    - Document novelty detection

Advantages:
    - Probabilistic calibration not required — direct boundary
    - RBF kernel captures nonlinear normal manifolds
    - No assumptions about data distribution

Disadvantages:
    - Sensitive to ν and γ hyperparameters
    - Poor scaling to very high dimensions (use Autoencoder AE instead)
    - Feature scaling is critical

Fine-Tuning Recommendations:
    - nu (ν): upper bound on training outlier fraction AND lower bound on
      support vector fraction. Start at 0.05. If many false positives → increase.
      If many false negatives → decrease. Typical range: 0.01–0.2.
    - kernel: 'rbf' for most cases. 'poly' with degree=3 for structured patterns.
    - gamma: 'scale' (1/(n_features * X.var())) is a good default. Tune via
      GridSearch: [1e-3, 1e-2, 0.1, 1.0, 'scale', 'auto'].
    - Always StandardScale before fitting — OCSVM is distance-based.
    - For high-dim data: apply PCA first (retain 95% variance) to reduce noise.
    - Threshold: decision_function() gives signed distance to hyperplane.
      Default decision boundary is at 0. Can shift by percentile if needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline

from deepdraft.core import save_figure
from deepdraft.evaluation import train_val_test_split


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

    Replace with your own dataset:
        import pandas as pd
        df = pd.read_csv("normal_operations.csv")
        X = df.values.astype(np.float32)
        # No labels needed for training; labels used only for evaluation

    Args:
        n_normal:     number of normal training samples
        n_anomalies:  number of anomalies (evaluation only)
        n_features:   feature dimensionality
        anomaly_type: 'random' (uniform outliers) | 'cluster' (tight outlier cluster)

    Returns:
        X:       all samples
        y:       labels (0=normal, 1=anomaly)
    """
    X_normal, _ = make_blobs(n_samples=n_normal, n_features=n_features,
                              centers=3, cluster_std=0.8, random_state=42)
    if anomaly_type == "random":
        rng = np.random.RandomState(7)
        X_anomalies = rng.uniform(
            low=X_normal.min(axis=0) * 3,
            high=X_normal.max(axis=0) * 3,
            size=(n_anomalies, n_features)
        )
    else:
        X_anomalies, _ = make_blobs(n_samples=n_anomalies, n_features=n_features,
                                     centers=[[10] * n_features], cluster_std=0.3,
                                     random_state=99)
    X = np.vstack([X_normal, X_anomalies]).astype(np.float32)
    y = np.array([0] * n_normal + [1] * n_anomalies)
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune nu (anomaly rate estimate) and gamma (RBF bandwidth)
OCSVM_CONFIG = {
    "kernel":  "rbf",
    "nu":      0.05,     # Placeholder: upper bound on outlier fraction (0.01–0.2)
    "gamma":   "scale",  # Placeholder: 'scale', 'auto', or float (e.g., 0.1)
    "cache_size": 500,
}


def build_pipeline(config: dict = None) -> Pipeline:
    """
    Build StandardScaler + OneClassSVM pipeline.

    Args:
        config: hyperparameter dict; defaults to OCSVM_CONFIG

    Returns:
        sklearn Pipeline (can be fit/predict directly)
    """
    if config is None:
        config = OCSVM_CONFIG
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm",  OneClassSVM(**config)),
    ])


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate One-Class SVM performance.

    sklearn OneClassSVM conventions:
        predict():           +1 = normal (inlier), -1 = anomaly (outlier)
        decision_function(): signed distance to boundary
                             positive = inside (normal), negative = outside (anomaly)

    We negate decision_function() so that higher score = more anomalous.

    Args:
        pipeline: fitted Pipeline(scaler, ocsvm)
        X:        held-out samples, none of them seen during fitting
        y:        true labels (0=normal, 1=anomaly)

    Returns:
        dict of metrics
    """
    # Anomaly scores: more positive = more anomalous
    scores  = -pipeline.decision_function(X)
    pred_raw = pipeline.predict(X)             # +1 or -1
    y_pred   = (pred_raw == -1).astype(int)    # 1 = anomaly

    auc = roc_auc_score(y, scores)
    ap  = average_precision_score(y, scores)
    cm  = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {"roc_auc": auc, "avg_precision": ap,
            "precision": precision, "recall": recall}


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def nu_gamma_sweep(X_train: np.ndarray, X_val: np.ndarray,
                   y_val: np.ndarray) -> dict:
    """
    Grid sweep over nu and gamma — plot heatmap of ROC-AUC.

    One-Class SVM has no sklearn-compatible GridSearchCV because there are
    no true labels during training. We score each setting on a small labeled
    VALIDATION set, never the test set: choosing hyperparameters by test score
    would make the reported test score optimistic.

    Returns:
        the best {"nu": ..., "gamma": ...} by validation ROC-AUC
    """
    nus    = [0.01, 0.05, 0.1, 0.15, 0.2]
    gammas = [0.001, 0.01, 0.1, 1.0, "scale"]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_tr_sc  = scaler.transform(X_train)
    X_ev_sc  = scaler.transform(X_val)

    results = np.zeros((len(nus), len(gammas)))
    for i, nu in enumerate(nus):
        for j, gamma in enumerate(gammas):
            model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            model.fit(X_tr_sc)
            scores = -model.decision_function(X_ev_sc)
            try:
                results[i, j] = roc_auc_score(y_val, scores)
            except Exception:
                results[i, j] = 0.5

    gamma_labels = [str(g) for g in gammas]
    plt.figure(figsize=(8, 5))
    im = plt.imshow(results, aspect="auto", cmap="viridis",
                    vmin=0.5, vmax=1.0)
    plt.colorbar(im, label="Validation ROC-AUC")
    plt.xticks(range(len(gammas)), gamma_labels, fontsize=9)
    plt.yticks(range(len(nus)), [str(n) for n in nus])
    plt.xlabel("gamma")
    plt.ylabel("nu")
    plt.title("One-Class SVM — nu/gamma ROC-AUC Heatmap")
    for i in range(len(nus)):
        for j in range(len(gammas)):
            plt.text(j, i, f"{results[i,j]:.2f}", ha="center", va="center",
                     color="white" if results[i,j] < 0.75 else "black", fontsize=8)
    plt.tight_layout()
    save_figure("ocsvm_heatmap.png", dpi=150)

    best_i, best_j = np.unravel_index(np.argmax(results), results.shape)
    print(f"Best: nu={nus[best_i]}, gamma={gammas[best_j]}, "
          f"validation ROC-AUC={results[best_i, best_j]:.4f}")
    return {"nu": nus[best_i], "gamma": gammas[best_j]}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_decision_scores(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """Histogram of decision function scores for normal vs. anomalous."""
    scores = -pipeline.decision_function(X)
    plt.figure(figsize=(9, 4))
    plt.hist(scores[y == 0], bins=50, alpha=0.6, label="Normal",  color="steelblue")
    plt.hist(scores[y == 1], bins=50, alpha=0.6, label="Anomaly", color="tomato")
    plt.axvline(0.0, color="black", linestyle="--", label="Default boundary (0)")
    plt.xlabel("Anomaly Score (negated decision function)")
    plt.ylabel("Count")
    plt.title("One-Class SVM — Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    save_figure("ocsvm_scores.png", dpi=150)


def plot_2d_boundary(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """
    2D decision boundary using first 2 PCA components.
    Contours show the anomaly boundary; red = predicted anomaly.
    """
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)

    if X.shape[1] > 2:
        pca  = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        title = "One-Class SVM — Boundary (PCA 2D)"
    else:
        X_2d  = X_scaled
        title = "One-Class SVM — Decision Boundary"

    scaler_2d = StandardScaler()
    X_2d_sc   = scaler_2d.fit_transform(X_2d)
    model_2d  = OneClassSVM(kernel="rbf",
                             nu=OCSVM_CONFIG["nu"],
                             gamma=OCSVM_CONFIG["gamma"])
    model_2d.fit(X_2d_sc)

    xx, yy = np.meshgrid(
        np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 200),
        np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 200),
    )
    Z = -model_2d.decision_function(scaler_2d.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z, levels=20, cmap="RdYlBu_r", alpha=0.7)
    plt.colorbar(label="Anomaly Score")
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c="steelblue", s=8,  alpha=0.5, label="Normal")
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c="red",       s=40, alpha=0.9, label="Anomaly", marker="x")
    plt.contour(xx, yy, Z, levels=[0], colors="black", linewidths=1.5)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_figure("ocsvm_boundary.png", dpi=150)


def plot_confusion_matrix(pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """Display confusion matrix at default decision boundary."""
    y_pred = (pipeline.predict(X) == -1).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred),
                           display_labels=["Normal", "Anomaly"]).plot(cmap="Oranges")
    plt.title("One-Class SVM — Confusion Matrix")
    plt.tight_layout()
    save_figure("ocsvm_confusion.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== One-Class SVM Anomaly Detection ===")

    # Placeholder: replace with your dataset
    X, y = generate_anomaly_data(n_normal=1000, n_anomalies=50)
    print(f"Dataset: {X.shape[0]} samples ({y.sum()} anomalies) | {X.shape[1]} features")

    # Stratified train / validation / test. One-Class SVM is a novelty
    # detector: it learns the boundary of clean, normal data, so keep only the
    # normal rows of the training split. Validation and test keep both
    # classes and contain no training rows.
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.2, test_size=0.3, stratify=y
    )
    X_train = X_train[y_train == 0]

    pipeline = build_pipeline()
    pipeline.fit(X_train)

    print("\nTest set, default config:")
    evaluate(pipeline, X_test, y_test)

    # Hyperparameter sweep, scored on the labeled validation set
    print("\nRunning nu/gamma sweep...")
    best = nu_gamma_sweep(X_train, X_val, y_val)
    tuned = build_pipeline({**OCSVM_CONFIG, **best})
    tuned.fit(X_train)
    print("\nTest set, tuned config:")
    evaluate(tuned, X_test, y_test)

    # Visualizations (held-out test data)
    plot_decision_scores(pipeline, X_test, y_test)
    plot_2d_boundary(pipeline, X_test, y_test)
    plot_confusion_matrix(pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
