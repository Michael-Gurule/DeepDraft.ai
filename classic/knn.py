"""
K-Nearest Neighbors (KNN)
=========================
A non-parametric, instance-based (lazy) learning algorithm. Makes predictions
by finding the K most similar training samples (neighbors) and aggregating
their labels (classification) or values (regression). No explicit training
phase — the model IS the training data.

Use Cases:
    - Recommendation systems (user/item similarity)
    - Anomaly detection (samples far from any neighbors)
    - Baseline classification for small-to-medium datasets
    - Image recognition on small datasets
    - Collaborative filtering

Advantages:
    - No training time (lazy learner)
    - Naturally handles multi-class problems
    - Adapts to complex, non-linear decision boundaries
    - Easy to understand and explain

Limitations:
    - Slow inference: O(n) per prediction (mitigated by ball tree / kd-tree)
    - Memory intensive: stores all training data
    - Degrades with high-dimensional data (curse of dimensionality)
    - Sensitive to irrelevant and scaled features

Fine-Tuning Recommendations:
    - k (n_neighbors): most important parameter.
        Small k → complex boundary, high variance (overfitting)
        Large k → smooth boundary, high bias (underfitting)
        Rule of thumb: start with k = sqrt(n_samples); tune via CV.
    - metric: Euclidean ('minkowski' p=2) is standard; use 'manhattan' (p=1)
      for sparse/high-dimensional data; 'cosine' for text/embeddings.
    - weights: 'uniform' = all neighbors equal; 'distance' = closer neighbors
      have more influence (generally better).
    - algorithm: 'auto' (auto-selects); 'ball_tree'/'kd_tree' for large datasets;
      'brute' for small datasets or high-dimensional data.
    - Feature scaling: CRITICAL — KNN uses distance, so unscaled features
      with large ranges dominate the distance metric entirely.
    - Dimensionality reduction (PCA) before KNN improves both speed and accuracy
      in high-dimensional settings.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification, make_regression

from deepdraft.core import save_figure


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 600, n_features: int = 6):
    """
    Generate synthetic classification data.

    Replace with your own:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["label"]).values
        y = df["label"].values

    Args:
        n_samples:  number of data points
        n_features: number of input features

    Returns:
        X: feature matrix
        y: class labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


def load_regression_data(n_samples: int = 300):
    """
    Generate synthetic regression data for KNeighborsRegressor demo.

    Returns:
        X: feature matrix
        y: continuous target
    """
    X, y = make_regression(n_samples=n_samples, n_features=4, noise=10.0, random_state=42)
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune k and distance metric for your dataset
CLASSIFIER_CONFIG = {
    "n_neighbors": 7,        # k; use odd numbers for binary to avoid ties
    "weights": "distance",   # 'uniform' or 'distance' (distance weighting usually better)
    "metric": "minkowski",   # 'minkowski' (p=2 = Euclidean), 'manhattan', 'cosine'
    "p": 2,                  # Minkowski power: 1=Manhattan, 2=Euclidean
    "algorithm": "auto",     # 'auto', 'ball_tree', 'kd_tree', 'brute'
    "n_jobs": -1,
}

REGRESSOR_CONFIG = {
    "n_neighbors": 7,
    "weights": "distance",
    "metric": "minkowski",
    "p": 2,
    "algorithm": "auto",
    "n_jobs": -1,
}


def build_classifier_pipeline(**overrides) -> Pipeline:
    """
    Build StandardScaler → KNeighborsClassifier pipeline.

    Feature scaling is mandatory for KNN. Without it, a feature measured
    in thousands (e.g., income) will completely dominate distance calculations
    over a feature measured in ones (e.g., number of children).

    Args:
        **overrides: override CLASSIFIER_CONFIG parameters

    Returns:
        sklearn Pipeline
    """
    config = {**CLASSIFIER_CONFIG, **overrides}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(**config)),
    ])


def build_regressor_pipeline(**overrides) -> Pipeline:
    """
    Build StandardScaler → KNeighborsRegressor pipeline.

    Args:
        **overrides: override REGRESSOR_CONFIG parameters

    Returns:
        sklearn Pipeline
    """
    config = {**REGRESSOR_CONFIG, **overrides}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(**config)),
    ])


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a KNN classifier on the test set.

    Args:
        model:  fitted pipeline
        X_test: test features
        y_test: true labels

    Returns:
        dict of metrics
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"accuracy": acc}


def cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    K-fold cross-validation.

    Fine-tuning tip: for KNN, cross-validate over a range of k values using
    the k_selection_curve() function below — this is the most effective way
    to choose k.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"\n{cv}-Fold CV Accuracy: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_k(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Grid search over k (n_neighbors), weights, and metric.

    This is the most important hyperparameter search for KNN.
    k directly controls the bias-variance trade-off.

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted pipeline
    """
    pipeline = build_classifier_pipeline()
    param_grid = {
        "model__n_neighbors": list(range(1, 31, 2)),  # odd values 1–29
        "model__weights":     ["uniform", "distance"],
        "model__metric":      ["minkowski", "manhattan"],
    }
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_k_selection_curve(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            max_k: int = 30) -> None:
    """
    Plot train and test accuracy for a range of k values.

    The k-selection curve is the primary diagnostic tool for KNN.
    The optimal k is where test accuracy peaks without underfitting.

    Pattern to look for:
        - k=1: often 100% train accuracy (memorizes training set), low test accuracy
        - Increasing k: test accuracy rises (reducing variance)
        - Too large k: test accuracy drops (increasing bias, underfitting)
        - Best k: near the peak of the test accuracy curve

    Fine-tuning tip: if test accuracy keeps rising to the maximum k tested,
    extend max_k or consider that KNN may not be the right model.

    Args:
        X_train, y_train: training data
        X_test, y_test:   test data
        max_k:            maximum k to evaluate
    """
    k_range = range(1, max_k + 1)
    train_acc, test_acc = [], []

    for k in k_range:
        pipeline = build_classifier_pipeline(n_neighbors=k)
        pipeline.fit(X_train, y_train)
        train_acc.append(pipeline.score(X_train, y_train))
        test_acc.append(pipeline.score(X_test, y_test))

    best_k = k_range[np.argmax(test_acc)]

    plt.figure(figsize=(10, 4))
    plt.plot(k_range, train_acc, label="Train Accuracy")
    plt.plot(k_range, test_acc, label="Test Accuracy")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Accuracy")
    plt.title("KNN — k Selection Curve")
    plt.legend()
    plt.tight_layout()
    save_figure("knn_k_selection.png", dpi=150)

    print(f"Best k: {best_k}  (test accuracy: {max(test_acc):.4f})")


def plot_decision_boundary_2d(pipeline, X: np.ndarray, y: np.ndarray) -> None:
    """
    2D decision boundary (requires exactly 2 features).

    KNN boundaries are non-parametric and can be very irregular for small k,
    becoming smoother as k increases.
    """
    if X.shape[1] != 2:
        print("Decision boundary plot requires exactly 2 features. Skipping.")
        return

    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", edgecolors="k", linewidths=0.3, s=20)
    plt.title(f"KNN — Decision Boundary (k={pipeline.named_steps['model'].n_neighbors})")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.tight_layout()
    save_figure("knn_decision_boundary.png", dpi=150)


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Display confusion matrix."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title("KNN — Confusion Matrix")
    plt.tight_layout()
    save_figure("knn_confusion.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification ──────────────────────────────────────────────
    print("=== K-Nearest Neighbors Classification ===")
    X, y = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_classifier_pipeline()
    pipeline.fit(X_train, y_train)

    print("\nTest Set Metrics:")
    evaluate_classifier(pipeline, X_test, y_test)
    cross_validate(pipeline, X, y)

    # k selection curve — most important diagnostic
    print("\n=== k Selection Analysis ===")
    plot_k_selection_curve(X_train, y_train, X_test, y_test)

    # Grid search
    print("\n=== Hyperparameter Tuning ===")
    best_model = tune_k(X_train, y_train)
    print("\nBest KNN — Test Metrics:")
    evaluate_classifier(best_model, X_test, y_test)

    # 2D boundary (subsample 2 features for visualization)
    pipeline_2d = build_classifier_pipeline()
    pipeline_2d.fit(X_train[:, :2], y_train)
    plot_decision_boundary_2d(pipeline_2d, X_train[:, :2], y_train)
    plot_confusion_matrix(pipeline, X_test, y_test)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== KNN Regression ===")
    Xr, yr = load_regression_data()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg_pipeline = build_regressor_pipeline()
    reg_pipeline.fit(Xr_train, yr_train)
    yr_pred = reg_pipeline.predict(Xr_test)
    print(f"  RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f}")
    print(f"  R²:   {r2_score(yr_test, yr_pred):.4f}")


if __name__ == "__main__":
    main()
