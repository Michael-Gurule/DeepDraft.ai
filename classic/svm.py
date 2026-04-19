"""
Support Vector Machine (SVM)
=============================
A powerful supervised learning algorithm that finds the maximum-margin hyperplane
separating classes. Can handle both linear and non-linear decision boundaries via
the kernel trick. Also supports regression (SVR).

Use Cases:
    - High-dimensional classification (text, genomics, image features)
    - Small-to-medium datasets where maximum-margin separation matters
    - Non-linear classification via kernel functions
    - SVR for regression tasks with outlier robustness

Advantages:
    - Effective in high-dimensional spaces (even when features > samples)
    - Memory efficient — only support vectors are stored
    - Versatile via kernel choice

Limitations:
    - Scales poorly to very large datasets (O(n²) to O(n³) training time)
    - Requires careful feature scaling
    - Probability estimates require Platt scaling (expensive, less reliable)
    - No direct feature importance — use kernel SHAP for interpretability

Fine-Tuning Recommendations:
    - Kernel selection:
        'linear'  — fast, interpretable, good for linearly separable or high-dim data
        'rbf'     — most versatile; controlled by gamma (bandwidth)
        'poly'    — captures polynomial feature interactions; degree param
        'sigmoid' — less common; behaves like a neural network activation
    - C parameter: controls soft-margin tolerance.
        Small C → wider margin, more misclassifications (underfitting)
        Large C → narrower margin, fewer misclassifications (overfitting)
    - gamma ('rbf'/'poly'/'sigmoid'): controls influence radius of each sample.
        'scale' (default) = 1/(n_features * X.var())
        'auto'            = 1/n_features
        Small gamma → smooth, broad decision boundary
        Large gamma → complex, tightly fitted boundary (overfitting risk)
    - Always normalize features — SVM is very sensitive to feature scale.
    - Use LinearSVC for large datasets (much faster than SVC with linear kernel).
    - class_weight='balanced' for imbalanced classes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 500, n_features: int = 8):
    """
    Generate synthetic binary classification data.

    Replace with your own loading logic. Feature scaling is handled in the
    pipeline, so raw values are fine here.

    Args:
        n_samples:  number of data points
        n_features: number of input features

    Returns:
        X: feature matrix
        y: binary class labels (0/1)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=1,
        class_sep=0.8,
        random_state=42,
    )
    return X, y


def load_regression_data(n_samples: int = 300):
    """
    Generate synthetic regression data for SVR demonstration.

    Returns:
        X: feature matrix
        y: continuous target
    """
    X, y = make_regression(n_samples=n_samples, n_features=5, noise=15.0, random_state=42)
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: adjust kernel and parameters here
SVC_CONFIG = {
    "kernel": "rbf",        # 'linear', 'rbf', 'poly', 'sigmoid'
    "C": 1.0,               # margin/misclassification trade-off
    "gamma": "scale",       # rbf/poly/sigmoid bandwidth; 'scale', 'auto', or float
    "degree": 3,            # only used when kernel='poly'
    "probability": True,    # enables predict_proba; costs ~5x training time
    "class_weight": None,   # set to 'balanced' for imbalanced classes
    "random_state": 42,
}

SVR_CONFIG = {
    "kernel": "rbf",
    "C": 1.0,
    "epsilon": 0.1,         # width of the epsilon-insensitive tube; larger = more tolerance
    "gamma": "scale",
}


def build_svc_pipeline(**overrides) -> Pipeline:
    """
    Build StandardScaler → SVC pipeline.

    Feature scaling is mandatory for SVM — the algorithm measures distances
    and unscaled features with large magnitudes dominate the margin calculation.

    Args:
        **overrides: override SVC_CONFIG parameters

    Returns:
        sklearn Pipeline
    """
    config = {**SVC_CONFIG, **overrides}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(**config)),
    ])


def build_svr_pipeline(**overrides) -> Pipeline:
    """
    Build StandardScaler → SVR pipeline for regression tasks.

    SVR finds a tube of width ±epsilon around the regression function and
    only penalizes points outside the tube.

    Args:
        **overrides: override SVR_CONFIG parameters

    Returns:
        sklearn Pipeline
    """
    config = {**SVR_CONFIG, **overrides}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(**config)),
    ])


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute classification metrics for SVC.

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

    metrics = {"accuracy": acc}

    if hasattr(model.named_steps["model"], "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        metrics["roc_auc"] = auc
        print(f"  ROC-AUC: {auc:.4f}")

    return metrics


def cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    K-fold cross-validation.

    Fine-tuning tip: SVM is sensitive to hyperparameters — always cross-validate
    after tuning C and gamma. A large standard deviation across folds suggests
    that the model is overfitting or the dataset is small.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"\n{cv}-Fold CV Accuracy: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_svc(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Grid search over C, gamma, and kernel for SVC.

    Search strategy:
        1. Start with a coarse grid on a log scale for C and gamma.
        2. Identify the best region, then narrow down with a finer grid.
        3. For large datasets, switch to RandomizedSearchCV.

    Fine-tuning tip: if the best params are at the boundary of your search
    range, extend the grid in that direction.

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted pipeline
    """
    pipeline = build_svc_pipeline(probability=False)  # faster without Platt scaling
    param_grid = [
        {
            "model__kernel": ["rbf"],
            "model__C":      [0.1, 1.0, 10.0, 100.0],
            "model__gamma":  ["scale", "auto", 0.01, 0.001],
        },
        {
            "model__kernel": ["linear"],
            "model__C":      [0.1, 1.0, 10.0],
        },
    ]
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_decision_boundary_2d(model, X: np.ndarray, y: np.ndarray) -> None:
    """
    2D decision boundary visualization (requires exactly 2 features).

    Shows how the SVM separates classes in feature space. The margin and
    support vectors are conceptually represented by the decision boundary shape.

    Note: for datasets with more than 2 features, consider PCA reduction first.
    """
    if X.shape[1] != 2:
        print("Decision boundary plot requires exactly 2 features. Skipping.")
        return

    h = 0.02  # mesh step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", linewidths=0.3)
    plt.title("SVM — Decision Boundary")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.tight_layout()
    plt.savefig("svm_decision_boundary.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Display confusion matrix for the classifier."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title("SVM — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("svm_confusion.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification ──────────────────────────────────────────────
    print("=== SVM Classification ===")
    X, y = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_svc_pipeline()
    pipeline.fit(X_train, y_train)

    print("\nTest Set Metrics:")
    evaluate_classifier(pipeline, X_test, y_test)

    cross_validate(pipeline, X, y)

    print("\n=== Hyperparameter Tuning ===")
    best_model = tune_svc(X_train, y_train)
    print("\nBest SVC — Test Metrics:")
    evaluate_classifier(best_model, X_test, y_test)

    plot_confusion_matrix(pipeline, X_test, y_test)

    # 2D boundary demo (subsample 2 features)
    plot_decision_boundary_2d(build_svc_pipeline().fit(X_train[:, :2], y_train),
                               X_train[:, :2], y_train)

    # ── Regression (SVR) ────────────────────────────────────────────
    print("\n=== SVR (Support Vector Regression) ===")
    from sklearn.metrics import mean_squared_error, r2_score

    Xr, yr = load_regression_data()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )

    svr_pipeline = build_svr_pipeline()
    svr_pipeline.fit(Xr_train, yr_train)

    yr_pred = svr_pipeline.predict(Xr_test)
    print(f"  RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f}")
    print(f"  R²:   {r2_score(yr_test, yr_pred):.4f}")


if __name__ == "__main__":
    main()
