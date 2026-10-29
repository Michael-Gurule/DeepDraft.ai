"""
Random Forest
=============
An ensemble method that builds many decision trees on bootstrapped subsets of
the data and averages their predictions (bagging + random feature selection).
Reduces the high variance of individual trees while maintaining low bias.

Use Cases:
    - Tabular data classification and regression (strong out-of-box baseline)
    - Feature importance ranking
    - Robust predictions with built-in uncertainty via OOB score
    - Missing value imputation (via proximity matrices)

Advantages:
    - Very strong out-of-the-box with little tuning
    - Built-in estimate of generalization via Out-Of-Bag (OOB) score
    - Parallelizable training (n_jobs=-1)
    - Handles non-linear relationships and feature interactions
    - Robust to outliers and noise

Limitations:
    - Less interpretable than a single decision tree
    - Slower inference than linear models (many trees to query)
    - Memory intensive for very large n_estimators
    - Can be outperformed by gradient boosting on structured data

Fine-Tuning Recommendations:
    - n_estimators: more trees = lower variance, diminishing returns after ~200.
      Start at 100, increase until OOB score plateaus.
    - max_features: most important tuning knob.
        'sqrt'  (default for classification) — reduces correlation between trees
        'log2'  — more aggressive feature sampling
        float   — fraction of features to consider (e.g., 0.5)
        None    — use all features (not recommended; trees become correlated)
    - max_depth: None (unlimited) works well with many trees; pruning via
      min_samples_leaf is often sufficient.
    - min_samples_leaf: 1 (default) can overfit; try 2–10 for noisy data.
    - bootstrap: True (default) — enables OOB score; set False for subspace methods.
    - oob_score: True gives a free generalization estimate without a validation set.
    - class_weight: 'balanced' or 'balanced_subsample' for imbalanced classes.
    - warm_start: True allows incremental training (add trees without retraining).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification, make_regression
from scipy.stats import randint

from deepdraft.core import fast_or, save_figure


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 1000, n_features: int = 15):
    """
    Generate synthetic classification data.

    Replace with your own dataset loading:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["label"]).values
        y = df["label"].values
        feature_names = df.drop(columns=["label"]).columns.tolist()

    Returns:
        X: feature matrix
        y: class labels
        feature_names: list of feature name strings
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=3,
        n_classes=2,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def load_regression_data(n_samples: int = 600):
    """
    Generate synthetic regression data for RandomForestRegressor demo.

    Returns:
        X: feature matrix
        y: continuous target
    """
    X, y = make_regression(
        n_samples=n_samples, n_features=10, n_informative=6, noise=10.0, random_state=42
    )
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: adjust these parameters for your dataset
CLASSIFIER_CONFIG = {
    "n_estimators": fast_or(200, 20),  # number of trees; more = better but slower
    "max_features": "sqrt",       # features per split: 'sqrt', 'log2', float, or None
    "max_depth": None,            # None = unlimited; restrict for interpretability
    "min_samples_leaf": 1,        # minimum samples at leaf; increase for noisy data
    "min_samples_split": 2,       # minimum samples to consider a split
    "bootstrap": True,            # sample with replacement (enables OOB score)
    "oob_score": True,            # free generalization estimate using out-of-bag samples
    "class_weight": None,         # 'balanced' or 'balanced_subsample' for imbalance
    "n_jobs": -1,                 # use all CPU cores
    "random_state": 42,
}

REGRESSOR_CONFIG = {
    "n_estimators": fast_or(200, 20),
    "max_features": 1.0,          # fraction of features; try 0.33–1.0 for regression
    "max_depth": None,
    "min_samples_leaf": 1,
    "bootstrap": True,
    "oob_score": True,
    "n_jobs": -1,
    "random_state": 42,
}


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a RandomForestClassifier.

    Also prints the OOB score if oob_score=True — this is the model's own
    estimate of generalization computed during training on unseen samples.

    Args:
        model:  fitted RandomForestClassifier
        X_test: test features
        y_test: true labels

    Returns:
        dict of metrics
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if hasattr(model, "oob_score_"):
        print(f"  OOB Score: {model.oob_score_:.4f}")

    print(f"  Test Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"accuracy": acc}


def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    K-fold cross-validation.

    Fine-tuning tip: if OOB score is significantly higher than CV score,
    the CV folds may be too small — try fewer folds or StratifiedKFold.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\n{cv}-Fold CV Accuracy: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_random_search(X_train: np.ndarray, y_train: np.ndarray,
                        n_iter: int = 30) -> RandomForestClassifier:
    """
    Randomized hyperparameter search (faster than grid search for RF).

    RandomizedSearchCV samples from distributions rather than exhaustive grid,
    making it practical for RF which has many hyperparameters.

    Fine-tuning strategy:
        1. Run a broad random search to identify promising regions.
        2. Narrow down with a grid search around the best params.
        3. Verify with cross-validation on the final configuration.

    Args:
        X_train: training features
        y_train: training labels
        n_iter:  number of parameter combinations to try

    Returns:
        best fitted RandomForestClassifier
    """
    param_dist = {
        "n_estimators":     randint(50, 400),
        "max_features":     ["sqrt", "log2", 0.3, 0.5, 0.7],
        "max_depth":        [None, 5, 10, 20, 30],
        "min_samples_leaf": randint(1, 20),
    }
    rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=n_iter, cv=5, scoring="accuracy",
        n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_gini_importance(model, feature_names: list, top_n: int = 15) -> None:
    """
    Gini-based (mean decrease impurity) feature importance.

    Built-in to the forest; fast to compute but biased toward high-cardinality
    features. Use permutation importance for a less biased estimate.

    Args:
        model:         fitted RandomForestClassifier
        feature_names: list of feature name strings
        top_n:         number of top features to display
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 5))
    plt.bar([feature_names[i] for i in sorted_idx], importances[sorted_idx],
            color="steelblue", edgecolor="k")
    plt.xlabel("Feature")
    plt.ylabel("Gini Importance")
    plt.title(f"Random Forest — Top {top_n} Feature Importances (Gini)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_figure("random_forest_gini_importance.png", dpi=150)


def plot_permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray,
                                 feature_names: list, top_n: int = 15) -> None:
    """
    Permutation-based feature importance (more reliable than Gini importance).

    Measures accuracy drop when a feature's values are randomly shuffled.
    Not biased toward high-cardinality features; accounts for feature interactions.

    Fine-tuning tip: features with importance near 0 or negative can often be
    removed without hurting performance (and may reduce overfitting).

    Args:
        model:         fitted model
        X_test:        test features (use held-out data, NOT training data)
        y_test:        true labels
        feature_names: list of feature name strings
        top_n:         number of top features to display
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = np.argsort(result.importances_mean)[::-1][:top_n]

    plt.figure(figsize=(10, 5))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=True,
        tick_labels=[feature_names[i] for i in sorted_idx],
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Feature")
    plt.ylabel("Accuracy Decrease")
    plt.title(f"Random Forest — Permutation Importance (top {top_n})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_figure("random_forest_permutation_importance.png", dpi=150)


def plot_n_estimators_curve(X_train, y_train, X_test, y_test,
                             max_trees: int = 300) -> None:
    """
    Learning curve: accuracy vs. number of trees.

    Useful for finding the point of diminishing returns — where adding more
    trees no longer improves accuracy. Use this to set n_estimators efficiently.

    Fine-tuning tip: if the curve is still rising steeply at max_trees,
    increase max_trees and re-run.
    """
    n_values = list(range(10, max_trees + 1, 10))
    oob_scores, test_scores = [], []

    for n in n_values:
        rf = RandomForestClassifier(
            n_estimators=n, oob_score=True, n_jobs=-1, random_state=42
        )
        rf.fit(X_train, y_train)
        oob_scores.append(rf.oob_score_)
        test_scores.append(rf.score(X_test, y_test))

    plt.figure(figsize=(8, 4))
    plt.plot(n_values, oob_scores, label="OOB Score")
    plt.plot(n_values, test_scores, label="Test Accuracy")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.title("Random Forest — n_estimators Learning Curve")
    plt.legend()
    plt.tight_layout()
    save_figure("random_forest_n_estimators.png", dpi=150)


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Display confusion matrix."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title("Random Forest — Confusion Matrix")
    plt.tight_layout()
    save_figure("random_forest_confusion.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification ──────────────────────────────────────────────
    print("=== Random Forest Classification ===")
    X, y, feature_names = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(**CLASSIFIER_CONFIG)
    clf.fit(X_train, y_train)

    print("\nTest Set Metrics:")
    evaluate_classifier(clf, X_test, y_test)
    cross_validate(clf, X, y)

    # Hyperparameter tuning
    print("\n=== Randomized Hyperparameter Search ===")
    best_clf = tune_random_search(X_train, y_train, n_iter=fast_or(30, 3))
    print("\nBest RF — Test Metrics:")
    evaluate_classifier(best_clf, X_test, y_test)

    # Visualizations
    plot_n_estimators_curve(X_train, y_train, X_test, y_test, max_trees=fast_or(300, 30))
    plot_gini_importance(clf, feature_names)
    plot_permutation_importance(clf, X_test, y_test, feature_names)
    plot_confusion_matrix(clf, X_test, y_test)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== Random Forest Regression ===")
    Xr, yr = load_regression_data()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg = RandomForestRegressor(**REGRESSOR_CONFIG)
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)
    print(f"  OOB R²:  {reg.oob_score_:.4f}")
    print(f"  RMSE:    {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f}")
    print(f"  Test R²: {r2_score(yr_test, yr_pred):.4f}")


if __name__ == "__main__":
    main()
