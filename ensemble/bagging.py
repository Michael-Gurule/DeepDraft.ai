"""
Bagging (Bootstrap Aggregating)
================================
An ensemble meta-algorithm that trains multiple base learners on independently
bootstrapped subsets of the training data and aggregates their predictions
(majority vote for classification, mean for regression). Reduces variance
without significantly increasing bias.

How it works:
    1. For each of n_estimators learners:
       a. Draw a bootstrap sample (n samples with replacement from training set)
       b. Optionally sample a random feature subset (max_features)
       c. Train a base learner on the bootstrap sample
    2. Aggregate predictions:
       - Classification: majority vote (hard) or probability averaging (soft)
       - Regression: mean of all learner predictions

Key insight: Random Forest IS Bagging + random feature selection per split.
BaggingClassifier provides a more general framework where any estimator can
be used as the base learner.

Use Cases:
    - Reducing variance of high-variance models (deep trees, SVMs, NNs)
    - Quick ensemble of any sklearn estimator
    - Understanding the contribution of bagging vs. other ensemble techniques
    - Parallel training of independent models

Fine-Tuning Recommendations:
    - base_estimator: the key choice. Default is DecisionTreeClassifier.
      Bagging benefits high-variance learners most. Low-variance learners
      (linear models) benefit little from bagging.
    - n_estimators: more → lower variance, diminishing returns after ~50–100.
      Use oob_score=True for a free validation estimate.
    - max_samples: fraction of training set per bootstrap. Default 1.0 (full).
      Reduce (0.5–0.8) for faster training and more diversity.
    - max_features: fraction of features per base learner. Reducing adds diversity.
    - bootstrap: True = sampling with replacement (true bagging); False = random subsets.
    - bootstrap_features: True = also sample features with replacement.
    - oob_score: True gives an estimate of generalization without a validation set,
      using the ~36.8% of samples not included in each bootstrap (out-of-bag).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_squared_error, r2_score
)
from sklearn.datasets import make_classification, make_regression

from deepdraft.core import fast_or, save_figure


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 800, n_features: int = 12):
    """
    Generate synthetic binary classification data.

    Replace with your own loading:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["label"]).values
        y = df["label"].values

    Returns:
        X: feature matrix
        y: binary class labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=7,
        n_redundant=2,
        random_state=42,
    )
    return X, y


def load_regression_data(n_samples: int = 500):
    """
    Generate synthetic regression data for BaggingRegressor.

    Returns:
        X: feature matrix
        y: continuous target
    """
    X, y = make_regression(n_samples=n_samples, n_features=8, noise=10.0, random_state=42)
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: swap the base_estimator to bag different learners
# Common choices:
#   DecisionTreeClassifier(max_depth=None)  — high-variance trees (classic bagging)
#   DecisionTreeClassifier(max_depth=5)     — partially-grown trees
#   SVC(kernel='rbf', probability=True)     — expensive but effective
#   KNeighborsClassifier(n_neighbors=5)     — KNN with diversity

BAGGING_CLASSIFIER_CONFIG = {
    "estimator":          DecisionTreeClassifier(max_depth=None, random_state=42),
    "n_estimators":       50,            # number of base learners
    "max_samples":        1.0,           # fraction of training samples per learner
    "max_features":       1.0,           # fraction of features per learner
    "bootstrap":          True,          # True = bootstrap sampling (bagging); False = random subsets
    "bootstrap_features": False,         # True = also bootstrap feature sampling
    "oob_score":          True,          # estimate generalization from out-of-bag samples
    "n_jobs":             -1,
    "random_state":       42,
}

BAGGING_REGRESSOR_CONFIG = {
    "estimator":    DecisionTreeRegressor(max_depth=None, random_state=42),
    "n_estimators": 50,
    "max_samples":  0.8,
    "max_features": 0.8,
    "bootstrap":    True,
    "oob_score":    True,
    "n_jobs":       -1,
    "random_state": 42,
}


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray,
                          model_name: str = "Bagging") -> dict:
    """
    Evaluate a BaggingClassifier.

    Also prints the OOB score if available — this is a near-unbiased
    estimate of generalization computed for free during training.

    Args:
        model:      fitted BaggingClassifier
        X_test:     test features
        y_test:     true labels
        model_name: string label for printing

    Returns:
        dict of metrics
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if hasattr(model, "oob_score_"):
        print(f"  OOB Score:     {model.oob_score_:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"accuracy": acc}


def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    K-fold cross-validation.

    Fine-tuning tip: compare the OOB score to the CV score. Large gap →
    OOB estimate may be unreliable (usually happens with very small datasets).
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\n{cv}-Fold CV Accuracy: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────

def compare_base_learners(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Compare bagging with different base learners.

    This demonstrates how bagging affects different model types:
        - High-variance models (deep trees) benefit the most
        - Low-variance models (linear models) benefit the least
        - KNN and SVM can also benefit significantly

    Args:
        X_train, y_train: training data
        X_test, y_test:   test data
    """
    base_learners = {
        "Single Deep Tree":        DecisionTreeClassifier(max_depth=None, random_state=42),
        "Bagging (Deep Trees)":    BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=None),
            n_estimators=50, oob_score=False, random_state=42, n_jobs=-1,
        ),
        "Bagging (Shallow Trees)": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=50, oob_score=False, random_state=42, n_jobs=-1,
        ),
        "Bagging (KNN)":           BaggingClassifier(
            estimator=KNeighborsClassifier(n_neighbors=5),
            n_estimators=20, max_samples=0.7, bootstrap=True,
            oob_score=False, random_state=42, n_jobs=-1,
        ),
    }

    results = {}
    for name, model in base_learners.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"  {name:<30}: {acc:.4f}")

    # Bar chart comparison
    plt.figure(figsize=(9, 4))
    bars = plt.bar(results.keys(), results.values(), color="steelblue", edgecolor="k")
    for bar, acc in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.1)
    plt.ylabel("Test Accuracy")
    plt.title("Bagging — Base Learner Comparison")
    plt.tight_layout()
    save_figure("bagging_base_learner_comparison.png", dpi=150)


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_bagging(X_train: np.ndarray, y_train: np.ndarray) -> BaggingClassifier:
    """
    Grid search over n_estimators, max_samples, and max_features.

    These parameters control the diversity among base learners:
        - More diversity (smaller max_samples, max_features) → lower variance
          but potentially higher bias
        - Less diversity (larger values) → closer to a single model

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted BaggingClassifier
    """
    param_grid = {
        "n_estimators": fast_or([20, 50, 100], [10]),
        "max_samples":  fast_or([0.5, 0.7, 1.0], [0.7, 1.0]),
        "max_features": fast_or([0.5, 0.7, 1.0], [1.0]),
    }
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None),
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )
    search = GridSearchCV(bag, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_oob_vs_n_estimators(X_train: np.ndarray, y_train: np.ndarray,
                               max_n: int = 100) -> None:
    """
    Plot OOB score as n_estimators increases.

    The OOB score is a free generalization estimate — use it to find the
    minimum n_estimators where the score stabilizes. No need for a
    separate validation set.

    Fine-tuning tip: if OOB score keeps rising beyond max_n, increase it.
    If it's flat after 30 trees, use 50 trees to be safe.
    """
    n_values = range(5, max_n + 1, 5)
    oob_scores = []

    for n in n_values:
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=None),
            n_estimators=n,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )
        bag.fit(X_train, y_train)
        oob_scores.append(bag.oob_score_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(n_values), oob_scores, marker="o", markersize=4, color="steelblue")
    plt.xlabel("n_estimators")
    plt.ylabel("OOB Score")
    plt.title("Bagging — OOB Score vs. n_estimators")
    plt.tight_layout()
    save_figure("bagging_oob_curve.png", dpi=150)


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Display confusion matrix."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title("Bagging — Confusion Matrix")
    plt.tight_layout()
    save_figure("bagging_confusion.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification ──────────────────────────────────────────────
    print("=== Bagging Classification ===")
    X, y = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    bag = BaggingClassifier(**BAGGING_CLASSIFIER_CONFIG)
    bag.fit(X_train, y_train)

    print("\nTest Set Metrics:")
    evaluate_classifier(bag, X_test, y_test)
    cross_validate(bag, X, y)

    # Base learner comparison
    print("\n=== Base Learner Comparison ===")
    compare_base_learners(X_train, y_train, X_test, y_test)

    # Hyperparameter tuning
    print("\n=== Hyperparameter Tuning ===")
    best_bag = tune_bagging(X_train, y_train)
    print("\nBest Bagging — Test Metrics:")
    evaluate_classifier(best_bag, X_test, y_test)

    # Visualizations
    plot_oob_vs_n_estimators(X_train, y_train, max_n=fast_or(100, 20))
    plot_confusion_matrix(bag, X_test, y_test)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== Bagging Regression ===")
    Xr, yr = load_regression_data()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg = BaggingRegressor(**BAGGING_REGRESSOR_CONFIG)
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)
    print(f"  OOB R²:  {reg.oob_score_:.4f}")
    print(f"  RMSE:    {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f}")
    print(f"  Test R²: {r2_score(yr_test, yr_pred):.4f}")


if __name__ == "__main__":
    main()
