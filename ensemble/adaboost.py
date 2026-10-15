"""
AdaBoost (Adaptive Boosting)
=============================
The original boosting algorithm. Trains weak learners sequentially, where each
new learner focuses more on the examples that previous learners misclassified
by re-weighting the training samples. Final prediction is a weighted majority
vote (classification) or weighted sum (regression) of all learners.

How it works:
    1. Assign equal weights to all training samples
    2. Train a weak learner (e.g., Decision Tree with max_depth=1 = "stump")
    3. Increase weights of misclassified samples; decrease weights of correct ones
    4. Train the next weak learner on the re-weighted distribution
    5. Combine all learners with weights proportional to their accuracy
    6. Final prediction = sign(weighted sum of all learner predictions)

Variants:
    SAMME    — multi-class extension (default in sklearn)
    SAMME.R  — uses class probability estimates (usually better)
    AdaBoostR2 — regression variant

Use Cases:
    - Binary and multi-class classification
    - Face detection (historically important: Viola-Jones algorithm used AdaBoost)
    - Robust classification when base learner is very simple
    - Interpretable ensemble (stumps are highly readable)

Advantages over basic boosting:
    - Does not require learning rate tuning (adaptive by design)
    - Works well with high-bias, low-variance weak learners (stumps)
    - Less prone to overfitting than single trees (when using stumps)

Limitations:
    - Very sensitive to noisy data and outliers (sample weights amplify noise)
    - Not competitive with GBM/XGBoost on most modern datasets
    - Slower than Random Forest for many weak learners
    - SAMME.R requires weak learners that output probabilities

Fine-Tuning Recommendations:
    - base_estimator (estimator): DecisionTreeClassifier(max_depth=1) (default).
      Increase max_depth (2–3) if underfitting. Use deeper trees carefully —
      AdaBoost can overfit quickly with strong base learners.
    - n_estimators: 50–200 typically. More estimators → better but slower.
      Use cross-validation to find the point where test loss plateaus.
    - learning_rate: shrinks each weak learner's contribution.
      Lower → less overfitting but needs more estimators.
      Typical: 0.5–1.0. For stumps, 1.0 often works fine.
    - algorithm: SAMME.R (default) is generally better when base learner
      supports predict_proba; use SAMME for learners without probabilities.
    - Noise sensitivity: if dataset is noisy, prefer GBM with subsampling.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_squared_error, r2_score
)
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 800, n_features: int = 10):
    """
    Generate synthetic binary classification data.

    Replace with your own data loading:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["label"]).values
        y = df["label"].values

    Returns:
        X: feature matrix
        y: binary labels (0/1)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    return X, y


def load_regression_data(n_samples: int = 400):
    """
    Generate synthetic regression data for AdaBoostRegressor.

    Returns:
        X: feature matrix
        y: continuous target
    """
    X, y = make_regression(n_samples=n_samples, n_features=6, noise=8.0, random_state=42)
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: adjust base estimator depth and n_estimators here
BASE_STUMP = DecisionTreeClassifier(
    max_depth=1,        # "decision stump" — the canonical weak learner for AdaBoost
    # Placeholder: increase max_depth to 2–3 if underfitting; watch for overfitting
)

BASE_DEEPER = DecisionTreeClassifier(
    max_depth=3,        # moderate weak learner — more expressive, may overfit faster
)

CLASSIFIER_CONFIG = {
    "estimator":     BASE_STUMP,    # weak learner; default = DecisionTreeClassifier(max_depth=1)
    "n_estimators":  100,           # number of boosting rounds
    "learning_rate": 1.0,           # shrinks each learner's contribution; try 0.1–1.0
    "algorithm":     "SAMME",       # 'SAMME' or 'SAMME.R' (needs predict_proba)
    "random_state":  42,
}

REGRESSOR_CONFIG = {
    "estimator":     DecisionTreeRegressor(max_depth=3),
    "n_estimators":  100,
    "learning_rate": 1.0,
    "loss":          "linear",     # 'linear', 'square', 'exponential'
    "random_state":  42,
}


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Args:
        model:  fitted AdaBoostClassifier
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


def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    K-fold cross-validation.

    Fine-tuning tip: if AdaBoost accuracy lags behind Random Forest by more
    than a few points, the task likely requires a stronger base learner or
    modern boosting (GBM/XGBoost) would be more appropriate.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n{cv}-Fold CV Accuracy: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_adaboost(X_train: np.ndarray, y_train: np.ndarray) -> AdaBoostClassifier:
    """
    Grid search over n_estimators, learning_rate, and base estimator depth.

    Fine-tuning strategy:
        1. Search over (n_estimators, learning_rate) pairs — they compensate each other.
        2. Try base estimator max_depth 1 (stump) vs. 2–3 (deeper tree).
        3. Pick the combination with best CV accuracy.

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted AdaBoostClassifier
    """
    param_grid = [
        {
            "estimator": [DecisionTreeClassifier(max_depth=d, random_state=42)
                          for d in [1, 2, 3]],
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.1, 0.5, 1.0],
            "algorithm":     ["SAMME"],
        }
    ]
    search = GridSearchCV(
        AdaBoostClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_staged_accuracy(model: AdaBoostClassifier,
                          X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot classification accuracy as a function of boosting stage.

    This shows how accuracy improves as more weak learners are added.
    Plateauing test accuracy suggests adding more estimators won't help.
    Rising test error (overfitting) is unusual in AdaBoost with stumps.

    Fine-tuning tip: the point where test accuracy stops improving is the
    optimal n_estimators — retrain with that value.
    """
    test_errors = [
        1.0 - accuracy_score(y_test, y_pred)
        for y_pred in model.staged_predict(X_test)
    ]

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(test_errors) + 1), test_errors,
             label="Test Error", color="tomato")
    best_n = np.argmin(test_errors) + 1
    plt.axvline(best_n, color="gray", linestyle="--",
                label=f"Best n={best_n} (err={min(test_errors):.4f})")
    plt.xlabel("Boosting Stage")
    plt.ylabel("Error Rate")
    plt.title("AdaBoost — Staged Test Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("adaboost_staged_error.png", dpi=150)
    plt.show()
    print(f"Optimal n_estimators from staged error: {best_n}")


def plot_estimator_weights(model: AdaBoostClassifier) -> None:
    """
    Bar chart of individual weak learner weights (alpha values).

    In AdaBoost, each weak learner receives a weight proportional to its accuracy.
    Strong learners get higher weights; weak/chance-level learners get near-zero weight.

    Fine-tuning tip: if most learners have similar low weights, the ensemble
    is struggling — increase base learner expressiveness or switch to GBM.
    """
    weights = model.estimator_weights_
    plt.figure(figsize=(10, 3))
    plt.bar(range(len(weights)), weights, color="steelblue", edgecolor="k", width=0.8)
    plt.xlabel("Weak Learner Index")
    plt.ylabel("Weight (alpha)")
    plt.title("AdaBoost — Weak Learner Weights")
    plt.tight_layout()
    plt.savefig("adaboost_weights.png", dpi=150)
    plt.show()


def plot_sample_weights_evolution(X_train: np.ndarray, y_train: np.ndarray,
                                   n_stages: int = 5) -> None:
    """
    Visualize how sample weights evolve across boosting stages.

    Shows which samples accumulate high weight (hardest to classify).
    High-weight samples after many stages are often:
        - Genuinely ambiguous (class boundary cases)
        - Outliers or label noise

    Fine-tuning tip: samples with persistently high weights across stages
    are worth manual inspection — they may be mislabeled or outliers.
    """
    fig, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))

    for stage, ax in enumerate(axes):
        n = stage + 1
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=n,
            algorithm="SAMME",
            random_state=42,
        )
        ada.fit(X_train[:, :2], y_train)  # use first 2 features for 2D viz

        # Recompute sample weights: last fitted estimator's sample weight
        # (approximated by the weight after n stages)
        sample_weight = np.ones(len(X_train)) / len(X_train)
        for i in range(n):
            y_pred_i = ada.estimators_[i].predict(X_train[:, :2])
            incorrect = (y_pred_i != y_train).astype(float)
            estimator_error = np.dot(sample_weight, incorrect)
            if estimator_error <= 0:
                break
            alpha = ada.estimator_weights_[i]
            sample_weight *= np.exp(alpha * incorrect * ((sample_weight > 0) | (alpha < 0)))
            sample_weight /= sample_weight.sum()

        scatter = ax.scatter(
            X_train[:, 0], X_train[:, 1],
            c=y_train, cmap="bwr",
            s=sample_weight * len(X_train) * 50 + 5,
            alpha=0.7, edgecolors="k", linewidths=0.2
        )
        ax.set_title(f"Stage {n}")
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")

    plt.suptitle("AdaBoost — Sample Weight Evolution (dot size ∝ weight)")
    plt.tight_layout()
    plt.savefig("adaboost_sample_weights.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Display confusion matrix."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title("AdaBoost — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("adaboost_confusion.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification ──────────────────────────────────────────────
    print("=== AdaBoost Classification ===")
    X, y = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ada = AdaBoostClassifier(**CLASSIFIER_CONFIG)
    ada.fit(X_train, y_train)

    print("\nTest Set Metrics (stumps, n=100):")
    evaluate_classifier(ada, X_test, y_test)
    cross_validate(ada, X, y)

    # Deeper base learner comparison
    print("\n=== Deeper Base Learner (max_depth=3) ===")
    ada_deep = AdaBoostClassifier(
        estimator=BASE_DEEPER, n_estimators=100,
        learning_rate=0.5, algorithm="SAMME", random_state=42
    )
    ada_deep.fit(X_train, y_train)
    print("\nDeeper Tree AdaBoost Metrics:")
    evaluate_classifier(ada_deep, X_test, y_test)

    # Hyperparameter tuning
    print("\n=== Hyperparameter Tuning ===")
    best_ada = tune_adaboost(X_train, y_train)
    print("\nBest AdaBoost — Test Metrics:")
    evaluate_classifier(best_ada, X_test, y_test)

    # Visualizations
    plot_staged_accuracy(ada, X_test, y_test)
    plot_estimator_weights(ada)
    plot_confusion_matrix(ada, X_test, y_test)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== AdaBoost Regression ===")
    Xr, yr = load_regression_data()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg = AdaBoostRegressor(**REGRESSOR_CONFIG)
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)
    print(f"  RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f}")
    print(f"  R²:   {r2_score(yr_test, yr_pred):.4f}")


if __name__ == "__main__":
    main()
