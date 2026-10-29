"""
Gradient Boosting (scikit-learn)
================================
An ensemble method that builds trees sequentially, where each new tree
corrects the residual errors of the previous ensemble. Uses gradient descent
in function space to minimize a differentiable loss function.

How it works:
    1. Fit an initial model (typically the mean for regression)
    2. Compute pseudo-residuals (negative gradient of the loss)
    3. Fit a new weak learner (shallow decision tree) to the residuals
    4. Add the new tree to the ensemble, scaled by learning_rate
    5. Repeat steps 2–4 for n_estimators iterations

Use Cases:
    - Tabular data regression and classification (industry standard)
    - Ranking problems (learning-to-rank)
    - Survival analysis
    - Anywhere XGBoost/LightGBM cannot be installed (sklearn-only environments)

Variants in this file:
    GradientBoostingClassifier  — traditional (tree-by-tree, slower)
    HistGradientBoostingClassifier — histogram-based (much faster, handles NaN)

Fine-Tuning Recommendations:
    - learning_rate vs n_estimators: these interact. Use a small learning_rate
      (0.01–0.1) with many trees (100–1000). Reduce learning_rate and increase
      n_estimators together for better generalization.
    - max_depth: 3–5 is typical (shallower than Random Forest). Deep trees
      → overfitting. Start with 3.
    - subsample: use < 1.0 (e.g., 0.8) for stochastic gradient boosting —
      reduces variance and speeds training. Similar to Random Forest bagging.
    - min_samples_leaf: increase to regularize (reduce overfitting).
    - max_features: 'sqrt' or a fraction < 1.0 adds randomization.
    - Early stopping: use validation_fraction + n_iter_no_change to stop
      training automatically when validation loss plateaus.
    - HistGradientBoosting: prefer over GradientBoosting for large datasets.
      Supports native NaN handling (no imputation needed) and categorical features.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_squared_error, r2_score, roc_auc_score
)
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification, make_regression


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 1000, n_features: int = 15):
    """
    Generate synthetic binary classification data.

    Replace with your own dataset. Note: HistGradientBoosting can handle
    NaN values natively — no imputation needed.

    Returns:
        X: feature matrix
        y: binary class labels
        feature_names: list of feature names
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=3,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def load_regression_data(n_samples: int = 600):
    """
    Generate synthetic regression data.

    Returns:
        X: feature matrix
        y: continuous target
    """
    X, y = make_regression(
        n_samples=n_samples, n_features=10, n_informative=6,
        noise=10.0, random_state=42
    )
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: learning_rate and n_estimators are the most impactful knobs.
# Use small learning_rate (0.05–0.1) with early stopping for best results.
GBM_CLASSIFIER_CONFIG = {
    "n_estimators": 200,         # number of boosting rounds (trees)
    "learning_rate": 0.1,        # step size shrinkage; smaller → more trees needed
    "max_depth": 4,              # depth per tree; 3–5 is typical
    "min_samples_leaf": 10,      # minimum samples at leaf (regularization)
    "subsample": 0.8,            # fraction of samples per tree (stochastic boosting)
    "max_features": "sqrt",      # features per split: 'sqrt', 'log2', or float
    "loss": "log_loss",          # 'log_loss' for binary; 'exponential' for AdaBoost-like
    "random_state": 42,
}

# HistGradientBoosting: much faster; preferred for large datasets
HIST_GBM_CONFIG = {
    "max_iter": 300,             # equivalent to n_estimators
    "learning_rate": 0.1,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "l2_regularization": 0.0,   # Placeholder: increase (0.01–1.0) to reduce overfitting
    "max_bins": 255,             # number of histogram bins per feature
    "early_stopping": True,      # stop when validation score plateaus
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 42,
}


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Args:
        model:  fitted classifier
        X_test: test features
        y_test: true labels

    Returns:
        dict of metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"accuracy": acc, "roc_auc": auc}


def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    K-fold cross-validation.

    Fine-tuning tip: gradient boosting is prone to overfitting on small datasets.
    If CV std is high, increase regularization (lower learning_rate, higher
    min_samples_leaf, or lower subsample).
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n{cv}-Fold CV AUC: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_gbm(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """
    Grid search over the most impactful GBM hyperparameters.

    Tuning strategy:
        1. Fix n_estimators=500 and find optimal learning_rate + max_depth.
        2. Once learning_rate and max_depth are set, tune subsample and
           min_samples_leaf.
        3. Lower learning_rate and increase n_estimators proportionally
           for a final fine-tuning pass.

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted GradientBoostingClassifier
    """
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth":     [3, 4, 5],
        "subsample":     [0.7, 0.8, 1.0],
    }
    gbm = GradientBoostingClassifier(n_estimators=200, random_state=42)
    search = GridSearchCV(gbm, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV AUC: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_staged_loss(model: GradientBoostingClassifier,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot train and test log-loss at each boosting stage.

    This is the primary diagnostic for gradient boosting:
        - Both curves decreasing: model is still learning
        - Test curve plateaus/rises while train continues dropping: overfitting
        - Optimal n_estimators: where test loss is minimized

    staged_predict_proba yields the ensemble's predictions after 1, 2, ...,
    n_estimators trees, so one fitted model gives the whole curve.

    Fine-tuning tip: if test loss plateaus early, reduce n_estimators or use
    early stopping (HistGradientBoosting supports this natively).
    """
    from sklearn.metrics import log_loss

    train_loss = [log_loss(y_train, p) for p in model.staged_predict_proba(X_train)]
    test_loss  = [log_loss(y_test, p) for p in model.staged_predict_proba(X_test)]
    stages = range(1, len(test_loss) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(stages, train_loss, label="Train Log-Loss")
    plt.plot(stages, test_loss, label="Test Log-Loss")
    plt.xlabel("Boosting Stage (n_estimators)")
    plt.ylabel("Log-Loss")
    plt.title("GBM — Train vs. Test Loss by Boosting Stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gbm_staged_loss.png", dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> None:
    """
    Gini-based feature importances from the gradient boosting ensemble.

    Fine-tuning tip: if a small number of features dominate, consider dropping
    the least important ones and retraining — this can speed up inference
    and reduce overfitting.

    Args:
        model:         fitted GradientBoostingClassifier
        feature_names: list of feature name strings
        top_n:         number of top features to display
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 5))
    plt.bar([feature_names[i] for i in sorted_idx], importances[sorted_idx],
            color="coral", edgecolor="k")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title(f"GBM — Top {top_n} Feature Importances")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("gbm_feature_importance.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Display confusion matrix."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Oranges")
    plt.title("GBM — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("gbm_confusion.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification: GradientBoostingClassifier ──────────────────
    print("=== Gradient Boosting Classification ===")
    X, y, feature_names = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    gbm = GradientBoostingClassifier(**GBM_CLASSIFIER_CONFIG)
    gbm.fit(X_train, y_train)

    print("\nTest Set Metrics:")
    evaluate_classifier(gbm, X_test, y_test)
    cross_validate(gbm, X, y)

    # ── HistGradientBoosting (faster, for larger datasets) ───────────
    print("\n=== HistGradientBoosting (fast variant) ===")
    hist_gbm = HistGradientBoostingClassifier(**HIST_GBM_CONFIG)
    hist_gbm.fit(X_train, y_train)
    print("\nHistGBM Test Metrics:")
    evaluate_classifier(hist_gbm, X_test, y_test)

    # Hyperparameter tuning
    print("\n=== Hyperparameter Tuning ===")
    best_gbm = tune_gbm(X_train, y_train)
    print("\nBest GBM — Test Metrics:")
    evaluate_classifier(best_gbm, X_test, y_test)

    # Visualizations
    plot_staged_loss(gbm, X_train, y_train, X_test, y_test)
    plot_feature_importance(gbm, feature_names)
    plot_confusion_matrix(gbm, X_test, y_test)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== Gradient Boosting Regression ===")
    Xr, yr = load_regression_data()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=4,
        subsample=0.8, random_state=42
    )
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)
    print(f"  RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f}")
    print(f"  R²:   {r2_score(yr_test, yr_pred):.4f}")


if __name__ == "__main__":
    main()
