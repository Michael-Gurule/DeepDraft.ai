"""
XGBoost (Extreme Gradient Boosting)
=====================================
An optimized gradient boosting library designed for speed and performance.
Extends gradient boosting with regularization (L1/L2), built-in cross-validation,
missing value handling, and hardware-level optimizations (cache-aware, out-of-core).

Key advantages over sklearn GBM:
    - Regularization (reg_alpha L1, reg_lambda L2) prevents overfitting
    - Handles missing values natively (no imputation needed)
    - Parallel tree construction (column block structure)
    - Pruning: max_delta_step, gamma stop unpromising splits early
    - Built-in cross-validation and early stopping via xgb.cv()
    - DART: Dropout Additive Regression Trees (booster='dart')

Use Cases:
    - Kaggle-style tabular competitions (one of the most dominant algorithms)
    - Click-through rate prediction
    - Fraud detection
    - Risk scoring
    - Any structured/tabular regression or classification problem

Fine-Tuning Recommendations (priority order):
    1. n_estimators + early_stopping_rounds: use 1000 trees + early stopping
       rather than tuning n_estimators manually.
    2. learning_rate (eta): 0.01–0.3. Lower → better generalization but
       needs more trees. Use 0.1 as default.
    3. max_depth: 3–6 for most tasks. 6 is aggressive; start at 4.
    4. subsample: 0.6–0.9. Stochastic boosting; reduces overfitting.
    5. colsample_bytree: 0.6–1.0. Fraction of features used per tree.
    6. min_child_weight: increase (3–10) to prevent overfitting on small clusters.
    7. gamma: minimum loss reduction required to make a split. 0 = any split.
       Increase (0.1–5) to make the algorithm more conservative.
    8. reg_alpha (L1): useful for feature selection (sparse models).
    9. reg_lambda (L2): default=1; increase for smoother models.
    10. scale_pos_weight: for imbalanced data, set to sum(neg)/sum(pos).

XGBoost vs. LightGBM — when to choose which:
    XGBoost:
        - Level-wise (depth-first) tree growth: more conservative, less overfit risk.
        - Better calibrated out-of-the-box on small-to-medium datasets (<100k rows).
        - Wider ecosystem: GPU, Spark, Dask integrations are more mature.
        - Preferred when interpretability or calibration matters more than raw AUC.

    LightGBM (see ensemble/lightgbm_model.py):
        - Leaf-wise (best-first) tree growth: finds better splits faster but can overfit
          more aggressively — use min_child_samples / num_leaves to constrain.
        - GOSS (Gradient-based One-Side Sampling): trains on a subset of samples,
          giving 10–20× speedup on large datasets (>500k rows) with minimal AUC loss.
        - EFB (Exclusive Feature Bundling): automatically handles high-cardinality sparse
          features (e.g., one-hot encoded categoricals) without increasing memory.
        - Native categorical support: pass integer columns + categorical_feature list;
          avoids the need for explicit one-hot encoding (often 1–3% AUC gain).
        - Preferred when: large n, high-cardinality categoricals, training speed is critical.

    In practice: benchmark both. On most Kaggle tabular datasets, LightGBM trains 3–10×
    faster than XGBoost with comparable or better AUC. On small datasets (<50k rows)
    XGBoost is often more robust.
"""

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_squared_error, r2_score, roc_auc_score
)
from sklearn.datasets import make_classification, make_regression
from scipy.stats import uniform, randint

from deepdraft.core import fast_or, save_figure


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 1200, n_features: int = 20):
    """
    Generate synthetic binary classification data.

    XGBoost can handle:
        - NaN values natively (no imputation needed)
        - Imbalanced classes (use scale_pos_weight)
        - Mixed dtypes (but prefers float32 for speed)

    Replace with your own loading. For DataFrames:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["target"])  # can keep as DataFrame
        y = df["target"].values

    Returns:
        X: feature matrix
        y: binary labels
        feature_names: list of feature name strings
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=4,
        random_state=42,
    )
    feature_names = [f"f{i}" for i in range(n_features)]
    return X.astype(np.float32), y, feature_names


def load_regression_data(n_samples: int = 800):
    """
    Generate synthetic regression data for XGBRegressor.

    Returns:
        X: feature matrix (float32)
        y: continuous target
    """
    X, y = make_regression(
        n_samples=n_samples, n_features=12, n_informative=8,
        noise=12.0, random_state=42
    )
    return X.astype(np.float32), y.astype(np.float32)


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune these parameters. See module docstring for priority order.
CLASSIFIER_CONFIG = {
    # Core parameters
    "n_estimators":      fast_or(500, 40),  # use large value + early_stopping_rounds
    "learning_rate":     0.05,         # eta; smaller requires more n_estimators
    "max_depth":         4,            # tree depth; 4–6 for most tasks

    # Stochastic regularization
    "subsample":         0.8,          # row sampling per tree
    "colsample_bytree":  0.8,          # feature sampling per tree
    "colsample_bylevel": 1.0,          # feature sampling per tree level
    "colsample_bynode":  1.0,          # feature sampling per node

    # Regularization
    "min_child_weight":  5,            # minimum sum of instance weight in child
    "gamma":             0.0,          # minimum loss reduction for split
    "reg_alpha":         0.0,          # L1 regularization on leaf weights
    "reg_lambda":        1.0,          # L2 regularization on leaf weights

    # Imbalanced data
    "scale_pos_weight":  1,            # Placeholder: set to sum(neg)/sum(pos) for imbalance

    # Misc
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "tree_method":       "hist",       # 'hist' (fastest), 'exact', 'approx'; use 'gpu_hist' with GPU
    "device":            "cpu",        # Placeholder: set to 'cuda' for GPU acceleration
    "verbosity":         0,
    "random_state":      42,
}

REGRESSOR_CONFIG = {
    "n_estimators":      fast_or(500, 40),
    "learning_rate":     0.05,
    "max_depth":         4,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "reg_lambda":        1.0,
    "objective":         "reg:squarederror",
    "eval_metric":       "rmse",
    "tree_method":       "hist",
    "device":            "cpu",
    "verbosity":         0,
    "random_state":      42,
}


# ─────────────────────────────────────────────
# TRAINING WITH EARLY STOPPING
# ─────────────────────────────────────────────

def train_with_early_stopping(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    early_stopping_rounds: int = 50,
    task: str = "classification",
) -> xgb.XGBClassifier | xgb.XGBRegressor:
    """
    Train XGBoost with early stopping on a validation set.

    Early stopping is the most practical way to set n_estimators:
        - Set n_estimators=1000 (or higher)
        - Monitor validation metric; stop when it hasn't improved for
          early_stopping_rounds consecutive rounds
        - The model uses the best iteration automatically

    Fine-tuning tip: use 10–20% of training data as validation set for
    early stopping, then retrain on the full dataset with the best n_estimators
    for production deployment.

    Args:
        X_train, y_train:  training data
        X_val, y_val:      validation data for early stopping
        early_stopping_rounds: stop after this many rounds without improvement
        task:              'classification' or 'regression'

    Returns:
        fitted XGBoost model
    """
    # Since xgboost 2.0, early_stopping_rounds is a constructor argument.
    if task == "classification":
        model = xgb.XGBClassifier(**CLASSIFIER_CONFIG, early_stopping_rounds=early_stopping_rounds)
    else:
        model = xgb.XGBRegressor(**REGRESSOR_CONFIG, early_stopping_rounds=early_stopping_rounds)

    # Track the training set too, for the learning curve. XGBoost early-stops
    # on the LAST entry of eval_set, so the validation set goes last.
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score:     {model.best_score:.4f}")
    return model


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Args:
        model:  fitted XGBClassifier
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


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  n_iter: int = 25) -> xgb.XGBClassifier:
    """
    Randomized search over XGBoost hyperparameters.

    Recommended search strategy:
        1. Broad random search (n_iter=50) to find promising regions
        2. Narrow grid search around best params
        3. Always use early stopping in final fit

    The most impactful parameters:
        learning_rate, max_depth, subsample, colsample_bytree, min_child_weight

    Args:
        X_train: training features
        y_train: training labels
        n_iter:  number of parameter combinations to evaluate

    Returns:
        best fitted XGBClassifier
    """
    param_dist = {
        "learning_rate":    uniform(0.01, 0.29),   # sample from [0.01, 0.30]
        "max_depth":        randint(3, 8),
        "subsample":        uniform(0.6, 0.4),      # sample from [0.6, 1.0]
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 15),
        "gamma":            uniform(0, 0.5),
        "reg_alpha":        uniform(0, 1.0),
        "reg_lambda":       uniform(0.5, 4.5),
    }

    model = xgb.XGBClassifier(
        n_estimators=fast_or(200, 30),
        objective="binary:logistic",
        tree_method="hist",
        verbosity=0,
        random_state=42,
    )

    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=5,
        scoring="roc_auc", n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV AUC: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, importance_type: str = "gain",
                             top_n: int = 15) -> None:
    """
    XGBoost feature importance chart.

    importance_type options:
        'weight'  — number of times a feature is used in splits (count)
        'gain'    — average gain per split (most informative; default)
        'cover'   — average number of samples impacted per split
        'total_gain' — total gain across all splits

    Fine-tuning tip: 'gain' is generally more reliable than 'weight' because
    it measures the actual improvement from each feature's splits.

    Args:
        model:           fitted XGBClassifier or XGBRegressor
        feature_names:   list of feature name strings
        importance_type: which type of importance to plot
        top_n:           number of top features to display
    """
    importance = model.get_booster().get_score(
        fmap="", importance_type=importance_type
    )

    if not importance:
        print("No feature importance available. Skipping.")
        return

    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [item[0] for item in sorted_items]
    scores   = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 5))
    plt.barh(features[::-1], scores[::-1], color="darkorange", edgecolor="k")
    plt.xlabel(f"Importance ({importance_type})")
    plt.title(f"XGBoost — Top {top_n} Feature Importances ({importance_type})")
    plt.tight_layout()
    save_figure(f"xgboost_importance_{importance_type}.png", dpi=150)


def plot_learning_curve(model) -> None:
    """
    Plot training and validation metric over boosting rounds.

    Requires the model to have been trained with
    eval_set=[(X_train, y_train), (X_val, y_val)], as train_with_early_stopping
    does. Shows when the model begins overfitting (validation curve starts rising).

    Fine-tuning tip: the best_iteration attribute shows the optimal stopping point.
    """
    results = model.evals_result()
    if not results:
        print("No eval results found. Train with eval_set to see learning curves.")
        return

    keys = list(results.keys())          # validation_0, validation_1, ... in eval_set order
    val_key = keys[-1]                   # the set early stopping watched
    metric  = list(results[val_key].keys())[0]
    iterations = range(1, len(results[val_key][metric]) + 1)

    plt.figure(figsize=(8, 4))
    if len(keys) > 1:
        plt.plot(iterations, results[keys[0]][metric], label="Train")
    plt.plot(iterations, results[val_key][metric], label="Validation")
    # best_iteration counts from 0; the x-axis counts rounds from 1
    plt.axvline(model.best_iteration + 1, color="red", linestyle="--",
                label=f"Best round: {model.best_iteration + 1}")
    plt.xlabel("Boosting Round")
    plt.ylabel(metric.upper())
    plt.title("XGBoost — Learning Curve")
    plt.legend()
    plt.tight_layout()
    save_figure("xgboost_learning_curve.png", dpi=150)


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Display confusion matrix."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Oranges")
    plt.title("XGBoost — Confusion Matrix")
    plt.tight_layout()
    save_figure("xgboost_confusion.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification ──────────────────────────────────────────────
    print("=== XGBoost Classification ===")
    X, y, feature_names = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Split train → train + validation for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print("\nTraining with Early Stopping:")
    model = train_with_early_stopping(X_tr, y_tr, X_val, y_val)

    print("\nTest Set Metrics:")
    evaluate_classifier(model, X_test, y_test)

    # Hyperparameter tuning
    print("\n=== Randomized Hyperparameter Search ===")
    best_model = tune_xgboost(X_train, y_train, n_iter=fast_or(25, 3))
    print("\nBest XGBoost — Test Metrics:")
    evaluate_classifier(best_model, X_test, y_test)

    # Visualizations
    plot_learning_curve(model)
    plot_feature_importance(model, feature_names, importance_type="gain")
    plot_feature_importance(model, feature_names, importance_type="weight")
    plot_confusion_matrix(model, X_test, y_test)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== XGBoost Regression ===")
    Xr, yr = load_regression_data()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    Xr_tr, Xr_val, yr_tr, yr_val = train_test_split(
        Xr_train, yr_train, test_size=0.15, random_state=42
    )
    reg = train_with_early_stopping(Xr_tr, yr_tr, Xr_val, yr_val, task="regression")
    yr_pred = reg.predict(Xr_test)
    print(f"  RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f}")
    print(f"  R²:   {r2_score(yr_test, yr_pred):.4f}")


if __name__ == "__main__":
    main()
