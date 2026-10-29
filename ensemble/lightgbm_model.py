"""
LightGBM
========

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework
that grows trees leaf-wise (best-first) rather than depth-wise (level-wise).
This makes it faster and often more accurate than XGBoost on tabular data.

Key differences from XGBoost:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Feature          │ LightGBM           │ XGBoost               │
  ├──────────────────────────────────────────────────────────────────┤
  │ Tree growth      │ Leaf-wise          │ Depth-wise            │
  │ Speed            │ Faster (GOSS+EFB)  │ Slower on large data  │
  │ Categoricals     │ Native (int32)     │ Manual encoding       │
  │ GPU              │ Yes (gpu device)   │ Yes (tree_method=gpu) │
  │ Max depth        │ Needs leaf limit   │ Controlled by depth   │
  └──────────────────────────────────────────────────────────────────┘

Algorithmic innovations:
  GOSS (Gradient-based One-Side Sampling):
    Keep all large-gradient instances; randomly sample small-gradient ones.
    Compensates with a constant multiplier to maintain distribution.

  EFB (Exclusive Feature Bundling):
    Bundle mutually exclusive sparse features into single dense features.
    Reduces feature count without information loss.

Regularization:
  - num_leaves:     controls model complexity (leaf-wise analog of max_depth)
  - min_data_in_leaf: prevents overfitting on small leaf nodes
  - lambda_l1/l2:  L1/L2 regularization on leaf weights
  - feature_fraction: column subsampling per tree
  - bagging_fraction: row subsampling per iteration

Fine-tuning recommendations (priority order):
  1. num_leaves (#Placeholder: 31): most important — keep < 2^max_depth to avoid overfitting
  2. learning_rate (#Placeholder: 0.05): lower + more n_estimators usually better
  3. min_data_in_leaf (#Placeholder: 20): larger for noisy data
  4. feature_fraction (#Placeholder: 0.9): use 0.6–0.9 for regularization
  5. bagging_fraction (#Placeholder: 0.8) + bagging_freq (#Placeholder: 5)
  6. lambda_l1/lambda_l2 (#Placeholder: 0.0): add if overfitting persists

References:
  Ke et al. (2017) — LightGBM: A Highly Efficient Gradient Boosting Decision Tree
  https://lightgbm.readthedocs.io/en/latest/Parameters.html
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    mean_squared_error, r2_score,
)

from deepdraft.core import fast_or, save_figure

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed. Run: pip install lightgbm")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG_CLF = {
    "objective":        "binary",         # # Placeholder: "binary" | "multiclass" | "cross_entropy"
    "metric":           "auc",            # # Placeholder: "auc" | "binary_logloss" | "multi_logloss"
    "num_leaves":       31,               # # Placeholder: primary complexity knob (< 2^max_depth)
    "max_depth":        -1,               # # Placeholder: -1 = unlimited; set 6-12 to cap depth
    "learning_rate":    0.05,             # # Placeholder:
    "n_estimators":     fast_or(500, 40),  # # Placeholder: use early stopping to find actual count
    "min_data_in_leaf": 20,               # # Placeholder:
    "feature_fraction": 0.9,             # # Placeholder: column subsampling per tree
    "bagging_fraction": 0.8,             # # Placeholder: row subsampling
    "bagging_freq":     5,               # # Placeholder: bagging every k iterations
    "lambda_l1":        0.0,             # # Placeholder:
    "lambda_l2":        0.0,             # # Placeholder:
    "min_gain_to_split": 0.0,            # # Placeholder:
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             42,
    "early_stopping_rounds": 50,         # # Placeholder:
}

CONFIG_REG = {**CONFIG_CLF, "objective": "regression", "metric": "rmse"}


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def generate_classification_data(
    n_samples: int = fast_or(5000, 800),
    n_features: int = 20,
    n_informative: int = 10,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple:
    """Replace with your real dataset."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=3,
        n_classes=n_classes, random_state=seed,
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def generate_regression_data(
    n_samples: int = fast_or(5000, 800),
    n_features: int = 20,
    seed: int = 42,
) -> tuple:
    """Replace with your real dataset."""
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=10, noise=0.1, random_state=seed,
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def make_categorical_dataset(n_samples: int = fast_or(3000, 600), seed: int = 42) -> tuple:
    """
    Demo dataset with mixed types — replace with real categorical data.
    LightGBM handles categoricals natively when columns are int32 dtype
    and passed as categorical_feature parameter.
    """
    rng = np.random.default_rng(seed)
    n = n_samples
    df = pd.DataFrame({
        "age":        rng.integers(18, 80, n).astype(np.int32),
        "income":     rng.normal(50000, 20000, n),
        "region":     rng.choice([0, 1, 2, 3], n).astype(np.int32),     # categorical
        "education":  rng.choice([0, 1, 2, 3, 4], n).astype(np.int32),  # categorical
        "score":      rng.normal(600, 100, n),
    })
    # Threshold at the median so both classes occur (a fixed threshold of
    # 30000 made nearly every label 1, leaving ROC-AUC undefined).
    signal = df["income"] * 0.3 + df["score"] * 100 + rng.normal(0, 5000, n)
    y = (signal > np.median(signal)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_classifier(
    X_train, y_train, X_val, y_val,
    cfg: dict = None,
    categorical_features: list = None,
) -> "lgb.LGBMClassifier":
    """
    Train LightGBM classifier with early stopping on validation AUC.
    """
    if not HAS_LGB:
        return None
    cfg = cfg or CONFIG_CLF

    model = lgb.LGBMClassifier(
        objective=cfg["objective"],
        metric=cfg["metric"],
        num_leaves=cfg["num_leaves"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"],
        min_child_samples=cfg["min_data_in_leaf"],
        colsample_bytree=cfg["feature_fraction"],
        subsample=cfg["bagging_fraction"],
        subsample_freq=cfg["bagging_freq"],
        reg_alpha=cfg["lambda_l1"],
        reg_lambda=cfg["lambda_l2"],
        min_split_gain=cfg["min_gain_to_split"],
        verbose=cfg["verbose"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["seed"],
    )

    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "callbacks": [
            lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=50),
        ],
    }
    if categorical_features:
        fit_kwargs["categorical_feature"] = categorical_features

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_regressor(
    X_train, y_train, X_val, y_val,
    cfg: dict = None,
) -> "lgb.LGBMRegressor":
    if not HAS_LGB:
        return None
    cfg = cfg or CONFIG_REG

    model = lgb.LGBMRegressor(
        objective=cfg["objective"],
        metric=cfg["metric"],
        num_leaves=cfg["num_leaves"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"],
        min_child_samples=cfg["min_data_in_leaf"],
        colsample_bytree=cfg["feature_fraction"],
        subsample=cfg["bagging_fraction"],
        subsample_freq=cfg["bagging_freq"],
        reg_alpha=cfg["lambda_l1"],
        reg_lambda=cfg["lambda_l2"],
        verbose=cfg["verbose"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["seed"],
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, label: str = "LightGBM") -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    n_trees = model.best_iteration_ or model.n_estimators

    print(f"\n{label} | Trees used: {n_trees}")
    print(f"  Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"  ROC-AUC:  {auc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"accuracy": acc, "auc": auc, "n_trees": n_trees}


def evaluate_regressor(model, X_test, y_test, label: str = "LightGBM") -> dict:
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    n_trees = model.best_iteration_ or model.n_estimators

    print(f"\n{label} | Trees used: {n_trees}")
    print(f"  RMSE: {rmse:.4f}  R²: {r2:.4f}")

    return {"rmse": rmse, "r2": r2, "n_trees": n_trees}


# ─────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list = None,
    importance_type: str = "gain",  # # Placeholder: "gain" | "split" | "weight"
    top_k: int = 20,
):
    """
    gain: total reduction in loss from splits on this feature (recommended)
    split: number of times feature used in splits
    """
    imp = model.booster_.feature_importance(importance_type=importance_type)
    names = feature_names or [f"f{i}" for i in range(len(imp))]

    df = pd.DataFrame({"feature": names, "importance": imp})
    df = df.sort_values("importance", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.35)))
    ax.barh(df["feature"][::-1], df["importance"][::-1], color="steelblue", edgecolor="white")
    ax.set_xlabel(f"Importance ({importance_type})")
    ax.set_title(f"LightGBM Feature Importance (top {top_k})")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_figure("lgbm_feature_importance.png", dpi=150)
    print("Saved: lgbm_feature_importance.png")
    return df


# ─────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────

def tune_num_leaves(
    X_train, y_train, X_val, y_val,
    leaf_values: list = None,
) -> pd.DataFrame:
    """
    Sweep num_leaves — the primary complexity parameter.
    Leaf-wise growth means num_leaves directly controls model complexity,
    unlike max_depth in depth-wise approaches.
    """
    if not HAS_LGB:
        return pd.DataFrame()
    leaf_values = leaf_values or fast_or([15, 31, 63, 127, 255], [15, 31])   # # Placeholder:
    results = []
    for nl in leaf_values:
        cfg = {**CONFIG_CLF, "num_leaves": nl}
        model = train_classifier(X_train, y_train, X_val, y_val, cfg)
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        trees = model.best_iteration_ or CONFIG_CLF["n_estimators"]
        results.append({"num_leaves": nl, "val_auc": auc, "n_trees": trees})
        print(f"  num_leaves={nl:4d} | AUC={auc:.4f} | trees={trees}")

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["num_leaves"], df["val_auc"], marker="o", color="steelblue")
    ax.set_xlabel("num_leaves"); ax.set_ylabel("Val AUC")
    ax.set_title("LightGBM — num_leaves Sweep")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure("lgbm_leaves_sweep.png", dpi=150)
    return df


def cross_validate_lgbm(X, y, cfg: dict = None, n_folds: int = 5) -> dict:
    """K-fold cross-validation using lgb.cv for efficiency."""
    if not HAS_LGB:
        return {}
    cfg = cfg or CONFIG_CLF
    dtrain = lgb.Dataset(X, label=y)
    params = {
        "objective": cfg["objective"],
        "metric": cfg["metric"],
        "num_leaves": cfg["num_leaves"],
        "learning_rate": cfg["learning_rate"],
        "feature_fraction": cfg["feature_fraction"],
        "bagging_fraction": cfg["bagging_fraction"],
        "bagging_freq": cfg["bagging_freq"],
        "verbose": -1,
        "seed": cfg["seed"],
    }
    cv_result = lgb.cv(
        params, dtrain,
        num_boost_round=cfg["n_estimators"],
        nfold=n_folds,
        stratified=True,
        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)],
        return_cvbooster=False,
    )
    metric_key = list(cv_result.keys())[0]   # e.g. "valid auc-mean"
    best_round = len(cv_result[metric_key])
    best_score = cv_result[metric_key][-1]
    std_key = metric_key.replace("mean", "stdv")
    best_std = cv_result.get(std_key, [0])[-1]

    print(f"\nCV ({n_folds}-fold) | Best round: {best_round} | "
          f"Score: {best_score:.5f} ± {best_std:.5f}")
    return {"best_round": best_round, "score": best_score, "std": best_std}


# ─────────────────────────────────────────────
# Native Categoricals Demo
# ─────────────────────────────────────────────

def demo_categorical_features():
    """
    Show LightGBM's native categorical feature handling.
    Pass categorical columns as int32 dtype + specify categorical_feature.
    No need for one-hot encoding — LGBM finds optimal splits.
    """
    if not HAS_LGB:
        return
    X_train, X_test, y_train, y_test = make_categorical_dataset()
    cat_cols = ["region", "education"]

    # Ensure int32 for native categorical treatment
    for col in cat_cols:
        X_train[col] = X_train[col].astype(np.int32)
        X_test[col]  = X_test[col].astype(np.int32)

    model = train_classifier(X_train, y_train, X_test, y_test,
                              categorical_features=cat_cols)
    metrics = evaluate_classifier(model, X_test, y_test, label="LGBM (native cats)")
    print(f"  Native categorical AUC: {metrics['auc']:.4f}")
    return metrics


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_learning_curves(model, X_val, y_val):
    """Plot AUC over boosting rounds from evals_result."""
    evals = model.evals_result_
    if not evals:
        print("No evals_result_ found.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for dataset_name, metrics in evals.items():
        for metric_name, values in metrics.items():
            ax.plot(values, label=f"{dataset_name} {metric_name}")

    ax.axvline(model.best_iteration_ or 0, color="red", linestyle="--",
               label=f"Best iter: {model.best_iteration_}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Metric")
    ax.set_title("LightGBM Learning Curves")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure("lgbm_learning_curves.png", dpi=150)
    print("Saved: lgbm_learning_curves.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if not HAS_LGB:
        print("Install LightGBM: pip install lightgbm")
        return

    np.random.seed(CONFIG_CLF["seed"])
    print(f"LightGBM version: {lgb.__version__}\n")

    # ── Classification ──────────────────────────────────────────────
    print("=== Classification ===")
    X_tr, X_te, y_tr, y_te = generate_classification_data()
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)

    clf = train_classifier(X_tr2, y_tr2, X_val, y_val)
    metrics = evaluate_classifier(clf, X_te, y_te)
    plot_learning_curves(clf, X_val, y_val)
    plot_feature_importance(clf, top_k=15)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== Regression ===")
    X_tr, X_te, y_tr, y_te = generate_regression_data()
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)
    reg = train_regressor(X_tr2, y_tr2, X_val, y_val)
    evaluate_regressor(reg, X_te, y_te)

    # ── num_leaves sweep ────────────────────────────────────────────
    print("\n=== num_leaves Sweep ===")
    X_tr, X_te, y_tr, y_te = generate_classification_data()
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)
    tune_num_leaves(X_tr2, y_tr2, X_val, y_val)

    # ── Cross-validation ────────────────────────────────────────────
    print("\n=== Cross-Validation ===")
    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    cross_validate_lgbm(X_all, y_all)

    # ── Native categoricals ─────────────────────────────────────────
    print("\n=== Native Categorical Features ===")
    demo_categorical_features()


if __name__ == "__main__":
    main()
