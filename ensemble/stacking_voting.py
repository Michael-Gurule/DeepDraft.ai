"""
Stacking and Voting Ensembles
==============================

Meta-learning strategies that combine predictions from multiple base models.

Voting Classifier / Regressor:
  Hard voting:  majority vote over predicted classes
  Soft voting:  average of predicted class probabilities (preferred — uses more info)
  Weighted:     w_i · p_i(y|x), sum over models

Stacking (Stacked Generalization):
  Level 0 (base learners): diverse models (LR, SVM, RF, GBM, NN, ...)
  Level 1 (meta-learner):  trained on out-of-fold predictions from Level 0
                           using k-fold cross-fitting to avoid label leakage

  Training procedure:
    1. Split training data into K folds
    2. For each fold k: train base learners on K-1 folds, predict on fold k
    3. Collect predictions → meta-features matrix M ∈ ℝ^{n × n_models}
    4. Train meta-learner on M (optionally include original features)
    5. For test: average base learner predictions over K models trained in step 2

  Why stacking outperforms voting:
    Voting weights all models equally; stacking learns optimal combination weights
    from data, and can learn non-linear combinations via a nonlinear meta-learner.

Blending (simplified stacking):
  Hold out a validation set; train base learners on train; get predictions on val;
  train meta-learner on val predictions. Simpler but wastes data.

Fine-tuning recommendations:
  - n_estimators base models (#Placeholder: mix of fast and slow learners)
  - cv (#Placeholder: 5): K-fold count for out-of-fold meta-features
  - meta_learner (#Placeholder): LogisticRegression is interpretable;
    Ridge is good for regression; GradientBoosting can overfit on small meta-feature sets
  - passthrough (#Placeholder: True/False): include original features in meta-learner input
  - weights (#Placeholder: None or list): voting weights per model (tune by val performance)

References:
  Wolpert (1992) — Stacked Generalization
  Breiman (1996) — Stacked Regressions
  sklearn.ensemble.StackingClassifier / VotingClassifier documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error, r2_score,
    classification_report,
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "n_samples":   5000,   # # Placeholder:
    "n_features":  20,     # # Placeholder:
    "cv":          5,      # # Placeholder: folds for stacking out-of-fold
    "seed":        42,
}


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def generate_classification_data(cfg: dict) -> tuple:
    """Replace with your real dataset."""
    X, y = make_classification(
        n_samples=cfg["n_samples"],
        n_features=cfg["n_features"],
        n_informative=10, n_redundant=4,
        random_state=cfg["seed"],
    )
    return train_test_split(X, y, test_size=0.2, random_state=cfg["seed"], stratify=y)


def generate_regression_data(cfg: dict) -> tuple:
    """Replace with your real dataset."""
    X, y = make_regression(
        n_samples=cfg["n_samples"],
        n_features=cfg["n_features"],
        n_informative=10, noise=0.2,
        random_state=cfg["seed"],
    )
    return train_test_split(X, y, test_size=0.2, random_state=cfg["seed"])


# ─────────────────────────────────────────────
# Base Learner Definitions
# ─────────────────────────────────────────────

def get_base_classifiers(seed: int = 42) -> list[tuple]:
    """
    Diverse set of base learners for stacking/voting.
    Diversity (different inductive biases) is key to ensemble gains.
    # Placeholder: swap or add models to match your problem
    """
    return [
        ("lr",   make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))),
        ("rf",   RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        ("gbm",  GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)),
        ("svm",  make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed))),
        ("knn",  make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))),
        ("mlp",  make_pipeline(StandardScaler(),
                               MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                             random_state=seed))),
    ]


def get_base_regressors(seed: int = 42) -> list[tuple]:
    """# Placeholder: swap or add regression models."""
    return [
        ("lr",  make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("rf",  RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
        ("gbm", GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)),
        ("svr", make_pipeline(StandardScaler(), SVR(kernel="rbf"))),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))),
    ]


# ─────────────────────────────────────────────
# Voting Ensemble
# ─────────────────────────────────────────────

def build_voting_classifier(
    estimators: list[tuple],
    voting: str = "soft",        # # Placeholder: "soft" (default) or "hard"
    weights: list = None,        # # Placeholder: per-model weights (None = equal)
) -> VotingClassifier:
    """
    Soft voting averages probability estimates → better calibration.
    Hard voting uses majority class predictions.
    """
    return VotingClassifier(estimators=estimators, voting=voting,
                             weights=weights, n_jobs=-1)


def build_voting_regressor(
    estimators: list[tuple],
    weights: list = None,
) -> VotingRegressor:
    return VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1)


# ─────────────────────────────────────────────
# Stacking Ensemble
# ─────────────────────────────────────────────

def build_stacking_classifier(
    estimators: list[tuple],
    final_estimator=None,        # # Placeholder: meta-learner
    cv: int = 5,                 # # Placeholder: out-of-fold folds
    passthrough: bool = False,   # # Placeholder: include original features in meta-input
    stack_method: str = "auto",  # # Placeholder: "predict_proba" | "decision_function" | "predict"
) -> StackingClassifier:
    """
    final_estimator is trained on out-of-fold predictions (meta-features).
    LogisticRegression is the standard meta-learner — interpretable, fast, hard to overfit.
    """
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=500)  # # Placeholder:
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        stack_method=stack_method,
        passthrough=passthrough,
        n_jobs=-1,
    )


def build_stacking_regressor(
    estimators: list[tuple],
    final_estimator=None,
    cv: int = 5,
    passthrough: bool = False,
) -> StackingRegressor:
    if final_estimator is None:
        final_estimator = Ridge(alpha=1.0)   # # Placeholder:
    return StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_classifiers(
    models: dict,
    X_train, y_train,
    X_test,  y_test,
    cv: int = 5,
) -> pd.DataFrame:
    """Train all models and evaluate on test set + cross-validation."""
    results = []
    for name, model in models.items():
        print(f"  Fitting {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="roc_auc", n_jobs=-1).mean()

        print(f"Acc={acc:.4f}  AUC={auc:.4f if auc else 'N/A'}  CV-AUC={cv_auc:.4f}")
        results.append({
            "model": name, "accuracy": acc,
            "test_auc": auc or 0.0, "cv_auc": cv_auc,
        })

    return pd.DataFrame(results).sort_values("test_auc", ascending=False)


def evaluate_regressors(
    models: dict,
    X_train, y_train,
    X_test,  y_test,
    cv: int = 5,
) -> pd.DataFrame:
    results = []
    for name, model in models.items():
        print(f"  Fitting {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="r2", n_jobs=-1).mean()
        print(f"RMSE={rmse:.4f}  R²={r2:.4f}  CV-R²={cv_r2:.4f}")
        results.append({"model": name, "rmse": rmse, "r2": r2, "cv_r2": cv_r2})

    return pd.DataFrame(results).sort_values("r2", ascending=False)


# ─────────────────────────────────────────────
# Meta-Learner Comparison
# ─────────────────────────────────────────────

def compare_meta_learners(X_train, y_train, X_test, y_test,
                           base_estimators: list, cv: int = 5) -> pd.DataFrame:
    """
    Compare different meta-learners on the same base estimators.
    Useful for selecting the best meta-learner without retraining base models.
    """
    meta_learners = {
        "LR (C=1)":    LogisticRegression(C=1.0, max_iter=500),        # # Placeholder:
        "LR (C=0.1)":  LogisticRegression(C=0.1, max_iter=500),
        "Ridge":       RidgeClassifier(alpha=1.0),
        "GBM":         GradientBoostingClassifier(n_estimators=50, max_depth=2),
    }
    results = []
    for meta_name, meta in meta_learners.items():
        stk = StackingClassifier(estimators=base_estimators, final_estimator=meta,
                                  cv=cv, n_jobs=-1)
        stk.fit(X_train, y_train)
        y_proba = stk.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"  Meta={meta_name:15s} | AUC={auc:.4f}")
        results.append({"meta_learner": meta_name, "auc": auc})
    return pd.DataFrame(results).sort_values("auc", ascending=False)


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_comparison(df_clf: pd.DataFrame, df_reg: pd.DataFrame = None):
    n_plots = 2 if df_reg is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots + 2, 5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Ensemble Comparison: Voting vs Stacking vs Base Models", fontsize=12)

    # Classification
    ax = axes[0]
    colors = ["#e74c3c" if "stack" in n else "#3498db" if "voting" in n else "#95a5a6"
              for n in df_clf["model"]]
    ax.barh(df_clf["model"][::-1], df_clf["test_auc"][::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("ROC-AUC (test)")
    ax.set_title("Classification — ROC-AUC")
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xlim(0.5, 1.0)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Stacking"),
                       Patch(facecolor="#3498db", label="Voting"),
                       Patch(facecolor="#95a5a6", label="Base model")]
    ax.legend(handles=legend_elements, loc="lower right")

    if df_reg is not None:
        ax2 = axes[1]
        colors2 = ["#e74c3c" if "stack" in n else "#3498db" if "voting" in n else "#95a5a6"
                   for n in df_reg["model"]]
        ax2.barh(df_reg["model"][::-1], df_reg["r2"][::-1], color=colors2[::-1], edgecolor="white")
        ax2.set_xlabel("R² (test)")
        ax2.set_title("Regression — R²")
        ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("stacking_voting_comparison.png", dpi=150)
    plt.show()
    print("Saved: stacking_voting_comparison.png")


def plot_meta_learner_comparison(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["meta_learner"], df["auc"], color="steelblue", edgecolor="white")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Stacking — Meta-Learner Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("stacking_meta_learners.png", dpi=150)
    plt.show()
    print("Saved: stacking_meta_learners.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(CONFIG["seed"])
    print(f"Config: {CONFIG}\n")

    # ── Classification ──────────────────────────────────────────────
    print("=== Classification: Stacking vs Voting vs Base Models ===")
    X_tr, X_te, y_tr, y_te = generate_classification_data(CONFIG)
    base_clf = get_base_classifiers(CONFIG["seed"])

    # Build ensemble models
    models_clf = dict(base_clf)   # individual base models
    models_clf["voting_soft"]  = build_voting_classifier(base_clf, voting="soft")
    models_clf["voting_hard"]  = build_voting_classifier(base_clf, voting="hard")
    models_clf["stacking_lr"]  = build_stacking_classifier(
        base_clf, cv=CONFIG["cv"],
        final_estimator=LogisticRegression(max_iter=500),
    )
    models_clf["stacking_gbm"] = build_stacking_classifier(
        base_clf, cv=CONFIG["cv"],
        final_estimator=GradientBoostingClassifier(n_estimators=50, max_depth=2),
        passthrough=True,    # include original features in meta-input
    )

    df_clf = evaluate_classifiers(models_clf, X_tr, y_tr, X_te, y_te, cv=CONFIG["cv"])
    print("\nClassification Results:")
    print(df_clf.to_string(index=False))

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== Regression: Stacking vs Voting vs Base Models ===")
    X_tr, X_te, y_tr, y_te = generate_regression_data(CONFIG)
    base_reg = get_base_regressors(CONFIG["seed"])

    models_reg = {name: est for name, est in base_reg}
    models_reg["voting_reg"]   = build_voting_regressor(base_reg)
    models_reg["stacking_reg"] = build_stacking_regressor(
        base_reg, cv=CONFIG["cv"],
        final_estimator=Ridge(alpha=1.0),
    )

    df_reg = evaluate_regressors(models_reg, X_tr, y_tr, X_te, y_te, cv=CONFIG["cv"])
    print("\nRegression Results:")
    print(df_reg.to_string(index=False))

    # ── Meta-learner comparison ──────────────────────────────────────
    print("\n=== Meta-Learner Comparison ===")
    X_tr, X_te, y_tr, y_te = generate_classification_data(CONFIG)
    df_meta = compare_meta_learners(X_tr, y_tr, X_te, y_te,
                                     base_clf, cv=CONFIG["cv"])

    # ── Visualization ────────────────────────────────────────────────
    plot_comparison(df_clf, df_reg)
    plot_meta_learner_comparison(df_meta)


if __name__ == "__main__":
    main()
