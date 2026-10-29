"""
Decision Tree
=============
A non-parametric supervised learning algorithm that learns a hierarchical
sequence of if/else rules by recursively splitting the feature space to
minimize impurity. Works for both classification and regression.

Use Cases:
    - Interpretable classification/regression (rules can be visualized)
    - Feature selection baseline (built-in feature importance)
    - Foundation for ensemble methods (Random Forest, Gradient Boosting)
    - Mixed data types (handles numerical and categorical naturally)

Advantages:
    - Highly interpretable — the learned rules can be printed/visualized
    - No feature scaling required
    - Handles multi-output problems natively
    - Captures non-linear and interaction effects

Limitations:
    - High variance — small data changes can produce very different trees
    - Prone to overfitting without pruning
    - Biased toward features with more split points (high cardinality)
    - Not competitive with ensembles on tabular data

Fine-Tuning Recommendations:
    - max_depth: most impactful parameter. None = unlimited (overfitting).
      Start with 3–5 and increase until validation loss plateaus.
    - min_samples_split: minimum samples to allow a split (reduces overfitting).
    - min_samples_leaf: minimum samples required at a leaf node.
    - max_features: number of features considered per split; reduces variance.
    - criterion: 'gini' (classification, faster) or 'entropy' (information gain);
      'squared_error' or 'absolute_error' (regression).
    - ccp_alpha: cost-complexity pruning; increase from 0 to prune more aggressively.
    - class_weight: 'balanced' for imbalanced datasets.
    - Post-pruning via ccp_alpha is often more effective than pre-pruning alone.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    plot_tree, export_text
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, mean_squared_error, r2_score
)
from sklearn.datasets import make_classification, make_regression
from sklearn.inspection import permutation_importance


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 800, n_features: int = 8):
    """
    Generate synthetic multi-class classification data.

    Replace with your own data loading:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        feature_names = df.drop(columns=["target"]).columns.tolist()

    Returns:
        X: feature matrix
        y: class labels
        feature_names: list of feature name strings
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def load_regression_data(n_samples: int = 400):
    """
    Generate synthetic regression data for DecisionTreeRegressor demo.

    Returns:
        X: feature matrix
        y: continuous target
    """
    X, y = make_regression(n_samples=n_samples, n_features=6, noise=10.0, random_state=42)
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: adjust these to control tree depth and pruning
CLASSIFIER_CONFIG = {
    "criterion": "gini",          # 'gini' or 'entropy' (information gain)
    "max_depth": 5,               # None = unlimited; start at 3–5
    "min_samples_split": 20,      # minimum samples required to split a node
    "min_samples_leaf": 10,       # minimum samples required at a leaf
    "max_features": None,         # None = all features; 'sqrt' reduces variance
    "class_weight": None,         # 'balanced' for imbalanced datasets
    "ccp_alpha": 0.0,             # cost-complexity pruning; 0 = no pruning
    "random_state": 42,
}

REGRESSOR_CONFIG = {
    "criterion": "squared_error", # 'squared_error', 'friedman_mse', 'absolute_error'
    "max_depth": 5,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "ccp_alpha": 0.0,
    "random_state": 42,
}


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a DecisionTreeClassifier on the test set.

    Args:
        model:  fitted DecisionTreeClassifier
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


def evaluate_regressor(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a DecisionTreeRegressor on the test set.

    Args:
        model:  fitted DecisionTreeRegressor
        X_test: test features
        y_test: true targets

    Returns:
        dict of metrics
    """
    y_pred = model.predict(X_test)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
    }
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    return metrics


def cross_validate(model, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "accuracy"):
    """
    K-fold cross-validation.

    Fine-tuning tip: if train accuracy >> CV accuracy → overfitting →
    reduce max_depth, increase min_samples_leaf, or use ccp_alpha pruning.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"\n{cv}-Fold CV {scoring}: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# PRUNING
# ─────────────────────────────────────────────

def find_optimal_ccp_alpha(X_train, y_train, X_test, y_test) -> float:
    """
    Cost-complexity pruning path to find the optimal ccp_alpha.

    How it works:
        - Train a full tree (alpha=0), then compute the pruning path.
        - For each alpha, retrain a pruned tree and measure test accuracy.
        - Pick the alpha that maximizes test accuracy while reducing complexity.

    Fine-tuning tip: the optimal alpha often lies near the "elbow" where test
    accuracy stops improving as alpha increases.

    Args:
        X_train, y_train: training data
        X_test, y_test:   test data

    Returns:
        the alpha value with best test accuracy
    """
    clf = DecisionTreeClassifier(random_state=42)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas[::5]  # sample every 5th alpha for speed

    train_scores, test_scores = [], []
    for alpha in alphas:
        t = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        t.fit(X_train, y_train)
        train_scores.append(t.score(X_train, y_train))
        test_scores.append(t.score(X_test, y_test))

    best_alpha = alphas[np.argmax(test_scores)]

    plt.figure(figsize=(8, 4))
    plt.plot(alphas, train_scores, marker="o", label="Train accuracy", markersize=3)
    plt.plot(alphas, test_scores, marker="o", label="Test accuracy", markersize=3)
    plt.axvline(best_alpha, color="red", linestyle="--", label=f"Best alpha={best_alpha:.5f}")
    plt.xlabel("ccp_alpha")
    plt.ylabel("Accuracy")
    plt.title("Cost-Complexity Pruning Path")
    plt.legend()
    plt.tight_layout()
    plt.savefig("decision_tree_pruning.png", dpi=150)
    plt.show()

    print(f"Optimal ccp_alpha: {best_alpha:.5f}  (test accuracy: {max(test_scores):.4f})")
    return best_alpha


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    """
    Grid search for best tree hyperparameters.

    The most impactful parameters to search:
        1. max_depth    — primary control over model complexity
        2. min_samples_leaf — controls leaf granularity
        3. ccp_alpha    — post-pruning strength

    Fine-tuning tip: use cross-validated scores to pick parameters, NOT
    training accuracy (trees can reach 100% train accuracy while overfitting).

    Returns:
        best fitted DecisionTreeClassifier
    """
    param_grid = {
        "max_depth":         [3, 5, 7, 10, None],
        "min_samples_leaf":  [5, 10, 20, 50],
        "criterion":         ["gini", "entropy"],
    }
    clf = DecisionTreeClassifier(random_state=42)
    search = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_tree_structure(model, feature_names: list, class_names: list | None = None,
                        max_depth_display: int = 3) -> None:
    """
    Visualize the learned decision tree structure.

    Each node shows: split condition, gini/entropy impurity, sample count,
    class distribution, and majority class.

    Fine-tuning tip: a tree with depth > 6 is usually too complex to read —
    prune it or limit max_depth_display for visualization.

    Args:
        model:             fitted DecisionTreeClassifier
        feature_names:     list of feature name strings
        class_names:       list of class name strings (optional)
        max_depth_display: how many levels to visualize (avoids clutter)
    """
    plt.figure(figsize=(20, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth_display,
        fontsize=8,
    )
    plt.title(f"Decision Tree (top {max_depth_display} levels)")
    plt.tight_layout()
    plt.savefig("decision_tree_structure.png", dpi=150)
    plt.show()


def print_rules(model, feature_names: list) -> None:
    """
    Print the decision tree as human-readable if/else rules.

    Useful for: code generation, rule extraction for business logic,
    communicating model logic to non-technical stakeholders.
    """
    rules = export_text(model, feature_names=feature_names, max_depth=5)
    print("\nDecision Tree Rules (top 5 levels):")
    print(rules)


def plot_feature_importance(model, feature_names: list) -> None:
    """
    Bar chart of Gini-based feature importances.

    Importance = total reduction in impurity from splits on that feature,
    weighted by number of samples reaching those splits.

    Warning: Gini importance is biased toward high-cardinality features.
    Use permutation importance for a more reliable estimate.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar([feature_names[i] for i in sorted_idx], importances[sorted_idx],
            color="steelblue", edgecolor="k")
    plt.xlabel("Feature")
    plt.ylabel("Gini Importance")
    plt.title("Decision Tree — Feature Importances")
    plt.tight_layout()
    plt.savefig("decision_tree_importance.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Display confusion matrix."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title("Decision Tree — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("decision_tree_confusion.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification ──────────────────────────────────────────────
    print("=== Decision Tree Classification ===")
    X, y, feature_names = load_classification_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(**CLASSIFIER_CONFIG)
    clf.fit(X_train, y_train)

    print("\nTest Set Metrics:")
    evaluate_classifier(clf, X_test, y_test)
    cross_validate(clf, X, y)

    # Post-pruning: find optimal ccp_alpha
    print("\n=== Pruning Analysis ===")
    best_alpha = find_optimal_ccp_alpha(X_train, y_train, X_test, y_test)
    pruned_clf = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
    pruned_clf.fit(X_train, y_train)
    print("\nPruned Tree — Test Metrics:")
    evaluate_classifier(pruned_clf, X_test, y_test)

    # Hyperparameter tuning
    print("\n=== Hyperparameter Tuning ===")
    best_clf = tune_hyperparameters(X_train, y_train)
    print("\nBest Tree — Test Metrics:")
    evaluate_classifier(best_clf, X_test, y_test)

    # Visualizations
    plot_tree_structure(clf, feature_names)
    print_rules(clf, feature_names)
    plot_feature_importance(clf, feature_names)
    plot_confusion_matrix(clf, X_test, y_test)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== Decision Tree Regression ===")
    Xr, yr = load_regression_data()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg = DecisionTreeRegressor(**REGRESSOR_CONFIG)
    reg.fit(Xr_train, yr_train)
    print("\nRegression Test Metrics:")
    evaluate_regressor(reg, Xr_test, yr_test)


if __name__ == "__main__":
    main()
