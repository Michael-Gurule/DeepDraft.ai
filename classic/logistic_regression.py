"""
Logistic Regression
===================
A fundamental supervised classification algorithm that models the probability
of class membership using the logistic (sigmoid) function. Despite its name,
it is a classification model, not a regression model.

Use Cases:
    - Binary classification (spam/not-spam, churn/no-churn, fraud detection)
    - Multi-class classification (with One-vs-Rest or softmax extension)
    - Probability calibration baseline
    - Interpretable models in regulated industries (finance, healthcare)

Assumptions:
    - Linear decision boundary in feature space
    - Little or no multicollinearity among features
    - Large sample size relative to number of features
    - Features and log-odds are linearly related

Fine-Tuning Recommendations:
    - Regularization: C is the inverse of regularization strength (C=1/alpha).
      Smaller C → stronger regularization. Default C=1.0 is a reasonable start.
    - Solver: 'lbfgs' for small-medium datasets; 'saga' for large/sparse datasets
      and L1 penalty; 'newton-cg' for multinomial problems.
    - Class imbalance: set class_weight='balanced' or use oversampling (SMOTE)
      for imbalanced datasets. Monitor precision/recall, not just accuracy.
    - Feature scaling: always scale features — gradient-based solvers converge
      much faster with normalized inputs.
    - Multi-class: scikit-learn fits a multinomial (softmax) model automatically
      for 3+ classes with the default 'lbfgs' solver. For one-vs-rest, wrap the
      estimator in sklearn.multiclass.OneVsRestClassifier.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_sample_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    class_sep: float = 1.0,
):
    """
    Generate synthetic classification data.

    Replace with your own data loading:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["label"]).values
        y = df["label"].values

    Args:
        n_samples:  total number of samples
        n_features: total number of features
        n_classes:  number of target classes (2 = binary)
        class_sep:  larger value → easier classification problem

    Returns:
        X: feature matrix
        y: integer class labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=42,
    )
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune these parameters
MODEL_CONFIG = {
    "C": 1.0,            # inverse regularization strength; try [0.001, 0.01, 0.1, 1, 10, 100]
    "penalty": "l2",     # 'l1' for sparsity/feature selection, 'l2' for weight shrinkage
    "solver": "lbfgs",   # 'lbfgs', 'liblinear' (small data), 'saga' (large/sparse + L1)
    "max_iter": 1000,    # increase if convergence warnings appear
    "class_weight": None,  # set to 'balanced' for imbalanced datasets
}


def build_pipeline(**model_kwargs) -> Pipeline:
    """
    Build a StandardScaler → LogisticRegression pipeline.

    Feature scaling is critical: without it, features with large magnitudes
    dominate the gradient and slow / prevent convergence.

    Args:
        **model_kwargs: passed directly to LogisticRegression

    Returns:
        sklearn Pipeline
    """
    config = {**MODEL_CONFIG, **model_kwargs}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**config)),
    ])


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Key metrics explained:
        Accuracy    — fraction correctly classified; misleading for imbalanced data
        Precision   — of predicted positives, how many are truly positive
        Recall      — of actual positives, how many were caught (sensitivity)
        F1 Score    — harmonic mean of precision and recall; balances both
        ROC-AUC     — area under the ROC curve; threshold-independent measure
                      (binary only — for multi-class, use macro/weighted average)

    Fine-tuning tip: if recall >> precision, lower the decision threshold;
    if precision >> recall, raise the threshold.

    Args:
        model:  fitted pipeline
        X_test: test features
        y_test: true labels

    Returns:
        dict of metric names to values
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {"accuracy": accuracy_score(y_test, y_pred)}

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ROC-AUC (binary only)
    if y_prob.shape[1] == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics


def cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    K-fold cross-validation.

    Uses 'roc_auc' scoring for binary problems — a more informative metric
    than accuracy for classification.

    Fine-tuning tip: large gap between train and CV score → overfitting →
    increase regularization (lower C) or reduce features.
    """
    scoring = "roc_auc" if len(np.unique(y)) == 2 else "accuracy"
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    print(f"\n{cv}-Fold CV {scoring} scores: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Grid search over regularization strength and penalty type.

    The C parameter is the most impactful knob:
        C > 1  → less regularization (higher variance, lower bias)
        C < 1  → more regularization (lower variance, higher bias)

    Fine-tuning tip: try logarithmically spaced values for C.
    Add 'solver' to the param grid if you need to test L1 vs L2.

    Args:
        X_train: training features
        y_train: training labels

    Returns:
        best fitted pipeline
    """
    pipeline = build_pipeline()
    param_grid = {
        "model__C":       [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__penalty": ["l1", "l2"],
        "model__solver":  ["saga"],   # saga supports both l1 and l2
    }
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV AUC: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_confusion_matrix(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Display confusion matrix heatmap.

    Rows = actual class, Columns = predicted class.
    Off-diagonal cells are misclassifications.

    Fine-tuning tip: if false negatives are costly (e.g., cancer detection),
    prioritize recall by lowering the decision threshold below 0.5.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Logistic Regression — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("logistic_regression_confusion.png", dpi=150)
    plt.show()


def plot_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    ROC curve for binary classification.

    The curve plots True Positive Rate (recall) vs. False Positive Rate at
    all decision thresholds. AUC = 1.0 is perfect; AUC = 0.5 is random guessing.

    Fine-tuning tip: choose the operating point on the curve that matches
    your precision/recall trade-off requirement.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logistic_regression_roc.png", dpi=150)
    plt.show()


def plot_coefficients(model, feature_names: list | None = None) -> None:
    """
    Horizontal bar chart of logistic regression coefficients.

    Positive coefficient → feature increases log-odds of the positive class.
    Negative coefficient → feature decreases log-odds of the positive class.
    Near-zero coefficient → feature has little linear effect.

    Note: only comparable after feature scaling (StandardScaler applied in pipeline).
    """
    coefs = model.named_steps["model"].coef_[0]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(len(coefs))]
    sorted_idx = np.argsort(coefs)
    plt.figure(figsize=(8, 5))
    plt.barh(
        [feature_names[i] for i in sorted_idx],
        coefs[sorted_idx],
        color=["salmon" if c < 0 else "steelblue" for c in coefs[sorted_idx]],
        edgecolor="k",
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient (log-odds)")
    plt.title("Logistic Regression — Feature Coefficients")
    plt.tight_layout()
    plt.savefig("logistic_regression_coefficients.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # 1. Load data
    X, y = load_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Build and train
    print("=== Logistic Regression ===")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 3. Evaluate
    print("\nTest Set Metrics:")
    evaluate(pipeline, X_test, y_test)

    # 4. Cross-validation
    cross_validate(pipeline, X, y)

    # 5. Hyperparameter tuning
    print("\n=== Hyperparameter Tuning ===")
    best_model = tune_hyperparameters(X_train, y_train)
    print("\nBest Model — Test Metrics:")
    evaluate(best_model, X_test, y_test)

    # 6. Visualizations
    plot_confusion_matrix(pipeline, X_test, y_test)
    plot_roc_curve(pipeline, X_test, y_test)
    plot_coefficients(pipeline)


if __name__ == "__main__":
    main()
