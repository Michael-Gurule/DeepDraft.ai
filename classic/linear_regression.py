"""
Linear Regression
=================
One of the most fundamental supervised learning algorithms. Models the linear
relationship between a continuous dependent variable and one or more independent
variables by minimizing the sum of squared residuals (Ordinary Least Squares).

Use Cases:
    - Predicting continuous outcomes (house prices, sales forecasting, risk scores)
    - Establishing baseline performance for regression tasks
    - Feature importance via coefficient analysis

Assumptions:
    - Linearity: relationship between features and target is linear
    - Independence: observations are independent of each other
    - Homoscedasticity: constant variance of residuals
    - Normality: residuals are normally distributed (for inference)
    - No multicollinearity: features are not highly correlated with each other

Fine-Tuning Recommendations:
    - Regularization: use Ridge (L2) if multicollinearity is present;
      use Lasso (L1) for automatic feature selection; use ElasticNet for both.
    - Feature scaling: StandardScaler or MinMaxScaler improves convergence
      when using gradient-based solvers.
    - Polynomial features: add PolynomialFeatures() to capture non-linear patterns.
    - Outlier handling: robust scalers or Huber regression reduce outlier sensitivity.
    - Feature engineering: interaction terms and log-transforms often improve fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_sample_data(n_samples: int = 500, n_features: int = 5, noise: float = 20.0):
    """
    Generate synthetic regression data.

    Replace this with your own dataset loading logic, e.g.:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["target"]).values
        y = df["target"].values

    Args:
        n_samples:  number of samples to generate
        n_features: number of input features
        noise:      standard deviation of Gaussian noise added to targets

    Returns:
        X: feature matrix (n_samples, n_features)
        y: target vector (n_samples,)
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,      # only 3 features actually drive the target
        noise=noise,
        random_state=42
    )
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: swap the estimator here to switch between variants.
#   LinearRegression() — no regularization (OLS)
#   Ridge(alpha=1.0)   — L2 regularization; shrinks coefficients, handles multicollinearity
#   Lasso(alpha=0.1)   — L1 regularization; zeros out weak features (feature selection)
#   ElasticNet(alpha=0.1, l1_ratio=0.5) — blend of L1 and L2

ESTIMATOR = LinearRegression()

# Placeholder: set to an integer > 1 to add polynomial interaction features.
# E.g., POLY_DEGREE = 2 adds x1^2, x2^2, x1*x2, etc.
POLY_DEGREE = 1  # 1 = no polynomial expansion


def build_pipeline(estimator=ESTIMATOR, poly_degree: int = POLY_DEGREE) -> Pipeline:
    """
    Assemble a preprocessing + model pipeline.

    Pipeline steps:
        1. StandardScaler   — zero mean, unit variance (required for regularized models)
        2. PolynomialFeatures (optional) — adds higher-order and interaction terms
        3. Estimator        — the actual regression model

    Args:
        estimator:   sklearn-compatible regression estimator
        poly_degree: degree of polynomial feature expansion (1 = linear)

    Returns:
        sklearn Pipeline
    """
    steps = [("scaler", StandardScaler())]

    if poly_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))

    steps.append(("model", estimator))
    return Pipeline(steps)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute standard regression metrics on a held-out test set.

    Metrics:
        RMSE — root mean squared error; same units as target, penalizes large errors
        MAE  — mean absolute error; robust to outliers, easier to interpret
        R²   — coefficient of determination; proportion of variance explained (1.0 = perfect)

    Args:
        model:  fitted sklearn estimator or pipeline
        X_test: test features
        y_test: test targets

    Returns:
        dict of metric names to values
    """
    y_pred = model.predict(X_test)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE":  mean_absolute_error(y_test, y_pred),
        "R2":   r2_score(y_test, y_pred),
    }
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")
    return metrics


def cross_validate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    """
    K-fold cross-validation to estimate generalization performance.

    Cross-validation is preferred over a single train/test split because it
    reduces variance in the performance estimate.

    Fine-tuning tip: if CV scores have high variance, increase regularization
    or collect more data.

    Args:
        pipeline: sklearn Pipeline
        X:        full feature matrix
        y:        full target vector
        cv:       number of folds (5 or 10 are common choices)
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    print(f"\n{cv}-Fold CV R² scores: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH (for regularized models)
# ─────────────────────────────────────────────

def tune_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Ridge:
    """
    Grid search over Ridge regularization strength.

    Alpha controls the penalty magnitude:
        - small alpha  → low regularization, closer to OLS (risk: overfitting)
        - large alpha  → strong regularization, shrinks coefficients to zero (risk: underfitting)

    Fine-tuning tip: extend the alpha grid if the best alpha is at either boundary.

    Args:
        X_train: training features
        y_train: training targets

    Returns:
        best fitted Ridge estimator
    """
    param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    pipeline = build_pipeline(estimator=Ridge())
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best Ridge alpha: {search.best_params_['model__alpha']}")
    print(f"Best CV R²: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_predictions(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Scatter plot of actual vs. predicted values.

    A perfect model would show all points on the diagonal line.
    Systematic deviations suggest the model is missing non-linear patterns.

    Fine-tuning tip: fan-shaped scatter → heteroscedasticity → try log-transforming target.
    """
    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Linear Regression — Actual vs. Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_predictions.png", dpi=150)
    plt.show()


def plot_residuals(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Residual plot: predicted values vs. residuals (actual - predicted).

    Desired pattern: residuals randomly scattered around zero (no structure).
    Warning signs:
        - curved pattern → missing non-linear term
        - funnel shape   → heteroscedasticity (non-constant variance)
        - outliers       → influential data points to investigate
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidths=0.3)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig("linear_regression_residuals.png", dpi=150)
    plt.show()


def plot_coefficients(model, feature_names: list | None = None) -> None:
    """
    Bar chart of model coefficients (feature importances).

    Larger absolute value → stronger linear effect on target.
    With Lasso: zero coefficients indicate features excluded by the model.

    Note: coefficients are only comparable when features are on the same scale
    (i.e., after StandardScaler).

    Args:
        model:         fitted pipeline (must include a 'model' step)
        feature_names: list of feature name strings; auto-generated if None
    """
    coefs = model.named_steps["model"].coef_
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(len(coefs))]
    plt.figure(figsize=(8, 4))
    plt.bar(feature_names, coefs, color="steelblue", edgecolor="k")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    plt.title("Linear Regression — Feature Coefficients")
    plt.tight_layout()
    plt.savefig("linear_regression_coefficients.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # 1. Load data
    X, y = load_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Build and train
    print("=== Linear Regression ===")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 3. Evaluate on test set
    print("\nTest Set Metrics:")
    evaluate(pipeline, X_test, y_test)

    # 4. Cross-validation
    cross_validate(pipeline, X, y)

    # 5. Hyperparameter tuning example (Ridge)
    print("\n=== Ridge Hyperparameter Tuning ===")
    best_ridge = tune_ridge(X_train, y_train)
    print("\nBest Ridge — Test Metrics:")
    evaluate(best_ridge, X_test, y_test)

    # 6. Visualizations
    plot_predictions(pipeline, X_test, y_test)
    plot_residuals(pipeline, X_test, y_test)
    plot_coefficients(pipeline)


if __name__ == "__main__":
    main()
