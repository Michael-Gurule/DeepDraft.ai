"""
Gaussian Process Regression
=============================
A non-parametric Bayesian model that places a prior distribution over functions.
A GP defines a distribution over functions f(x) such that any finite collection
of function values follows a joint Gaussian distribution.

Mathematical formulation:
    Prior:     f(x) ~ GP(m(x), k(x, x'))
    Posterior: f(x*) | X, y ~ N(μ*, Σ*)

    Where:
        m(x)      = mean function (often 0)
        k(x, x')  = kernel/covariance function
        μ*        = K(X*, X) [K(X,X) + σ²I]⁻¹ y
        Σ*        = K(X*,X*) - K(X*,X) [K(X,X) + σ²I]⁻¹ K(X,X*)

    The posterior uncertainty grows away from observed data — key for
    uncertainty-aware predictions (sensor fusion, active learning).

Kernel Choices:
    RBF (Squared Exponential): smooth functions, length_scale controls correlation range.
    Matérn (ν=1.5 or 2.5): rougher functions; ν=2.5 is often best in practice.
    Rational Quadratic: mixture of RBF kernels at different scales (multi-scale).
    Periodic: cyclic data (e.g., daily/weekly seasonality in time series).
    Composite: RBF + WhiteKernel (noise), or RBF * Periodic (smooth+cyclic).

Use Cases:
    - Uncertainty-aware regression (sensor readings, financial forecasting)
    - Time series interpolation / gap filling
    - Hyperparameter optimization (as surrogate in Bayesian Optimization)
    - Multi-output regression (with MultiOutputRegressor wrapper)
    - Active learning (query points with highest uncertainty)
    - Spatial statistics / Kriging

Advantages:
    - Principled uncertainty estimates (not just point predictions)
    - Flexible via kernel engineering
    - Works well with small-to-medium datasets

Limitations:
    - O(n³) training complexity — scales poorly past ~10k samples
    - Kernel selection is non-trivial; wrong kernel → poor calibration
    - High-dimensional inputs require ARD kernels (per-feature length scales)

Fine-Tuning Recommendations:
    - Kernel: start with Matern(nu=2.5) + WhiteKernel. Matern handles
      non-smooth real-world functions better than RBF.
    - length_scale: governs smoothness. Tune via log-marginal likelihood
      optimization (sklearn does this automatically with optimizer='fmin_l_bfgs_b').
    - n_restarts_optimizer: 5–10 to avoid local optima in kernel hyperparameters.
    - normalize_y=True: almost always beneficial — centers the output.
    - For large n: use GPyTorch (sparse GP, inducing points) or GPflow.
    - ARD (Automatic Relevance Determination): use separate length_scale per
      feature to perform implicit feature selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_regression_data(
    n_train: int = 100,
    n_test:  int = 200,
    noise:   float = 0.3,
):
    """
    Synthetic 1D regression data: f(x) = sin(x) + 0.5*sin(3x).

    Replace with your own data:
        import pandas as pd
        df = pd.read_csv("sensor_readings.csv")
        X_train = df[["feature_1", "feature_2"]].values
        y_train = df["target"].values

    Returns:
        X_train, y_train, X_test, y_test, X_full (for plotting)
    """
    rng = np.random.RandomState(42)
    X_train = np.sort(rng.uniform(-5, 5, n_train))[:, None].astype(np.float32)
    y_train = (np.sin(X_train.ravel()) + 0.5 * np.sin(3 * X_train.ravel())
               + rng.normal(0, noise, n_train)).astype(np.float32)

    X_test  = np.linspace(-7, 7, n_test)[:, None].astype(np.float32)
    y_test  = (np.sin(X_test.ravel()) + 0.5 * np.sin(3 * X_test.ravel())).astype(np.float32)

    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# KERNEL LIBRARY
# ─────────────────────────────────────────────

def get_kernels() -> dict:
    """
    Common GP kernels for experimentation.

    Placeholder: swap kernel in GP_CONFIG to match your data's smoothness.
    """
    return {
        "rbf": C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        "matern_15": C(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1),
        "matern_25": C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
        "rational_quadratic": C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=0.1),
    }


# Placeholder: swap kernel name or tune bounds
GP_CONFIG = {
    "kernel":              "matern_25",    # Placeholder: 'rbf', 'matern_15', 'matern_25', 'rational_quadratic'
    "n_restarts_optimizer": 5,             # Placeholder: 5–15 restarts for robust hyperparameter optimization
    "normalize_y":          True,          # Placeholder: almost always True
    "alpha":                1e-10,         # Placeholder: jitter for numerical stability (increase if Cholesky fails)
}


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_gp(config: dict = None) -> GaussianProcessRegressor:
    """
    Build a GaussianProcessRegressor with the specified kernel.

    The kernel hyperparameters (length_scale, noise_level, etc.) are optimized
    by maximizing the log-marginal likelihood during fit().
    """
    if config is None:
        config = GP_CONFIG
    kernel = get_kernels()[config["kernel"]]
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=config["n_restarts_optimizer"],
        normalize_y=config["normalize_y"],
        alpha=config["alpha"],
    )


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model: GaussianProcessRegressor,
             X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray,  y_test: np.ndarray) -> dict:
    """
    Evaluate GP regression: RMSE, MAE, R², plus log-marginal likelihood.

    Returns:
        dict of metrics
    """
    y_pred, y_std = model.predict(X_test, return_std=True)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    lml  = model.log_marginal_likelihood_value_

    # Calibration: what % of test points fall within 95% confidence interval
    in_ci = np.mean(np.abs(y_test - y_pred) < 1.96 * y_std)

    print(f"  RMSE:       {rmse:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"  R²:         {r2:.4f}")
    print(f"  Log MLL:    {lml:.4f}")
    print(f"  95% CI cov: {in_ci*100:.1f}%  (ideal: ~95%)")
    print(f"  Kernel:     {model.kernel_}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "log_mll": lml, "ci_coverage": in_ci}


def cross_validate(X: np.ndarray, y: np.ndarray, config: dict = None,
                   n_splits: int = 5) -> None:
    """K-fold cross-validation for GP regression."""
    model = build_gp(config)
    cv    = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    neg_rmse = cross_val_score(model, X, y, cv=cv,
                                scoring="neg_root_mean_squared_error")
    rmse_scores = -neg_rmse
    print(f"  {n_splits}-Fold CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")


# ─────────────────────────────────────────────
# KERNEL COMPARISON
# ─────────────────────────────────────────────

def compare_kernels(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray,  y_test: np.ndarray) -> None:
    """
    Fit GP with each available kernel and report test RMSE + log-MLL.
    Use this to select the best kernel for your data.
    """
    results = {}
    for name, kernel in get_kernels().items():
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True)
        gp.fit(X_train, y_train)
        y_pred = gp.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        lml    = gp.log_marginal_likelihood_value_
        results[name] = {"rmse": rmse, "log_mll": lml}
        print(f"  {name:<25} RMSE={rmse:.4f}  LogMLL={lml:.2f}")

    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n  Best kernel by RMSE: {best}")


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_gp_posterior(model: GaussianProcessRegressor,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray,  y_test: np.ndarray) -> None:
    """
    Plot GP posterior: mean prediction, uncertainty bands, training points.

    Shaded region = 2σ (95% credible interval).
    Widening bands far from training data = epistemic uncertainty.
    """
    y_pred, y_std = model.predict(X_test, return_std=True)

    plt.figure(figsize=(12, 5))
    plt.plot(X_test.ravel(), y_test,  "k--", linewidth=1.0, label="True function", alpha=0.6)
    plt.plot(X_test.ravel(), y_pred,  "steelblue",  linewidth=2.0, label="GP mean")
    plt.fill_between(X_test.ravel(),
                     y_pred - 2 * y_std,
                     y_pred + 2 * y_std,
                     alpha=0.25, color="steelblue", label="95% CI (±2σ)")
    plt.scatter(X_train.ravel(), y_train, c="tomato", s=30, zorder=5, label="Training data")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Gaussian Process Regression — Posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gp_posterior.png", dpi=150)
    plt.show()


def plot_uncertainty(model: GaussianProcessRegressor,
                     X_test: np.ndarray) -> None:
    """
    Plot predictive standard deviation across input space.
    High std = uncertain region (far from training data or complex structure).
    """
    _, y_std = model.predict(X_test, return_std=True)

    plt.figure(figsize=(10, 3))
    plt.plot(X_test.ravel(), y_std, color="darkorange", linewidth=2.0)
    plt.fill_between(X_test.ravel(), 0, y_std, alpha=0.3, color="orange")
    plt.xlabel("x")
    plt.ylabel("Predictive Std σ(x)")
    plt.title("Gaussian Process — Predictive Uncertainty")
    plt.tight_layout()
    plt.savefig("gp_uncertainty.png", dpi=150)
    plt.show()


def plot_kernel_comparison(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray) -> None:
    """Plot GP posteriors for all kernels in a single figure."""
    kernels = get_kernels()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for ax, (name, kernel) in zip(axes, kernels.items()):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True)
        gp.fit(X_train, y_train)
        y_pred, y_std = gp.predict(X_test, return_std=True)

        ax.plot(X_test.ravel(), y_pred, "steelblue", linewidth=2)
        ax.fill_between(X_test.ravel(), y_pred - 2*y_std, y_pred + 2*y_std,
                        alpha=0.25, color="steelblue")
        ax.scatter(X_train.ravel(), y_train, c="tomato", s=20, zorder=5)
        ax.set_title(f"{name}\nkernel: {gp.kernel_}", fontsize=8)

    plt.suptitle("GP Kernel Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig("gp_kernel_comparison.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# GP CLASSIFICATION (Laplace approximation)
# ─────────────────────────────────────────────

def generate_classification_data(
    n_samples: int = 200,
    n_features: int = 2,
    seed: int = 42,
):
    """
    Two-class synthetic dataset with a non-linear boundary.

    Replace with your own data:
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
    """
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=seed)
    return X.astype(np.float32), y


def build_gp_classifier(
    kernel_name: str = "rbf",
    n_restarts_optimizer: int = 3,          # Placeholder: 5–10 for production
    multi_class: str = "one_vs_rest",       # Placeholder: 'one_vs_rest' or 'one_vs_one'
    max_iter_predict: int = 100,            # Placeholder: increase for convergence on hard problems
) -> GaussianProcessClassifier:
    """
    GP classifier using Laplace approximation for the posterior.

    sklearn's GaussianProcessClassifier approximates the non-Gaussian
    posterior p(f|X,y) ∝ p(y|f) p(f) with a Gaussian via Laplace:
        - Finds the MAP estimate f* of the latent function
        - Approximates the posterior as N(f*, (∇²log p)⁻¹)
        - Squashes through the sigmoid (binary) or softmax (multiclass)

    Laplace is fast but can underestimate uncertainty near the decision
    boundary. Alternatives: EP (more accurate), MCMC (exact but slow).

    Kernel choice for classification:
        - RBF: smooth decision boundaries (default)
        - Matérn ν=1.5: slightly rougher; good for noisy data
        - DotProduct: linear classifier (use when linear boundary suffices)

    Args:
        kernel_name:         one of 'rbf', 'matern_15', 'matern_25'
        multi_class:         strategy for >2 classes
        max_iter_predict:    Newton-CG iterations for Laplace approximation

    Returns:
        Unfitted GaussianProcessClassifier
    """
    kernel_map = {
        "rbf": C(1.0) * RBF(length_scale=1.0),
        "matern_15": C(1.0) * Matern(length_scale=1.0, nu=1.5),
        "matern_25": C(1.0) * Matern(length_scale=1.0, nu=2.5),
    }
    kernel = kernel_map.get(kernel_name, kernel_map["rbf"])
    return GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        multi_class=multi_class,
        max_iter_predict=max_iter_predict,
        random_state=42,
    )


def evaluate_gp_classifier(
    clf: GaussianProcessClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Accuracy, ROC-AUC, and calibration check for GP classifier."""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Expected Calibration: fraction of test points where argmax probability > 0.5
    confident = np.mean(np.max(clf.predict_proba(X_test), axis=1) > 0.5)

    print(f"  Accuracy:      {acc:.4f}")
    print(f"  ROC-AUC:       {auc:.4f}")
    print(f"  Kernel:        {clf.kernel_}")
    print(f"  Log MLL:       {clf.log_marginal_likelihood_value_:.4f}")
    print(f"  Confident preds (>0.5): {confident*100:.1f}%")

    cv_scores = cross_val_score(clf.__class__(kernel=clf.kernel_),
                                X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {"accuracy": acc, "roc_auc": auc}


def plot_gp_classification(
    clf: GaussianProcessClassifier,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "GP Classification — Decision Boundary",
) -> None:
    """
    Plot GP classifier decision boundary and class probabilities on 2D data.
    Colour intensity encodes the predicted probability P(y=1|x).
    Contour line at P=0.5 is the decision boundary.
    """
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.8)
    plt.colorbar(label="P(y=1 | x)")
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5, linestyles="--")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r",
                          edgecolors="k", linewidths=0.5, s=40, zorder=5)
    plt.title(title)
    plt.xlabel("x₁"); plt.ylabel("x₂")
    plt.tight_layout()
    plt.savefig("gp_classification.png", dpi=150)
    plt.show()
    print("Saved gp_classification.png")


def compare_classification_kernels(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> None:
    """Fit GP classifier with each kernel and compare ROC-AUC."""
    print("\n--- GP Classification Kernel Comparison ---")
    for kname in ["rbf", "matern_15", "matern_25"]:
        clf = build_gp_classifier(kname)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print(f"  {kname:<15}  AUC={auc:.4f}  kernel={clf.kernel_}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Gaussian Process Regression ===")

    # Placeholder: replace with your dataset
    X_train, y_train, X_test, y_test = generate_regression_data()
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Build and fit GP
    model = build_gp()
    model.fit(X_train, y_train)

    print("\nEvaluation:")
    evaluate(model, X_train, y_train, X_test, y_test)

    print("\nCross-validation:")
    cross_validate(X_train, y_train)

    print("\nKernel comparison:")
    compare_kernels(X_train, y_train, X_test, y_test)

    # Visualizations
    plot_gp_posterior(model, X_train, y_train, X_test, y_test)
    plot_uncertainty(model, X_test)
    plot_kernel_comparison(X_train, y_train, X_test)

    # ── GP Classification ────────────────────────────────────────────────────
    print("\n=== Gaussian Process Classification (Laplace Approximation) ===")
    from sklearn.model_selection import train_test_split

    X_cls, y_cls = generate_classification_data(n_samples=300)
    X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.25,
                                               random_state=42, stratify=y_cls)
    print(f"Train: {X_tr.shape}  |  Test: {X_te.shape}")

    clf = build_gp_classifier(kernel_name="rbf")   # Placeholder: try 'matern_25'
    clf.fit(X_tr, y_tr)

    print("\nClassification metrics:")
    evaluate_gp_classifier(clf, X_tr, y_tr, X_te, y_te)

    plot_gp_classification(clf, X_cls, y_cls)
    compare_classification_kernels(X_tr, y_tr, X_te, y_te)


if __name__ == "__main__":
    main()
