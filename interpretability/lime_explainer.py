"""
LIME — Local Interpretable Model-agnostic Explanations
========================================================
Explains individual predictions of any black-box model by fitting a simple,
interpretable surrogate model (linear regression) in the local neighborhood
of a specific input instance.

Core idea (Ribeiro et al., 2016):
    For a black-box model f and an input x, LIME:
    1. Perturbs x by randomly sampling z in the feature space near x.
    2. Queries f(z) for each perturbed sample.
    3. Weights perturbations by proximity to x: w(z) = exp(-d(x,z)² / σ²)
    4. Fits a weighted linear model g on {(z, f(z), w(z))}.
    5. Returns g's coefficients as local feature attributions.

Formal objective:
    ξ(x) = argmin_{g ∈ G} L(f, g, π_x) + Ω(g)

    Where:
        L:   fidelity — how well g approximates f near x (weighted MSE)
        π_x: proximity kernel — Gaussian weights by distance from x
        Ω:   complexity penalty — sparsity via Lasso (limits number of features)
        G:   class of interpretable models (linear)

Tabular LIME:
    Features are perturbed by replacing feature values with samples from the
    training distribution. Each perturbation is represented as a binary vector
    (feature present/absent in its original value).

Text LIME:
    Words are the "features"; each perturbation hides a subset of words.

Image LIME (superpixels):
    Contiguous image regions (superpixels) are the features; perturbations
    hide/show superpixel groups.

Use Cases:
    - Explaining any classification or regression model (RF, XGBoost, DNN, SVM)
    - Healthcare: why did the model flag this patient as high risk?
    - Finance: what drove this loan application rejection?
    - Debugging: find which features cause counterintuitive predictions

Advantages:
    - Model-agnostic: works for any f with a predict() interface
    - Intuitive: linear surrogate is human-readable
    - Local: explanations are tailored to each instance

Limitations:
    - Instability: perturbing different random samples can give different explanations
    - The locality assumption (linear ≈ f near x) may fail for highly nonlinear f
    - n_samples trade-off: more samples = more stable but slower
    - Surrogate fidelity (R²) is not always high — check before trusting

Fine-Tuning Recommendations:
    1. n_samples: 500–5000. More is better. Use 2000+ for stable explanations.
    2. kernel_width σ: controls locality. Smaller σ = more local (but noisy);
       larger σ = more global. Default: 0.75 * sqrt(n_features).
    3. n_features_explanation: 5–10 for readability; fewer = sparser Lasso.
    4. Feature perturbation: for correlated features, segment-based perturbation
       (hide whole groups) is more stable than individual feature perturbation.
    5. Stability check: run LIME 5–10 times on the same instance; report
       coefficient standard deviation. High std = unreliable explanation.
    6. Surrogate R²: report it alongside explanations. R² < 0.7 = low fidelity.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data() -> tuple:
    """Load breast cancer dataset for binary classification."""
    data = load_breast_cancer()
    scaler = StandardScaler()
    X = scaler.fit_transform(data.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data.target, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names), scaler


def load_regression_data() -> tuple:
    """Load California housing dataset for regression."""
    data = fetch_california_housing()
    scaler = StandardScaler()
    X = scaler.fit_transform(data.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data.target, test_size=0.2, random_state=42,
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names), scaler


# ─────────────────────────────────────────────
# LIME EXPLAINER
# ─────────────────────────────────────────────

class LIMETabularExplainer:
    """
    LIME for tabular data (continuous features).

    Explains a single prediction by fitting a local linear surrogate model
    on perturbed samples weighted by their proximity to the explained instance.

    Perturbation strategy (tabular):
        For each feature i, sample from N(x_i, σ_i) where σ_i is the
        training set standard deviation of feature i. This keeps perturbations
        in the plausible data range while exploring the local neighborhood.

    Args:
        model:          fitted black-box model with predict() or predict_proba()
        X_train:        (n, p) training data (used for feature statistics)
        feature_names:  list of feature names
        n_samples:      number of perturbation samples per explanation
        kernel_width:   bandwidth σ of the exponential kernel; None = auto
        n_features_exp: number of features to include in the sparse explanation
        mode:           'classification' (uses predict_proba) or 'regression'
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: list[str],
        n_samples: int = 1000,          # Placeholder: 500 (fast), 5000 (stable)
        kernel_width: float | None = None,  # Placeholder: 0.25–2.0 × sqrt(n_features)
        n_features_exp: int = 10,       # Placeholder: 5–15 for readable explanations
        mode: str = "classification",
    ):
        self.model       = model
        self.X_train     = X_train
        self.feature_names = feature_names
        self.n_samples   = n_samples
        self.n_features_exp = n_features_exp
        self.mode        = mode

        # Feature statistics for perturbation
        self._means  = X_train.mean(axis=0)
        self._stds   = X_train.std(axis=0) + 1e-10

        # Kernel width: Gaussian proximity kernel
        n_features = X_train.shape[1]
        self.kernel_width = kernel_width or 0.75 * np.sqrt(n_features)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Query the black-box model; returns probability for class 1 or regression value."""
        if self.mode == "classification" and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def _perturb(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample perturbations around x.

        Strategy: z_i ~ N(x_i, σ_i) for continuous features.
        The first row is always x itself (so g(x) = f(x) is anchored).

        Returns:
            Z:         (n_samples, n_features) perturbed samples in feature space
            Z_binary:  (n_samples, n_features) binary representation (1 = close to x_i)
        """
        n_features = len(x)
        Z = np.random.randn(self.n_samples, n_features) * self._stds + x
        Z[0] = x  # anchor the explained instance

        # Binary representation: 1 if within 1 std of x, else 0
        Z_binary = (np.abs(Z - x) <= self._stds).astype(float)
        return Z, Z_binary

    def _kernel(self, Z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Exponential (Gaussian) proximity kernel: π_x(z) = exp(-d(x,z)² / σ²)

        Weights perturbations by their Euclidean distance from x in feature space.
        Closer perturbations get higher weight → local fidelity.
        """
        dists = np.sqrt(np.sum((Z - x) ** 2, axis=1))
        return np.exp(-(dists ** 2) / (self.kernel_width ** 2))

    def explain_instance(
        self,
        x: np.ndarray,
        label: int | None = None,
    ) -> dict:
        """
        Explain the prediction at a single instance x.

        Algorithm:
            1. Perturb x → Z (n_samples, n_features)
            2. Query black-box: f_Z = f(Z)
            3. Compute weights: w = kernel(Z, x)
            4. Fit sparse local linear model: g = Lasso(Z_binary, f_Z, weights=w)
            5. Return g's non-zero coefficients as local feature importances

        Args:
            x:     (n_features,) instance to explain (standardized)
            label: class index for classification (None = class 1)

        Returns:
            dict with:
                feature_importances: dict {feature_name: coefficient}
                prediction:          model's prediction at x
                surrogate_r2:        fidelity of the local surrogate
                intercept:           surrogate model intercept
        """
        Z, Z_binary = self._perturb(x)
        f_Z         = self._predict(Z)
        weights     = self._kernel(Z, x)

        # Fit sparse surrogate with weighted Lasso
        # Lasso promotes sparsity — only n_features_exp features get non-zero coefs
        alpha = self._select_lasso_alpha(Z_binary, f_Z, weights)
        surrogate = Lasso(alpha=alpha, max_iter=5000, fit_intercept=True)
        surrogate.fit(Z_binary, f_Z, sample_weight=weights)

        # If Lasso zeros too many features, fall back to Ridge
        n_nonzero = np.sum(surrogate.coef_ != 0)
        if n_nonzero < 2:
            surrogate = Ridge(alpha=0.01)
            surrogate.fit(Z_binary, f_Z, sample_weight=weights)

        # Surrogate fidelity: weighted R²
        f_pred = surrogate.predict(Z_binary)
        ss_res = np.sum(weights * (f_Z - f_pred) ** 2)
        ss_tot = np.sum(weights * (f_Z - np.average(f_Z, weights=weights)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))

        # Top features by coefficient magnitude
        coefs = surrogate.coef_
        top_k = np.argsort(np.abs(coefs))[::-1][:self.n_features_exp]
        feature_importances = {
            self.feature_names[i]: float(coefs[i]) for i in top_k if coefs[i] != 0
        }

        return {
            "feature_importances": feature_importances,
            "prediction":          float(self._predict(x.reshape(1, -1))[0]),
            "surrogate_r2":        r2,
            "intercept":           float(surrogate.intercept_),
            "all_coefs":           coefs,
        }

    def _select_lasso_alpha(
        self,
        Z_binary: np.ndarray,
        f_Z: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Heuristic to select Lasso alpha that retains approximately n_features_exp features.

        Tries a grid of alphas; picks the one that gives the closest non-zero count.
        Avoids expensive cross-validation for speed.
        """
        target = self.n_features_exp
        for alpha in [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]:
            m = Lasso(alpha=alpha, max_iter=2000, fit_intercept=True)
            m.fit(Z_binary, f_Z, sample_weight=weights)
            n_nonzero = np.sum(m.coef_ != 0)
            if n_nonzero <= target:
                return alpha
        return 0.01

    def stability_report(
        self,
        x: np.ndarray,
        n_runs: int = 5,
    ) -> dict:
        """
        Estimate explanation stability by running LIME n_runs times.

        Computes mean coefficient per feature and coefficient of variation (CV).
        High CV = unreliable explanation for that feature.

        Args:
            x:      instance to explain
            n_runs: number of independent LIME runs

        Returns:
            dict with mean_coefs, std_coefs, and cv per feature
        """
        all_coefs = []
        for _ in range(n_runs):
            result = self.explain_instance(x)
            all_coefs.append(result["all_coefs"])

        all_coefs = np.array(all_coefs)   # (n_runs, n_features)
        mean_coefs = all_coefs.mean(axis=0)
        std_coefs  = all_coefs.std(axis=0)
        cv = np.where(np.abs(mean_coefs) > 1e-6, std_coefs / (np.abs(mean_coefs) + 1e-10), 0.0)

        return {
            "mean_coefs": mean_coefs,
            "std_coefs":  std_coefs,
            "cv":         cv,
        }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_local_explanation(
    result: dict,
    sample_idx: int = 0,
    title: str = "LIME — Local Explanation",
    filename: str = "lime_explanation.png",
) -> None:
    """
    Horizontal bar chart of a single LIME explanation.

    Red bars = features pushing prediction toward class 1 (or higher value).
    Blue bars = features pushing toward class 0 (or lower value).
    Width = magnitude of local influence.

    Args:
        result: output of LIMETabularExplainer.explain_instance()
    """
    fi = result["feature_importances"]
    pred = result["prediction"]
    r2   = result["surrogate_r2"]

    features = list(fi.keys())
    values   = list(fi.values())
    order    = np.argsort(values)

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.45)))
    colors = ["tomato" if v > 0 else "steelblue" for v in np.array(values)[order]]
    ax.barh(range(len(features)), np.array(values)[order], color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in order], fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Local coefficient (attribution)")
    ax.set_title(f"{title}\nprediction={pred:.3f} | surrogate R²={r2:.3f}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_global_lime(
    explainer: LIMETabularExplainer,
    X_explain: np.ndarray,
    title: str = "LIME — Global Feature Importance",
    filename: str = "lime_global.png",
    top_k: int = 15,
) -> None:
    """
    Aggregate LIME explanations across many samples to estimate global importance.

    Mean |LIME coefficient| across all explained instances.
    Less theoretically grounded than SHAP global, but complements local analysis.

    Args:
        X_explain: (n, p) samples to explain (subsample for speed)
    """
    n_features = X_explain.shape[1]
    all_coefs = []
    for i in range(len(X_explain)):
        r = explainer.explain_instance(X_explain[i])
        all_coefs.append(r["all_coefs"])

    all_coefs = np.abs(np.array(all_coefs))   # (n, p)
    mean_abs = all_coefs.mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]

    plt.figure(figsize=(8, max(4, top_k * 0.35)))
    plt.barh(range(top_k), mean_abs[order][::-1], color="steelblue")
    plt.yticks(range(top_k), [explainer.feature_names[i] for i in order[::-1]], fontsize=9)
    plt.xlabel("Mean |LIME coefficient|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_stability(
    stability: dict,
    feature_names: list[str],
    top_k: int = 15,
    filename: str = "lime_stability.png",
) -> None:
    """
    Bar chart with error bars showing LIME coefficient stability.

    Each bar = mean coefficient; error bar = ±1 std across n_runs.
    Short error bars = stable explanation; long error bars = unreliable.

    Coefficient of variation (CV) is shown in the title.
    """
    mean_coefs = stability["mean_coefs"]
    std_coefs  = stability["std_coefs"]

    order = np.argsort(np.abs(mean_coefs))[::-1][:top_k]
    mn = mean_coefs[order]
    sd = std_coefs[order]
    fn = [feature_names[i] for i in order]
    colors = ["tomato" if v > 0 else "steelblue" for v in mn]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.45)))
    ax.barh(range(top_k), mn[::-1], xerr=sd[::-1], color=colors[::-1],
            error_kw={"elinewidth": 1.5, "capsize": 3})
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(fn[::-1], fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("LIME coefficient (mean ± std across runs)")
    ax.set_title("LIME Stability — Coefficient Variability")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_fidelity_distribution(
    fidelities: list[float],
    filename: str = "lime_fidelity.png",
) -> None:
    """
    Histogram of surrogate model R² across many explained instances.

    R² distribution tells you how often the local linear model is a good
    approximation. High R² throughout = LIME is trustworthy for this model.
    Low R² = the model is too nonlinear for a linear surrogate locally.
    """
    plt.figure(figsize=(7, 4))
    plt.hist(fidelities, bins=20, color="steelblue", edgecolor="white", density=True)
    plt.axvline(np.mean(fidelities), color="darkorange", linestyle="--",
                label=f"Mean R² = {np.mean(fidelities):.3f}")
    plt.axvline(0.7, color="red", linestyle=":", label="R² = 0.7 threshold")
    plt.xlabel("Surrogate R² (local fidelity)")
    plt.ylabel("Density")
    plt.title("LIME — Surrogate Fidelity Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== LIME — Local Interpretable Model-agnostic Explanations ===")

    # 1. Classification
    print("\n--- Classification: Breast Cancer ---")
    X_train, X_test, y_train, y_test, feat_names, _ = load_classification_data()

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")

    explainer = LIMETabularExplainer(
        clf, X_train, feat_names,
        n_samples=1000, n_features_exp=10, mode="classification",
    )

    # Explain a single instance
    result = explainer.explain_instance(X_test[0])
    print(f"\nSample 0 — prediction: {result['prediction']:.3f} | "
          f"surrogate R²: {result['surrogate_r2']:.3f}")
    print(f"Top features: {dict(list(result['feature_importances'].items())[:5])}")
    plot_local_explanation(result, sample_idx=0, filename="lime_explanation_clf.png")

    # Stability check
    print("\nStability analysis (5 runs on same sample)...")
    stability = explainer.stability_report(X_test[0], n_runs=5)
    plot_stability(stability, feat_names)

    # Global importance (over 50 test samples)
    print("\nComputing global LIME importance (50 samples)...")
    plot_global_lime(explainer, X_test[:50],
                     title="LIME Global Importance — Breast Cancer",
                     filename="lime_global_clf.png")

    # Fidelity distribution
    print("Computing fidelity distribution (100 samples)...")
    fidelities = [explainer.explain_instance(X_test[i])["surrogate_r2"]
                  for i in range(100)]
    plot_fidelity_distribution(fidelities)

    # 2. Regression
    print("\n--- Regression: California Housing ---")
    X_tr, X_te, y_tr, y_te, reg_feat_names, _ = load_regression_data()

    reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    reg.fit(X_tr, y_tr)
    print(f"Test R²: {reg.score(X_te, y_te):.3f}")

    reg_explainer = LIMETabularExplainer(
        reg, X_tr, reg_feat_names,
        n_samples=1000, n_features_exp=8, mode="regression",
    )
    reg_result = reg_explainer.explain_instance(X_te[0])
    print(f"Sample 0 — prediction: {reg_result['prediction']:.3f} | "
          f"R²: {reg_result['surrogate_r2']:.3f}")
    plot_local_explanation(reg_result, title="LIME — California Housing",
                           filename="lime_explanation_reg.png")


if __name__ == "__main__":
    main()
