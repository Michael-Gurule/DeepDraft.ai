"""
SHAP Values — SHapley Additive exPlanations
=============================================
A unified framework for explaining model predictions by assigning each feature
a contribution value based on Shapley values from cooperative game theory.

Core idea:
    For a prediction f(x), the SHAP value φ_i of feature i is the average
    marginal contribution of feature i across all possible feature coalitions:

    φ_i = Σ_{S ⊆ F\\{i}} [|S|!(|F|-|S|-1)! / |F|!] * [f(S∪{i}) - f(S)]

    where F is the full feature set and S ranges over all subsets not containing i.
    This is the unique attribution satisfying: efficiency, symmetry, dummy, and additivity.

Additive explanation property:
    f(x) = E[f(X)] + Σ_i φ_i(x)
    The prediction equals the base value (average) plus the sum of all SHAP values.
    This enables exact decomposition — SHAP values sum to the prediction gap from average.

Main estimators:
    TreeSHAP (Lundberg 2018):
        Exact O(TLD²) algorithm for tree ensembles (RF, XGBoost, LightGBM).
        No sampling — computes exact Shapley values via tree traversal.
    KernelSHAP:
        Model-agnostic; approximates Shapley values via a weighted linear regression
        on coalitions. Slow but works for any model.
    DeepSHAP / GradientSHAP:
        For neural networks; uses backprop rules to propagate SHAP values.
    PermutationSHAP (implemented here):
        Estimates by randomly permuting features; unbiased but high variance.
        Useful for any model when exact methods are unavailable.

Use Cases:
    - Feature importance (global: |mean SHAP|)
    - Instance-level debugging ("why did the model predict X for this sample?")
    - Regulatory compliance (GDPR "right to explanation")
    - Model auditing for bias detection
    - Feature selection based on SHAP importance

Advantages:
    - Theoretically grounded (unique satisfying four axioms)
    - Works for any model (KernelSHAP / PermutationSHAP)
    - Consistent — a feature that increases model output always gets positive SHAP
    - Enables both local (per-instance) and global (dataset-level) explanations

Limitations:
    - KernelSHAP is slow (exponential number of coalitions, approximated)
    - Assumes feature independence when computing conditional expectations
      (violated for correlated features — TreeSHAP has a variant that handles this)
    - SHAP values can be misleading when features are highly correlated

Fine-Tuning Recommendations:
    1. Use TreeSHAP (via the `shap` library) for tree models — it's exact and fast.
    2. n_permutations: 50–200 for PermutationSHAP. Higher = lower variance.
    3. background_samples: 50–200 samples for the baseline distribution (KernelSHAP).
    4. For classification: explain log-odds (link='logit') for better interpretability.
    5. Interaction values: TreeSHAP can compute φ_ij (feature pair interactions) —
       useful to detect features that only matter together.
    6. Summary plots: use beeswarm + absolute mean bar as a two-plot standard.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load breast cancer dataset for binary classification.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, list(data.feature_names)


def load_regression_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load California housing dataset for regression.

    Falls back to scikit-learn's bundled diabetes dataset when the download
    is unavailable (offline machines, CI runners).

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    try:
        data = fetch_california_housing()
    except Exception as exc:  # offline, firewalled, or download failed
        print(f"California housing download failed ({type(exc).__name__}); "
              "falling back to the bundled diabetes dataset.")
        data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, list(data.feature_names)


# ─────────────────────────────────────────────
# SHAP ESTIMATORS
# ─────────────────────────────────────────────

class PermutationSHAP:
    """
    Model-agnostic SHAP estimator via random feature permutations.

    Algorithm (per sample x):
        Repeat n_permutations times:
            1. Draw a random permutation π of feature indices.
            2. For each feature i in order of π:
               - Compute f(features before i in π = from x; rest = from background)
               - Compute f(features at i and before = from x; rest = from background)
               - Marginal contribution = difference of the two
            3. Average marginal contributions over all permutations.

    This is an unbiased estimator of Shapley values but has high variance.
    Use n_permutations ≥ 100 for stable estimates.

    Note: for production use, the `shap` library's TreeExplainer is recommended
    for tree models and is orders of magnitude faster.

    Args:
        model:              any fitted model with a .predict() or .predict_proba() method
        background:         (n_bg, n_features) background dataset for marginal expectations
        n_permutations:     number of random permutations per sample
        use_proba:          if True, explain predict_proba[:,1] (for classifiers)
    """

    def __init__(
        self,
        model,
        background: np.ndarray,
        n_permutations: int = 100,  # Placeholder: 50 (fast), 200 (accurate)
        use_proba: bool = False,
    ):
        self.model = model
        self.background = background
        self.n_permutations = n_permutations
        self.use_proba = use_proba
        self.n_features = background.shape[1]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with optional probability output."""
        if self.use_proba and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def _expected_value(self) -> float:
        """Base value: E[f(X)] estimated on background samples."""
        return float(np.mean(self._predict(self.background)))

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for all samples in X.

        Args:
            X: (n_samples, n_features) instances to explain

        Returns:
            shap_vals: (n_samples, n_features) SHAP values
        """
        n_samples, n_features = X.shape
        shap_vals = np.zeros((n_samples, n_features))

        # Use a small random background for speed
        n_bg = min(50, len(self.background))
        bg_idx = np.random.choice(len(self.background), n_bg, replace=False)
        bg = self.background[bg_idx]

        for s_idx in range(n_samples):
            x = X[s_idx]
            contrib = np.zeros(n_features)

            for _ in range(self.n_permutations):
                perm = np.random.permutation(n_features)
                # Pick one background sample
                bg_sample = bg[np.random.randint(n_bg)].copy()
                x_with    = bg_sample.copy()
                x_without = bg_sample.copy()

                prev_val = float(np.mean(self._predict(bg_sample.reshape(1, -1))))

                for i in perm:
                    x_with[i] = x[i]
                    curr_val = float(np.mean(self._predict(x_with.reshape(1, -1))))
                    contrib[i] += curr_val - prev_val
                    prev_val = curr_val

            shap_vals[s_idx] = contrib / self.n_permutations

        return shap_vals

    @property
    def expected_value(self) -> float:
        """Base value: average model output on background."""
        return self._expected_value()


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_explanation_consistency(
    shap_vals: np.ndarray,
    X: np.ndarray,
    model,
    explainer: PermutationSHAP,
    n_check: int = 20,
) -> dict[str, float]:
    """
    Verify the efficiency axiom: f(x) ≈ E[f(X)] + Σ_i φ_i(x)

    For each of n_check samples, checks that the sum of SHAP values plus
    the base value is close to the actual prediction.

    Args:
        shap_vals: (n_samples, n_features) SHAP values
        X:         (n_samples, n_features) input data
        model:     fitted model
        explainer: PermutationSHAP instance (for base value and predict)
        n_check:   number of samples to verify

    Returns:
        dict with mean_absolute_error and max_absolute_error of the efficiency check
    """
    idx = np.random.choice(len(X), min(n_check, len(X)), replace=False)
    preds  = explainer._predict(X[idx])
    approx = explainer.expected_value + shap_vals[idx].sum(axis=1)
    errors = np.abs(preds - approx)
    return {
        "efficiency_mae": float(np.mean(errors)),
        "efficiency_max": float(np.max(errors)),
    }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_global_importance(
    shap_vals: np.ndarray,
    feature_names: list[str],
    title: str = "SHAP Global Feature Importance",
    filename: str = "shap_global_importance.png",
    top_k: int = 15,
) -> None:
    """
    Bar plot of mean |SHAP| per feature — global feature importance.

    This is the SHAP analogue of permutation importance or MDI (mean decrease impurity).
    Unlike MDI, SHAP importance is consistent and not inflated for high-cardinality features.

    Args:
        shap_vals:    (n_samples, n_features) SHAP values
        feature_names: list of feature names
        top_k:        show only top K features
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]

    plt.figure(figsize=(8, max(4, top_k * 0.35)))
    plt.barh(range(top_k), mean_abs[order][::-1], color="steelblue")
    plt.yticks(range(top_k), [feature_names[i] for i in order[::-1]], fontsize=9)
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_beeswarm(
    shap_vals: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    title: str = "SHAP Beeswarm",
    filename: str = "shap_beeswarm.png",
    top_k: int = 12,
    max_display_points: int = 200,
) -> None:
    """
    Beeswarm plot: each dot is one sample, x-position = SHAP value, color = feature value.

    Combines feature importance (x-range) with feature effect direction (positive/negative)
    and feature value (color: blue=low, red=high). The standard SHAP summary plot.

    Args:
        top_k:               number of top features to show (by mean |SHAP|)
        max_display_points:  subsample for display if n_samples > this
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]

    # Subsample for display
    if shap_vals.shape[0] > max_display_points:
        idx = np.random.choice(shap_vals.shape[0], max_display_points, replace=False)
        sv_display = shap_vals[idx]
        X_display  = X[idx]
    else:
        sv_display, X_display = shap_vals, X

    fig, ax = plt.subplots(figsize=(9, max(5, top_k * 0.45)))
    cmap = plt.cm.RdBu_r

    for plot_row, feat_idx in enumerate(reversed(order)):
        sv_feat   = sv_display[:, feat_idx]
        feat_vals = X_display[:, feat_idx]
        # Normalize feature values to [0,1] for color
        fv_norm = (feat_vals - feat_vals.min()) / (np.ptp(feat_vals) + 1e-10)
        colors  = cmap(fv_norm)

        # Jitter y-axis to spread points
        y_jitter = plot_row + np.random.uniform(-0.3, 0.3, len(sv_feat))
        ax.scatter(sv_feat, y_jitter, c=colors, alpha=0.5, s=10, linewidths=0)

    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in reversed(order)], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Feature value (low → high)", shrink=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_waterfall(
    shap_vals: np.ndarray,
    base_value: float,
    feature_names: list[str],
    sample_idx: int = 0,
    title: str = "SHAP Waterfall",
    filename: str = "shap_waterfall.png",
    top_k: int = 10,
) -> None:
    """
    Waterfall plot for a single prediction: shows how each feature shifts
    the prediction from the base value to the final output.

    Starting from E[f(X)] at the bottom, each bar shows one feature's contribution.
    Red bars push the prediction up; blue bars push it down.

    Args:
        shap_vals:     (n_samples, n_features) SHAP values
        base_value:    E[f(X)] — the model's average prediction
        sample_idx:    which sample to explain
        top_k:         show top K features by absolute SHAP value
    """
    sv = shap_vals[sample_idx]
    order = np.argsort(np.abs(sv))[::-1][:top_k]

    sv_top  = sv[order]
    fn_top  = [feature_names[i] for i in order]
    cumsum  = np.cumsum(sv_top)
    starts  = np.concatenate([[base_value], base_value + cumsum[:-1]])

    colors = ["tomato" if v > 0 else "steelblue" for v in sv_top]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.5)))
    ax.barh(range(top_k), sv_top, left=starts, color=colors, edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(fn_top[::-1][::-1], fontsize=9)
    ax.axvline(base_value, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Model output")
    ax.set_title(f"{title} (sample {sample_idx})")

    final_pred = base_value + sv.sum()
    ax.annotate(f"f(x) = {final_pred:.3f}", xy=(final_pred, -0.5),
                fontsize=9, color="black", ha="center")
    ax.annotate(f"E[f(x)] = {base_value:.3f}", xy=(base_value, top_k),
                fontsize=8, color="gray", ha="center")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_dependence(
    shap_vals: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    feature_idx: int = 0,
    interaction_idx: int | None = None,
    filename: str = "shap_dependence.png",
) -> None:
    """
    SHAP dependence plot: SHAP value of one feature vs. its raw value.

    Reveals the shape of the feature's effect (linear, nonlinear, threshold).
    Color by an interaction feature to detect interaction effects.

    Args:
        feature_idx:     primary feature to plot
        interaction_idx: feature to use for coloring (None = auto-select most correlated)
    """
    sv_feat   = shap_vals[:, feature_idx]
    feat_vals = X[:, feature_idx]

    if interaction_idx is None:
        # Auto-select the feature most correlated with shap residuals
        corrs = [abs(np.corrcoef(X[:, j], sv_feat)[0, 1]) for j in range(X.shape[1])
                 if j != feature_idx]
        interaction_idx = [j for j in range(X.shape[1]) if j != feature_idx][int(np.argmax(corrs))]

    color_vals = X[:, interaction_idx]
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(feat_vals, sv_feat, c=color_vals, cmap="RdBu_r", alpha=0.5, s=15)
    plt.colorbar(sc, label=feature_names[interaction_idx])
    plt.xlabel(feature_names[feature_idx])
    plt.ylabel(f"SHAP({feature_names[feature_idx]})")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(f"SHAP Dependence: {feature_names[feature_idx]}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== SHAP Values — SHapley Additive exPlanations ===")

    # 1. Classification task (breast cancer)
    print("\n--- Classification: Breast Cancer ---")
    X_train, X_test, y_train, y_test, feat_names = load_classification_data()

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")

    # PermutationSHAP (model-agnostic; use shap.TreeExplainer for production speed)
    print("Computing SHAP values (this may take a moment)...")
    explainer = PermutationSHAP(clf, background=X_train, n_permutations=50, use_proba=True)
    shap_vals = explainer.shap_values(X_test[:100])   # explain first 100 test samples

    consistency = evaluate_explanation_consistency(shap_vals, X_test[:100], clf, explainer)
    print(f"Efficiency axiom check — MAE: {consistency['efficiency_mae']:.4f} "
          f"(should be near 0 for exact methods)")

    plot_global_importance(shap_vals, feat_names, title="SHAP Importance — Breast Cancer",
                           filename="shap_global_importance_clf.png")
    plot_beeswarm(shap_vals, X_test[:100], feat_names,
                  title="SHAP Beeswarm — Breast Cancer",
                  filename="shap_beeswarm_clf.png")
    plot_waterfall(shap_vals, explainer.expected_value, feat_names, sample_idx=0,
                   title="SHAP Waterfall — Breast Cancer",
                   filename="shap_waterfall_clf.png")
    plot_dependence(shap_vals, X_test[:100], feat_names, feature_idx=0,
                    filename="shap_dependence_clf.png")

    # 2. Regression task (housing)
    print("\n--- Regression: California Housing ---")
    X_tr, X_te, y_tr, y_te, reg_feat_names = load_regression_data()

    reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    reg.fit(X_tr, y_tr)
    r2 = reg.score(X_te, y_te)
    print(f"Test R²: {r2:.3f}")

    reg_explainer = PermutationSHAP(reg, background=X_tr[:100], n_permutations=50)
    reg_shap_vals = reg_explainer.shap_values(X_te[:50])

    plot_global_importance(reg_shap_vals, reg_feat_names,
                           title="SHAP Importance — California Housing",
                           filename="shap_global_importance_reg.png")
    plot_beeswarm(reg_shap_vals, X_te[:50], reg_feat_names,
                  title="SHAP Beeswarm — California Housing",
                  filename="shap_beeswarm_reg.png")


if __name__ == "__main__":
    main()
