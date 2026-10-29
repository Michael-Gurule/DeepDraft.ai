"""
Double Machine Learning (DML) — Causal Effect Estimation
==========================================================
A semiparametric framework for estimating causal/treatment effects from
observational data using machine learning for nuisance function estimation,
while maintaining valid statistical inference (confidence intervals, p-values).

The fundamental problem:
    We want to estimate θ in: Y = θ D + g(X) + ε
    where Y is the outcome, D is the treatment, X are confounders.
    Naively regressing Y on D ignores confounding: E[D ε] ≠ 0.

Frisch-Waugh-Lovell (FWL) theorem:
    θ can be recovered by:
    1. Partial out X from Y: Ỹ = Y - E[Y|X]
    2. Partial out X from D: D̃ = D - E[D|X]
    3. Regress Ỹ on D̃: θ̂ = (D̃'D̃)⁻¹ D̃'Ỹ
    This removes confounding without requiring a parametric model for g(X).

Cross-fitting (the DML innovation):
    If E[Y|X] and E[D|X] are fit on the same data used to estimate θ,
    regularization bias contaminates θ̂ (Neyman orthogonality fails).
    Fix: K-fold cross-fitting — fit nuisance on fold k, predict on held-out fold.
    This achieves √n-consistent, asymptotically normal estimation of θ
    even when the nuisance functions converge at slower rates (e.g., n^{-1/3}).

Neyman orthogonality:
    The moment condition ∂/∂η E[ψ(W; θ, η)] = 0 at the true η.
    DML's FWL structure satisfies this — first-order nuisance errors cancel,
    enabling valid inference despite nonparametric estimation.

Use Cases:
    - Program evaluation (treatment effect of a policy/intervention)
    - Pricing (causal effect of price on demand, controlling for seasonality)
    - Medicine (effect of a drug controlling for patient characteristics)
    - Economics (wage returns to education controlling for ability)

Advantages:
    - Uses flexible ML for nuisance estimation (no misspecification)
    - Valid confidence intervals and hypothesis tests
    - Works with high-dimensional confounders
    - Handles heterogeneous treatment effects (CATE) via extensions

Limitations:
    - Assumes no unmeasured confounding (unconfoundedness / selection on observables)
    - Requires overlap: 0 < P(D=1|X) < 1 for binary treatment
    - Treatment effect θ is a single summary (ATE) — extensions needed for CATE
    - Cross-fitting reduces effective sample size

Fine-Tuning Recommendations:
    1. n_splits: 5 (standard). More folds → more data for each nuisance model.
    2. Nuisance model: try GradientBoosting (default), RandomForest, Lasso.
       More flexible models help with many confounders (high-dimensional X).
    3. Binary treatment: ensure propensity score E[D|X] is well-calibrated;
       check overlap by plotting propensity score histograms by treatment group.
    4. Continuous treatment: use a regression model for both E[Y|X] and E[D|X].
    5. Heterogeneous effects: after DML, run a CATE model on D̃, Ỹ by subgroup.
    6. Robustness: bootstrap θ̂ confidence intervals as a sanity check.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats

from deepdraft.core import fast_or, save_figure


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_observational_data(
    n_samples: int = 2000,
    n_confounders: int = 10,
    true_ate: float = 2.0,
    treatment_type: str = "continuous",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate observational data with confounding.

    Data generating process:
        X ~ N(0, I)             (confounders)
        D = f(X) + ε_D          (treatment depends on X — confounding)
        Y = θ D + g(X) + ε_Y    (outcome depends on D and X)

    Naive OLS on (Y, D) will be biased because D correlates with ε_Y through X.
    DML should recover the true ATE = θ.

    Args:
        n_samples:       number of observations
        n_confounders:   number of confounding variables
        true_ate:        true average treatment effect θ
        treatment_type:  'continuous' or 'binary'
        seed:            random seed

    Returns:
        X: (n_samples, n_confounders) confounders
        D: (n_samples,) treatment
        Y: (n_samples,) outcome
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_confounders))

    # Nonlinear confounder effects
    g_X = (0.5 * X[:, 0] ** 2 + 0.3 * X[:, 1] * X[:, 2]
           - 0.4 * np.abs(X[:, 3]) + 0.2 * X[:, 4])

    if treatment_type == "continuous":
        D = 1.5 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] ** 2 + rng.standard_normal(n_samples)
    else:
        propensity = 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2])))
        D = rng.binomial(1, propensity).astype(float)

    Y = true_ate * D + g_X + rng.standard_normal(n_samples)
    return X, D, Y


# ─────────────────────────────────────────────
# DML ESTIMATOR
# ─────────────────────────────────────────────

class DoubleMLEstimator:
    """
    Partially Linear DML estimator for the Average Treatment Effect (ATE).

    Implements the "DML2" variant (Robinson 1988 + Chernozhukov et al. 2018):
        Pool cross-fitted residuals across all folds before final OLS.

    The estimator is:
        θ̂ = (Σ D̃_i²)⁻¹ Σ D̃_i Ỹ_i
    with the heteroskedasticity-robust (sandwich) variance of
    Chernozhukov et al. (2018):
        Var(θ̂) = Σ D̃_i² ε̂_i² / (Σ D̃_i²)²,   ε̂_i = Ỹ_i - θ̂ D̃_i
    The simpler σ² / Σ D̃_i² assumes the outcome noise has the same variance
    for every unit; it understates the standard error when it does not.

    With a binary treatment and a heterogeneous effect, this partially linear
    θ̂ is a variance-weighted average of the unit effects, not the ATE. Use
    AIPWEstimator below for the ATE in that case.

    Args:
        n_splits:         K-fold cross-fitting splits
        outcome_model:    sklearn regressor for E[Y|X]
        treatment_model:  sklearn regressor/classifier for E[D|X]
        treatment_type:   'continuous' (regression) or 'binary' (classification)
        confidence_level: for confidence interval construction
    """

    def __init__(
        self,
        n_splits: int = 5,          # Placeholder: 5 is standard; 10 for small samples
        outcome_model=None,         # Placeholder: GradientBoosting, RandomForest, Lasso
        treatment_model=None,       # Placeholder: same options; LogisticRegression for binary
        treatment_type: str = "continuous",
        confidence_level: float = 0.95,
    ):
        self.n_splits = n_splits
        self.treatment_type = treatment_type
        self.confidence_level = confidence_level

        self.outcome_model   = outcome_model   or GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        self.treatment_model = treatment_model or (
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            if treatment_type == "binary"
            else GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        )

        # Fitted attributes
        self.theta_: float | None = None
        self.se_: float | None = None
        self.ci_: tuple | None = None
        self.p_value_: float | None = None
        self.Y_residuals_: np.ndarray | None = None
        self.D_residuals_: np.ndarray | None = None

    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> "DoubleMLEstimator":
        """
        Fit DML via K-fold cross-fitting.

        Phase 1 (nuisance estimation):
            For each fold k:
                - Fit outcome model  on train_k → predict on test_k → Ỹ_k = Y_k - Ŷ_k
                - Fit treatment model on train_k → predict on test_k → D̃_k = D_k - D̂_k

        Phase 2 (final estimation):
            Pool all (D̃, Ỹ) residuals → OLS → θ̂

        Args:
            X: (n, p) confounder matrix
            D: (n,) treatment vector
            Y: (n,) outcome vector

        Returns:
            self (fitted)
        """
        import copy
        n = len(Y)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        Y_resid = np.zeros(n)
        D_resid = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            D_train, D_test = D[train_idx], D[test_idx]

            # Outcome model: E[Y|X]
            out_model = copy.deepcopy(self.outcome_model)
            out_model.fit(X_train, Y_train)
            Y_hat = out_model.predict(X_test)
            Y_resid[test_idx] = Y_test - Y_hat

            # Treatment model: E[D|X]
            trt_model = copy.deepcopy(self.treatment_model)
            trt_model.fit(X_train, D_train)
            if self.treatment_type == "binary" and hasattr(trt_model, "predict_proba"):
                D_hat = trt_model.predict_proba(X_test)[:, 1]
            else:
                D_hat = trt_model.predict(X_test)
            D_resid[test_idx] = D_test - D_hat

        # Final OLS: θ̂ = (D̃'D̃)⁻¹ D̃'Ỹ
        denom = np.dot(D_resid, D_resid)
        theta = np.dot(D_resid, Y_resid) / denom

        # Heteroskedasticity-robust (sandwich) standard error
        residuals = Y_resid - theta * D_resid
        se = np.sqrt(np.sum(D_resid ** 2 * residuals ** 2)) / denom

        # Confidence interval and t-test
        z_crit = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci = (theta - z_crit * se, theta + z_crit * se)
        t_stat = theta / se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        self.theta_      = theta
        self.se_         = se
        self.ci_         = ci
        self.p_value_    = p_value
        self.Y_residuals_ = Y_resid
        self.D_residuals_ = D_resid
        return self

    def summary(self) -> dict:
        """Return estimated ATE with inference statistics."""
        return {
            "ATE":      self.theta_,
            "SE":       self.se_,
            "CI_lower": self.ci_[0],
            "CI_upper": self.ci_[1],
            "t_stat":   self.theta_ / self.se_,
            "p_value":  self.p_value_,
        }


class AIPWEstimator:
    """
    Augmented inverse-propensity-weighted (AIPW) estimator of the ATE for a
    binary treatment, cross-fitted. Also called the DML interactive
    regression model (IRM).

    For each unit, with cross-fitted nuisances g_d(X) = E[Y | D=d, X] and
    propensity m(X) = P(D=1 | X):
        ψ_i = g_1(X_i) - g_0(X_i)
              + D_i (Y_i - g_1(X_i)) / m(X_i)
              - (1 - D_i)(Y_i - g_0(X_i)) / (1 - m(X_i))
        θ̂ = mean(ψ),   SE = std(ψ) / √n

    ψ is doubly robust: θ̂ stays consistent if either the outcome models or
    the propensity model is right. Unlike the partially linear model, it
    targets the ATE even when the treatment effect varies across units.

    Args:
        n_splits:         K-fold cross-fitting splits
        outcome_model:    sklearn regressor for E[Y | D=d, X]
        propensity_model: sklearn classifier for P(D=1 | X)
        clip:             propensity clipping bound; weights explode near 0 or 1
        confidence_level: for confidence interval construction
    """

    def __init__(self, n_splits: int = 5, outcome_model=None, propensity_model=None,
                 clip: float = 0.01, confidence_level: float = 0.95):
        self.n_splits = n_splits
        self.outcome_model = outcome_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42)
        self.propensity_model = propensity_model or GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42)
        self.clip = clip
        self.confidence_level = confidence_level

    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> "AIPWEstimator":
        import copy
        n = len(Y)
        g0, g1, m = np.zeros(n), np.zeros(n), np.zeros(n)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            X_tr, D_tr, Y_tr = X[train_idx], D[train_idx], Y[train_idx]
            for d, g in ((0, g0), (1, g1)):
                model = copy.deepcopy(self.outcome_model)
                model.fit(X_tr[D_tr == d], Y_tr[D_tr == d])
                g[test_idx] = model.predict(X[test_idx])
            ps = copy.deepcopy(self.propensity_model)
            ps.fit(X_tr, D_tr)
            m[test_idx] = ps.predict_proba(X[test_idx])[:, 1]
        m = np.clip(m, self.clip, 1 - self.clip)

        psi = g1 - g0 + D * (Y - g1) / m - (1 - D) * (Y - g0) / (1 - m)
        self.theta_ = float(psi.mean())
        self.se_ = float(psi.std(ddof=1) / np.sqrt(n))
        z_crit = stats.norm.ppf((1 + self.confidence_level) / 2)
        self.ci_ = (self.theta_ - z_crit * self.se_, self.theta_ + z_crit * self.se_)
        self.p_value_ = float(2 * (1 - stats.norm.cdf(abs(self.theta_ / self.se_))))
        self.propensity_ = m
        return self

    def summary(self) -> dict:
        """Return estimated ATE with inference statistics."""
        return {
            "ATE":      self.theta_,
            "SE":       self.se_,
            "CI_lower": self.ci_[0],
            "CI_upper": self.ci_[1],
            "t_stat":   self.theta_ / self.se_,
            "p_value":  self.p_value_,
        }


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_ate_estimators(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    true_ate: float,
    treatment_type: str = "continuous",
) -> dict[str, dict]:
    """
    Compare DML against naive OLS and a single-split (non-cross-fitted) baseline.

    Metrics:
        bias:           |θ̂ - θ_true|
        CI_coverage:    True if θ_true ∈ [CI_lower, CI_upper]
        CI_width:       confidence interval width (narrower = more precise)

    Args:
        X:              confounders
        D:              treatment
        Y:              outcome
        true_ate:       ground truth θ
        treatment_type: 'continuous' or 'binary'

    Returns:
        dict of method name → summary dict
    """
    results = {}

    # 1. Naive OLS (biased because it ignores confounding)
    D_col = D.reshape(-1, 1)
    XD = np.column_stack([D_col, X])
    coef, _, _, _ = np.linalg.lstsq(XD, Y, rcond=None)
    naive_theta = coef[0]
    results["Naive OLS"] = {
        "ATE":      naive_theta,
        "SE":       float("nan"),
        "CI_lower": float("nan"),
        "CI_upper": float("nan"),
        "p_value":  float("nan"),
        "bias":     abs(naive_theta - true_ate),
    }

    # 2. DML with GradientBoosting
    dml = DoubleMLEstimator(n_splits=fast_or(5, 2), treatment_type=treatment_type)
    dml.fit(X, D, Y)
    s = dml.summary()
    s["bias"] = abs(s["ATE"] - true_ate)
    s["CI_coverage"] = s["CI_lower"] <= true_ate <= s["CI_upper"]
    results["DML (GradientBoosting)"] = s

    # 3. DML with Lasso nuisance
    lasso_out = LassoCV(cv=3, random_state=42, max_iter=2000)
    lasso_trt = LassoCV(cv=3, random_state=42, max_iter=2000) if treatment_type == "continuous" \
                else LogisticRegressionCV(cv=3, max_iter=500, random_state=42)
    dml_lasso = DoubleMLEstimator(n_splits=fast_or(5, 2), outcome_model=lasso_out,
                                   treatment_model=lasso_trt, treatment_type=treatment_type)
    dml_lasso.fit(X, D, Y)
    s2 = dml_lasso.summary()
    s2["bias"] = abs(s2["ATE"] - true_ate)
    s2["CI_coverage"] = s2["CI_lower"] <= true_ate <= s2["CI_upper"]
    results["DML (Lasso)"] = s2

    return results


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_residualization(dml: DoubleMLEstimator, true_ate: float) -> None:
    """
    Plot the FWL residuals D̃ vs Ỹ and the estimated treatment effect line.

    The slope of this scatter (after partialling out X) equals θ̂.
    Compare with the raw D vs Y scatter to see how confounding inflated the raw slope.
    """
    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    D_resid = dml.D_residuals_
    Y_resid = dml.Y_residuals_

    axes.scatter(D_resid, Y_resid, alpha=0.2, s=8, color="steelblue", label="Residuals")

    # Fitted line: slope = θ̂
    d_range = np.linspace(D_resid.min(), D_resid.max(), 100)
    axes.plot(d_range, dml.theta_ * d_range, color="darkorange", linewidth=2,
              label=f"DML slope θ̂ = {dml.theta_:.3f}")
    axes.plot(d_range, true_ate * d_range, color="green", linewidth=2, linestyle="--",
              label=f"True ATE θ = {true_ate:.3f}")

    axes.set_xlabel("D̃ (treatment residual)")
    axes.set_ylabel("Ỹ (outcome residual)")
    axes.set_title("Double ML — FWL Residuals")
    axes.legend()
    plt.tight_layout()
    save_figure("dml_residuals.png", dpi=150)


def plot_estimator_comparison(results: dict, true_ate: float) -> None:
    """
    Forest plot comparing ATE estimates and confidence intervals across methods.

    Methods closer to the true ATE dashed line are less biased.
    Wider intervals indicate less precision.
    """
    methods = list(results.keys())
    ates    = [results[m]["ATE"]      for m in methods]
    lowers  = [results[m]["CI_lower"] for m in methods]
    uppers  = [results[m]["CI_upper"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, max(3, len(methods) * 1.0)))
    y_pos = range(len(methods))

    for i, (m, ate, lo, hi) in enumerate(zip(methods, ates, lowers, uppers)):
        ax.plot(ate, i, "o", color="steelblue", markersize=8)
        if not (np.isnan(lo) or np.isnan(hi)):
            ax.hlines(i, lo, hi, color="steelblue", linewidth=2)

    ax.axvline(true_ate, color="red", linestyle="--", label=f"True ATE = {true_ate:.1f}")
    ax.set_yticks(y_pos); ax.set_yticklabels(methods)
    ax.set_xlabel("Estimated ATE")
    ax.set_title("ATE Estimator Comparison — Forest Plot")
    ax.legend()
    plt.tight_layout()
    save_figure("dml_comparison.png", dpi=150)


def plot_overlap(D: np.ndarray, propensity: np.ndarray | None = None) -> None:
    """
    Overlap (common support) check for binary treatment.

    Propensity score histograms should overlap between treated and control.
    Poor overlap (no shared support) violates the positivity assumption and
    makes causal estimates unreliable.

    Args:
        D:          (n,) binary treatment indicator
        propensity: (n,) estimated propensity scores (if None, skip plot)
    """
    if propensity is None:
        print("No propensity scores provided — overlap check skipped.")
        return

    plt.figure(figsize=(7, 4))
    plt.hist(propensity[D == 0], bins=30, alpha=0.5, color="steelblue",
             density=True, label="Control (D=0)")
    plt.hist(propensity[D == 1], bins=30, alpha=0.5, color="darkorange",
             density=True, label="Treated (D=1)")
    plt.xlabel("Propensity score P(D=1|X)")
    plt.ylabel("Density")
    plt.title("Overlap Check — Propensity Score Distribution")
    plt.legend()
    plt.tight_layout()
    save_figure("dml_overlap.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Double Machine Learning (DML) — ATE Estimation ===")

    true_ate = 2.0

    # 1. Continuous treatment
    print("\n--- Continuous Treatment ---")
    X, D, Y = generate_observational_data(
        n_samples=fast_or(2000, 400), n_confounders=10,
        true_ate=true_ate, treatment_type="continuous",
    )
    print(f"Data shape: X={X.shape}, D={D.shape}, Y={Y.shape}")

    dml = DoubleMLEstimator(n_splits=fast_or(5, 2), treatment_type="continuous")
    dml.fit(X, D, Y)
    s = dml.summary()
    print(f"DML ATE:    {s['ATE']:.4f} (true: {true_ate})")
    print(f"95% CI:     [{s['CI_lower']:.4f}, {s['CI_upper']:.4f}]")
    print(f"p-value:    {s['p_value']:.4f}")

    results = evaluate_ate_estimators(X, D, Y, true_ate, "continuous")
    for method, res in results.items():
        print(f"  {method}: ATE={res['ATE']:.3f}, bias={res['bias']:.3f}")

    plot_residualization(dml, true_ate)
    plot_estimator_comparison(results, true_ate)

    # 2. Binary treatment
    print("\n--- Binary Treatment ---")
    X_b, D_b, Y_b = generate_observational_data(
        n_samples=fast_or(2000, 400), n_confounders=10,
        true_ate=true_ate, treatment_type="binary",
    )
    dml_b = DoubleMLEstimator(n_splits=fast_or(5, 2), treatment_type="binary")
    dml_b.fit(X_b, D_b, Y_b)
    s_b = dml_b.summary()
    print(f"DML ATE (binary): {s_b['ATE']:.4f} (true: {true_ate})")
    print(f"95% CI:           [{s_b['CI_lower']:.4f}, {s_b['CI_upper']:.4f}]")

    # AIPW targets the ATE directly; it matches the partially linear estimate
    # here because the simulated effect is the same for every unit.
    aipw = AIPWEstimator(n_splits=fast_or(5, 2)).fit(X_b, D_b, Y_b)
    s_a = aipw.summary()
    print(f"AIPW ATE:         {s_a['ATE']:.4f} (true: {true_ate})")
    print(f"95% CI:           [{s_a['CI_lower']:.4f}, {s_a['CI_upper']:.4f}]")

    # Overlap check with the cross-fitted propensity scores. In-sample scores
    # from a flexible classifier overfit toward 0 and 1 and exaggerate poor overlap.
    plot_overlap(D_b, aipw.propensity_)


if __name__ == "__main__":
    main()
