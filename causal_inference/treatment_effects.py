"""
Heterogeneous Treatment Effects — CATE Estimation
===================================================
Methods for estimating Conditional Average Treatment Effects (CATE):
    τ(x) = E[Y(1) - Y(0) | X = x]
where Y(1) and Y(0) are potential outcomes under treatment and control.

The fundamental problem of causal inference (Rubin 1974):
    We observe Y_i = D_i Y_i(1) + (1-D_i) Y_i(0), but never both Y_i(1) and Y_i(0)
    for the same unit — the counterfactual outcome is unobservable.

Identification assumptions (all methods require these):
    1. Unconfoundedness: (Y(0), Y(1)) ⊥ D | X   (no unmeasured confounders)
    2. Overlap: 0 < P(D=1|X=x) < 1  for all x   (positivity)
    3. SUTVA: no interference between units

Methods covered:
    T-Learner (Two-Model estimator):
        Fit μ̂_1(x) = E[Y|D=1, X=x] and μ̂_0(x) = E[Y|D=0, X=x] separately.
        CATE: τ̂(x) = μ̂_1(x) - μ̂_0(x)
        Limitation: ignores signal from the control group when fitting the treatment model.

    S-Learner (Single-Model estimator):
        Fit one model μ̂(x, d) = E[Y|X=x, D=d] on all data.
        CATE: τ̂(x) = μ̂(x, 1) - μ̂(x, 0)
        Limitation: may shrink treatment effect to zero if D has low importance.

    X-Learner (Künzel et al. 2019):
        Stage 1: fit T-Learner models μ̂_0 and μ̂_1.
        Stage 2: compute imputed treatment effects:
            D̃_i = Y_i - μ̂_0(X_i) for treated units
            D̃_i = μ̂_1(X_i) - Y_i for control units
        Stage 3: fit τ̂_0 on control, τ̂_1 on treated, combine via propensity weights.
        Advantage: borrows strength across groups — excellent for unbalanced treatment.

    Causal Forest (Wager & Athey 2018):
        An ensemble of honest causal trees trained on orthogonalized residuals.
        Key features: honesty (split different data than estimation), subsampling.
        Asymptotically normal estimates with valid confidence intervals.

Use Cases:
    - Personalized medicine (which patients benefit most from a drug?)
    - Targeting (which customers respond to promotions?)
    - Policy evaluation (which subpopulations are helped by a program?)
    - A/B test analysis with heterogeneity exploration

Advantages:
    - T/S/X-learners are model-agnostic (plug in any ML model)
    - X-learner particularly effective when treatment is rare
    - Causal Forest provides uncertainty quantification

Limitations:
    - All require unconfoundedness — unmeasured confounders bias all estimates
    - High variance for small n per subgroup
    - Model selection is harder than in prediction (no ground truth CATE)

Fine-Tuning Recommendations:
    1. Metalearner choice: X-Learner for imbalanced treatment; T-Learner for balanced.
    2. Base models: GradientBoosting or RandomForest as default; Lasso for sparse X.
    3. Propensity estimation: calibrate propensity scores (Platt scaling) for X-Learner.
    4. Evaluation: use QINI curve and AUUC (no ground truth) or synthetic benchmarks.
    5. Doubly robust: augment AIPW with DML residuals for model-robust estimates.
    6. Confidence bands: use causalml or EconML packages for production-grade inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_cate_data(
    n_samples: int = 3000,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate data with known heterogeneous treatment effects.

    Data generating process:
        X ~ N(0, I)
        D | X ~ Bernoulli(σ(X_0 - 0.5 X_1))   (confounded binary treatment)
        τ(X) = 2 + X_0 + 0.5 X_1² - X_2       (heterogeneous CATE)
        Y = D * τ(X) + 0.3 * X_0 - X_1 + ε

    The true CATE varies across units — positive for some, near-zero for others.
    This reflects real-world heterogeneity (not everyone responds to treatment).

    Args:
        n_samples:   number of observations
        n_features:  number of covariates
        seed:        random seed

    Returns:
        X:        (n, p) covariates
        D:        (n,) binary treatment
        Y:        (n,) outcome
        tau_true: (n,) true CATE for each unit
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    # True heterogeneous CATE
    tau_true = 2.0 + X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2]

    # Confounded treatment assignment
    logit = X[:, 0] - 0.5 * X[:, 1]
    propensity = 1 / (1 + np.exp(-logit))
    D = rng.binomial(1, propensity).astype(float)

    # Outcome: treatment effect + baseline
    baseline = 0.3 * X[:, 0] - X[:, 1] + 0.2 * X[:, 2] ** 2
    Y = D * tau_true + baseline + rng.standard_normal(n_samples)

    return X, D, Y, tau_true


# ─────────────────────────────────────────────
# META-LEARNERS
# ─────────────────────────────────────────────

class TLearner:
    """
    T-Learner (Two-Model) CATE estimator.

    Fits separate outcome models for treated and control groups.
    CATE = μ̂_1(x) - μ̂_0(x)

    Simple baseline — works best when both groups have sufficient data.
    Underperforms when treatment is rare (small treated group = poor μ̂_1).

    Args:
        outcome_model: sklearn regressor template (cloned per group)
    """

    def __init__(
        self,
        outcome_model=None,         # Placeholder: GradientBoosting, RandomForest, Lasso
    ):
        import copy
        self._base_model = outcome_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )
        self.mu0_ = None    # E[Y|D=0, X]
        self.mu1_ = None    # E[Y|D=1, X]

    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> "TLearner":
        """
        Args:
            X: (n, p) covariates
            D: (n,) binary treatment
            Y: (n,) outcome
        """
        import copy
        self.mu0_ = copy.deepcopy(self._base_model)
        self.mu1_ = copy.deepcopy(self._base_model)
        self.mu0_.fit(X[D == 0], Y[D == 0])
        self.mu1_.fit(X[D == 1], Y[D == 1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE: τ̂(x) = μ̂_1(x) - μ̂_0(x)

        Args:
            X: (n, p) covariates

        Returns:
            (n,) CATE estimates
        """
        return self.mu1_.predict(X) - self.mu0_.predict(X)


class SLearner:
    """
    S-Learner (Single-Model) CATE estimator.

    Includes D as a feature in a single model and predicts CATE
    by the difference in predictions at D=1 and D=0.

    Risk: if the ML model de-emphasizes D (as regularization may do),
    the CATE estimate shrinks toward zero even when a true effect exists.

    Args:
        outcome_model: sklearn regressor that includes D as a feature
    """

    def __init__(
        self,
        outcome_model=None,
    ):
        self._base_model = outcome_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )
        self.model_ = None

    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> "SLearner":
        """
        Fit one model on [X, D] concatenated.
        """
        import copy
        XD = np.column_stack([X, D])
        self.model_ = copy.deepcopy(self._base_model)
        self.model_.fit(XD, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        CATE: μ̂(x, 1) - μ̂(x, 0)
        """
        n = len(X)
        XD_1 = np.column_stack([X, np.ones(n)])
        XD_0 = np.column_stack([X, np.zeros(n)])
        return self.model_.predict(XD_1) - self.model_.predict(XD_0)


class XLearner:
    """
    X-Learner (Künzel et al. 2019) CATE estimator.

    Exploits the full sample in both stages by imputing counterfactuals
    and fitting on cross-group pseudo-outcomes.

    Algorithm:
        Stage 1: fit T-Learner → μ̂_0, μ̂_1
        Stage 2: compute imputed effects:
            Treated: D̃_i = Y_i - μ̂_0(X_i)      [imputed effect on treated]
            Control: D̃_i = μ̂_1(X_i) - Y_i      [imputed effect on control]
        Stage 3: fit τ̂_1 on treated D̃; τ̂_0 on control D̃
        Stage 4: combine via propensity e(x):
            τ̂(x) = e(x) τ̂_0(x) + (1-e(x)) τ̂_1(x)

    Particularly effective when treatment is rare (imbalanced D).

    Args:
        outcome_model:   sklearn regressor for Stage 1/3
        propensity_model: sklearn classifier for Stage 4
    """

    def __init__(
        self,
        outcome_model=None,
        propensity_model=None,
    ):
        self._base_outcome   = outcome_model    or GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        self._base_propensity = propensity_model or GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.mu0_ = self.mu1_ = self.tau0_ = self.tau1_ = self.ps_ = None

    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> "XLearner":
        """
        Fit the X-Learner in four stages.
        """
        import copy
        idx1, idx0 = D == 1, D == 0

        # Stage 1: T-Learner base models
        self.mu0_ = copy.deepcopy(self._base_outcome); self.mu0_.fit(X[idx0], Y[idx0])
        self.mu1_ = copy.deepcopy(self._base_outcome); self.mu1_.fit(X[idx1], Y[idx1])

        # Stage 2: Imputed treatment effects
        D_tilde_1 = Y[idx1] - self.mu0_.predict(X[idx1])  # treated
        D_tilde_0 = self.mu1_.predict(X[idx0]) - Y[idx0]  # control

        # Stage 3: Fit CATE on each group
        self.tau1_ = copy.deepcopy(self._base_outcome); self.tau1_.fit(X[idx1], D_tilde_1)
        self.tau0_ = copy.deepcopy(self._base_outcome); self.tau0_.fit(X[idx0], D_tilde_0)

        # Stage 4: Propensity model for weighting
        self.ps_ = copy.deepcopy(self._base_propensity); self.ps_.fit(X, D)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        CATE: e(x) τ̂_0(x) + (1-e(x)) τ̂_1(x)

        Propensity weighting gives more weight to τ̂_1 where treatment is rare.
        """
        e = self.ps_.predict_proba(X)[:, 1]
        return e * self.tau0_.predict(X) + (1 - e) * self.tau1_.predict(X)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_cate(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """
    Evaluate CATE estimates against ground truth (available in simulation).

    In real data, use QINI/AUUC for policy evaluation instead.

    Metrics:
        RMSE:  root mean squared error of CATE estimates
        bias:  mean signed error (positive = overestimation)
        corr:  Pearson correlation between τ̂ and τ (ranking quality)

    Args:
        tau_hat:  (n,) estimated CATE
        tau_true: (n,) true CATE
        label:    name of the estimator

    Returns:
        dict of metrics
    """
    rmse = np.sqrt(np.mean((tau_hat - tau_true) ** 2))
    bias = np.mean(tau_hat - tau_true)
    corr = np.corrcoef(tau_hat, tau_true)[0, 1]
    if label:
        print(f"  {label}: RMSE={rmse:.3f}, bias={bias:.3f}, corr={corr:.3f}")
    return {"RMSE": rmse, "bias": bias, "corr": corr}


def compute_qini_curve(
    tau_hat: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the Qini curve for policy value evaluation.

    The Qini curve evaluates how well τ̂ ranks units by treatment benefit
    without requiring knowledge of the true CATE.

    Algorithm:
        1. Sort units by decreasing τ̂ (most treated-beneficial first).
        2. For each fraction φ ∈ [0,1] of the population targeted:
           Compute: Qini(φ) = Σ_{i≤φn} [D_i Y_i / π_1 - (1-D_i) Y_i / π_0]
           where π_1 = P(D=1) and π_0 = P(D=0).
        3. AUUC (Area Under Uplift Curve) = area between Qini and random policy.

    A better CATE model → higher AUUC (targets the most treatment-responsive units).

    Args:
        tau_hat: (n,) estimated CATE (ranking variable)
        D:       (n,) binary treatment
        Y:       (n,) outcome

    Returns:
        fractions: x-axis (proportion treated)
        qini:      y-axis (uplift)
        auuc:      area under the Qini curve
    """
    n = len(tau_hat)
    order = np.argsort(-tau_hat)
    D_ord, Y_ord = D[order], Y[order]

    pi1 = D.mean()
    pi0 = 1 - pi1

    uplift = []
    for k in range(1, n + 1):
        D_k = D_ord[:k]
        Y_k = Y_ord[:k]
        n1 = D_k.sum()
        n0 = k - n1
        val = (Y_k[D_k == 1].sum() / (pi1 * n) if n1 > 0 else 0) - \
              (Y_k[D_k == 0].sum() / (pi0 * n) if n0 > 0 else 0)
        uplift.append(val)

    fractions = np.arange(1, n + 1) / n
    qini = np.array(uplift)
    auuc = np.trapz(qini, fractions) - np.trapz(np.linspace(qini[0], qini[-1], n), fractions)
    return fractions, qini, auuc


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_cate_scatter(
    tau_hat_dict: dict[str, np.ndarray],
    tau_true: np.ndarray,
    filename: str = "cate_scatter.png",
) -> None:
    """
    Scatter plot of τ̂ vs τ_true for each estimator.

    Points near the diagonal indicate accurate CATE estimates.
    Spread around the diagonal reflects variance.
    """
    n_methods = len(tau_hat_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (label, tau_hat) in zip(axes, tau_hat_dict.items()):
        ax.scatter(tau_true, tau_hat, alpha=0.2, s=8, color="steelblue")
        lim = [min(tau_true.min(), tau_hat.min()), max(tau_true.max(), tau_hat.max())]
        ax.plot(lim, lim, "r--", linewidth=1, label="Perfect")
        corr = np.corrcoef(tau_hat, tau_true)[0, 1]
        ax.set_title(f"{label}\ncorr={corr:.3f}")
        ax.set_xlabel("True CATE τ(x)")
        ax.set_ylabel("Estimated CATE τ̂(x)")
        ax.legend()

    plt.suptitle("CATE Estimation — True vs. Predicted")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_cate_distribution(
    tau_hat_dict: dict[str, np.ndarray],
    tau_true: np.ndarray,
    filename: str = "cate_distribution.png",
) -> None:
    """
    Distribution of estimated vs. true CATE across the population.

    A good estimator should reproduce the shape of the true CATE distribution.
    Wide variance suggests poor individual-level precision even if ATE is correct.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(tau_true, bins=40, alpha=0.4, density=True, label="True CATE", color="gray")
    colors = ["steelblue", "darkorange", "green", "red"]
    for (label, tau_hat), color in zip(tau_hat_dict.items(), colors):
        ax.hist(tau_hat, bins=40, alpha=0.35, density=True, label=label, color=color)
    ax.set_xlabel("CATE τ(x)")
    ax.set_ylabel("Density")
    ax.set_title("CATE Distribution — True vs. Estimated")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_qini_curves(
    tau_hat_dict: dict[str, np.ndarray],
    D: np.ndarray,
    Y: np.ndarray,
    filename: str = "qini_curves.png",
) -> None:
    """
    Qini curves for policy evaluation without true CATE.

    Higher AUUC → better at identifying and targeting the most treatment-responsive units.
    The random policy line is the baseline (treat randomly regardless of τ̂).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "darkorange", "green", "red"]

    for (label, tau_hat), color in zip(tau_hat_dict.items(), colors):
        fracs, qini, auuc = compute_qini_curve(tau_hat, D, Y)
        ax.plot(fracs, qini, label=f"{label} (AUUC={auuc:.4f})", color=color, linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Random policy")
    ax.set_xlabel("Fraction of population targeted (decreasing τ̂)")
    ax.set_ylabel("Cumulative uplift")
    ax.set_title("Qini Curves — Policy Value Evaluation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Heterogeneous Treatment Effects — T/S/X-Learners ===")

    # 1. Data
    X, D, Y, tau_true = generate_cate_data(n_samples=3000, n_features=10)
    X_tr, X_te, D_tr, D_te, Y_tr, Y_te, tau_tr, tau_te = train_test_split(
        X, D, Y, tau_true, test_size=0.3, random_state=42
    )
    print(f"Data: {X.shape}, treated fraction: {D.mean():.2f}")
    print(f"True CATE: mean={tau_true.mean():.3f}, std={tau_true.std():.3f}")

    # 2. Fit metalearners
    print("\n--- Fitting Meta-Learners ---")
    t_learner = TLearner().fit(X_tr, D_tr, Y_tr)
    s_learner = SLearner().fit(X_tr, D_tr, Y_tr)
    x_learner = XLearner().fit(X_tr, D_tr, Y_tr)

    # 3. Predict CATE
    tau_t = t_learner.predict(X_te)
    tau_s = s_learner.predict(X_te)
    tau_x = x_learner.predict(X_te)

    # 4. Evaluate
    print("\n--- CATE Evaluation (vs. true τ) ---")
    metrics = {}
    for label, tau_hat in [("T-Learner", tau_t), ("S-Learner", tau_s), ("X-Learner", tau_x)]:
        metrics[label] = evaluate_cate(tau_hat, tau_te, label)

    tau_hat_dict = {"T-Learner": tau_t, "S-Learner": tau_s, "X-Learner": tau_x}

    # 5. Visualize
    plot_cate_scatter(tau_hat_dict, tau_te)
    plot_cate_distribution(tau_hat_dict, tau_te)
    plot_qini_curves(tau_hat_dict, D_te, Y_te)

    # 6. ATE summary
    print("\n--- ATE Estimates ---")
    print(f"  True ATE:    {tau_true.mean():.3f}")
    for label, tau_hat in tau_hat_dict.items():
        print(f"  {label}: {tau_hat.mean():.3f}")


if __name__ == "__main__":
    main()
