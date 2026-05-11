"""
Bayesian Optimization
======================
An efficient strategy for optimizing expensive black-box functions using a
probabilistic surrogate model (Gaussian Process) and an acquisition function
that balances exploration and exploitation.

Algorithm:
    1. Build a GP surrogate of the objective function f(x).
    2. Maximize an acquisition function α(x) to propose the next query point.
    3. Evaluate f(x_next) (the expensive objective).
    4. Update the GP with the new observation.
    5. Repeat until budget exhausted.

Acquisition Functions:
    Expected Improvement (EI):   α(x) = E[max(f(x) - f*, 0)]
        Picks points expected to improve over current best f*.
        EI = (μ - f* - ξ) Φ(Z) + σ φ(Z),  Z = (μ - f* - ξ) / σ

    Upper Confidence Bound (UCB): α(x) = μ(x) + κ · σ(x)
        Optimistic estimate; κ controls exploration. High κ → explore more.

    Probability of Improvement (PI): α(x) = Φ((μ - f* - ξ) / σ)
        Greedy; useful when f* is not well-established.

Use Cases:
    - Hyperparameter tuning (neural network lr, batch_size, dropout)
    - Chemical/material discovery
    - A/B test parameter selection
    - Marketing budget allocation across channels (Hyperion use case)
    - Engineering design optimization (simulation parameters)
    - Drug dosage optimization

Advantages:
    - Sample-efficient: gets close to optimum in far fewer evaluations than
      grid/random search
    - Uncertainty-aware: knows where it is uncertain
    - Works for non-differentiable, noisy, stochastic objectives

Limitations:
    - GP surrogate: O(n³) — limited to ~1000 observations
    - Not parallelizable out-of-the-box (batched BO requires q-EI)
    - Works best in ~20 or fewer dimensions

Fine-Tuning Recommendations:
    - n_initial_points: 5–10 random points to seed the GP.
    - acquisition: EI is default choice. UCB with κ=2–5 for high uncertainty.
    - xi (ξ) in EI: trade-off parameter. Higher → more exploration (default 0.01).
    - kappa (κ) in UCB: typically 2.0. Increase to 5.0 for more exploration.
    - Kernel: Matern(nu=2.5) + WhiteKernel is robust default.
    - n_restarts_optimizer: 5–10 to escape kernel local optima.
    - For parallelism: use scikit-optimize's ask/tell interface or BoTorch (PyTorch).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# OBJECTIVE FUNCTION (replace with your own)
# ─────────────────────────────────────────────

def objective_function(x: np.ndarray, noise: float = 0.1) -> float:
    """
    Placeholder black-box objective to maximize.

    Replace with your actual expensive function, e.g.:
        - train_and_evaluate_model(hyperparams) → validation AUC
        - run_simulation(params) → output metric
        - marketing_roi(channel_budget) → total revenue

    Args:
        x:     1D or 2D parameter vector
        noise: observation noise

    Returns:
        scalar function value (higher = better for maximization)
    """
    x = np.atleast_1d(x).ravel()
    # Multi-modal 1D test function: sin(3x) + x**2*0.1
    val = np.sin(3 * x[0]) - x[0] ** 2 * 0.1
    val += np.random.normal(0, noise)
    return float(val)


# ─────────────────────────────────────────────
# ACQUISITION FUNCTIONS
# ─────────────────────────────────────────────

def expected_improvement(X_cand: np.ndarray, gp: GaussianProcessRegressor,
                          y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    EI(x) = (μ(x) - f* - ξ) · Φ(Z) + σ(x) · φ(Z)
    where Z = (μ(x) - f* - ξ) / σ(x)

    Args:
        X_cand: candidate points (n, d)
        gp:     fitted GaussianProcessRegressor
        y_best: current best observed value f*
        xi:     exploration-exploitation tradeoff (Placeholder: 0.01–0.1)

    Returns:
        EI values (n,) — higher = better candidate
    """
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    Z    = (mu - y_best - xi) / sigma
    ei   = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-8] = 0.0
    return ei


def upper_confidence_bound(X_cand: np.ndarray, gp: GaussianProcessRegressor,
                            kappa: float = 2.0) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) acquisition.

    UCB(x) = μ(x) + κ · σ(x)

    Args:
        X_cand: candidate points (n, d)
        gp:     fitted GP
        kappa:  exploration weight (Placeholder: 1.0–5.0)

    Returns:
        UCB values (n,)
    """
    mu, sigma = gp.predict(X_cand, return_std=True)
    return mu + kappa * sigma


def probability_of_improvement(X_cand: np.ndarray, gp: GaussianProcessRegressor,
                                 y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Probability of Improvement (PI) acquisition.

    PI(x) = Φ((μ(x) - f* - ξ) / σ(x))
    """
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-8)
    Z    = (mu - y_best - xi) / sigma
    return norm.cdf(Z)


# ─────────────────────────────────────────────
# BAYESIAN OPTIMIZATION
# ─────────────────────────────────────────────

# Placeholder: tune bounds, n_initial_points, n_iterations, acquisition
BO_CONFIG = {
    "bounds":           [(-5.0, 5.0)],    # Placeholder: [(low, high)] per dimension
    "n_initial_points": 5,                # Placeholder: 5–10 random seed points
    "n_iterations":     25,               # Placeholder: budget of expensive evaluations
    "acquisition":      "ei",             # Placeholder: 'ei', 'ucb', 'pi'
    "xi":               0.01,             # Placeholder: EI/PI exploration param
    "kappa":            2.0,              # Placeholder: UCB exploration param
    "n_restarts":       10,               # restarts for acquisition maximization
}

GP_KERNEL = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)


class BayesianOptimizer:
    """
    Gaussian Process-based Bayesian Optimization.

    Supports 1D and multi-dimensional search spaces.
    Objective is maximized (negate if minimizing).

    Usage:
        opt = BayesianOptimizer(objective_function, bounds=[(-5, 5)])
        opt.run()
        print(opt.best_x, opt.best_y)
    """

    def __init__(self, func, bounds: list, config: dict = None):
        """
        Args:
            func:   callable f(x) → scalar (to MAXIMIZE)
            bounds: list of (low, high) tuples per dimension
            config: BO_CONFIG dict
        """
        self.func   = func
        self.bounds = np.array(bounds)   # shape (d, 2)
        self.cfg    = config or BO_CONFIG
        self.dim    = len(bounds)

        self.X_obs  = []   # observed inputs
        self.y_obs  = []   # observed outputs
        self.gp     = GaussianProcessRegressor(
            kernel=GP_KERNEL,
            n_restarts_optimizer=5,
            normalize_y=True,
        )

    @property
    def best_y(self) -> float:
        return max(self.y_obs) if self.y_obs else -np.inf

    @property
    def best_x(self) -> np.ndarray:
        if not self.y_obs:
            return None
        return self.X_obs[int(np.argmax(self.y_obs))]

    def _random_sample(self) -> np.ndarray:
        """Uniform random sample within bounds."""
        return np.array([np.random.uniform(lo, hi) for lo, hi in self.bounds])

    def _maximize_acquisition(self) -> np.ndarray:
        """
        Maximize acquisition function via multi-start L-BFGS-B.
        Returns the best candidate x.
        """
        acq = self.cfg["acquisition"]

        def neg_acq(x):
            x = x.reshape(1, -1)
            if acq == "ei":
                return -expected_improvement(x, self.gp, self.best_y, self.cfg["xi"])[0]
            elif acq == "ucb":
                return -upper_confidence_bound(x, self.gp, self.cfg["kappa"])[0]
            elif acq == "pi":
                return -probability_of_improvement(x, self.gp, self.best_y, self.cfg["xi"])[0]
            else:
                raise ValueError(f"Unknown acquisition: {acq}")

        best_val = np.inf
        best_x   = self._random_sample()

        for _ in range(self.cfg["n_restarts"]):
            x0 = self._random_sample()
            res = minimize(neg_acq, x0, bounds=self.bounds,
                           method="L-BFGS-B")
            if res.fun < best_val:
                best_val = res.fun
                best_x   = res.x

        return best_x

    def run(self) -> dict:
        """
        Run full Bayesian Optimization loop.

        Returns:
            dict with best_x, best_y, X_obs, y_obs history
        """
        print(f"  Acquisition: {self.cfg['acquisition']} | "
              f"Budget: {self.cfg['n_initial_points']} init + {self.cfg['n_iterations']} BO")

        # Phase 1: random initialization
        for i in range(self.cfg["n_initial_points"]):
            x   = self._random_sample()
            y   = self.func(x)
            self.X_obs.append(x)
            self.y_obs.append(y)
            print(f"  Init {i+1:2d}: x={x.round(3)}  y={y:.4f}  best={self.best_y:.4f}")

        # Phase 2: BO loop
        for i in range(self.cfg["n_iterations"]):
            X_np = np.array(self.X_obs)
            y_np = np.array(self.y_obs)
            self.gp.fit(X_np, y_np)

            x_next = self._maximize_acquisition()
            y_next = self.func(x_next)
            self.X_obs.append(x_next)
            self.y_obs.append(y_next)

            print(f"  BO  {i+1:2d}: x={x_next.round(3)}  y={y_next:.4f}  best={self.best_y:.4f}")

        print(f"\n  Optimum found: x={self.best_x.round(4)}  f(x)={self.best_y:.4f}")
        return {"best_x": self.best_x, "best_y": self.best_y,
                "X_obs": np.array(self.X_obs), "y_obs": np.array(self.y_obs)}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_optimization_history(opt: BayesianOptimizer) -> None:
    """
    Plot observed values over iterations and best-so-far curve.
    Shows convergence behavior.
    """
    y_obs  = np.array(opt.y_obs)
    best_so_far = np.maximum.accumulate(y_obs)
    iters  = np.arange(1, len(y_obs) + 1)
    n_init = opt.cfg["n_initial_points"]

    plt.figure(figsize=(10, 4))
    plt.scatter(iters[:n_init], y_obs[:n_init], c="gray", s=40, label="Random init", zorder=5)
    plt.scatter(iters[n_init:], y_obs[n_init:], c="steelblue", s=40, label="BO proposals", zorder=5)
    plt.plot(iters, best_so_far, "tomato", linewidth=2, label="Best so far")
    plt.axvline(n_init + 0.5, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Objective f(x)")
    plt.title("Bayesian Optimization — Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bo_convergence.png", dpi=150)
    plt.show()


def plot_surrogate_1d(opt: BayesianOptimizer, iteration: int = -1) -> None:
    """
    For 1D problems: plot GP surrogate + acquisition function at final iteration.
    """
    if opt.dim != 1:
        print("  plot_surrogate_1d: skipped (not 1D)")
        return

    lo, hi = opt.bounds[0]
    X_grid = np.linspace(lo, hi, 300)[:, None]
    X_np   = np.array(opt.X_obs)
    y_np   = np.array(opt.y_obs)
    opt.gp.fit(X_np, y_np)

    mu, sigma = opt.gp.predict(X_grid, return_std=True)
    acq_vals  = expected_improvement(X_grid, opt.gp, max(y_np), opt.cfg["xi"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # True function (dashed)
    true_y = [objective_function(x, noise=0) for x in X_grid.ravel()]
    ax1.plot(X_grid.ravel(), true_y, "k--", linewidth=1, alpha=0.5, label="True f(x)")
    ax1.plot(X_grid.ravel(), mu, "steelblue", linewidth=2, label="GP mean")
    ax1.fill_between(X_grid.ravel(), mu - 2*sigma, mu + 2*sigma,
                     alpha=0.25, color="steelblue", label="95% CI")
    ax1.scatter(X_np.ravel(), y_np, c="tomato", s=30, zorder=5, label="Observations")
    ax1.scatter([opt.best_x[0]], [opt.best_y], c="gold", s=100, zorder=6,
                marker="*", label=f"Best x={opt.best_x[0]:.3f}")
    ax1.set_ylabel("f(x)")
    ax1.legend(fontsize=8)
    ax1.set_title("Bayesian Optimization — GP Surrogate")

    ax2.plot(X_grid.ravel(), acq_vals, "darkorange", linewidth=2)
    ax2.fill_between(X_grid.ravel(), 0, acq_vals, alpha=0.3, color="orange")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Expected Improvement")
    ax2.set_title("Acquisition Function (EI)")

    plt.tight_layout()
    plt.savefig("bo_surrogate.png", dpi=150)
    plt.show()


def plot_acquisition_comparison(opt: BayesianOptimizer) -> None:
    """Compare EI, UCB, PI acquisition functions on the final surrogate."""
    if opt.dim != 1:
        return

    lo, hi  = opt.bounds[0]
    X_grid  = np.linspace(lo, hi, 300)[:, None]
    X_np    = np.array(opt.X_obs)
    y_np    = np.array(opt.y_obs)
    opt.gp.fit(X_np, y_np)
    y_best  = max(y_np)

    ei  = expected_improvement(X_grid, opt.gp, y_best, xi=0.01)
    ucb = upper_confidence_bound(X_grid, opt.gp, kappa=2.0)
    pi  = probability_of_improvement(X_grid, opt.gp, y_best, xi=0.01)

    # Normalize each to [0, 1] for overlay
    def norm01(a): return (a - a.min()) / (a.max() - a.min() + 1e-10)

    plt.figure(figsize=(10, 4))
    plt.plot(X_grid.ravel(), norm01(ei),  label="EI (norm)", color="steelblue")
    plt.plot(X_grid.ravel(), norm01(ucb), label="UCB (norm)", color="tomato")
    plt.plot(X_grid.ravel(), norm01(pi),  label="PI (norm)",  color="darkgreen")
    plt.xlabel("x")
    plt.ylabel("Acquisition (normalized)")
    plt.title("Bayesian Optimization — Acquisition Function Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bo_acquisitions.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Bayesian Optimization ===")
    np.random.seed(42)

    # Placeholder: replace objective_function with your expensive evaluation
    # e.g., objective = lambda x: train_model(lr=10**x[0], dropout=x[1])
    config = {**BO_CONFIG, "bounds": [(-5.0, 5.0)]}
    opt = BayesianOptimizer(objective_function, bounds=config["bounds"], config=config)
    results = opt.run()

    print(f"\nFinal: best_x = {results['best_x'].round(4)},  "
          f"best_y = {results['best_y']:.4f}")

    plot_optimization_history(opt)
    plot_surrogate_1d(opt)
    plot_acquisition_comparison(opt)


if __name__ == "__main__":
    main()
