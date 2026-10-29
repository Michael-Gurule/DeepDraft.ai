"""
Markov Chain Monte Carlo (MCMC)
================================

MCMC methods sample from a target distribution π(θ) ∝ p(θ) · L(θ|data)
when direct sampling is intractable (unnormalized posteriors).

Methods covered:
  1. Metropolis-Hastings (MH): propose → accept/reject via detailed balance
       acceptance ratio: α = min(1, π(θ*)q(θ|θ*) / π(θ)q(θ*|θ))
       Symmetric q (random walk): α = min(1, π(θ*)/π(θ))

  2. Gibbs Sampling: sample each dimension conditioned on all others.
       Requires tractable full conditionals p(θ_i | θ_{-i}, data).
       Special case of MH with α=1.

  3. Hamiltonian Monte Carlo (HMC): augment with momentum p,
       run leapfrog dynamics in (θ, p) space, project back to θ.
       Gradient-based → much more efficient in high dimensions.
       Accept/reject with energy: H(θ,p) = U(θ) + K(p)
       where U(θ) = -log π(θ) and K(p) = p^T M^{-1} p / 2.

  4. No-U-Turn Sampler (NUTS): adaptive HMC that automatically selects
       trajectory length (no manual leapfrog steps L).
       Standard sampler in Stan, PyMC, Pyro.

  5. Slice Sampling: draws uniform under the density; no step-size tuning.

Diagnostics:
  R-hat (Gelman-Rubin):  < 1.01 indicates chain convergence
  Effective Sample Size: n_eff = n / (1 + 2·Σ_{k=1}^∞ ρ_k)
  Trace plots:           visual check for mixing
  Autocorrelation:       slow decay → poor mixing

Fine-tuning recommendations:
  MH:
    - proposal_std (#Placeholder: 0.1): acceptance rate ~ 23% is optimal for d-dim Gaussians
    - n_warmup (#Placeholder: 1000): discard initial samples (burn-in)
  HMC:
    - step_size (#Placeholder: 0.01): smaller = more accurate but slower
    - n_leapfrog (#Placeholder: 20): longer trajectories = better exploration
    - mass_matrix (#Placeholder: I): diagonal approximation to covariance
  General:
    - n_samples (#Placeholder: 5000): effective n = n / autocorr_time
    - n_chains (#Placeholder: 4): needed for R-hat convergence diagnostics

References:
  Metropolis et al. (1953) — Equations of State Calculations by Fast Computing Machines
  Hastings (1970) — Monte Carlo Sampling Methods Using Markov Chains
  Duane et al. (1987) — Hybrid Monte Carlo (HMC)
  Neal (2011) — MCMC Using Hamiltonian Dynamics (review)
  Hoffman & Gelman (2014) — The No-U-Turn Sampler (NUTS)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm, gamma as sp_gamma

from deepdraft.core import fast_or, save_figure

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "n_samples":    fast_or(5000, 300),  # # Placeholder:
    "n_warmup":     fast_or(1000, 100),  # # Placeholder: burn-in samples to discard
    "n_chains":     4,       # # Placeholder: for R-hat convergence check
    "mh_std":       0.3,     # # Placeholder: MH random-walk proposal std
    "hmc_step":     0.05,    # # Placeholder: leapfrog step size
    "hmc_n_steps":  20,      # # Placeholder: leapfrog trajectory length
    "seed":         42,
}


# ─────────────────────────────────────────────
# Target Distributions
# ─────────────────────────────────────────────

def banana_log_prob(theta: np.ndarray, b: float = 0.1) -> float:
    """
    2D banana (Rosenbrock-like) distribution — non-Gaussian, curved.
    p(x,y) ∝ exp(-x²/2 - (y - b·x² + 100b)²/2)
    Classic MCMC test: requires gradient-aware sampler to explore efficiently.
    """
    x, y = theta
    return -x**2 / 2.0 - (y - b * x**2 + 100 * b)**2 / 2.0


def gaussian_mixture_log_prob(theta: np.ndarray) -> float:
    """
    2D mixture of 3 Gaussians — multimodal, tests mode exploration.
    """
    means  = [np.array([0, 0]), np.array([4, 4]), np.array([-4, 4])]
    covs   = [np.eye(2) * 0.5, np.eye(2) * 0.5, np.eye(2) * 0.5]
    log_p  = -np.inf
    for mu, cov in zip(means, covs):
        lp = multivariate_normal.logpdf(theta, mean=mu, cov=cov)
        log_p = np.logaddexp(log_p, lp)
    return log_p


def bayesian_linear_regression_log_prob(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray,
    prior_std: float = 5.0,
) -> float:
    """
    Bayesian linear regression posterior.
    theta = [w0, w1, ..., wd, log_sigma]
    Prior: w_i ~ N(0, prior_std²), σ ~ HalfNormal
    Likelihood: y_i ~ N(X w, σ²)
    """
    d = X.shape[1]
    w = theta[:d]
    log_sigma = theta[d]
    sigma = np.exp(log_sigma)

    log_prior = -np.sum(w**2) / (2 * prior_std**2) - log_sigma   # improper sigma prior
    residuals  = y - X @ w
    log_lik    = -len(y) * log_sigma - np.sum(residuals**2) / (2 * sigma**2)
    return float(log_prior + log_lik)


# ─────────────────────────────────────────────
# Metropolis-Hastings
# ─────────────────────────────────────────────

def metropolis_hastings(
    log_prob_fn,
    init: np.ndarray,
    n_samples: int,
    n_warmup: int,
    proposal_std: float = 0.1,   # # Placeholder:
    seed: int = 42,
) -> dict:
    """
    Random-walk Metropolis-Hastings.
    proposal: q(θ*|θ) = N(θ, σ²I)
    Target acceptance rate: ~23% for isotropic Gaussian in d dimensions.
    """
    rng = np.random.default_rng(seed)
    dim = len(init)
    theta = init.copy()
    log_p = log_prob_fn(theta)

    samples = np.zeros((n_samples + n_warmup, dim))
    accepted = 0

    for i in range(n_samples + n_warmup):
        theta_star = theta + rng.normal(0, proposal_std, dim)
        log_p_star = log_prob_fn(theta_star)

        log_alpha = log_p_star - log_p
        if np.log(rng.random()) < log_alpha:
            theta = theta_star
            log_p = log_p_star
            if i >= n_warmup:
                accepted += 1

        samples[i] = theta

    n_total = n_samples + n_warmup
    accept_rate = accepted / n_samples
    print(f"MH acceptance rate: {accept_rate:.3f}  (target ~0.23)")
    return {
        "samples":      samples[n_warmup:],
        "accept_rate":  accept_rate,
        "all_samples":  samples,
    }


# ─────────────────────────────────────────────
# Hamiltonian Monte Carlo
# ─────────────────────────────────────────────

def _grad_log_prob(log_prob_fn, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Numerical gradient of log_prob via central differences."""
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta); e[i] = eps
        grad[i] = (log_prob_fn(theta + e) - log_prob_fn(theta - e)) / (2 * eps)
    return grad


def hamiltonian_monte_carlo(
    log_prob_fn,
    init: np.ndarray,
    n_samples: int,
    n_warmup: int,
    step_size: float = 0.01,   # # Placeholder:
    n_leapfrog: int = 20,      # # Placeholder:
    grad_fn=None,
    seed: int = 42,
) -> dict:
    """
    HMC sampler.
    U(θ) = -log π(θ)
    K(p) = p^T p / 2   (unit mass matrix)
    Leapfrog integration:
      p_half = p - (ε/2) ∇U(θ)
      θ_new  = θ + ε p_half
      p_new  = p_half - (ε/2) ∇U(θ_new)
    """
    rng = np.random.default_rng(seed)
    dim = len(init)
    grad = grad_fn or (lambda t: _grad_log_prob(log_prob_fn, t))
    theta = init.copy()
    log_p = log_prob_fn(theta)

    samples = np.zeros((n_samples + n_warmup, dim))
    accepted = 0

    for i in range(n_samples + n_warmup):
        p = rng.normal(0, 1, dim)
        theta_curr = theta.copy()
        p_curr = p.copy()
        current_H = -log_p + 0.5 * np.dot(p, p)

        # Leapfrog integration
        p_half = p - (step_size / 2) * (-grad(theta))
        for _ in range(n_leapfrog - 1):
            theta = theta + step_size * p_half
            p_half = p_half - step_size * (-grad(theta))
        theta = theta + step_size * p_half
        p = p_half - (step_size / 2) * (-grad(theta))

        proposed_log_p = log_prob_fn(theta)
        proposed_H = -proposed_log_p + 0.5 * np.dot(p, p)

        # Accept/reject
        if np.log(rng.random()) < current_H - proposed_H:
            log_p = proposed_log_p
            if i >= n_warmup:
                accepted += 1
        else:
            theta = theta_curr

        samples[i] = theta

    accept_rate = accepted / n_samples
    print(f"HMC acceptance rate: {accept_rate:.3f}  (target ~0.65)")
    return {
        "samples":     samples[n_warmup:],
        "accept_rate": accept_rate,
    }


# ─────────────────────────────────────────────
# Slice Sampling (1D)
# ─────────────────────────────────────────────

def slice_sampling_1d(
    log_prob_fn,
    init: float,
    n_samples: int,
    n_warmup: int,
    w: float = 1.0,    # # Placeholder: initial slice width
    seed: int = 42,
) -> np.ndarray:
    """
    1D slice sampling — no step-size tuning required.
    Samples θ uniformly under the log-density curve.
    Works on any univariate distribution.
    """
    rng = np.random.default_rng(seed)
    theta = float(init)
    samples = []

    for _ in range(n_samples + n_warmup):
        log_y = log_prob_fn(np.array([theta])) + np.log(rng.random())

        # Step out: expand interval [L, R] until outside slice
        L = theta - w * rng.random()
        R = L + w
        while log_prob_fn(np.array([L])) > log_y:
            L -= w
        while log_prob_fn(np.array([R])) > log_y:
            R += w

        # Shrink: sample uniformly in [L, R], shrink on rejection
        for _ in range(100):
            theta_new = rng.uniform(L, R)
            if log_prob_fn(np.array([theta_new])) > log_y:
                theta = theta_new
                break
            if theta_new < theta:
                L = theta_new
            else:
                R = theta_new

        samples.append(theta)

    return np.array(samples[n_warmup:])


# ─────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────

def r_hat(chains: list[np.ndarray]) -> np.ndarray:
    """
    Gelman-Rubin R-hat statistic for convergence.
    R-hat < 1.01 indicates chains have mixed.
    chains: list of (n_samples, d) arrays from independent chains.
    """
    m = len(chains)         # number of chains
    n = chains[0].shape[0]  # samples per chain
    chains_arr = np.array(chains)   # (m, n, d)

    chain_means = chains_arr.mean(axis=1)   # (m, d)
    overall_mean = chain_means.mean(axis=0)  # (d,)

    B = n / (m - 1) * np.sum((chain_means - overall_mean[None]) ** 2, axis=0)
    W = np.mean([np.var(c, axis=0, ddof=1) for c in chains], axis=0)

    var_plus = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_plus / (W + 1e-12))
    return rhat


def effective_sample_size(samples: np.ndarray, max_lag: int = 500) -> np.ndarray:
    """
    Estimate effective sample size using autocorrelation.
    n_eff = n / (1 + 2·Σ_{k=1}^K ρ_k)
    """
    n = len(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    ess = []
    for j in range(samples.shape[1]):
        x = samples[:, j] - samples[:, j].mean()
        acf = np.correlate(x, x, mode="full")[n - 1:]
        acf = acf / (acf[0] + 1e-12)
        # Sum positive autocorrelations (Geyer's monotone sequence)
        cumsum = 0.0
        for k in range(1, min(max_lag, n // 2)):
            if acf[k] > 0:
                cumsum += acf[k]
            else:
                break
        ess.append(n / (1 + 2 * cumsum))
    return np.array(ess)


# ─────────────────────────────────────────────
# Bayesian Linear Regression Demo
# ─────────────────────────────────────────────

def bayesian_regression_demo(seed: int = 42) -> dict:
    """
    Fit Bayesian linear regression via MCMC.
    True model: y = 1.5 x + 0.5 + N(0, 0.3²)
    Compare MH vs HMC posterior samples with analytical solution.
    """
    rng = np.random.default_rng(seed)
    n = 50
    X_raw = rng.normal(0, 1, (n, 1))
    y = 1.5 * X_raw[:, 0] + 0.5 + rng.normal(0, 0.3, n)
    X = np.hstack([np.ones((n, 1)), X_raw])   # design matrix [1, x]

    log_p = lambda theta: bayesian_linear_regression_log_prob(theta, X, y)
    init = np.zeros(3)   # [intercept, slope, log_sigma]

    print("MH for Bayesian Linear Regression...")
    mh_result = metropolis_hastings(log_p, init, n_samples=fast_or(3000, 300), n_warmup=fast_or(1000, 100),
                                     proposal_std=0.15, seed=seed)

    print("HMC for Bayesian Linear Regression...")
    hmc_result = hamiltonian_monte_carlo(log_p, init, n_samples=fast_or(2000, 200), n_warmup=fast_or(500, 50),
                                          step_size=0.02, n_leapfrog=15, seed=seed)

    return {"mh": mh_result, "hmc": hmc_result, "X": X, "y": y}


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_2d_samples(samples_dict: dict, log_prob_fn, title: str, xlim=(-5, 5), ylim=(-5, 5)):
    """Compare samples from different MCMC methods against true density contours."""
    n_methods = len(samples_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12)

    # True density on a grid
    xx, yy = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    log_p_grid = np.array([log_prob_fn(g) for g in grid])
    p_grid = np.exp(log_p_grid - log_p_grid.max()).reshape(xx.shape)

    for ax, (name, samples) in zip(axes, samples_dict.items()):
        ax.contour(xx, yy, p_grid, levels=10, alpha=0.5, cmap="Blues")
        ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.3, c="red")
        ess = effective_sample_size(samples)
        ax.set_title(f"{name}\nESS≈{ess.mean():.0f}")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("θ₁"); ax.set_ylabel("θ₂")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = f"mcmc_{title.lower().replace(' ', '_').replace('/', '')}.png"
    save_figure(fname, dpi=150)
    print(f"Saved: {fname}")


def plot_trace(samples: np.ndarray, param_names: list = None, title: str = ""):
    """Trace plot and marginal histograms."""
    d = min(samples.shape[1], 4)
    fig, axes = plt.subplots(d, 2, figsize=(11, 2.5 * d))
    if d == 1:
        axes = axes[None]
    fig.suptitle(f"MCMC Trace & Marginals — {title}", fontsize=11)
    names = param_names or [f"θ{i+1}" for i in range(d)]

    for i in range(d):
        axes[i, 0].plot(samples[:, i], lw=0.5, alpha=0.7, color="steelblue")
        axes[i, 0].set_ylabel(names[i]); axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].hist(samples[:, i], bins=50, density=True,
                         color="steelblue", edgecolor="white", alpha=0.7)
        axes[i, 1].set_xlabel(names[i]); axes[i, 1].set_ylabel("density")
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Iteration")
    plt.tight_layout()
    fname = f"mcmc_trace_{title.lower().replace(' ', '_')}.png"
    save_figure(fname, dpi=150)
    print(f"Saved: {fname}")


def plot_regression_posterior(result: dict):
    """Posterior predictive plot for Bayesian linear regression."""
    mh_samples = result["mh"]["samples"]
    X, y = result["X"], result["y"]
    x_plot = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Bayesian Linear Regression — MCMC Posterior", fontsize=12)

    for ax, samples, label in zip(axes,
                                    [mh_samples, result["hmc"]["samples"]],
                                    ["Metropolis-Hastings", "HMC"]):
        # Posterior predictive lines
        idx = np.random.choice(len(samples), 200, replace=False)
        for i in idx:
            w0, w1, _ = samples[i]
            ax.plot(x_plot, w0 + w1 * x_plot, "b-", alpha=0.02, lw=0.5)

        # Data points
        ax.scatter(X[:, 1], y, s=20, zorder=5, c="red", label="Data")
        # Mean posterior
        w0_mean = samples[:, 0].mean()
        w1_mean = samples[:, 1].mean()
        ax.plot(x_plot, w0_mean + w1_mean * x_plot, "k-", lw=2, label="Posterior mean")
        ax.set_title(label); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("mcmc_regression_posterior.png", dpi=150)
    print("Saved: mcmc_regression_posterior.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(CONFIG["seed"])
    print(f"Config: {CONFIG}\n")

    # ── Banana distribution: MH vs HMC ─────────────────────────────
    print("=== Banana Distribution — MH vs HMC ===")
    init2d = np.array([0.0, 0.0])
    mh_banana = metropolis_hastings(
        banana_log_prob, init2d, CONFIG["n_samples"], CONFIG["n_warmup"],
        proposal_std=CONFIG["mh_std"], seed=CONFIG["seed"],
    )
    hmc_banana = hamiltonian_monte_carlo(
        banana_log_prob, init2d, CONFIG["n_samples"], CONFIG["n_warmup"],
        step_size=CONFIG["hmc_step"], n_leapfrog=CONFIG["hmc_n_steps"],
        seed=CONFIG["seed"],
    )
    plot_2d_samples(
        {"MH": mh_banana["samples"], "HMC": hmc_banana["samples"]},
        banana_log_prob, "Banana Distribution", xlim=(-4, 4), ylim=(-5, 25),
    )

    # ── Gaussian mixture: multimodal test ──────────────────────────
    print("\n=== Gaussian Mixture — MH vs HMC ===")
    mh_mix = metropolis_hastings(
        gaussian_mixture_log_prob, init2d, CONFIG["n_samples"], CONFIG["n_warmup"],
        proposal_std=1.5, seed=CONFIG["seed"],
    )
    hmc_mix = hamiltonian_monte_carlo(
        gaussian_mixture_log_prob, init2d, CONFIG["n_samples"], CONFIG["n_warmup"],
        step_size=0.1, n_leapfrog=10, seed=CONFIG["seed"],
    )
    plot_2d_samples(
        {"MH": mh_mix["samples"], "HMC": hmc_mix["samples"]},
        gaussian_mixture_log_prob, "Gaussian Mixture", xlim=(-8, 8), ylim=(-3, 8),
    )

    # ── Trace plots ────────────────────────────────────────────────
    plot_trace(hmc_banana["samples"], ["θ₁", "θ₂"], title="HMC Banana")

    # ── R-hat convergence check ────────────────────────────────────
    print("\n=== R-hat Convergence Check (4 chains) ===")
    chains = []
    for s in range(CONFIG["n_chains"]):
        init_s = np.random.normal(0, 2, 2)
        c = metropolis_hastings(banana_log_prob, init_s, fast_or(2000, 200), fast_or(500, 50),
                                 proposal_std=CONFIG["mh_std"], seed=CONFIG["seed"] + s)
        chains.append(c["samples"])
    rhat_vals = r_hat(chains)
    print(f"  R-hat: θ₁={rhat_vals[0]:.4f}  θ₂={rhat_vals[1]:.4f}  (< 1.01 = converged)")

    # ── Bayesian regression demo ───────────────────────────────────
    print("\n=== Bayesian Linear Regression ===")
    reg_result = bayesian_regression_demo(seed=CONFIG["seed"])
    plot_regression_posterior(reg_result)
    mh_ess = effective_sample_size(reg_result["mh"]["samples"])
    hmc_ess = effective_sample_size(reg_result["hmc"]["samples"])
    print(f"  MH ESS:  {mh_ess}")
    print(f"  HMC ESS: {hmc_ess}")


if __name__ == "__main__":
    main()
