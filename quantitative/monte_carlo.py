"""
Monte Carlo Simulation for Finance
=====================================
Monte Carlo (MC) methods approximate complex quantities by repeated random sampling.
In finance, MC simulates many possible future price paths to estimate:

    - Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
    - Option pricing (especially path-dependent options)
    - Portfolio scenario analysis
    - Risk of ruin / drawdown distributions

Geometric Brownian Motion (GBM) for asset price simulation:
    dS = μ·S·dt + σ·S·dW                (SDE)
    S(t) = S(0) · exp((μ - σ²/2)·t + σ·√t·Z)   (analytical solution)

    where:
        μ  = drift (expected annual return)
        σ  = annual volatility
        dW = Wiener process increment ~ N(0, dt)
        Z  ~ N(0, 1) standard normal shock

Multi-asset extension:
    Use Cholesky decomposition of correlation matrix to generate correlated shocks:
        L = chol(Σ)
        dW_corr = L · z,  z ~ N(0, I)

VaR and CVaR:
    VaR_α:  the α-quantile of the loss distribution (e.g., 95% or 99%)
    CVaR_α: mean of losses beyond VaR_α (Expected Shortfall)
    CVaR is a coherent risk measure; VaR is not (does not capture tail severity)

Variance Reduction Techniques:
    - Antithetic variates: simulate z and -z, average results → halves variance
    - Control variates: use known analytical price to reduce error
    - Quasi-Monte Carlo: use low-discrepancy sequences (Sobol, Halton)

Use Cases:
    - Portfolio VaR / CVaR for risk management
    - Option pricing (path-dependent: Asian, barrier, American via LSM)
    - Stress testing (tail scenarios)
    - Project NPV distributions under uncertainty
    - Scenario analysis for any simulated outcome distribution

Fine-Tuning Recommendations:
    - n_simulations: 10k for rough estimates, 100k for production VaR.
      Confidence interval width ∝ 1/√N.
    - Use antithetic variates to double effective N at no extra cost.
    - For high accuracy: use Quasi-Monte Carlo (scipy.stats.qmc.Sobol).
    - VaR confidence: 95% is standard; 99% is used for bank regulatory capital.
    - CVaR is always preferred over VaR for tail risk characterization.
    - Correlations: calibrate from rolling historical windows, not full history.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from deepdraft.core import fast_or, save_figure


# ─────────────────────────────────────────────
# SIMULATION CONFIG
# ─────────────────────────────────────────────

# Placeholder: set parameters from your asset data
MC_CONFIG = {
    "n_simulations": fast_or(50_000, 2_000),  # Placeholder: 10k (fast) to 100k (accurate)
    "n_days":        252,      # Placeholder: forecast horizon in trading days
    "dt":            1 / 252,  # daily time step
    "antithetic":    True,     # variance reduction: simulate z and -z pairs
    "seed":          42,
}

# Example single-asset parameters
ASSET_CONFIG = {
    "S0":    100.0,    # Placeholder: current price or portfolio value
    "mu":    0.08,     # Placeholder: annual expected return
    "sigma": 0.20,     # Placeholder: annual volatility (from GARCH or hist. std)
    "rf":    0.04,     # Placeholder: risk-free rate (for option pricing)
}

# Example multi-asset portfolio
PORTFOLIO_CONFIG = {
    "S0":         np.array([100.0, 100.0, 100.0]),  # Placeholder: current prices
    "weights":    np.array([0.40,  0.35,  0.25]),   # Placeholder: portfolio weights
    "mus":        np.array([0.08,  0.06,  0.10]),   # Placeholder: expected annual returns
    "sigmas":     np.array([0.20,  0.15,  0.25]),   # Placeholder: annual vols
    "corr": np.array([                              # Placeholder: correlation matrix
        [1.00, 0.50, 0.30],
        [0.50, 1.00, 0.20],
        [0.30, 0.20, 1.00],
    ]),
}


# ─────────────────────────────────────────────
# SINGLE-ASSET GBM SIMULATION
# ─────────────────────────────────────────────

def simulate_gbm(S0: float, mu: float, sigma: float,
                  n_days: int, n_sims: int, dt: float = 1/252,
                  antithetic: bool = True, seed: int = 42) -> np.ndarray:
    """
    Simulate asset price paths using Geometric Brownian Motion.

    With antithetic=True, row i and row i + n_sims/2 are a pair driven by
    shocks z and -z, and an odd n_sims is rounded up to keep every path paired.
    Pass the same flag to the pricing functions so their standard errors
    account for the pairing.

    Returns:
        paths: (n_sims, n_days + 1) array of price paths
               paths[:, 0] = S0 (starting price)
    """
    rng = np.random.RandomState(seed)

    if antithetic:
        n_raw = (n_sims + 1) // 2
        z_raw = rng.standard_normal((n_raw, n_days))
        z     = np.vstack([z_raw, -z_raw])  # antithetic pairs
    else:
        z = rng.standard_normal((n_sims, n_days))
    n_sims = z.shape[0]

    # GBM analytical formula: log-normal increments
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol   = sigma * np.sqrt(dt)

    log_returns = drift + vol * z              # (n_sims, n_days)
    log_paths   = np.cumsum(log_returns, axis=1)
    paths       = S0 * np.exp(np.hstack([np.zeros((n_sims, 1)), log_paths]))

    return paths


def simulate_portfolio_gbm(config: dict, n_days: int, n_sims: int,
                             dt: float = 1/252, seed: int = 42) -> np.ndarray:
    """
    Simulate correlated multi-asset portfolio value paths.

    Uses Cholesky decomposition to generate correlated normal shocks.

    Returns:
        portfolio_paths: (n_sims, n_days + 1) — total portfolio value
    """
    rng   = np.random.RandomState(seed)
    n_a   = len(config["S0"])
    L     = np.linalg.cholesky(config["corr"])

    # Generate correlated shocks
    z_raw = rng.standard_normal((n_sims, n_days, n_a))
    z_corr = z_raw @ L.T   # (n_sims, n_days, n_a) — correlated

    drifts = (config["mus"] - 0.5 * config["sigmas"] ** 2) * dt
    vols   = config["sigmas"] * np.sqrt(dt)

    log_ret = drifts + vols * z_corr       # (n_sims, n_days, n_a)
    log_cum = np.cumsum(log_ret, axis=1)
    price_paths = config["S0"] * np.exp(
        np.concatenate([np.zeros((n_sims, 1, n_a)), log_cum], axis=1)
    )  # (n_sims, n_days+1, n_a)

    # Portfolio value = w^T * prices (normalized to initial weights)
    initial_value = float(config["S0"] @ config["weights"])
    portfolio_paths = (price_paths * config["weights"]).sum(axis=2)

    return portfolio_paths


# ─────────────────────────────────────────────
# VaR & CVaR
# ─────────────────────────────────────────────

def calculate_var_cvar(paths: np.ndarray, S0: float = None,
                        horizon: int = None,
                        confidence: float = 0.95) -> dict:
    """
    Calculate VaR and CVaR from simulated terminal values.

    Args:
        paths:      (n_sims, n_days + 1) price paths
        S0:         initial value; if None, uses paths[:, 0].mean()
        horizon:    day index to evaluate; if None, uses last
        confidence: VaR confidence level (e.g., 0.95 or 0.99)

    Returns:
        dict with var, cvar, mean_return, prob_loss
    """
    if S0 is None:
        S0 = paths[:, 0].mean()
    if horizon is None:
        terminal = paths[:, -1]
    else:
        terminal = paths[:, horizon]

    pnl       = terminal - S0  # profit/loss
    returns_  = pnl / S0       # return

    alpha     = 1 - confidence
    var_loss  = float(-np.percentile(pnl, alpha * 100))
    cvar_loss = float(-pnl[pnl < -var_loss].mean()) if (pnl < -var_loss).any() else var_loss

    print(f"  Confidence:  {confidence*100:.0f}%")
    print(f"  VaR:         ${var_loss:.2f}  ({-np.percentile(returns_, alpha*100)*100:.2f}%)")
    print(f"  CVaR (ES):   ${cvar_loss:.2f}  ({cvar_loss/S0*100:.2f}%)")
    print(f"  Mean P&L:    ${pnl.mean():.2f}")
    print(f"  P(Loss):     {(pnl < 0).mean()*100:.1f}%")

    return {
        "var": var_loss, "cvar": cvar_loss,
        "mean_pnl": float(pnl.mean()),
        "prob_loss": float((pnl < 0).mean()),
        "pnl": pnl,
    }


# ─────────────────────────────────────────────
# OPTION PRICING
# ─────────────────────────────────────────────

def price_european_option(paths: np.ndarray, K: float, rf: float,
                            T: float, option_type: str = "call",
                            antithetic: bool = False) -> dict:
    """
    Price European option via Monte Carlo (risk-neutral pricing).

    Under risk-neutral measure, drift μ → rf.
    Payoff at expiry discounted back at risk-free rate.

    Args:
        paths:       (n_sims, n_days+1) risk-neutral price paths
        K:           strike price
        rf:          risk-free rate (annual)
        T:           time to expiry (years)
        option_type: 'call' or 'put'
        antithetic:  True if paths come in antithetic pairs (simulate_gbm)

    Returns:
        dict with price, std_error, confidence_interval
    """
    S_T = paths[:, -1]  # terminal prices

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)

    discount      = np.exp(-rf * T)
    mc_price, std_error = mc_mean_and_se(discount * payoff, antithetic)
    ci            = (mc_price - 1.96 * std_error, mc_price + 1.96 * std_error)

    print(f"  {option_type.capitalize()} option price: ${mc_price:.4f}  "
          f"± {1.96*std_error:.4f}  "
          f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

    return {"price": mc_price, "std_error": std_error, "ci_95": ci}


def price_asian_option(paths: np.ndarray, K: float, rf: float,
                        T: float, option_type: str = "call",
                        antithetic: bool = False) -> dict:
    """
    Price Asian (average-price) option via Monte Carlo.

    Asian options use the arithmetic mean of prices along the path.
    They are path-dependent — no closed-form Black-Scholes solution.

    Payoff(call) = max(S_avg - K, 0)
    Payoff(put)  = max(K - S_avg, 0)
    """
    S_avg = paths[:, 1:].mean(axis=1)  # arithmetic average (exclude t=0)

    if option_type == "call":
        payoff = np.maximum(S_avg - K, 0)
    else:
        payoff = np.maximum(K - S_avg, 0)

    discount   = np.exp(-rf * T)
    mc_price, std_error = mc_mean_and_se(discount * payoff, antithetic)
    ci         = (mc_price - 1.96 * std_error, mc_price + 1.96 * std_error)

    print(f"  Asian {option_type} price: ${mc_price:.4f}  ± {1.96*std_error:.4f}")
    return {"price": mc_price, "std_error": std_error, "ci_95": ci}


def mc_mean_and_se(values: np.ndarray, antithetic: bool = False) -> tuple[float, float]:
    """
    Monte Carlo estimate of E[values] and its standard error.

    Plain sampling: SE = std(values) / √N.

    Antithetic sampling: the two halves of `values` are pairs (see
    simulate_gbm). The pair members are negatively correlated, which is the
    whole point, so they are not N independent draws and std / √N is wrong.
    Average each pair first; the N/2 pair means ARE independent:
        SE = std(pair means) / √(N/2)
    """
    values = np.asarray(values, dtype=float)
    if not antithetic:
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
    half = len(values) // 2
    pair_means = (values[:half] + values[half:2 * half]) / 2
    return float(pair_means.mean()), float(pair_means.std(ddof=1) / np.sqrt(half))


# ─────────────────────────────────────────────
# CONVERGENCE ANALYSIS
# ─────────────────────────────────────────────

def convergence_analysis(S0: float, mu: float, sigma: float,
                           n_days: int, confidence: float = 0.95,
                           n_range: list = None) -> None:
    """
    Show how VaR estimate converges as N increases.
    Demonstrates the 1/√N convergence of MC error.
    """
    if n_range is None:
        n_range = fast_or([500, 1000, 5000, 10000, 50000], [200, 500])

    vars_  = []
    cvars_ = []

    for n in n_range:
        paths = simulate_gbm(S0, mu, sigma, n_days, n)
        r     = calculate_var_cvar(paths, S0, confidence=confidence)
        vars_.append(r["var"])
        cvars_.append(r["cvar"])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogx(n_range, vars_,  "o-", color="steelblue", label="VaR")
    plt.semilogx(n_range, cvars_, "s--", color="tomato",   label="CVaR")
    plt.xlabel("N Simulations (log)")
    plt.ylabel("Loss ($)")
    plt.title(f"VaR/CVaR Convergence ({confidence*100:.0f}%)")
    plt.legend()

    # Std error ∝ 1/sqrt(N)
    plt.subplot(1, 2, 2)
    expected_error = np.array(n_range, dtype=float) ** (-0.5)
    expected_error *= vars_[0] * np.sqrt(n_range[0])
    plt.loglog(n_range, expected_error, "k--", label="1/√N reference")
    plt.xlabel("N Simulations (log)")
    plt.ylabel("Expected MC std error (log)")
    plt.title("MC Standard Error vs N")
    plt.legend()

    plt.tight_layout()
    save_figure("mc_convergence.png", dpi=150)


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_price_paths(paths: np.ndarray, n_show: int = 200,
                      S0: float = None, title: str = "GBM Price Paths") -> None:
    """
    Plot a sample of simulated price paths.
    """
    if S0 is None:
        S0 = paths[:, 0].mean()
    t  = np.arange(paths.shape[1])
    idx = np.random.choice(len(paths), min(n_show, len(paths)), replace=False)

    plt.figure(figsize=(11, 5))
    plt.plot(t, paths[idx].T, linewidth=0.4, alpha=0.3, color="steelblue")
    plt.plot(t, np.percentile(paths, 5,  axis=0), "r--", linewidth=1.5, label="5th pct")
    plt.plot(t, np.percentile(paths, 50, axis=0), "k-",  linewidth=1.5, label="Median")
    plt.plot(t, np.percentile(paths, 95, axis=0), "g--", linewidth=1.5, label="95th pct")
    plt.axhline(S0, color="gray", linestyle="--", alpha=0.5, label=f"S0={S0}")
    plt.xlabel("Trading Days")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_figure("mc_paths.png", dpi=150)


def plot_pnl_distribution(pnl: np.ndarray, var: float,
                           cvar: float, confidence: float = 0.95) -> None:
    """
    Histogram of P&L distribution with VaR and CVaR marked.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(pnl, bins=100, density=True, color="steelblue", alpha=0.7)
    plt.axvline(-var,  color="red",    linewidth=2, linestyle="--",
                label=f"VaR({confidence*100:.0f}%) = ${var:.2f}")
    plt.axvline(-cvar, color="darkred", linewidth=2, linestyle="-.",
                label=f"CVaR({confidence*100:.0f}%) = ${cvar:.2f}")
    plt.axvline(0, color="black", linewidth=1, alpha=0.5)
    plt.xlabel("P&L ($)")
    plt.ylabel("Density")
    plt.title("Monte Carlo — P&L Distribution")
    plt.legend()
    plt.tight_layout()
    save_figure("mc_pnl_distribution.png", dpi=150)


def plot_var_by_horizon(S0: float, mu: float, sigma: float,
                         horizons: list = None, confidence: float = 0.95) -> None:
    """
    Plot VaR and CVaR as a function of forecast horizon.
    Shows how risk grows with time horizon.
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 63, 126, 252]

    vars_  = []
    cvars_ = []

    for h in horizons:
        paths = simulate_gbm(S0, mu, sigma, h, n_sims=fast_or(20000, 1000), dt=1/252)
        r     = calculate_var_cvar(paths, S0, confidence=confidence)
        vars_.append(r["var"])
        cvars_.append(r["cvar"])

    plt.figure(figsize=(9, 4))
    plt.plot(horizons, vars_,  "o-", color="tomato",    label=f"VaR ({confidence*100:.0f}%)")
    plt.plot(horizons, cvars_, "s--", color="darkred",   label=f"CVaR ({confidence*100:.0f}%)")
    plt.xlabel("Forecast Horizon (trading days)")
    plt.ylabel("Loss ($)")
    plt.title("Monte Carlo — VaR/CVaR vs Horizon")
    plt.legend()
    plt.tight_layout()
    save_figure("mc_var_horizon.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Monte Carlo Simulation — Finance ===")
    cfg = MC_CONFIG
    ac  = ASSET_CONFIG
    np.random.seed(cfg["seed"])

    # 1. Single-asset price paths
    print(f"\nSimulating {cfg['n_simulations']:,} GBM paths "
          f"(μ={ac['mu']:.0%}, σ={ac['sigma']:.0%}, T={cfg['n_days']} days)...")
    paths = simulate_gbm(
        S0=ac["S0"], mu=ac["mu"], sigma=ac["sigma"],
        n_days=cfg["n_days"], n_sims=cfg["n_simulations"],
        dt=cfg["dt"], antithetic=cfg["antithetic"], seed=cfg["seed"],
    )

    # 2. VaR / CVaR
    print("\n1-Year VaR/CVaR (95% and 99%):")
    r95 = calculate_var_cvar(paths, ac["S0"], confidence=0.95)
    r99 = calculate_var_cvar(paths, ac["S0"], confidence=0.99)

    # 3. Option pricing (risk-neutral: drift = rf)
    print("\nEuropean Option Pricing (K=100, T=1yr, rf=4%):")
    anti = cfg["antithetic"]
    paths_rn = simulate_gbm(ac["S0"], ac["rf"], ac["sigma"],
                              cfg["n_days"], cfg["n_simulations"],
                              antithetic=anti, seed=cfg["seed"])
    price_european_option(paths_rn, K=100, rf=ac["rf"], T=1.0, option_type="call", antithetic=anti)
    price_european_option(paths_rn, K=100, rf=ac["rf"], T=1.0, option_type="put", antithetic=anti)

    print("\nAsian Call Option:")
    price_asian_option(paths_rn, K=100, rf=ac["rf"], T=1.0, option_type="call", antithetic=anti)

    # 4. Portfolio VaR
    print("\nMulti-Asset Portfolio VaR:")
    port_paths = simulate_portfolio_gbm(PORTFOLIO_CONFIG, cfg["n_days"],
                                         cfg["n_simulations"], seed=cfg["seed"])
    port_S0    = (PORTFOLIO_CONFIG["S0"] * PORTFOLIO_CONFIG["weights"]).sum()
    calculate_var_cvar(port_paths, S0=port_S0, confidence=0.95)

    # 5. Convergence
    print("\nConvergence analysis:")
    convergence_analysis(ac["S0"], ac["mu"], ac["sigma"], cfg["n_days"])

    # Visualizations
    plot_price_paths(paths, n_show=300)
    plot_pnl_distribution(r95["pnl"], r95["var"], r95["cvar"])
    plot_var_by_horizon(ac["S0"], ac["mu"], ac["sigma"])


if __name__ == "__main__":
    main()
