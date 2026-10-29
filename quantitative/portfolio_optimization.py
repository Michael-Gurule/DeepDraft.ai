"""
Markowitz Portfolio Optimization (Mean-Variance)
==================================================
Modern Portfolio Theory (MPT) — find the asset allocation that maximizes
expected return for a given level of risk (or minimizes risk for a given return).

Framework:
    Portfolio return:   E[R_p] = w^T μ
    Portfolio variance: σ²_p   = w^T Σ w

    where:
        w  = weight vector (n_assets,), Σw = 1, w ≥ 0 (long-only)
        μ  = expected returns vector
        Σ  = covariance matrix of asset returns

Efficient Frontier:
    The set of portfolios that are Pareto-optimal in the risk-return space.
    For each risk level, no portfolio achieves higher return; for each return
    target, no portfolio achieves lower risk.

Special portfolios:
    Minimum Variance Portfolio (MVP):    argmin w^T Σ w
    Maximum Sharpe Ratio (Tangency):     argmax (w^T μ - r_f) / sqrt(w^T Σ w)
    Equal Weight:                        w_i = 1/n
    Risk Parity:                         allocate so each asset contributes equally to σ²_p

Sharpe Ratio:
    S = (E[R_p] - r_f) / σ_p   (reward per unit of risk)

Use Cases:
    - Asset allocation across equities, bonds, alternatives
    - Budget allocation across marketing channels or projects
    - Portfolio rebalancing given updated expected returns
    - Combine with Black-Litterman views for more stable allocations

Limitations:
    - Sensitive to input estimates (μ, Σ) — garbage in, garbage out
    - Historical μ estimation is noisy; covariance is more stable
    - Short-selling constraint (w ≥ 0) significantly restricts the frontier
    - Transaction costs and liquidity not modeled

Fine-Tuning Recommendations:
    - Use shrinkage estimators for Σ: LedoitWolf or OAS (sklearn)
      to reduce estimation error in high dimensions.
    - For μ: use Black-Litterman (blend prior with views) or just use
      0 expected return (minimum variance only) for more stability.
    - Add constraints: max position size (w_i ≤ 0.3), sector constraints,
      turnover constraints via CVXPY.
    - Robustness: optimize for worst-case over a set of μ scenarios.
    - Rebalance frequency: monthly or quarterly; too frequent → transaction costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from deepdraft.core import save_figure


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_return_data(
    n_assets:  int = 10,
    n_periods: int = 252 * 5,   # 5 years of daily returns
    seed:      int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic multi-asset daily return matrix.

    Replace with real data:
        import yfinance as yf
        tickers = ["SPY", "TLT", "GLD", "QQQ", "IWM"]
        prices = yf.download(tickers, start="2018-01-01")["Adj Close"]
        returns = prices.pct_change().dropna()

    Or from CSV:
        prices  = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
        returns = prices.pct_change().dropna()

    Returns:
        pd.DataFrame of daily returns (n_periods, n_assets)
    """
    rng = np.random.RandomState(seed)

    # Random correlation matrix
    A    = rng.randn(n_assets, n_assets)
    corr = A @ A.T
    np.fill_diagonal(corr, 1.0)
    corr /= np.diag(corr)[:, None]  # normalize to correlation

    # Asset volatilities (annual, then scaled to daily)
    annual_vols = rng.uniform(0.10, 0.35, n_assets)
    daily_vols  = annual_vols / np.sqrt(252)

    # Scale correlation to covariance
    cov = np.diag(daily_vols) @ corr @ np.diag(daily_vols)

    # Daily expected returns (slight positive drift)
    daily_mu = rng.uniform(0.0001, 0.0008, n_assets)

    # Multivariate normal returns
    returns = rng.multivariate_normal(daily_mu, cov, n_periods)
    cols    = [f"Asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=cols)


# ─────────────────────────────────────────────
# RETURN & RISK ESTIMATION
# ─────────────────────────────────────────────

def estimate_params(returns: pd.DataFrame,
                    annualize: bool = True,
                    shrink_cov: bool = True) -> tuple:
    """
    Estimate expected returns and covariance matrix.

    Args:
        returns:    daily return DataFrame
        annualize:  if True, scale to annual figures (×252)
        shrink_cov: use Ledoit-Wolf shrinkage for Σ (recommended)

    Returns:
        mu:   expected returns (n_assets,)
        cov:  covariance matrix (n_assets, n_assets)
    """
    mu = returns.mean().values
    if shrink_cov:
        lw  = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
    else:
        cov = returns.cov().values

    if annualize:
        mu  = mu * 252
        cov = cov * 252

    return mu, cov


# ─────────────────────────────────────────────
# PORTFOLIO METRICS
# ─────────────────────────────────────────────

def portfolio_metrics(w: np.ndarray, mu: np.ndarray,
                       cov: np.ndarray, rf: float = 0.0) -> tuple:
    """
    Compute portfolio return, volatility, and Sharpe ratio.

    Args:
        w:   weight vector (n_assets,)
        mu:  expected annual returns
        cov: annual covariance matrix
        rf:  risk-free rate (annual)

    Returns:
        ret, vol, sharpe
    """
    ret    = float(w @ mu)
    var    = float(w @ cov @ w)
    vol    = float(np.sqrt(var))
    sharpe = (ret - rf) / vol if vol > 1e-8 else 0.0
    return ret, vol, sharpe


# ─────────────────────────────────────────────
# OPTIMIZATION
# ─────────────────────────────────────────────

# Placeholder: tune rf (risk-free rate) to current rate
OPT_CONFIG = {
    "rf":             0.04,    # Placeholder: current risk-free rate (e.g., 3-month T-bill)
    "n_frontier_pts": 200,     # points on efficient frontier
    "max_weight":     0.30,    # Placeholder: maximum allocation per asset (0=no limit)
    "min_weight":     0.0,     # 0 = long-only
}


def _constraints(n_assets: int) -> list:
    """Standard portfolio constraints: weights sum to 1."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _bounds(n_assets: int, min_w: float = 0.0, max_w: float = 1.0) -> list:
    """Weight bounds: [min_weight, max_weight] per asset."""
    return [(min_w, max_w)] * n_assets


def minimum_variance_portfolio(mu: np.ndarray, cov: np.ndarray,
                                 config: dict = None) -> np.ndarray:
    """
    Find the Global Minimum Variance Portfolio (MVP).
    No return target required — just minimize σ²_p.
    """
    if config is None:
        config = OPT_CONFIG
    n = len(mu)
    w0 = np.ones(n) / n

    res = minimize(
        lambda w: w @ cov @ w,
        w0,
        method="SLSQP",
        bounds=_bounds(n, config["min_weight"], config["max_weight"]),
        constraints=_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x


def maximum_sharpe_portfolio(mu: np.ndarray, cov: np.ndarray,
                               config: dict = None) -> np.ndarray:
    """
    Find the Maximum Sharpe Ratio (Tangency) Portfolio.
    """
    if config is None:
        config = OPT_CONFIG
    rf = config["rf"]
    n  = len(mu)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        ret, vol, _ = portfolio_metrics(w, mu, cov, rf)
        return -(ret - rf) / (vol + 1e-8)

    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=_bounds(n, config["min_weight"], config["max_weight"]),
        constraints=_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x


def efficient_frontier(mu: np.ndarray, cov: np.ndarray,
                         config: dict = None) -> pd.DataFrame:
    """
    Trace the efficient frontier by solving minimum variance for each return target.

    Returns DataFrame with columns: return, vol, sharpe, weights.
    """
    if config is None:
        config = OPT_CONFIG
    n      = len(mu)
    w_mvp  = minimum_variance_portfolio(mu, cov, config)
    ret_mvp, _, _ = portfolio_metrics(w_mvp, mu, cov)
    ret_max = mu.max() * 0.99  # cap at max single asset return

    targets = np.linspace(ret_mvp, ret_max, config["n_frontier_pts"])
    records = []

    for target in targets:
        constraints = _constraints(n) + [
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t}
        ]
        w0  = np.ones(n) / n
        res = minimize(
            lambda w: w @ cov @ w,
            w0,
            method="SLSQP",
            bounds=_bounds(n, config["min_weight"], config["max_weight"]),
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if res.success:
            ret, vol, sharpe = portfolio_metrics(res.x, mu, cov, config["rf"])
            records.append({"return": ret, "vol": vol, "sharpe": sharpe,
                             "weights": res.x})

    return pd.DataFrame(records)


def risk_parity_portfolio(cov: np.ndarray) -> np.ndarray:
    """
    Risk Parity: each asset contributes equally to total portfolio variance.

    Marginal risk contribution: MRC_i = (Σ w)_i
    Total risk contribution:    TRC_i = w_i · MRC_i
    Objective: minimize variance of TRC_i across assets.
    """
    n  = cov.shape[0]
    w0 = np.ones(n) / n

    def risk_contrib_variance(w):
        sigma2 = w @ cov @ w
        mrc    = cov @ w               # marginal risk contributions
        trc    = w * mrc               # total risk contributions
        target = sigma2 / n            # equal contribution target
        return np.sum((trc - target) ** 2)

    res = minimize(
        risk_contrib_variance,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=_constraints(n),
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    return res.x


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_efficient_frontier(frontier_df: pd.DataFrame,
                             mu: np.ndarray, cov: np.ndarray,
                             asset_names: list = None,
                             config: dict = None) -> None:
    """
    Plot efficient frontier with special portfolios highlighted.
    """
    if config is None:
        config = OPT_CONFIG
    rf = config["rf"]
    n  = len(mu)
    if asset_names is None:
        asset_names = [f"A{i}" for i in range(n)]

    w_mvp     = minimum_variance_portfolio(mu, cov, config)
    w_sharpe  = maximum_sharpe_portfolio(mu, cov, config)
    w_eq      = np.ones(n) / n
    w_rp      = risk_parity_portfolio(cov)

    def p(w): return portfolio_metrics(w, mu, cov, rf)

    plt.figure(figsize=(10, 7))

    # Frontier curve
    plt.plot(frontier_df["vol"], frontier_df["return"],
             "steelblue", linewidth=2.5, label="Efficient Frontier")

    # Individual assets
    for i in range(n):
        w = np.zeros(n); w[i] = 1.0
        r, v, _ = p(w)
        plt.scatter(v, r, c="gray", s=30, zorder=4, alpha=0.7)
        plt.annotate(asset_names[i], (v, r), fontsize=7, alpha=0.6)

    # Special portfolios
    for w, name, color, marker in [
        (w_mvp,    "Min Variance",       "darkgreen", "D"),
        (w_sharpe, "Max Sharpe",         "tomato",    "*"),
        (w_eq,     "Equal Weight",       "purple",    "s"),
        (w_rp,     "Risk Parity",        "orange",    "^"),
    ]:
        r, v, sr = p(w)
        plt.scatter(v, r, c=color, s=150, zorder=5, marker=marker,
                    label=f"{name} (SR={sr:.2f})")

    # Capital Market Line
    _, vol_sharpe, _ = p(w_sharpe)
    ret_sharpe, _, _ = p(w_sharpe)
    x_cml = np.array([0, vol_sharpe * 1.5])
    y_cml = rf + (ret_sharpe - rf) / vol_sharpe * x_cml
    plt.plot(x_cml, y_cml, "k--", linewidth=1.0, alpha=0.5, label="Capital Market Line")
    plt.scatter(0, rf, c="black", s=50, zorder=5, label=f"Risk-free ({rf*100:.1f}%)")

    plt.xlabel("Annual Volatility (std)")
    plt.ylabel("Annual Expected Return")
    plt.title("Markowitz — Efficient Frontier")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_figure("portfolio_frontier.png", dpi=150)


def plot_portfolio_weights(weights_dict: dict) -> None:
    """
    Side-by-side bar chart comparing allocations across portfolios.
    """
    n_portfolios = len(weights_dict)
    n_assets     = len(next(iter(weights_dict.values())))
    fig, axes    = plt.subplots(1, n_portfolios, figsize=(4 * n_portfolios, 5))
    if n_portfolios == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab20")

    for ax, (name, w) in zip(axes, weights_dict.items()):
        colors = [cmap(i) for i in range(n_assets)]
        ax.bar(range(n_assets), w * 100, color=colors, alpha=0.85)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Asset")
        ax.set_ylabel("Weight (%)")
        ax.set_xticks(range(n_assets))
        ax.set_ylim(0, 60)

    plt.suptitle("Portfolio Allocations", y=1.02)
    plt.tight_layout()
    save_figure("portfolio_weights.png", dpi=150)


def plot_risk_contributions(w: np.ndarray, cov: np.ndarray,
                             asset_names: list = None) -> None:
    """
    Bar chart of each asset's share of portfolio variance.

    Contributions can be negative when an asset is negatively correlated with
    the rest of the portfolio (it hedges risk), so a pie chart cannot show them.
    """
    mrc     = cov @ w
    trc     = w * mrc
    trc_pct = 100 * trc / trc.sum()

    if asset_names is None:
        asset_names = [f"A{i}" for i in range(len(w))]

    plt.figure(figsize=(8, 4))
    colors = ["tomato" if v < 0 else "steelblue" for v in trc_pct]
    plt.bar(asset_names, trc_pct, color=colors, edgecolor="k")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Share of portfolio variance (%)")
    plt.title("Portfolio — Risk Contributions")
    plt.tight_layout()
    save_figure("portfolio_risk_contrib.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Markowitz Portfolio Optimization ===")

    # Placeholder: replace with your asset return data
    returns = generate_return_data(n_assets=10, n_periods=252 * 5)
    print(f"Assets: {returns.shape[1]}  |  Periods: {returns.shape[0]}")

    mu, cov = estimate_params(returns, annualize=True, shrink_cov=True)
    asset_names = returns.columns.tolist()
    rf = OPT_CONFIG["rf"]
    print(f"\nAnnualized returns: {(mu * 100).round(1)} %")
    print(f"Risk-free rate: {rf*100:.1f}%")

    # Compute special portfolios
    w_mvp    = minimum_variance_portfolio(mu, cov)
    w_sharpe = maximum_sharpe_portfolio(mu, cov)
    w_eq     = np.ones(len(mu)) / len(mu)
    w_rp     = risk_parity_portfolio(cov)

    print("\nPortfolio Metrics (Return, Vol, Sharpe):")
    for name, w in [("Min Variance", w_mvp), ("Max Sharpe", w_sharpe),
                     ("Equal Weight", w_eq), ("Risk Parity", w_rp)]:
        r, v, s = portfolio_metrics(w, mu, cov, rf)
        print(f"  {name:<18} Return={r*100:.2f}%  Vol={v*100:.2f}%  Sharpe={s:.3f}")

    # Efficient frontier
    print("\nTracing efficient frontier...")
    frontier_df = efficient_frontier(mu, cov)
    print(f"  Frontier points computed: {len(frontier_df)}")

    # Visualizations
    plot_efficient_frontier(frontier_df, mu, cov, asset_names)
    plot_portfolio_weights({
        "Min Variance": w_mvp,
        "Max Sharpe": w_sharpe,
        "Risk Parity": w_rp,
    })
    plot_risk_contributions(w_sharpe, cov, asset_names)


if __name__ == "__main__":
    main()
