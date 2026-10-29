"""
GARCH(1,1) Volatility Modeling
================================
Generalized AutoRegressive Conditional Heteroskedasticity — models the
time-varying variance (volatility) of financial return series.

Financial returns exhibit:
    - Volatility clustering: high-vol periods follow high-vol periods
    - Fat tails: more extreme moves than Gaussian
    - Mean reversion: volatility reverts to a long-run average

GARCH(p,q) Model:
    Returns:     r_t = μ + ε_t,  ε_t = σ_t · z_t,  z_t ~ N(0,1)
    Variance:    σ²_t = ω + Σᵢ αᵢ ε²_{t-i} + Σⱼ βⱼ σ²_{t-j}

GARCH(1,1) — most widely used:
    σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}

    ω:  baseline variance (ω > 0)
    α:  ARCH term — reaction to recent shocks (0 < α < 1)
    β:  GARCH term — volatility persistence (0 < β < 1)
    α+β < 1 for stationarity; close to 1 → high persistence

Long-run variance:  σ²_∞ = ω / (1 - α - β)
Half-life of shock: t½ = log(0.5) / log(α + β)

Variants:
    EGARCH:  asymmetric leverage effect (bad news → more vol than good news)
    GJR-GARCH (TGARCH): asymmetric response via indicator term
    GARCH-M:  risk premium in mean (μ includes σ_t term)

Distribution choices:
    Normal:      underestimates tail risk
    Student-t:   better fat tails (most common in practice)
    Skewed-t:    captures both fat tails and skewness

Use Cases:
    - VaR / CVaR calculation (risk management)
    - Option pricing (local vol surface)
    - Dynamic portfolio optimization (time-varying covariance)
    - Trading signal: enter/exit based on vol regime
    - Volatility-adjusted position sizing

Fine-Tuning Recommendations:
    - GARCH(1,1) is sufficient for most financial series. Higher p,q rarely help.
    - Distribution: use Student-t (dist='t') to capture fat tails.
    - Use arch_model().fit(update_freq=5, disp='off') to suppress output.
    - AIC/BIC: compare GARCH(1,1) vs GARCH(2,1) vs EGARCH(1,1).
    - Forecast: use model.forecast(horizon=h, reindex=False) for multi-step.
    - Standardized residuals should be N(0,1) — check QQ-plot + Ljung-Box.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from arch.__future__ import reindexing
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from deepdraft.core import save_figure


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_return_series(
    n_obs:  int = 2000,
    seed:   int = 42,
) -> pd.Series:
    """
    Generate synthetic financial return series with GARCH(1,1) dynamics.

    Replace with real data:
        import yfinance as yf
        df = yf.download("SPY", start="2010-01-01", end="2024-01-01")
        returns = df["Adj Close"].pct_change().dropna() * 100  # in %

    Or from CSV:
        df = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
        returns = df["close"].pct_change().dropna() * 100

    Returns:
        pd.Series of daily percentage returns (multiply by 100 for arch library)
    """
    rng = np.random.RandomState(seed)

    # True GARCH(1,1) parameters
    omega, alpha, beta = 0.05, 0.10, 0.85
    n = n_obs

    returns = np.zeros(n)
    sigma2  = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_t(df=6)

    dates = pd.date_range(start="2014-01-01", periods=n, freq="B")
    return pd.Series(returns, index=dates, name="return_pct")


# ─────────────────────────────────────────────
# GARCH MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune p, q, dist, and model type
GARCH_CONFIG = {
    "vol":   "Garch",     # Placeholder: 'Garch', 'EGARCH', 'GJR-Garch'
    "p":     1,           # Placeholder: GARCH lag order (usually 1)
    "q":     1,           # Placeholder: ARCH lag order (usually 1)
    "dist":  "t",         # Placeholder: 'normal', 't' (Student-t), 'skewt'
    "mean":  "Constant",  # Placeholder: 'Constant', 'AR', 'Zero'
    "rescale": False,     # True if returns not already in % units
}


def build_garch(returns: pd.Series, config: dict = None):
    """
    Build GARCH model specification.

    Args:
        returns: return series in % (e.g., 1.5 for +1.5%)
        config:  GARCH_CONFIG dict

    Returns:
        arch ARCHModel object (unfitted)
    """
    if config is None:
        config = GARCH_CONFIG
    return arch_model(
        returns,
        vol=config["vol"],
        p=config["p"],
        q=config["q"],
        dist=config["dist"],
        mean=config["mean"],
        rescale=config["rescale"],
    )


# ─────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────

def compare_models(returns: pd.Series) -> None:
    """
    Compare GARCH variants by AIC/BIC and log-likelihood.
    """
    specs = [
        ("GARCH(1,1)-N",   {"vol": "Garch",     "p": 1, "q": 1, "dist": "normal"}),
        ("GARCH(1,1)-t",   {"vol": "Garch",     "p": 1, "q": 1, "dist": "t"}),
        ("GARCH(2,1)-t",   {"vol": "Garch",     "p": 2, "q": 1, "dist": "t"}),
        ("EGARCH(1,1)-t",  {"vol": "EGARCH",    "p": 1, "q": 1, "dist": "t"}),
        ("GJR-GARCH(1,1)", {"vol": "GJR-Garch", "p": 1, "q": 1, "dist": "t"}),
    ]
    results = []
    for name, cfg in specs:
        try:
            m = arch_model(returns, **cfg).fit(disp="off", update_freq=0)
            results.append((name, m.aic, m.bic, m.loglikelihood))
            print(f"  {name:<22} AIC={m.aic:.2f}  BIC={m.bic:.2f}  LogL={m.loglikelihood:.2f}")
        except Exception as e:
            print(f"  {name} failed: {e}")

    if results:
        best = min(results, key=lambda r: r[1])  # by AIC
        print(f"\n  Best model (AIC): {best[0]}")


# ─────────────────────────────────────────────
# EVALUATION & DIAGNOSTICS
# ─────────────────────────────────────────────

def fit_and_evaluate(returns: pd.Series, config: dict = None) -> tuple:
    """
    Fit GARCH model and run diagnostics.

    Returns:
        fitted result, conditional volatility series
    """
    if config is None:
        config = GARCH_CONFIG
    model  = build_garch(returns, config)
    result = model.fit(disp="off", update_freq=0)

    print(result.summary())

    # Extract parameters
    params = result.params
    print(f"\n  omega:  {params.get('omega', 'N/A'):.6f}")
    print(f"  alpha:  {params.get('alpha[1]', 'N/A'):.6f}")
    print(f"  beta:   {params.get('beta[1]',  'N/A'):.6f}")

    alpha = params.get("alpha[1]", 0)
    beta  = params.get("beta[1]",  0)
    if alpha > 0 and beta > 0 and alpha + beta < 1:
        lr_var  = params["omega"] / (1 - alpha - beta)
        half_life = np.log(0.5) / np.log(alpha + beta)
        print(f"  Long-run vol: {np.sqrt(lr_var):.4f}%")
        print(f"  Shock half-life: {half_life:.1f} days")

    cond_vol = result.conditional_volatility
    return result, cond_vol


def residual_diagnostics(result) -> None:
    """
    Check standardized residuals: should be i.i.d. N(0,1).

    Tests:
        - Ljung-Box on squared residuals (no ARCH effect remaining)
        - QQ-plot for normality
    """
    std_resid = result.std_resid.dropna()

    # Ljung-Box on squared residuals
    lb = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
    p_val = lb["lb_pvalue"].values[0]
    print(f"  Ljung-Box (sq. resid, lag 10): p={p_val:.4f}  "
          f"({'PASS' if p_val > 0.05 else 'FAIL — remaining ARCH'})")

    # Jarque-Bera normality test
    jb_stat, jb_p = stats.jarque_bera(std_resid)
    print(f"  Jarque-Bera: stat={jb_stat:.2f}  p={jb_p:.4f}  "
          f"({'NORMAL' if jb_p > 0.05 else 'NON-NORMAL (fat tails expected)'})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # QQ-plot
    stats.probplot(std_resid, dist="norm", plot=ax1)
    ax1.set_title("Standardized Residuals — QQ Plot")

    # Histogram
    ax2.hist(std_resid, bins=50, density=True, alpha=0.6, color="steelblue")
    x = np.linspace(-4, 4, 200)
    ax2.plot(x, stats.norm.pdf(x), "r--", linewidth=2, label="N(0,1)")
    ax2.set_title("Standardized Residuals — Histogram")
    ax2.legend()

    plt.tight_layout()
    save_figure("garch_diagnostics.png", dpi=150)


# ─────────────────────────────────────────────
# FORECASTING & VaR
# ─────────────────────────────────────────────

def forecast_volatility(result, horizon: int = 10) -> pd.DataFrame:
    """
    Multi-step conditional variance forecast.

    Returns DataFrame with forecast mean, variance, and volatility.
    """
    fc = result.forecast(horizon=horizon, reindex=False)
    forecast_df = pd.DataFrame({
        "mean_forecast":     fc.mean.values[-1],
        "variance_forecast": fc.variance.values[-1],
        "vol_forecast":      np.sqrt(fc.variance.values[-1]),
    })
    print(f"\n  {horizon}-step volatility forecast:")
    print(forecast_df.round(4).to_string())
    return forecast_df


def calculate_var(result, returns: pd.Series,
                   confidence: float = 0.99) -> pd.Series:
    """
    1-day Value at Risk (VaR) at given confidence level.

    VaR_t = μ_t + σ_t · z_α
    where z_α is the quantile of the assumed return distribution.

    Args:
        confidence: e.g., 0.99 for 99% VaR
    Returns:
        VaR series (negative = expected maximum loss)
    """
    alpha     = 1 - confidence
    cond_mean = result.params.get("mu", 0)
    cond_vol  = result.conditional_volatility

    # Use t-distribution quantile if t dist was used
    dist_name = GARCH_CONFIG["dist"]
    if dist_name == "t":
        nu = result.params.get("nu", 8)
        z  = stats.t.ppf(alpha, df=nu)
    else:
        z = stats.norm.ppf(alpha)

    var_series = cond_mean + cond_vol * z
    print(f"\n  VaR({confidence*100:.0f}%) stats:")
    print(f"    Mean:    {var_series.mean():.4f}%")
    print(f"    Min:     {var_series.min():.4f}%  (worst day)")
    print(f"    Max:     {var_series.max():.4f}%  (calmest day)")

    # Kupiec test: expected VaR violations
    violations = (returns < var_series).mean()
    expected   = alpha
    print(f"\n  VaR violations: {violations*100:.2f}%  (expected: {expected*100:.2f}%)")
    return var_series


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_returns_and_volatility(returns: pd.Series,
                                 cond_vol: pd.Series) -> None:
    """Plot return series with conditional volatility bands."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(returns.index, returns, linewidth=0.5, color="steelblue", alpha=0.7)
    ax1.set_ylabel("Return (%)")
    ax1.set_title("Daily Returns")

    ax2.plot(cond_vol.index, cond_vol, color="tomato", linewidth=1.0)
    ax2.fill_between(cond_vol.index, 0, cond_vol, alpha=0.3, color="tomato")
    ax2.axhline(cond_vol.mean(), color="gray", linestyle="--",
                label=f"Mean vol: {cond_vol.mean():.2f}%")
    ax2.set_ylabel("Conditional Volatility (%)")
    ax2.set_title("GARCH(1,1) Conditional Volatility")
    ax2.legend()

    plt.suptitle("GARCH — Returns & Conditional Volatility")
    plt.tight_layout()
    save_figure("garch_volatility.png", dpi=150)


def plot_var(returns: pd.Series, var_series: pd.Series,
              confidence: float = 0.99) -> None:
    """Plot returns vs. VaR bands, highlight violations."""
    violations_mask = returns < var_series

    plt.figure(figsize=(12, 5))
    plt.plot(returns.index, returns, linewidth=0.5, color="steelblue",
             alpha=0.6, label="Returns")
    plt.plot(var_series.index, var_series, color="tomato", linewidth=1.0,
             linestyle="--", label=f"VaR({confidence*100:.0f}%)")
    plt.scatter(returns.index[violations_mask], returns[violations_mask],
                c="red", s=10, zorder=5, label=f"Violations ({violations_mask.sum()})")
    plt.ylabel("Return (%)")
    plt.title(f"GARCH — VaR({confidence*100:.0f}%) Backtesting")
    plt.legend()
    plt.tight_layout()
    save_figure("garch_var.png", dpi=150)


def plot_volatility_forecast(forecast_df: pd.DataFrame) -> None:
    """Bar chart of multi-step volatility forecast."""
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(forecast_df) + 1), forecast_df["vol_forecast"],
            color="steelblue", alpha=0.8)
    plt.xlabel("Forecast Horizon (days)")
    plt.ylabel("Forecasted Volatility (%)")
    plt.title("GARCH — Multi-Step Volatility Forecast")
    plt.tight_layout()
    save_figure("garch_forecast.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== GARCH(1,1) Volatility Modeling ===")

    # Placeholder: replace with real return series
    returns = generate_return_series(n_obs=2000)
    print(f"Returns: {len(returns)} obs  "
          f"[{returns.index[0].date()} → {returns.index[-1].date()}]")
    print(f"  Mean: {returns.mean():.4f}%  |  Std: {returns.std():.4f}%  "
          f"|  Skew: {returns.skew():.3f}  |  Kurt: {returns.kurt():.3f}")

    # Model comparison
    print("\nModel comparison:")
    compare_models(returns)

    # Fit GARCH(1,1) with Student-t
    print("\nFitting GARCH(1,1)-t:")
    result, cond_vol = fit_and_evaluate(returns)

    # Residual diagnostics
    print("\nResidual diagnostics:")
    residual_diagnostics(result)

    # Volatility forecast
    forecast_df = forecast_volatility(result, horizon=10)

    # VaR
    var_99 = calculate_var(result, returns, confidence=0.99)

    # Visualizations
    plot_returns_and_volatility(returns, cond_vol)
    plot_var(returns, var_99)
    plot_volatility_forecast(forecast_df)


if __name__ == "__main__":
    main()
