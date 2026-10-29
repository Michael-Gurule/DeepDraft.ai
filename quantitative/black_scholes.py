"""
Black-Scholes Option Pricing Model
=====================================
The Black-Scholes-Merton (BSM) model gives closed-form prices for European options
under the assumption that the underlying follows Geometric Brownian Motion.

Black-Scholes PDE:
    ∂V/∂t + (1/2)σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0

Call price:    C = S·N(d₁) - K·e^{-rT}·N(d₂)
Put price:     P = K·e^{-rT}·N(-d₂) - S·N(-d₁)

Where:
    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
    N(·) = standard normal CDF
    S    = current underlying price
    K    = strike price
    T    = time to expiry (years)
    r    = risk-free rate (continuously compounded)
    σ    = volatility (annual)

The Greeks (risk sensitivities):
    Delta (Δ):   ∂V/∂S    — sensitivity to underlying price
    Gamma (Γ):   ∂²V/∂S²  — rate of change of Delta
    Theta (Θ):   ∂V/∂t    — time decay per day
    Vega  (ν):   ∂V/∂σ    — sensitivity to volatility (per 1% move in vol)
    Rho   (ρ):   ∂V/∂r    — sensitivity to interest rate

Implied Volatility:
    The market-implied σ that makes BS price = market price.
    No closed form; solve numerically (Newton-Raphson / Brent's method).

Put-Call Parity:
    C - P = S·e^{-qT} - K·e^{-rT}  (with continuous dividend yield q)

Use Cases:
    - Options trading: price + hedge vanilla options
    - Risk management: Greek-based hedging (delta-neutral, gamma-neutral)
    - Volatility surface: plot implied vol vs. strike (volatility smile/skew)
    - P&L attribution: isolate Delta/Gamma/Vega/Theta P&L components

Limitations:
    - Assumes constant σ (real vol is stochastic — see GARCH)
    - Assumes log-normal returns (real returns have fat tails)
    - No dividends in base model (extend with continuous yield q)
    - European exercise only (American options require numerical methods)

Fine-Tuning / Extensions:
    - Add dividend yield q: replace S → S·e^{-qT} in d₁ formula
    - Stochastic vol: Heston model (mean-reverting vol) for vol smile
    - Jump diffusion: Merton/Bates model for fat tails
    - Local vol: Dupire model — σ(S,t) calibrated from market
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

from deepdraft.core import save_figure


# ─────────────────────────────────────────────
# CORE BLACK-SCHOLES FORMULAS
# ─────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float,
            sigma: float, q: float = 0.0) -> tuple:
    """
    Compute d1 and d2 for Black-Scholes.

    Args:
        S:     current price
        K:     strike
        T:     time to expiry (years)
        r:     risk-free rate
        sigma: volatility (annual)
        q:     continuous dividend yield

    Returns:
        d1, d2
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    log_moneyness = np.log(S / K)
    d1 = (log_moneyness + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = "call",
              q: float = 0.0) -> float:
    """
    Black-Scholes option price.

    Args:
        S:           current underlying price
        K:           strike price
        T:           time to expiry (years)
        r:           annual risk-free rate (continuously compounded)
        sigma:       annual implied/realized volatility
        option_type: 'call' or 'put'
        q:           continuous dividend yield (Placeholder: 0 if no dividends)

    Returns:
        option price
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    discount = np.exp(-r * T)
    S_adj    = S * np.exp(-q * T)

    if option_type == "call":
        return S_adj * norm.cdf(d1) - K * discount * norm.cdf(d2)
    elif option_type == "put":
        return K * discount * norm.cdf(-d2) - S_adj * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# ─────────────────────────────────────────────
# THE GREEKS
# ─────────────────────────────────────────────

def bs_greeks(S: float, K: float, T: float, r: float,
               sigma: float, option_type: str = "call",
               q: float = 0.0) -> dict:
    """
    Compute Black-Scholes Greeks.

    Delta:  ∂V/∂S     — hedge ratio; shares of underlying to delta-hedge
    Gamma:  ∂²V/∂S²   — how fast Delta changes; large Γ → expensive to gamma-hedge
    Theta:  ∂V/∂t     — daily time decay (in $ per day)
    Vega:   ∂V/∂σ     — $ change per 1% (0.01) move in vol
    Rho:    ∂V/∂r     — $ change per 1% (0.01) move in rates

    Returns:
        dict of Greeks
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_T  = np.sqrt(T)
    discount = np.exp(-r * T)
    S_adj    = S * np.exp(-q * T)
    pdf_d1   = norm.pdf(d1)

    # Delta
    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

    # Gamma (same for call and put)
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * sqrt_T)

    # Theta (per calendar day = divide by 365; trading day = 252)
    theta_annual = (
        -S_adj * pdf_d1 * sigma / (2 * sqrt_T)
        - r * K * discount * (norm.cdf(d2) if option_type == "call" else -norm.cdf(-d2))
        + q * S_adj * (norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1))
    )
    theta = theta_annual / 365  # per calendar day

    # Vega (per 1% change in vol = multiply by 0.01)
    vega = S_adj * pdf_d1 * sqrt_T * 0.01  # scaled to 1% vol move

    # Rho (per 1% change in rate)
    if option_type == "call":
        rho = K * T * discount * norm.cdf(d2) * 0.01
    else:
        rho = -K * T * discount * norm.cdf(-d2) * 0.01

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega":  vega,
        "rho":   rho,
        "d1":    d1,
        "d2":    d2,
    }


# ─────────────────────────────────────────────
# IMPLIED VOLATILITY
# ─────────────────────────────────────────────

def implied_volatility(market_price: float, S: float, K: float,
                        T: float, r: float, option_type: str = "call",
                        q: float = 0.0,
                        tol: float = 1e-6) -> float:
    """
    Compute implied volatility via Brent's root-finding method.

    IV is the σ that makes BS price = observed market price.

    Args:
        market_price: observed option price from market
        S, K, T, r:   standard BS inputs
        option_type:  'call' or 'put'

    Returns:
        implied_vol (or NaN if not found)
    """
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == "call" else \
                max(K * np.exp(-r * T) - S, 0)

    if market_price < intrinsic:
        return float("nan")

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=200)
    except ValueError:
        iv = float("nan")
    return iv


def volatility_surface(S: float, T_range: list, K_range: list,
                         r: float, market_prices: np.ndarray,
                         option_type: str = "call") -> np.ndarray:
    """
    Build implied volatility surface from matrix of market prices.

    Args:
        S:             current spot price
        T_range:       list of expiry times (years)
        K_range:       list of strikes
        r:             risk-free rate
        market_prices: (n_T, n_K) matrix of observed prices
        option_type:   'call' or 'put'

    Returns:
        iv_surface: (n_T, n_K) implied volatility matrix
    """
    n_T = len(T_range)
    n_K = len(K_range)
    iv_surface = np.full((n_T, n_K), np.nan)

    for i, T in enumerate(T_range):
        for j, K in enumerate(K_range):
            iv_surface[i, j] = implied_volatility(
                market_prices[i, j], S, K, T, r, option_type
            )
    return iv_surface


# ─────────────────────────────────────────────
# PUT-CALL PARITY CHECK
# ─────────────────────────────────────────────

def verify_put_call_parity(S: float, K: float, T: float, r: float,
                             sigma: float, q: float = 0.0) -> None:
    """
    Verify: C - P = S·e^{-qT} - K·e^{-rT}

    Any deviation in market prices from parity = arbitrage opportunity.
    """
    call = bs_price(S, K, T, r, sigma, "call", q)
    put  = bs_price(S, K, T, r, sigma, "put",  q)
    lhs  = call - put
    rhs  = S * np.exp(-q * T) - K * np.exp(-r * T)
    print(f"  C - P = {lhs:.6f}  |  S·e^(-qT) - K·e^(-rT) = {rhs:.6f}  "
          f"|  Diff = {abs(lhs - rhs):.2e}  "
          f"({'PASS' if abs(lhs - rhs) < 1e-8 else 'FAIL'})")


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_option_payoff(S: float, K: float, T: float, r: float,
                        sigma: float) -> None:
    """
    Plot option price and intrinsic value as function of spot price.
    Shows time value = option price - intrinsic value.
    """
    S_range = np.linspace(0.5 * K, 1.5 * K, 200)

    call_prices = [bs_price(s, K, T, r, sigma, "call") for s in S_range]
    put_prices  = [bs_price(s, K, T, r, sigma, "put")  for s in S_range]
    call_intr   = np.maximum(S_range - K, 0)
    put_intr    = np.maximum(K - S_range, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(S_range, call_prices, "steelblue", linewidth=2, label="Call Price")
    ax1.plot(S_range, call_intr,   "k--",       linewidth=1, label="Intrinsic Value")
    ax1.fill_between(S_range, call_intr, call_prices, alpha=0.2, color="steelblue",
                     label="Time Value")
    ax1.axvline(K, color="gray", linestyle=":", alpha=0.7, label=f"Strike K={K}")
    ax1.set_title("Call Option")
    ax1.legend(fontsize=8)

    ax2.plot(S_range, put_prices, "tomato", linewidth=2, label="Put Price")
    ax2.plot(S_range, put_intr,   "k--",    linewidth=1, label="Intrinsic Value")
    ax2.fill_between(S_range, put_intr, put_prices, alpha=0.2, color="tomato",
                     label="Time Value")
    ax2.axvline(K, color="gray", linestyle=":", alpha=0.7, label=f"Strike K={K}")
    ax2.set_title("Put Option")
    ax2.legend(fontsize=8)

    for ax in [ax1, ax2]:
        ax.set_xlabel("Spot Price S")
        ax.set_ylabel("Option Price")
        ax.axvline(S, color="green", linestyle="--", alpha=0.5, label=f"Current S={S}")

    plt.suptitle(f"Black-Scholes Option Price  (σ={sigma:.0%}, T={T:.2f}yr, r={r:.0%})")
    plt.tight_layout()
    save_figure("bs_payoff.png", dpi=150)


def plot_greeks_profile(S: float, K: float, T: float, r: float,
                         sigma: float) -> None:
    """
    Plot all Greeks as a function of spot price.
    """
    S_range = np.linspace(0.5 * K, 1.5 * K, 200)
    greek_names = ["delta", "gamma", "theta", "vega", "rho"]

    call_greeks = {g: [] for g in greek_names}
    put_greeks  = {g: [] for g in greek_names}

    for s in S_range:
        cg = bs_greeks(s, K, T, r, sigma, "call")
        pg = bs_greeks(s, K, T, r, sigma, "put")
        for g in greek_names:
            call_greeks[g].append(cg[g])
            put_greeks[g].append(pg[g])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, g in zip(axes, greek_names):
        ax.plot(S_range, call_greeks[g], "steelblue", linewidth=1.5, label="Call")
        ax.plot(S_range, put_greeks[g],  "tomato",    linewidth=1.5, label="Put")
        ax.axvline(K, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(g.capitalize())
        ax.legend(fontsize=8)
        ax.set_xlabel("Spot Price")

    axes[-1].axis("off")
    plt.suptitle(f"Black-Scholes Greeks  (K={K}, T={T:.2f}yr, σ={sigma:.0%})")
    plt.tight_layout()
    save_figure("bs_greeks.png", dpi=150)


def plot_volatility_smile(S: float, T: float, r: float,
                           base_sigma: float = 0.20) -> None:
    """
    Simulate a volatility smile by generating market prices with stochastic vol
    and computing implied vols. In reality, the smile arises from jumps / fat tails.
    """
    # Simulate "market prices" with a Heston-like vol smile (manually injected)
    K_range   = np.linspace(0.7 * S, 1.3 * S, 30)
    moneyness = np.log(K_range / S)

    # Stylized vol smile: ATM base vol, wings up
    smiles = base_sigma + 0.05 * moneyness ** 2 + 0.02 * moneyness

    # Generate "market prices" from these smiles
    market_prices = np.array([
        bs_price(S, K, T, r, vol, "call")
        for K, vol in zip(K_range, smiles)
    ])

    # Recover implied vols
    iv_list = [implied_volatility(p, S, K, T, r, "call")
               for p, K in zip(market_prices, K_range)]

    plt.figure(figsize=(9, 4))
    plt.plot(K_range / S, np.array(iv_list) * 100, "steelblue", linewidth=2)
    plt.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="ATM (K/S=1)")
    plt.xlabel("Moneyness (K/S)")
    plt.ylabel("Implied Volatility (%)")
    plt.title(f"Implied Volatility Smile  (T={T:.2f}yr)")
    plt.legend()
    plt.tight_layout()
    save_figure("bs_vol_smile.png", dpi=150)


def plot_price_vs_volatility(S: float, K: float, T: float, r: float) -> None:
    """
    Plot option price as a function of implied volatility.
    Shows Vega (the slope) at the current vol level.
    """
    sigma_range = np.linspace(0.01, 1.0, 200)
    call_prices = [bs_price(S, K, T, r, s, "call") for s in sigma_range]
    put_prices  = [bs_price(S, K, T, r, s, "put")  for s in sigma_range]

    plt.figure(figsize=(9, 4))
    plt.plot(sigma_range * 100, call_prices, "steelblue", linewidth=2, label="Call")
    plt.plot(sigma_range * 100, put_prices,  "tomato",    linewidth=2, label="Put")
    plt.xlabel("Volatility (%)")
    plt.ylabel("Option Price")
    plt.title(f"Option Price vs Volatility  (S={S}, K={K}, T={T:.2f}yr)")
    plt.legend()
    plt.tight_layout()
    save_figure("bs_price_vs_vol.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Black-Scholes Option Pricing & Greeks ===")

    # Placeholder: replace with your option parameters
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.04, 0.20

    print(f"\nParameters: S={S}, K={K}, T={T}yr, r={r:.0%}, σ={sigma:.0%}")

    # Option prices
    call_price = bs_price(S, K, T, r, sigma, "call")
    put_price  = bs_price(S, K, T, r, sigma, "put")
    print(f"\nOption Prices:")
    print(f"  Call: ${call_price:.4f}")
    print(f"  Put:  ${put_price:.4f}")

    # Greeks
    print("\nCall Greeks:")
    call_g = bs_greeks(S, K, T, r, sigma, "call")
    for name, val in call_g.items():
        if name not in ("d1", "d2"):
            print(f"  {name.capitalize():<7} = {val:.6f}")

    print("\nPut Greeks:")
    put_g = bs_greeks(S, K, T, r, sigma, "put")
    for name, val in put_g.items():
        if name not in ("d1", "d2"):
            print(f"  {name.capitalize():<7} = {val:.6f}")

    # Put-Call Parity
    print("\nPut-Call Parity check:")
    verify_put_call_parity(S, K, T, r, sigma)

    # Implied Volatility round-trip
    market_price = call_price * 1.05  # 5% above BS price
    iv = implied_volatility(market_price, S, K, T, r, "call")
    print(f"\nImplied Vol (market price {market_price:.4f}): {iv:.4f}  "
          f"(true: {sigma:.4f})")

    # Multiple strikes and expiries
    print("\nOption price table (call, ATM=100):")
    print(f"{'Strike':>10} {'T=0.25':>10} {'T=0.5':>10} {'T=1.0':>10}")
    for K_ in [85, 90, 95, 100, 105, 110, 115]:
        row = [f"{bs_price(S, K_, T_, r, sigma, 'call'):>10.4f}"
               for T_ in [0.25, 0.5, 1.0]]
        print(f"{K_:>10}" + "".join(row))

    # Visualizations
    plot_option_payoff(S, K, T, r, sigma)
    plot_greeks_profile(S, K, T, r, sigma)
    plot_volatility_smile(S, T, r, sigma)
    plot_price_vs_volatility(S, K, T, r)


if __name__ == "__main__":
    main()
