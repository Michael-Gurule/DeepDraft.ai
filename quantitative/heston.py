"""
Heston Stochastic Volatility Model
=====================================

The Heston model (1993) extends Black-Scholes by modeling volatility as a
mean-reverting CIR process. This captures the volatility smile/skew that
Black-Scholes cannot reproduce with a single constant σ.

SDEs (risk-neutral measure Q):
  dS_t = r S_t dt + √V_t S_t dW_t^S
  dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V

  Correlation: dW_t^S · dW_t^V = ρ dt

Parameters:
  S₀    — initial spot price
  V₀    — initial variance (V₀ = σ₀²)
  κ     — mean reversion speed  (# Placeholder: 2–5 typical)
  θ     — long-run variance     (# Placeholder: implied vol² at long end)
  ξ     — volatility of variance / vol-of-vol  (# Placeholder: 0.1–1.0)
  ρ     — spot-vol correlation  (# Placeholder: -0.9 to 0 for equities)
  r     — risk-free rate
  q     — dividend yield

Feller condition: 2κθ > ξ²  → variance stays positive.
If violated, variance can hit zero (handled via reflection/absorption).

Pricing approaches:
  1. Analytic (Heston 1993): semi-closed form via characteristic function + FFT
       C = S₀ e^{-qT} P₁ - K e^{-rT} P₂
       P_j = 1/2 + 1/π ∫₀^∞ Re[e^{-iφ ln K} f_j(φ)] / (iφ) dφ
       where f_j are known characteristic functions.

  2. Monte Carlo simulation: Euler-Milstein discretization of the SDEs.
       Milstein correction for variance: + ξ²/4 (ΔW_V² - Δt)

  3. COS method (Fang & Oosterlee 2008): fast Fourier series option pricing
     (not implemented here; see reference).

Calibration:
  Minimize sum of squared implied vol differences vs market quotes.
  Minimize Σ (σ_market(K,T) - σ_heston(K,T))² over (κ, θ, ξ, ρ, V₀).

Fine-tuning recommendations:
  - kappa (#Placeholder: 2.0): larger = faster mean reversion, less persistence
  - theta (#Placeholder: 0.04): = long-run σ² (e.g. 0.04 = 20% vol)
  - xi (#Placeholder: 0.3): larger = fatter tails / more vol-of-vol
  - rho (#Placeholder: -0.7): negative = left skew (equity-typical)
  - V0 (#Placeholder: 0.04): start near theta for stationary regime
  - n_paths (#Placeholder: 100_000): more paths → tighter MC intervals
  - n_steps (#Placeholder: 252): daily steps per year typically sufficient

References:
  Heston (1993) — A Closed-Form Solution for Options with Stochastic Volatility
  Broadie & Kaya (2006) — Exact Simulation of Stochastic Volatility (BK scheme)
  Fang & Oosterlee (2008) — A Novel Pricing Method via Fourier-Cosine Series (COS)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # Heston model parameters
    "S0":    100.0,   # # Placeholder: initial spot price
    "V0":    0.04,    # # Placeholder: initial variance (= σ₀²)
    "kappa": 2.0,     # # Placeholder: mean reversion speed
    "theta": 0.04,    # # Placeholder: long-run variance
    "xi":    0.3,     # # Placeholder: vol-of-vol
    "rho":   -0.7,    # # Placeholder: spot-vol correlation
    "r":     0.05,    # # Placeholder: risk-free rate
    "q":     0.0,     # # Placeholder: dividend yield
    # Monte Carlo
    "n_paths": 50_000,   # # Placeholder:
    "n_steps": 252,      # # Placeholder: time steps per year
    "seed":    42,
}


# ─────────────────────────────────────────────
# Characteristic Function
# ─────────────────────────────────────────────

def heston_char_fn(
    phi: complex,
    S0: float, V0: float, kappa: float, theta: float,
    xi: float, rho: float, r: float, q: float, T: float,
) -> complex:
    """
    Heston characteristic function E[e^{i phi ln S_T}].
    Uses Albrecher et al. (2007) formula to avoid branch-cut issues.
    """
    i = 1j
    a = kappa * theta
    u = -0.5
    b = kappa - rho * xi * phi * i

    d = np.sqrt((rho * xi * phi * i - b)**2 - xi**2 * (2 * u * phi * i - phi**2))
    g = (b - rho * xi * phi * i + d) / (b - rho * xi * phi * i - d)

    exp_dT  = np.exp(d * T)
    C = (r - q) * phi * i * T + (a / xi**2) * (
        (b - rho * xi * phi * i + d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((b - rho * xi * phi * i + d) / xi**2) * (1 - exp_dT) / (1 - g * exp_dT)
    return np.exp(C + D * V0 + i * phi * np.log(S0))


def heston_price_analytic(
    S0: float, K: float, T: float, r: float, q: float,
    V0: float, kappa: float, theta: float, xi: float, rho: float,
    option_type: str = "call",
    n_quad: int = 200,    # # Placeholder: quadrature points
) -> float:
    """
    Semi-analytic Heston option price via numerical integration of
    the characteristic function.

    C = S₀ e^{-qT} P₁ - K e^{-rT} P₂
    """
    def integrand(phi, j):
        cf = heston_char_fn(phi - j * 1j, S0, V0, kappa, theta, xi, rho, r, q, T)
        cf0 = heston_char_fn(-j * 1j, S0, V0, kappa, theta, xi, rho, r, q, T)
        psi = cf / cf0
        return np.real(np.exp(-1j * phi * np.log(K)) * psi / (1j * phi))

    P1, _ = quad(integrand, 1e-10, 200, args=(1,), limit=n_quad)
    P2, _ = quad(integrand, 1e-10, 200, args=(0,), limit=n_quad)
    P1 = 0.5 + P1 / np.pi
    P2 = 0.5 + P2 / np.pi

    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    if option_type == "call":
        return max(call, 0.0)
    else:  # put via put-call parity
        return max(call - S0 * np.exp(-q * T) + K * np.exp(-r * T), 0.0)


# ─────────────────────────────────────────────
# Monte Carlo Simulation
# ─────────────────────────────────────────────

def simulate_heston(
    S0: float, V0: float, kappa: float, theta: float,
    xi: float, rho: float, r: float, q: float,
    T: float, n_paths: int, n_steps: int, seed: int = 42,
    scheme: str = "milstein",   # # Placeholder: "euler" | "milstein"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation of Heston model.
    Returns (S_paths, V_paths) each of shape (n_paths, n_steps+1).

    Milstein scheme adds ξ²/4 (ΔW_V² - Δt) correction for variance.
    Reflection: ensure V ≥ 0 via abs() (Full Truncation variant).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    S = np.full(n_paths, S0)
    V = np.full(n_paths, V0)

    # Correlated Brownian motions via Cholesky
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])

    S_paths = np.zeros((n_paths, n_steps + 1)); S_paths[:, 0] = S
    V_paths = np.zeros((n_paths, n_steps + 1)); V_paths[:, 0] = V

    for t in range(n_steps):
        Z = rng.standard_normal((n_paths, 2)) @ L.T
        Z_S = Z[:, 0]; Z_V = Z[:, 1]

        V_pos = np.maximum(V, 0)   # full truncation

        if scheme == "milstein":
            dV = (kappa * (theta - V_pos) * dt
                  + xi * np.sqrt(V_pos * dt) * Z_V
                  + 0.25 * xi**2 * dt * (Z_V**2 - 1))
        else:   # Euler
            dV = kappa * (theta - V_pos) * dt + xi * np.sqrt(V_pos * dt) * Z_V

        dS = (r - q) * S * dt + np.sqrt(V_pos * dt) * S * Z_S

        V = np.abs(V + dV)    # reflection to ensure non-negative variance
        S = S + dS

        S_paths[:, t+1] = S
        V_paths[:, t+1] = V

    return S_paths, V_paths


def price_option_mc(
    S_paths: np.ndarray, K: float, r: float, q: float, T: float,
    option_type: str = "call",
    payoff_type: str = "european",  # # Placeholder: "european" | "asian"
) -> tuple[float, float]:
    """
    Price option from simulated terminal prices.
    Returns (price, std_error).
    """
    S_T = S_paths[:, -1]

    if payoff_type == "european":
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
    elif payoff_type == "asian":
        S_avg = S_paths.mean(axis=1)
        if option_type == "call":
            payoffs = np.maximum(S_avg - K, 0)
        else:
            payoffs = np.maximum(K - S_avg, 0)
    else:
        raise ValueError(payoff_type)

    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    stderr = discount * payoffs.std() / np.sqrt(len(payoffs))
    return price, stderr


# ─────────────────────────────────────────────
# Implied Volatility from Heston Price
# ─────────────────────────────────────────────

def bs_call(S, K, T, r, q, sigma):
    """Black-Scholes call price (for IV inversion)."""
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def implied_vol(S0, K, T, r, q, price, tol: float = 1e-6) -> float:
    """Bisection root-find for Black-Scholes IV from a given option price."""
    lo, hi = 1e-6, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if bs_call(S0, K, T, r, q, mid) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def heston_vol_surface(
    S0: float, V0: float, kappa: float, theta: float,
    xi: float, rho: float, r: float, q: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """
    Compute implied volatility surface from Heston model prices.
    Returns (len(maturities), len(strikes)) IV matrix.
    """
    iv_surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            try:
                price = heston_price_analytic(S0, K, T, r, q, V0, kappa, theta, xi, rho)
                iv_surface[i, j] = implied_vol(S0, K, T, r, q, price)
            except Exception:
                iv_surface[i, j] = np.nan
    return iv_surface


# ─────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────

def calibrate_heston(
    market_prices: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    S0: float, r: float, q: float,
    method: str = "differential_evolution",  # # Placeholder: "Nelder-Mead" | "differential_evolution"
) -> dict:
    """
    Calibrate Heston parameters to market option prices.
    Minimizes sum of squared price differences.
    Parameters: (V0, kappa, theta, xi, rho)
    Bounds: ensure Feller condition 2κθ > ξ².
    """
    def objective(params):
        V0, kappa, theta, xi, rho = params
        if 2 * kappa * theta < xi**2:   # Feller condition
            return 1e10
        total_err = 0.0
        for (T, K, mkt_price) in zip(maturities, strikes, market_prices):
            try:
                model_price = heston_price_analytic(S0, K, T, r, q, V0, kappa, theta, xi, rho)
                total_err += (model_price - mkt_price) ** 2
            except Exception:
                total_err += 1e4
        return total_err

    bounds = [
        (0.001, 0.5),     # V0
        (0.1,   10.0),    # kappa
        (0.001, 0.5),     # theta
        (0.01,  2.0),     # xi
        (-0.99, -0.01),   # rho (negative for equity)
    ]

    print("Calibrating Heston model...")
    if method == "differential_evolution":
        result = differential_evolution(objective, bounds, seed=42, maxiter=100,
                                         tol=1e-6, popsize=10, workers=1)
    else:
        x0 = [0.04, 2.0, 0.04, 0.3, -0.7]
        result = minimize(objective, x0, method=method,
                          bounds=bounds, options={"maxiter": 1000})

    V0, kappa, theta, xi, rho = result.x
    print(f"Calibrated: V0={V0:.4f} κ={kappa:.4f} θ={theta:.4f} ξ={xi:.4f} ρ={rho:.4f}")
    print(f"Feller:  2κθ={2*kappa*theta:.4f} > ξ²={xi**2:.4f}: {2*kappa*theta > xi**2}")
    print(f"Objective: {result.fun:.6f}")

    return {"V0": V0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho,
            "success": result.success, "fun": result.fun}


def generate_synthetic_market(cfg: dict) -> dict:
    """
    Generate synthetic market prices using a reference Heston model.
    Used to test calibration.
    """
    strikes    = np.array([80, 90, 95, 100, 105, 110, 120])   # # Placeholder:
    maturities = np.array([0.25, 0.5, 1.0])                    # # Placeholder:
    all_prices, all_K, all_T = [], [], []
    for T in maturities:
        for K in strikes:
            p = heston_price_analytic(
                cfg["S0"], K, T, cfg["r"], cfg["q"],
                cfg["V0"], cfg["kappa"], cfg["theta"], cfg["xi"], cfg["rho"],
            )
            # Add 1% noise for realistic calibration test
            p *= (1 + np.random.normal(0, 0.01))
            all_prices.append(p); all_K.append(K); all_T.append(T)
    return {"prices": np.array(all_prices), "strikes": np.array(all_K),
            "maturities": np.array(all_T)}


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_vol_surface(cfg: dict):
    """3D implied volatility surface."""
    strikes    = np.linspace(70, 130, 20)
    maturities = np.linspace(0.1, 2.0, 10)
    iv = heston_vol_surface(
        cfg["S0"], cfg["V0"], cfg["kappa"], cfg["theta"],
        cfg["xi"], cfg["rho"], cfg["r"], cfg["q"],
        strikes, maturities,
    )

    K_grid, T_grid = np.meshgrid(strikes, maturities)
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_grid, T_grid, iv * 100, cmap="viridis", alpha=0.9)
    fig.colorbar(surf, ax=ax, label="Implied Vol (%)")
    ax.set_xlabel("Strike"); ax.set_ylabel("Maturity (yr)"); ax.set_zlabel("IV (%)")
    ax.set_title("Heston Implied Volatility Surface")
    plt.tight_layout()
    plt.savefig("heston_vol_surface.png", dpi=150)
    plt.show()
    print("Saved: heston_vol_surface.png")


def plot_vol_smile(cfg: dict):
    """Implied vol smile for different maturities — compares Heston vs BS."""
    strikes  = np.linspace(70, 130, 30)
    maturities = [0.25, 0.5, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(maturities)))

    for T, col in zip(maturities, colors):
        ivs = []
        for K in strikes:
            try:
                price = heston_price_analytic(
                    cfg["S0"], K, T, cfg["r"], cfg["q"],
                    cfg["V0"], cfg["kappa"], cfg["theta"], cfg["xi"], cfg["rho"],
                )
                iv = implied_vol(cfg["S0"], K, T, cfg["r"], cfg["q"], price)
                ivs.append(iv * 100)
            except Exception:
                ivs.append(np.nan)
        moneyness = strikes / cfg["S0"]
        ax.plot(moneyness, ivs, color=col, lw=2, label=f"T={T}y")

    ax.axhline(np.sqrt(cfg["theta"]) * 100, color="gray", linestyle="--",
               label=f"√θ = {np.sqrt(cfg['theta'])*100:.1f}% (long-run)")
    ax.set_xlabel("Moneyness (K/S₀)"); ax.set_ylabel("Implied Vol (%)")
    ax.set_title("Heston Model — Volatility Smile")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("heston_vol_smile.png", dpi=150)
    plt.show()
    print("Saved: heston_vol_smile.png")


def plot_sample_paths(cfg: dict, T: float = 1.0):
    """Plot sample S and V paths from Heston simulation."""
    S_paths, V_paths = simulate_heston(
        cfg["S0"], cfg["V0"], cfg["kappa"], cfg["theta"],
        cfg["xi"], cfg["rho"], cfg["r"], cfg["q"],
        T=T, n_paths=20, n_steps=252, seed=cfg["seed"],
    )
    t = np.linspace(0, T, S_paths.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Heston Model — Sample Paths", fontsize=12)
    for i in range(20):
        axes[0].plot(t, S_paths[i], lw=0.7, alpha=0.5)
        axes[1].plot(t, np.sqrt(V_paths[i]) * 100, lw=0.7, alpha=0.5)
    axes[0].set_ylabel("Stock Price S_t"); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Instantaneous Vol (%)"); axes[1].set_xlabel("Time (yr)")
    axes[1].axhline(np.sqrt(cfg["theta"]) * 100, color="red", linestyle="--",
                    label=f"Long-run vol = {np.sqrt(cfg['theta'])*100:.0f}%")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("heston_paths.png", dpi=150)
    plt.show()
    print("Saved: heston_paths.png")


def plot_analytic_vs_mc(cfg: dict):
    """Validate analytic formula against Monte Carlo prices."""
    strikes = np.linspace(80, 120, 15)
    T = 1.0
    analytic, mc_prices, mc_errors = [], [], []

    S_paths, _ = simulate_heston(
        cfg["S0"], cfg["V0"], cfg["kappa"], cfg["theta"],
        cfg["xi"], cfg["rho"], cfg["r"], cfg["q"],
        T=T, n_paths=cfg["n_paths"], n_steps=cfg["n_steps"], seed=cfg["seed"],
    )

    for K in strikes:
        an = heston_price_analytic(cfg["S0"], K, T, cfg["r"], cfg["q"],
                                    cfg["V0"], cfg["kappa"], cfg["theta"], cfg["xi"], cfg["rho"])
        mc, se = price_option_mc(S_paths, K, cfg["r"], cfg["q"], T)
        analytic.append(an); mc_prices.append(mc); mc_errors.append(2 * se)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Heston — Analytic vs Monte Carlo", fontsize=12)
    axes[0].plot(strikes, analytic, "b-", lw=2, label="Analytic")
    axes[0].errorbar(strikes, mc_prices, yerr=mc_errors, fmt="ro", ms=5,
                     label="MC (±2σ)")
    axes[0].set_xlabel("Strike K"); axes[0].set_ylabel("Call Price")
    axes[0].set_title("Option Prices"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    rel_err = np.abs(np.array(analytic) - np.array(mc_prices)) / np.array(analytic) * 100
    axes[1].bar(strikes, rel_err, width=1.5, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Strike K"); axes[1].set_ylabel("Relative Error (%)")
    axes[1].set_title("Analytic vs MC Relative Error")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("heston_analytic_vs_mc.png", dpi=150)
    plt.show()
    print("Saved: heston_analytic_vs_mc.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(CONFIG["seed"])
    print(f"Config: {CONFIG}")
    print(f"Feller condition: 2κθ={2*CONFIG['kappa']*CONFIG['theta']:.4f} "
          f"> ξ²={CONFIG['xi']**2:.4f}: {2*CONFIG['kappa']*CONFIG['theta'] > CONFIG['xi']**2}\n")

    # ── Analytic pricing ────────────────────────────────────────────
    K, T = 100.0, 1.0
    call = heston_price_analytic(CONFIG["S0"], K, T, CONFIG["r"], CONFIG["q"],
                                  CONFIG["V0"], CONFIG["kappa"], CONFIG["theta"],
                                  CONFIG["xi"], CONFIG["rho"])
    put  = heston_price_analytic(CONFIG["S0"], K, T, CONFIG["r"], CONFIG["q"],
                                  CONFIG["V0"], CONFIG["kappa"], CONFIG["theta"],
                                  CONFIG["xi"], CONFIG["rho"], option_type="put")
    print(f"Heston Call (K={K}, T={T}): {call:.4f}")
    print(f"Heston Put  (K={K}, T={T}): {put:.4f}")
    pcp_err = abs(call - put - CONFIG["S0"]*np.exp(-CONFIG["q"]*T) + K*np.exp(-CONFIG["r"]*T))
    print(f"Put-call parity error: {pcp_err:.2e}")

    # ── Visualizations ──────────────────────────────────────────────
    print("\n=== Sample Paths ===")
    plot_sample_paths(CONFIG)

    print("\n=== Volatility Smile ===")
    plot_vol_smile(CONFIG)

    print("\n=== Volatility Surface ===")
    plot_vol_surface(CONFIG)

    print("\n=== Analytic vs Monte Carlo Validation ===")
    plot_analytic_vs_mc(CONFIG)

    # ── Calibration demo ────────────────────────────────────────────
    print("\n=== Calibration Demo ===")
    np.random.seed(CONFIG["seed"])
    market = generate_synthetic_market(CONFIG)
    cal = calibrate_heston(
        market["prices"], market["strikes"], market["maturities"],
        CONFIG["S0"], CONFIG["r"], CONFIG["q"],
    )
    print(f"\nTrue:       V0={CONFIG['V0']} κ={CONFIG['kappa']} θ={CONFIG['theta']} "
          f"ξ={CONFIG['xi']} ρ={CONFIG['rho']}")
    print(f"Calibrated: V0={cal['V0']:.4f} κ={cal['kappa']:.4f} θ={cal['theta']:.4f} "
          f"ξ={cal['xi']:.4f} ρ={cal['rho']:.4f}")


if __name__ == "__main__":
    main()
