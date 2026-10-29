import numpy as np
import pytest
from scipy import stats

from conftest import load_module

monte_carlo = load_module("quantitative/monte_carlo.py")


# ── GARCH ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def garch():
    pytest.importorskip("arch")
    return load_module("quantitative/garch.py")


def test_garch_recovers_simulation_parameters(garch):
    returns = garch.generate_return_series(n_obs=8000, seed=3)
    result = garch.build_garch(returns).fit(disp="off")
    p = result.params
    # Simulated with omega=0.05, alpha=0.10, beta=0.85, Student-t nu=6
    assert p["alpha[1]"] == pytest.approx(0.10, abs=0.03)
    assert p["beta[1]"] == pytest.approx(0.85, abs=0.04)
    assert p["omega"] / (1 - p["alpha[1]"] - p["beta[1]"]) == pytest.approx(1.0, rel=0.25)
    assert p["nu"] == pytest.approx(6.0, abs=1.5)


def test_simulated_shocks_have_unit_variance(garch):
    returns = garch.generate_return_series(n_obs=20000, seed=0)
    # Unconditional variance omega / (1 - alpha - beta) = 0.05 / 0.05 = 1
    assert returns.var() == pytest.approx(1.0, rel=0.15)


def test_var_quantile_uses_the_standardized_t(garch):
    result = garch.build_garch(garch.generate_return_series(n_obs=2000)).fit(disp="off")
    nu = result.params["nu"]
    expected = stats.t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu)
    assert garch.var_quantile(result, confidence=0.99) == pytest.approx(expected, rel=1e-8)


def test_kupiec_test_exact_cases(garch):
    on_target = garch.kupiec_test(np.r_[np.ones(10), np.zeros(990)].astype(bool), 0.99)
    assert on_target["LR"] == pytest.approx(0.0, abs=1e-12) and on_target["p_value"] == pytest.approx(1.0)

    none = garch.kupiec_test(np.zeros(500, dtype=bool), 0.99)
    assert none["LR"] == pytest.approx(-2 * 500 * np.log(0.99))

    x, T, a = 25, 1000, 0.01
    many = garch.kupiec_test(np.r_[np.ones(x), np.zeros(T - x)].astype(bool), 1 - a)
    lr = -2 * (x * np.log(a) + (T - x) * np.log(1 - a)
               - x * np.log(x / T) - (T - x) * np.log(1 - x / T))
    assert many["LR"] == pytest.approx(lr) and many["p_value"] < 1e-4


def test_out_of_sample_var_has_correct_coverage(garch, no_figures):
    returns = garch.generate_return_series(n_obs=6000, seed=11)
    _, kupiec = garch.backtest_var(returns, confidence=0.99, test_size=0.5)
    assert kupiec["p_value"] > 0.01


# ── Monte Carlo ──────────────────────────────────────────────────

def _black_scholes_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def _call_price(antithetic, seed, n_sims=2000):
    # One step of length T gives the exact terminal distribution.
    paths = monte_carlo.simulate_gbm(100.0, 0.04, 0.2, n_days=1, n_sims=n_sims,
                                     dt=1.0, antithetic=antithetic, seed=seed)
    payoff = np.exp(-0.04) * np.maximum(paths[:, -1] - 100.0, 0)
    return monte_carlo.mc_mean_and_se(payoff, antithetic)


@pytest.mark.parametrize("antithetic", [False, True])
def test_monte_carlo_standard_error_matches_replication_spread(antithetic):
    estimates, ses = zip(*(_call_price(antithetic, seed) for seed in range(300)))
    ratio = np.std(estimates, ddof=1) / np.mean(ses)
    assert ratio == pytest.approx(1.0, abs=0.15)
    bs = _black_scholes_call(100.0, 100.0, 0.04, 0.2, 1.0)
    assert np.mean(estimates) == pytest.approx(bs, abs=4 * np.mean(ses) / np.sqrt(300))


def test_antithetic_handles_odd_path_counts():
    paths = monte_carlo.simulate_gbm(100.0, 0.05, 0.2, n_days=5, n_sims=7, antithetic=True)
    assert paths.shape == (8, 6)
    np.testing.assert_allclose(np.log(paths[:4, 1:] / 100) + np.log(paths[4:, 1:] / 100),
                               2 * np.cumsum(np.full((4, 5), (0.05 - 0.02) / 252), axis=1))


# ── Black-Scholes ────────────────────────────────────────────────

black_scholes = load_module("quantitative/black_scholes.py")


def test_black_scholes_textbook_values():
    # Hull, Options, Futures, and Other Derivatives: S=K=100, T=1, r=5%, sigma=20%
    assert black_scholes.bs_price(100, 100, 1.0, 0.05, 0.2, "call") == pytest.approx(10.4506, abs=1e-4)
    assert black_scholes.bs_price(100, 100, 1.0, 0.05, 0.2, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("K, T, r, sigma, q", [(80, 0.5, 0.03, 0.25, 0.0), (120, 2.0, 0.01, 0.4, 0.02)])
def test_black_scholes_put_call_parity(K, T, r, sigma, q):
    S = 100.0
    call = black_scholes.bs_price(S, K, T, r, sigma, "call", q=q)
    put = black_scholes.bs_price(S, K, T, r, sigma, "put", q=q)
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_black_scholes_greeks_match_finite_differences():
    S, K, T, r, sigma, h = 100.0, 105.0, 0.75, 0.02, 0.3, 1e-3
    price = lambda **kw: black_scholes.bs_price(**{"S": S, "K": K, "T": T, "r": r,
                                                   "sigma": sigma, **kw})
    g = black_scholes.bs_greeks(S, K, T, r, sigma, "call")
    assert g["delta"] == pytest.approx((price(S=S + h) - price(S=S - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((price(S=S + h) - 2 * price() + price(S=S - h)) / h ** 2, rel=1e-3)
    assert g["vega"] == pytest.approx(0.01 * (price(sigma=sigma + h) - price(sigma=sigma - h)) / (2 * h), rel=1e-5)


def test_implied_volatility_inverts_the_price():
    price = black_scholes.bs_price(100, 90, 0.5, 0.03, 0.37, "put")
    assert black_scholes.implied_volatility(price, 100, 90, 0.5, 0.03, "put") == pytest.approx(0.37, abs=1e-5)


# ── Heston ───────────────────────────────────────────────────────

heston = load_module("quantitative/heston.py")
HESTON = dict(S0=100.0, T=1.0, r=0.03, q=0.0, V0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


@pytest.mark.parametrize("K, reference", [(80, 23.8732), (100, 8.9294), (120, 1.2935)])
def test_heston_matches_an_independent_pricer(K, reference):
    # Reference: an independent implementation of Gatheral's "little trap"
    # characteristic function, confirmed by 200,000-path Monte Carlo.
    p = HESTON
    price = heston.heston_price_analytic(p["S0"], K, p["T"], p["r"], p["q"], p["V0"],
                                         p["kappa"], p["theta"], p["xi"], p["rho"])
    assert price == pytest.approx(reference, abs=2e-3)


def test_heston_reduces_to_black_scholes_without_vol_of_vol():
    # With xi -> 0 and V0 = theta, variance stays at theta: Black-Scholes with sigma = sqrt(theta)
    p = {**HESTON, "xi": 1e-4, "rho": 0.0}
    for K in (85, 100, 115):
        h_price = heston.heston_price_analytic(p["S0"], K, p["T"], p["r"], p["q"], p["V0"],
                                               p["kappa"], p["theta"], p["xi"], p["rho"])
        bs = black_scholes.bs_price(p["S0"], K, p["T"], p["r"], np.sqrt(p["theta"]), "call")
        assert h_price == pytest.approx(bs, abs=1e-3)


def test_heston_put_call_parity():
    p = HESTON
    args = (p["S0"], 95, p["T"], p["r"], p["q"], p["V0"], p["kappa"], p["theta"], p["xi"], p["rho"])
    call = heston.heston_price_analytic(*args, option_type="call")
    put = heston.heston_price_analytic(*args, option_type="put")
    assert call - put == pytest.approx(p["S0"] - 95 * np.exp(-p["r"] * p["T"]), abs=1e-8)
