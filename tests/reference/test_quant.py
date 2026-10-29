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
