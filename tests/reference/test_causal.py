import numpy as np
import pytest

from conftest import load_module

pytest.importorskip("networkx")

discovery = load_module("causal_inference/causal_discovery.py")
dml = load_module("causal_inference/dml_estimation.py")
effects = load_module("causal_inference/treatment_effects.py")


# ── Causal discovery ─────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_direct_lingam_recovers_the_true_dag(seed):
    X, truth = discovery.generate_linear_dag_data(n_samples=2000, seed=seed)
    model = discovery.DirectLiNGAM(threshold=0.15).fit(X)
    np.testing.assert_array_equal(model.adjacency_matrix_, truth["adjacency"])


def test_direct_lingam_order_follows_variables_not_columns():
    X, truth = discovery.generate_linear_dag_data(n_samples=2000, seed=1)
    perm = np.array([4, 2, 0, 3, 1])
    model = discovery.DirectLiNGAM(threshold=0.15).fit(X[:, perm])
    np.testing.assert_array_equal(model.adjacency_matrix_, truth["adjacency"][np.ix_(perm, perm)])


def test_direct_lingam_coefficients_use_j_to_i_convention():
    rng = np.random.default_rng(0)
    x0 = rng.laplace(size=5000)
    x1 = 0.8 * x0 + rng.uniform(-1, 1, 5000)
    model = discovery.DirectLiNGAM(threshold=0.1).fit(np.column_stack([x1, x0]))
    assert model.causal_order_ == [1, 0]
    assert model.B_[0, 1] == pytest.approx(0.8, abs=0.05)     # x0 (column 1) → x1 (column 0)
    assert model.adjacency_matrix_[1, 0] == 1 and model.adjacency_matrix_[0, 1] == 0


def _adj(n, edges):
    a = np.zeros((n, n), dtype=int)
    for i, j in edges:
        a[i, j] = 1
    return a


def test_graph_metrics_on_exact_cases():
    truth = _adj(3, [(0, 1), (1, 2)])

    perfect = discovery.evaluate_graph(truth, truth)
    assert perfect["SHD"] == 0 and perfect["F1"] == 1.0 and perfect["orientation_accuracy"] == 1.0
    assert (perfect["TP"], perfect["FP"], perfect["FN"]) == (2, 0, 0)    # each pair counted once

    reversed_edge = discovery.evaluate_graph(_adj(3, [(1, 0), (1, 2)]), truth)
    assert reversed_edge["SHD"] == 1 and reversed_edge["F1"] == 1.0
    assert reversed_edge["orientation_accuracy"] == 0.5

    undirected = discovery.evaluate_graph(_adj(3, [(0, 1), (1, 0), (1, 2)]), truth)
    assert undirected["SHD"] == 1 and undirected["orientation_accuracy"] == 0.5

    missing_and_extra = discovery.evaluate_graph(_adj(3, [(0, 1), (0, 2)]), truth)
    assert missing_and_extra["SHD"] == 2
    assert (missing_and_extra["TP"], missing_and_extra["FP"], missing_and_extra["FN"]) == (1, 1, 1)


# ── Double machine learning ──────────────────────────────────────

def test_dml_standard_error_matches_hc0_sandwich():
    sm = pytest.importorskip("statsmodels.api")
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 2000
    X = rng.standard_normal((n, 3))
    D = X[:, 0] + rng.standard_normal(n)
    # Heteroskedastic noise: the case the homoskedastic formula gets wrong
    Y = 2.0 * D + X[:, 1] + rng.standard_normal(n) * (0.5 + np.abs(D))
    est = dml.DoubleMLEstimator(n_splits=3, outcome_model=LinearRegression(),
                                treatment_model=LinearRegression()).fit(X, D, Y)

    ols = sm.OLS(est.Y_residuals_, est.D_residuals_).fit(cov_type="HC0")
    assert est.theta_ == pytest.approx(ols.params[0], rel=1e-10)
    assert est.se_ == pytest.approx(ols.bse[0], rel=1e-10)
    assert est.ci_[0] < 2.0 < est.ci_[1]


def test_aipw_recovers_a_heterogeneous_ate():
    from sklearn.linear_model import LinearRegression, LogisticRegression

    rng = np.random.default_rng(1)
    n = 20000
    X = rng.standard_normal((n, 3))
    e = 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))
    D = rng.binomial(1, e).astype(float)
    tau = 1.0 + X[:, 0]                       # effect varies with a confounder
    Y = D * tau + X[:, 0] + 0.5 * X[:, 2] + rng.standard_normal(n)

    est = dml.AIPWEstimator(n_splits=3, outcome_model=LinearRegression(),
                            propensity_model=LogisticRegression()).fit(X, D, Y)
    assert est.ci_[0] < 1.0 < est.ci_[1]      # true ATE = E[1 + X0] = 1
    assert est.theta_ == pytest.approx(1.0, abs=0.05)


# ── Uplift evaluation ────────────────────────────────────────────

def test_qini_with_propensities_recovers_ate_under_confounding():
    X, D, Y, tau = effects.generate_cate_data(n_samples=200_000, seed=1)
    e = 1 / (1 + np.exp(-(X[:, 0] - 0.5 * X[:, 1])))     # the simulation's propensity

    _, weighted, _ = effects.compute_qini_curve(tau, D, Y, propensity=e)
    _, unweighted, _ = effects.compute_qini_curve(tau, D, Y)

    assert weighted[-1] == pytest.approx(tau.mean(), abs=0.05)
    assert abs(unweighted[-1] - tau.mean()) > 0.5        # a constant share is badly biased here

    half = len(tau) // 2
    best_half = np.sort(tau)[::-1][:half].sum() / len(tau)
    assert weighted[half - 1] == pytest.approx(best_half, abs=0.05)


def test_qini_auuc_is_zero_for_a_random_ranking_in_expectation():
    rng = np.random.default_rng(0)
    n = 100_000
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = D * 1.0 + rng.standard_normal(n)
    _, _, auuc = effects.compute_qini_curve(rng.standard_normal(n), D, Y)
    assert abs(auuc) < 0.02
