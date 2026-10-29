import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from conftest import load_module

shap_values = load_module("interpretability/shap_values.py")


def _linear_model(n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, n_features))
    w = np.array([2.0, -1.0, 0.5, 0.0])[:n_features]
    model = LinearRegression().fit(X, X @ w + 3.0)
    return model, X, model.coef_


def test_permutation_shap_is_exact_for_a_linear_model_with_one_baseline():
    # With a single background row b, every permutation gives the same
    # marginal contributions, and the exact Shapley value is w_i (x_i - b_i).
    model, X, w = _linear_model()
    background = X[:1]
    explainer = shap_values.PermutationSHAP(model, background, n_permutations=5)
    np.random.seed(0)
    phi = explainer.shap_values(X[10:15])
    np.testing.assert_allclose(phi, w * (X[10:15] - background), atol=1e-10)
    # Efficiency: contributions add up to f(x) - f(baseline)
    np.testing.assert_allclose(phi.sum(axis=1),
                               model.predict(X[10:15]) - model.predict(background)[0], atol=1e-10)


def test_permutation_shap_converges_to_interventional_shapley_values():
    # Exact interventional SHAP for f(x) = w·x + b: phi_i = w_i (x_i - E_bg[x_i])
    model, X, w = _linear_model()
    background = X[:40]
    explainer = shap_values.PermutationSHAP(model, background, n_permutations=2_000)
    np.random.seed(1)
    phi = explainer.shap_values(X[100:103])
    exact = w * (X[100:103] - background.mean(axis=0))
    np.testing.assert_allclose(phi, exact, atol=0.12)
    assert explainer.expected_value == pytest.approx(model.predict(background).mean())
