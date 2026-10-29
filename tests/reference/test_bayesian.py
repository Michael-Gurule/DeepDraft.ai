import numpy as np
import pytest

from conftest import load_module

kalman = load_module("bayesian/kalman_filter.py")
mcmc = load_module("bayesian/mcmc.py")


# ── Kalman filter and RTS smoother vs. the exact Gaussian posterior ──

def _exact_posteriors(F, H, Q, R, m0, P0, ys):
    """
    Condition the joint Gaussian of (x_1..x_T, y_1..y_T) directly.

    Every state and observation is a linear function of the independent
    Gaussians z = (x_0, w_1..w_T, v_1..v_T), so the joint covariance is
    A Σ_z Aᵀ. Conditioning on a prefix of the observations gives the
    filtering distribution; conditioning on all of them gives the smoother.
    """
    T, m = ys.shape
    n = F.shape[0]
    dz = n + T * n + T * m
    A_x = np.zeros((T * n, dz))
    A_y = np.zeros((T * m, dz))
    for t in range(1, T + 1):
        row = slice((t - 1) * n, t * n)
        A_x[row, :n] = np.linalg.matrix_power(F, t)
        for k in range(1, t + 1):
            A_x[row, n + (k - 1) * n: n + k * n] = np.linalg.matrix_power(F, t - k)
        yrow = slice((t - 1) * m, t * m)
        A_y[yrow] = H @ A_x[row]
        A_y[yrow, n + T * n + (t - 1) * m: n + T * n + t * m] = np.eye(m)

    cov_z = np.zeros((dz, dz))
    cov_z[:n, :n] = P0
    for k in range(T):
        cov_z[n + k * n: n + (k + 1) * n, n + k * n: n + (k + 1) * n] = Q
        cov_z[n + T * n + k * m: n + T * n + (k + 1) * m,
              n + T * n + k * m: n + T * n + (k + 1) * m] = R
    mean_z = np.zeros(dz)
    mean_z[:n] = m0

    mu_x, mu_y = A_x @ mean_z, A_y @ mean_z
    S_xx, S_xy, S_yy = A_x @ cov_z @ A_x.T, A_x @ cov_z @ A_y.T, A_y @ cov_z @ A_y.T
    y = ys.ravel()

    def condition(t_state, n_obs):
        rx = slice((t_state - 1) * n, t_state * n)
        ry = slice(0, n_obs * m)
        gain = S_xy[rx, ry] @ np.linalg.inv(S_yy[ry, ry])
        mean = mu_x[rx] + gain @ (y[ry] - mu_y[ry])
        cov = S_xx[rx, rx] - gain @ S_xy[rx, ry].T
        return mean, cov

    filtered = [condition(t, t) for t in range(1, T + 1)]
    smoothed = [condition(t, T) for t in range(1, T + 1)]
    return filtered, smoothed


def test_kalman_filter_and_smoother_match_exact_posterior():
    rng = np.random.default_rng(0)
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1.0, 0.0]])
    Q = 0.05 * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    R = np.array([[0.5]])
    m0, P0 = np.array([0.0, 1.0]), np.diag([1.0, 0.5])
    ys = rng.normal(size=(12, 1)) + np.arange(12)[:, None]

    kf = kalman.KalmanFilter(F=F, H=H, Q=Q, R=R, x0=m0.copy(), P0=P0.copy())
    filtered = kf.filter(ys)
    smoothed = kf.rts_smoother(filtered)
    exact_f, exact_s = _exact_posteriors(F, H, Q, R, m0, P0, ys)

    for t in range(len(ys)):
        np.testing.assert_allclose(filtered["means"][t], exact_f[t][0], atol=1e-8)
        np.testing.assert_allclose(filtered["covs"][t], exact_f[t][1], atol=1e-8)
        np.testing.assert_allclose(smoothed["means"][t], exact_s[t][0], atol=1e-8)
        np.testing.assert_allclose(smoothed["covs"][t], exact_s[t][1], atol=1e-8)


# ── MCMC samplers recover a known Gaussian ──

MEAN = np.array([1.0, -2.0])
COV = np.array([[1.0, 0.6], [0.6, 2.0]])
PREC = np.linalg.inv(COV)


def _log_prob(theta):
    d = theta - MEAN
    return -0.5 * d @ PREC @ d


def _check_moments(samples, mean_tol, cov_tol):
    np.testing.assert_allclose(samples.mean(axis=0), MEAN, atol=mean_tol)
    np.testing.assert_allclose(np.cov(samples.T), COV, atol=cov_tol)


def test_metropolis_hastings_recovers_a_gaussian():
    out = mcmc.metropolis_hastings(_log_prob, np.zeros(2), n_samples=40_000,
                                   n_warmup=2_000, proposal_std=1.2, seed=0)
    _check_moments(out["samples"], mean_tol=0.15, cov_tol=0.25)


def test_hmc_recovers_a_gaussian():
    out = mcmc.hamiltonian_monte_carlo(_log_prob, np.zeros(2), n_samples=3_000,
                                       n_warmup=500, step_size=0.3, n_leapfrog=10,
                                       grad_fn=lambda th: -PREC @ (th - MEAN), seed=0)
    _check_moments(out["samples"], mean_tol=0.12, cov_tol=0.25)
