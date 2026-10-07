"""
Kalman Filter / Extended Kalman Filter / Unscented Kalman Filter
================================================================

The Kalman Filter (KF) is the optimal linear state estimator under Gaussian
noise assumptions. It maintains a belief over hidden state x_t as a Gaussian:
  p(x_t | y_{1:t}) = N(x_t; μ_t, Σ_t)

State-space model:
  Transition:   x_t = F x_{t-1} + B u_t + w_t,   w_t ~ N(0, Q)
  Observation:  y_t = H x_t + v_t,                v_t ~ N(0, R)

Kalman Filter equations (two-step):
  PREDICT:
    μ_t|t-1 = F μ_{t-1} + B u_t
    Σ_t|t-1 = F Σ_{t-1} F^T + Q

  UPDATE:
    S_t     = H Σ_t|t-1 H^T + R        (innovation covariance)
    K_t     = Σ_t|t-1 H^T S_t^{-1}     (Kalman gain)
    μ_t     = μ_t|t-1 + K_t (y_t - H μ_t|t-1)
    Σ_t     = (I - K_t H) Σ_t|t-1

Extended Kalman Filter (EKF):
  Linearize nonlinear f(x), h(x) via first-order Taylor expansion:
    F_t = ∂f/∂x |_{x=μ_{t-1}}
    H_t = ∂h/∂x |_{x=μ_t|t-1}
  Same predict/update equations but with Jacobians in place of F, H.

Unscented Kalman Filter (UKF):
  Uses sigma points to propagate the Gaussian through nonlinear functions
  without linearization (exact for 3rd-order Taylor terms for polynomial f).
  Sigma points:  X_0 = μ
                 X_i = μ ± sqrt((n+λ) Σ)_i   for i = 1..n
  Weights: W_m_0 = λ/(n+λ),  W_c_0 = λ/(n+λ) + (1-α²+β)
           W_m_i = W_c_i = 1/(2(n+λ))

Rauch-Tung-Striebel (RTS) Smoother:
  Backward pass that improves estimates using future observations:
    G_t   = Σ_t F^T (Σ_{t+1|t})^{-1}
    μ_t^s = μ_t + G_t (μ_{t+1}^s - μ_{t+1|t})
    Σ_t^s = Σ_t + G_t (Σ_{t+1}^s - Σ_{t+1|t}) G_t^T

Applications:
  - Object tracking (radar, GPS)
  - Robot localization (SLAM)
  - Financial state estimation (hidden volatility)
  - Sensor fusion (IMU + GPS)

Fine-tuning recommendations:
  - Q (#Placeholder): process noise covariance — larger = trust dynamics less, adapt faster
  - R (#Placeholder): observation noise covariance — larger = trust observations less
  - P0 (#Placeholder): initial state covariance — reflects initial uncertainty
  - alpha, beta, kappa (#Placeholder): UKF sigma-point spread parameters
    alpha controls spread (1e-3 to 1), beta=2 is optimal for Gaussians

References:
  Kalman (1960) — A New Approach to Linear Filtering and Prediction Problems
  Julier & Uhlmann (1997) — A New Extension of the Kalman Filter (UKF)
  Rauch, Tung & Striebel (1965) — Maximum Likelihood Estimates of Linear Systems
  Thrun, Burgard & Fox — Probabilistic Robotics (Ch. 3–4)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    # UKF sigma-point parameters
    "alpha":  1e-3,     # # Placeholder: spread of sigma points (1e-4 to 1)
    "beta":   2.0,      # # Placeholder: 2 is optimal for Gaussian distributions
    "kappa":  0.0,      # # Placeholder: secondary scaling (often 0 or 3-n)
    "seed":   42,
}


# ─────────────────────────────────────────────
# Linear Kalman Filter
# ─────────────────────────────────────────────

@dataclass
class KalmanFilter:
    """
    Linear Kalman Filter for the model:
      x_t = F x_{t-1} + B u_t + w_t,  w ~ N(0, Q)
      y_t = H x_t + v_t,               v ~ N(0, R)

    All matrices are numpy arrays.
    """
    F: np.ndarray           # (n, n) state transition matrix
    H: np.ndarray           # (m, n) observation matrix
    Q: np.ndarray           # (n, n) process noise covariance   # Placeholder: tune
    R: np.ndarray           # (m, m) observation noise covariance  # Placeholder: tune
    B: Optional[np.ndarray] = None   # (n, p) control input matrix
    x0: Optional[np.ndarray] = None  # (n,) initial state mean
    P0: Optional[np.ndarray] = None  # (n, n) initial state covariance  # Placeholder: tune

    def __post_init__(self):
        n = self.F.shape[0]
        self.x = self.x0 if self.x0 is not None else np.zeros(n)
        self.P = self.P0 if self.P0 is not None else np.eye(n)
        self._I = np.eye(n)

    def predict(self, u: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Predict step. Returns (x_pred, P_pred)."""
        self.x = self.F @ self.x
        if u is not None and self.B is not None:
            self.x += self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update step given observation y. Returns (x_updated, P_updated)."""
        innovation = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (self._I - K @ self.H) @ self.P
        # Joseph form for numerical stability
        IKH = self._I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        return self.x.copy(), self.P.copy()

    def filter(
        self,
        observations: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run full filter pass over T observations.
        Args:
            observations: (T, m) array — use np.nan for missing observations
            controls:     (T, p) or None
        Returns dict with means, covariances, predicted means/covs.
        """
        T = len(observations)
        n = self.F.shape[0]
        m = self.H.shape[0]

        means  = np.zeros((T, n))
        covs   = np.zeros((T, n, n))
        pred_means = np.zeros((T, n))
        pred_covs  = np.zeros((T, n, n))

        for t in range(T):
            u = controls[t] if controls is not None else None
            x_pred, P_pred = self.predict(u)
            pred_means[t] = x_pred
            pred_covs[t]  = P_pred

            y = observations[t]
            if not np.any(np.isnan(y)):
                self.update(y)

            means[t] = self.x
            covs[t]  = self.P

        return {"means": means, "covs": covs,
                "pred_means": pred_means, "pred_covs": pred_covs}

    def rts_smoother(self, filtered: dict) -> dict:
        """
        Rauch-Tung-Striebel backward smoother.
        Improves filter estimates using all future observations.
        """
        means = filtered["means"]
        covs  = filtered["covs"]
        pred_means = filtered["pred_means"]
        pred_covs  = filtered["pred_covs"]
        T = len(means)

        smooth_means = means.copy()
        smooth_covs  = covs.copy()

        for t in range(T - 2, -1, -1):
            G = covs[t] @ self.F.T @ np.linalg.inv(pred_covs[t + 1])
            smooth_means[t] = means[t] + G @ (smooth_means[t + 1] - pred_means[t + 1])
            smooth_covs[t]  = covs[t] + G @ (smooth_covs[t + 1] - pred_covs[t + 1]) @ G.T

        return {"means": smooth_means, "covs": smooth_covs}


# ─────────────────────────────────────────────
# Extended Kalman Filter
# ─────────────────────────────────────────────

@dataclass
class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems:
      x_t = f(x_{t-1}, u_t) + w_t
      y_t = h(x_t) + v_t

    Linearizes via Jacobians at the current estimate.
    Better than KF for mild nonlinearities; UKF is preferred for strong ones.
    """
    f: Callable                  # state transition function f(x, u) → x
    h: Callable                  # observation function h(x) → y
    jac_f: Callable              # Jacobian of f w.r.t. x: (x, u) → (n, n)
    jac_h: Callable              # Jacobian of h w.r.t. x: (x,) → (m, n)
    Q: np.ndarray                # process noise covariance   # Placeholder: tune
    R: np.ndarray                # observation noise covariance  # Placeholder: tune
    x0: np.ndarray = field(default_factory=lambda: np.zeros(1))
    P0: np.ndarray = field(default_factory=lambda: np.eye(1))

    def __post_init__(self):
        self.x = self.x0.copy()
        self.P = self.P0.copy()

    def predict(self, u: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        F = self.jac_f(self.x, u)
        self.x = self.f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = self.jac_h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.h(self.x))
        IKH = np.eye(len(self.x)) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        return self.x.copy(), self.P.copy()

    def filter(self, observations: np.ndarray, controls=None) -> dict:
        T = len(observations)
        n = len(self.x)
        means = np.zeros((T, n))
        covs  = np.zeros((T, n, n))
        for t in range(T):
            u = controls[t] if controls is not None else None
            self.predict(u)
            if not np.any(np.isnan(observations[t])):
                self.update(observations[t])
            means[t] = self.x
            covs[t]  = self.P
        return {"means": means, "covs": covs}


# ─────────────────────────────────────────────
# Unscented Kalman Filter
# ─────────────────────────────────────────────

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter — propagates Gaussian uncertainty through
    nonlinear functions using deterministically chosen sigma points.
    No Jacobians required; more accurate than EKF for strong nonlinearities.
    """

    def __init__(
        self,
        f: Callable,
        h: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        alpha: float = 1e-3,   # # Placeholder: sigma-point spread
        beta:  float = 2.0,    # # Placeholder: 2 optimal for Gaussian
        kappa: float = 0.0,    # # Placeholder: secondary scale
    ):
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = P0.copy()

        n = len(x0)
        self.n = n
        lam = alpha**2 * (n + kappa) - n
        self.lam = lam

        # Weights for mean and covariance
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    def _sigma_points(self) -> np.ndarray:
        """Compute 2n+1 sigma points around current mean."""
        n = self.n
        try:
            sqrt_P = np.linalg.cholesky((n + self.lam) * self.P)
        except np.linalg.LinAlgError:
            sqrt_P = np.linalg.cholesky((n + self.lam) * (self.P + 1e-6 * np.eye(n)))
        X = np.zeros((2 * n + 1, n))
        X[0] = self.x
        for i in range(n):
            X[i + 1]     = self.x + sqrt_P[i]
            X[n + i + 1] = self.x - sqrt_P[i]
        return X

    def predict(self, u=None) -> tuple[np.ndarray, np.ndarray]:
        X = self._sigma_points()
        X_pred = np.array([self.f(xi, u) for xi in X])

        x_pred = np.sum(self.Wm[:, None] * X_pred, axis=0)
        P_pred = self.Q.copy()
        for i, xi in enumerate(X_pred):
            d = (xi - x_pred)[:, None]
            P_pred += self.Wc[i] * (d @ d.T)

        self.x = x_pred
        self.P = P_pred
        self._X_pred = X_pred
        return self.x.copy(), self.P.copy()

    def update(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._X_pred
        Y = np.array([self.h(xi) for xi in X])

        y_pred = np.sum(self.Wm[:, None] * Y, axis=0)
        S = self.R.copy()
        Pxy = np.zeros((self.n, len(y)))
        for i in range(len(X)):
            dy = (Y[i] - y_pred)[:, None]
            dx = (X[i] - self.x)[:, None]
            S   += self.Wc[i] * (dy @ dy.T)
            Pxy += self.Wc[i] * (dx @ dy.T)

        K = Pxy @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_pred)
        self.P = self.P - K @ S @ K.T
        return self.x.copy(), self.P.copy()

    def filter(self, observations: np.ndarray, controls=None) -> dict:
        T = len(observations)
        n = self.n
        means = np.zeros((T, n))
        covs  = np.zeros((T, n, n))
        for t in range(T):
            u = controls[t] if controls is not None else None
            self.predict(u)
            if not np.any(np.isnan(observations[t])):
                self.update(observations[t])
            means[t] = self.x
            covs[t]  = self.P
        return {"means": means, "covs": covs}


# ─────────────────────────────────────────────
# Synthetic Data Generators
# ─────────────────────────────────────────────

def generate_tracking_data(
    T: int = 200,
    dt: float = 0.1,
    process_noise: float = 0.1,    # # Placeholder:
    obs_noise: float = 1.0,        # # Placeholder:
    seed: int = 42,
) -> dict:
    """
    Constant-velocity 2D tracking problem.
    State: [x, ẋ, y, ẏ]  (position + velocity in 2D)
    Observation: [x, y]   (noisy position only)
    Replace with your real sensor data.
    """
    rng = np.random.default_rng(seed)
    # State transition: constant velocity + process noise
    F = np.array([[1, dt, 0,  0],
                  [0,  1, 0,  0],
                  [0,  0, 1, dt],
                  [0,  0, 0,  1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    Q = process_noise ** 2 * np.eye(4)
    R = obs_noise ** 2 * np.eye(2)

    # True trajectory: circular motion
    t = np.arange(T) * dt
    x_true = np.stack([
        5 * np.cos(0.5 * t),
        -5 * 0.5 * np.sin(0.5 * t),
        5 * np.sin(0.5 * t),
         5 * 0.5 * np.cos(0.5 * t),
    ], axis=1)

    # Add process noise to true trajectory
    x_true += rng.normal(0, process_noise * 0.1, x_true.shape)

    # Observations with noise
    obs = x_true @ H.T + rng.normal(0, obs_noise, (T, 2))
    # Randomly drop 10% of observations (simulate missed detections)
    missing = rng.random(T) < 0.1
    obs[missing] = np.nan

    return {"F": F, "H": H, "Q": Q, "R": R,
            "x_true": x_true, "obs": obs, "t": t}


def generate_pendulum_data(
    T: int = 150,
    dt: float = 0.05,
    obs_noise: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Nonlinear pendulum for EKF/UKF demo.
    State: [θ, θ̇]
    Dynamics: θ̈ = -(g/L) sin(θ) - b θ̇
    Observation: [sin(θ)]  (nonlinear)
    """
    rng = np.random.default_rng(seed)
    g, L, b = 9.81, 1.0, 0.1
    x = np.array([1.5, 0.0])   # initial angle, angular velocity
    states = [x.copy()]

    # Euler integration for true trajectory
    for _ in range(T - 1):
        theta, theta_dot = x
        theta_ddot = -(g / L) * np.sin(theta) - b * theta_dot
        x = x + dt * np.array([theta_dot, theta_ddot])
        x += rng.normal(0, 0.005, 2)
        states.append(x.copy())

    x_true = np.array(states)
    obs = np.sin(x_true[:, 0:1]) + rng.normal(0, obs_noise, (T, 1))

    # Jacobians for EKF
    def f(x, u=None):
        theta, theta_dot = x
        theta_ddot = -(g / L) * np.sin(theta) - b * theta_dot
        return x + dt * np.array([theta_dot, theta_ddot])

    def h(x):
        return np.array([np.sin(x[0])])

    def jac_f(x, u=None):
        theta, theta_dot = x
        return np.array([[1, dt],
                         [-dt * (g / L) * np.cos(theta), 1 - dt * b]])

    def jac_h(x):
        return np.array([[np.cos(x[0]), 0]])

    Q = (0.005 ** 2) * np.eye(2)
    R = (obs_noise ** 2) * np.eye(1)
    x0 = np.array([1.4, 0.1])
    P0 = np.diag([0.1, 0.1])

    return {"f": f, "h": h, "jac_f": jac_f, "jac_h": jac_h,
            "Q": Q, "R": R, "x0": x0, "P0": P0,
            "x_true": x_true, "obs": obs, "dt": dt}


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def compare_filters(data: dict) -> dict:
    """Run KF, EKF (on pendulum), or KF + RTS on tracking data."""
    results = {}

    # ── Linear KF + RTS smoother on tracking data ──────────────────
    d = generate_tracking_data(seed=CONFIG["seed"])
    kf = KalmanFilter(
        F=d["F"], H=d["H"], Q=d["Q"], R=d["R"],
        x0=np.zeros(4), P0=np.eye(4) * 10,
    )
    filtered = kf.filter(d["obs"])

    # Reset and re-run for smoother (needs filter pass first)
    kf2 = KalmanFilter(F=d["F"], H=d["H"], Q=d["Q"], R=d["R"],
                        x0=np.zeros(4), P0=np.eye(4) * 10)
    filtered2 = kf2.filter(d["obs"])
    smoothed  = kf2.rts_smoother(filtered2)

    pos_filter = filtered["means"][:, [0, 2]]   # x, y position
    pos_smooth = smoothed["means"][:, [0, 2]]
    pos_true   = d["x_true"][:, [0, 2]]

    rmse_filter = rmse(pos_filter, pos_true)
    rmse_smooth = rmse(pos_smooth, pos_true)
    print(f"KF  Filter RMSE: {rmse_filter:.4f}")
    print(f"KF  Smoother RMSE (RTS): {rmse_smooth:.4f}")
    results["kf_filter"] = {"rmse": rmse_filter, "means": filtered["means"],
                             "covs": filtered["covs"], "data": d}
    results["kf_smooth"] = {"rmse": rmse_smooth, "means": smoothed["means"],
                             "covs": smoothed["covs"], "data": d}

    # ── EKF on nonlinear pendulum ────────────────────────────────────
    pd = generate_pendulum_data(seed=CONFIG["seed"])
    ekf = ExtendedKalmanFilter(
        f=pd["f"], h=pd["h"], jac_f=pd["jac_f"], jac_h=pd["jac_h"],
        Q=pd["Q"], R=pd["R"], x0=pd["x0"], P0=pd["P0"],
    )
    ekf_out = ekf.filter(pd["obs"])
    rmse_ekf = rmse(ekf_out["means"][:, 0], pd["x_true"][:, 0])
    print(f"EKF Pendulum angle RMSE: {rmse_ekf:.4f}")
    results["ekf"] = {"rmse": rmse_ekf, "means": ekf_out["means"], "data": pd}

    # ── UKF on nonlinear pendulum ────────────────────────────────────
    ukf = UnscentedKalmanFilter(
        f=pd["f"], h=pd["h"], Q=pd["Q"], R=pd["R"],
        x0=pd["x0"], P0=pd["P0"],
        alpha=CONFIG["alpha"], beta=CONFIG["beta"], kappa=CONFIG["kappa"],
    )
    ukf_out = ukf.filter(pd["obs"])
    rmse_ukf = rmse(ukf_out["means"][:, 0], pd["x_true"][:, 0])
    print(f"UKF Pendulum angle RMSE: {rmse_ukf:.4f}")
    results["ukf"] = {"rmse": rmse_ukf, "means": ukf_out["means"], "data": pd}

    return results


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_tracking(results: dict):
    """2D tracking plot: true path vs KF filter vs RTS smoother."""
    d = results["kf_filter"]["data"]
    kf_m  = results["kf_filter"]["means"]
    rts_m = results["kf_smooth"]["means"]
    obs   = d["obs"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Kalman Filter — 2D Tracking", fontsize=12)

    # 2D trajectory
    ax = axes[0]
    ax.plot(d["x_true"][:, 0], d["x_true"][:, 2], "k-", lw=1.5, label="True path")
    ax.scatter(obs[~np.isnan(obs[:, 0]), 0], obs[~np.isnan(obs[:, 0]), 1],
               s=8, alpha=0.4, c="gray", label="Observations")
    ax.plot(kf_m[:, 0], kf_m[:, 2], "b--", lw=1.2, label="KF filter")
    ax.plot(rts_m[:, 0], rts_m[:, 2], "r-", lw=1.2, label="RTS smoother")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("2D Position Estimate")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Position error over time
    t = d["t"]
    true_pos = d["x_true"][:, [0, 2]]
    err_kf  = np.sqrt(((kf_m[:, [0, 2]] - true_pos) ** 2).sum(1))
    err_rts = np.sqrt(((rts_m[:, [0, 2]] - true_pos) ** 2).sum(1))
    ax2 = axes[1]
    ax2.plot(t, err_kf,  "b-", label=f"KF  (RMSE={results['kf_filter']['rmse']:.3f})")
    ax2.plot(t, err_rts, "r-", label=f"RTS (RMSE={results['kf_smooth']['rmse']:.3f})")
    ax2.set_xlabel("Time"); ax2.set_ylabel("Position Error")
    ax2.set_title("Filter vs Smoother Error")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kalman_tracking.png", dpi=150)
    plt.show()
    print("Saved: kalman_tracking.png")


def plot_pendulum(results: dict):
    """Compare EKF vs UKF on nonlinear pendulum angle estimation."""
    pd = results["ekf"]["data"]
    t  = np.arange(len(pd["x_true"])) * pd["dt"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("EKF vs UKF — Nonlinear Pendulum", fontsize=12)

    for ax, key, label, color in zip(
        axes,
        ["ekf", "ukf"],
        ["Extended Kalman Filter", "Unscented Kalman Filter"],
        ["steelblue", "darkorange"],
    ):
        est = results[key]["means"][:, 0]
        ax.plot(t, pd["x_true"][:, 0], "k-", lw=1.5, label="True θ")
        ax.scatter(t, pd["obs"][:, 0], s=5, alpha=0.3, c="gray", label="Obs (sin θ)")
        ax.plot(t, est, color=color, lw=1.5, label=label)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("θ (rad)")
        ax.set_title(f"{label}  (RMSE={results[key]['rmse']:.4f})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kalman_pendulum.png", dpi=150)
    plt.show()
    print("Saved: kalman_pendulum.png")


def plot_uncertainty_bands(results: dict):
    """Show ±2σ uncertainty bands for KF filter vs RTS smoother."""
    d    = results["kf_filter"]["data"]
    kf_m = results["kf_filter"]["means"]
    kf_c = results["kf_filter"]["covs"]
    rt_m = results["kf_smooth"]["means"]
    rt_c = results["kf_smooth"]["covs"]
    t    = d["t"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Kalman Filter Uncertainty Bands (x-position)", fontsize=12)

    for ax, means, covs, label, color in zip(
        axes,
        [kf_m, rt_m],
        [kf_c, rt_c],
        ["KF Filter", "RTS Smoother"],
        ["steelblue", "darkorange"],
    ):
        std = np.sqrt(covs[:, 0, 0])
        ax.fill_between(t, means[:, 0] - 2*std, means[:, 0] + 2*std,
                        alpha=0.2, color=color, label="±2σ")
        ax.plot(t, means[:, 0], color=color, lw=1.5, label=label)
        ax.plot(t, d["x_true"][:, 0], "k--", lw=1, alpha=0.7, label="True x")
        ax.set_ylabel("x position")
        ax.set_title(label)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig("kalman_uncertainty.png", dpi=150)
    plt.show()
    print("Saved: kalman_uncertainty.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(CONFIG["seed"])
    print(f"Config: {CONFIG}\n")

    print("=== Kalman Filter Comparison ===")
    results = compare_filters({})

    plot_tracking(results)
    plot_pendulum(results)
    plot_uncertainty_bands(results)

    print("\nSummary:")
    print(f"  KF  Filter  RMSE (position): {results['kf_filter']['rmse']:.4f}")
    print(f"  RTS Smoother RMSE (position): {results['kf_smooth']['rmse']:.4f}")
    print(f"  EKF Pendulum angle RMSE:     {results['ekf']['rmse']:.4f}")
    print(f"  UKF Pendulum angle RMSE:     {results['ukf']['rmse']:.4f}")


if __name__ == "__main__":
    main()
