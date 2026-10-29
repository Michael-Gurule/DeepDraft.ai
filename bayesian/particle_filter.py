"""
Particle Filter (Sequential Monte Carlo)
==========================================
A sequential Bayesian estimation algorithm that represents the posterior
distribution over hidden states as a set of weighted particles (samples).
Unlike the Kalman Filter (which assumes linear-Gaussian models), the Particle
Filter handles nonlinear, non-Gaussian state-space models.

State-Space Model:
    State transition:   x_t ~ p(x_t | x_{t-1})          (process model)
    Observation model:  z_t ~ p(z_t | x_t)               (likelihood)
    Prior:              x_0 ~ p(x_0)

Algorithm (Bootstrap Particle Filter / SIR):
    1. Predict:    propagate each particle through the process model
                   x_t^(i) ~ p(x_t | x_{t-1}^(i))
    2. Update:     weight each particle by the likelihood
                   w_t^(i) ∝ p(z_t | x_t^(i))
    3. Normalize:  w_t^(i) = w_t^(i) / Σ w_t^(j)
    4. Estimate:   E[x_t] ≈ Σ w_t^(i) · x_t^(i)
    5. Resample:   if N_eff = 1/Σ(w^(i))² < threshold → resample to prevent degeneracy

Resampling Strategies:
    - Multinomial (naive): O(n log n), introduces extra variance
    - Systematic (default): O(n), lower variance, preferred
    - Residual: hybrid that keeps high-weight particles deterministically

Use Cases:
    - Multi-target tracking (radar, drone, and sensor localization)
    - Robot localization (Monte Carlo Localization)
    - Financial state estimation (stochastic volatility models)
    - GPS/IMU fusion under non-linear dynamics
    - Fault detection (latent system state estimation)

Advantages:
    - Handles any nonlinear, non-Gaussian system
    - Can represent multi-modal posteriors (unlike EKF/UKF)
    - Conceptually simple and parallelizable

Limitations:
    - Computational: O(N) per time step; degrades in high dimensions
    - Particle degeneracy: requires resampling
    - Curse of dimensionality: N grows exponentially with state dimension

Fine-Tuning Recommendations:
    - n_particles: 100–10000. More particles → better approximation.
      For 1D/2D: 500–1000 sufficient. Higher dim: 5000–50000+.
    - resample_threshold: typically 0.5 × N (resample when N_eff < 50% of N).
    - Process noise (sigma_process): tunes prediction spread. Too small → filter
      loses track. Too large → high variance estimates.
    - Measurement noise (sigma_obs): must match actual sensor noise level.
      Use RMSE of residuals to calibrate.
    - For high-dim: use Auxiliary Particle Filter or SMC² to reduce degeneracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Optional

from deepdraft.core import save_figure


# ─────────────────────────────────────────────
# STATE SPACE MODEL (replace with your dynamics)
# ─────────────────────────────────────────────

@dataclass
class StateSpaceModel:
    """
    Defines the generative model used by the particle filter.

    Replace transition_fn and observation_fn with your actual system dynamics.
    """
    state_dim:   int = 2    # [position, velocity]
    obs_dim:     int = 1    # observed position only

    # Process noise: how much does the state "wander" each step?
    sigma_process: float = 0.1   # Placeholder: calibrate from system data

    # Observation noise: sensor measurement noise
    sigma_obs: float = 0.5       # Placeholder: set to known sensor noise std

    def transition(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        State transition: x_{t+1} = f(x_t) + noise.

        Example: constant-velocity model
            position  += velocity * dt + process_noise
            velocity  += process_noise (random walk)

        Replace with your system dynamics (e.g., a vehicle or robot motion model).
        """
        x_next = x.copy()
        x_next[:, 0] += x_next[:, 1] * dt  # pos += vel
        noise = np.random.normal(0, self.sigma_process, x.shape)
        return x_next + noise

    def observation_likelihood(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Likelihood: p(z | x) — Gaussian observation model.

        Args:
            x: particles (N, state_dim)
            z: current measurement (obs_dim,)

        Returns:
            weights (N,) — likelihood of each particle given observation z
        """
        # Observe only position (first state dimension)
        predicted_z = x[:, 0]
        diff = z[0] - predicted_z
        return np.exp(-0.5 * (diff / self.sigma_obs) ** 2) / (
            self.sigma_obs * np.sqrt(2 * np.pi)
        )

    def sample_prior(self, n: int) -> np.ndarray:
        """
        Sample N particles from the prior p(x_0).
        Replace with your domain-specific prior.
        """
        x = np.zeros((n, self.state_dim))
        x[:, 0] = np.random.normal(0, 1.0, n)   # initial position
        x[:, 1] = np.random.normal(0, 0.5, n)   # initial velocity
        return x


# ─────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────

def generate_tracking_data(
    n_steps: int = 100,
    model: StateSpaceModel = None,
    seed: int = 42,
) -> tuple:
    """
    Generate ground truth trajectory + noisy observations.

    True state: constant-velocity motion with random velocity perturbations.
    Observations: noisy position measurements.

    Replace with real sensor data:
        observations = sensor_df["position"].values[:, None]

    Returns:
        true_states:  (n_steps, state_dim)
        observations: (n_steps, obs_dim)
    """
    if model is None:
        model = StateSpaceModel()
    rng = np.random.RandomState(seed)

    true_states  = np.zeros((n_steps, model.state_dim))
    observations = np.zeros((n_steps, model.obs_dim))

    # Initial state
    true_states[0] = [0.0, 0.5]  # position=0, velocity=0.5

    for t in range(1, n_steps):
        true_states[t, 0] = true_states[t-1, 0] + true_states[t-1, 1]
        true_states[t, 1] = true_states[t-1, 1] + rng.normal(0, model.sigma_process * 0.5)

    observations[:, 0] = true_states[:, 0] + rng.normal(0, model.sigma_obs, n_steps)
    return true_states, observations


# ─────────────────────────────────────────────
# PARTICLE FILTER
# ─────────────────────────────────────────────

# Placeholder: tune n_particles and resample_threshold
PF_CONFIG = {
    "n_particles":        500,   # Placeholder: 100–10000
    "resample_threshold": 0.5,   # fraction of N below which to resample
    "resampling":         "systematic",  # 'systematic', 'multinomial'
}


class ParticleFilter:
    """
    Bootstrap Particle Filter (SIR — Sequential Importance Resampling).

    Maintains N weighted particles approximating the posterior p(x_t | z_{1:t}).
    Handles nonlinear, non-Gaussian state-space models.
    """

    def __init__(self, model: StateSpaceModel, config: dict = None):
        self.model  = model
        self.cfg    = config or PF_CONFIG
        self.N      = self.cfg["n_particles"]
        self.thresh = self.cfg["resample_threshold"] * self.N

        # Initialize particles from prior
        self.particles = model.sample_prior(self.N)   # (N, state_dim)
        self.weights   = np.ones(self.N) / self.N      # uniform initially

        # History for diagnostics
        self.estimates   = []   # E[x_t]
        self.stds        = []   # std of particles
        self.n_eff_hist  = []

    @property
    def n_eff(self) -> float:
        """Effective sample size: N_eff = 1 / Σ w_i²."""
        return 1.0 / (np.sum(self.weights ** 2) + 1e-10)

    def _resample_systematic(self) -> np.ndarray:
        """
        Systematic resampling — O(N), lower variance than multinomial.
        Returns indices of selected particles.
        """
        positions = (np.arange(self.N) + np.random.uniform(0, 1)) / self.N
        cumsum    = np.cumsum(self.weights)
        indices   = np.searchsorted(cumsum, positions)
        return np.clip(indices, 0, self.N - 1)

    def _resample_multinomial(self) -> np.ndarray:
        """Multinomial resampling — simple but higher variance."""
        return np.random.choice(self.N, size=self.N, replace=True, p=self.weights)

    def step(self, observation: np.ndarray) -> np.ndarray:
        """
        One filter step: predict → update → (conditionally) resample.

        Args:
            observation: current measurement z_t (obs_dim,)

        Returns:
            estimate: posterior mean E[x_t] (state_dim,)
        """
        # 1. Predict: propagate particles through process model
        self.particles = self.model.transition(self.particles)

        # 2. Update: weight by observation likelihood
        likelihoods  = self.model.observation_likelihood(self.particles, observation)
        self.weights *= likelihoods
        total = self.weights.sum()
        if total < 1e-300:
            # Degeneracy — reinitialize weights uniformly
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights /= total

        # 3. Estimate: posterior mean and std
        estimate = np.average(self.particles, weights=self.weights, axis=0)
        std      = np.sqrt(np.average((self.particles - estimate)**2,
                                       weights=self.weights, axis=0))

        # 4. Resample if needed
        neff = self.n_eff
        if neff < self.thresh:
            if self.cfg["resampling"] == "systematic":
                idx = self._resample_systematic()
            else:
                idx = self._resample_multinomial()
            self.particles = self.particles[idx]
            self.weights   = np.ones(self.N) / self.N

        # Log history
        self.estimates.append(estimate)
        self.stds.append(std)
        self.n_eff_hist.append(neff)

        return estimate

    def run(self, observations: np.ndarray) -> np.ndarray:
        """
        Run particle filter over a full observation sequence.

        Args:
            observations: (T, obs_dim)

        Returns:
            estimates: (T, state_dim) — posterior means over time
        """
        for t, z in enumerate(observations):
            self.step(z)
            if (t + 1) % 20 == 0:
                print(f"  t={t+1:4d}  estimate={self.estimates[-1].round(3)}  "
                      f"N_eff={self.n_eff_hist[-1]:.0f}")
        return np.array(self.estimates)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(estimates: np.ndarray, true_states: np.ndarray) -> dict:
    """
    Evaluate filter accuracy: RMSE per state dimension.
    """
    rmse_per_dim = np.sqrt(np.mean((estimates - true_states)**2, axis=0))
    overall_rmse = np.sqrt(np.mean((estimates - true_states)**2))

    for i, r in enumerate(rmse_per_dim):
        print(f"  RMSE dim {i}: {r:.4f}")
    print(f"  Overall RMSE: {overall_rmse:.4f}")

    return {"rmse": overall_rmse, "rmse_per_dim": rmse_per_dim.tolist()}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_tracking(estimates: np.ndarray, true_states: np.ndarray,
                  observations: np.ndarray, pf: ParticleFilter) -> None:
    """
    Plot tracking results: true trajectory, particle filter estimates, observations.
    """
    T    = len(true_states)
    t    = np.arange(T)
    stds = np.array(pf.stds)

    fig, axes = plt.subplots(true_states.shape[1], 1,
                              figsize=(12, 3 * true_states.shape[1]), sharex=True)
    if true_states.shape[1] == 1:
        axes = [axes]

    labels = ["Position", "Velocity"]
    for i, ax in enumerate(axes):
        ax.plot(t, true_states[:, i], "k-",  linewidth=1.5, label="True", alpha=0.8)
        ax.plot(t, estimates[:, i],   "steelblue", linewidth=1.5, label="PF Estimate")
        ax.fill_between(t,
                        estimates[:, i] - 2 * stds[:, i],
                        estimates[:, i] + 2 * stds[:, i],
                        alpha=0.25, color="steelblue", label="±2σ")
        if i == 0 and observations is not None:
            ax.scatter(t, observations[:, 0], s=8, c="tomato", alpha=0.5,
                       label="Observations", zorder=3)
        ax.set_ylabel(labels[i] if i < len(labels) else f"Dim {i}")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Particle Filter — State Tracking", y=1.01)
    plt.tight_layout()
    save_figure("pf_tracking.png", dpi=150)


def plot_effective_sample_size(pf: ParticleFilter) -> None:
    """
    Plot effective sample size N_eff over time.
    Drops below threshold → triggered resampling.
    """
    t    = np.arange(len(pf.n_eff_hist))
    neff = np.array(pf.n_eff_hist)

    plt.figure(figsize=(10, 3))
    plt.plot(t, neff, color="steelblue", linewidth=1.0)
    plt.axhline(pf.thresh, color="red", linestyle="--",
                label=f"Resample threshold ({pf.thresh:.0f})")
    plt.xlabel("Time Step")
    plt.ylabel("Effective Sample Size N_eff")
    plt.title("Particle Filter — Effective Sample Size")
    plt.legend()
    plt.tight_layout()
    save_figure("pf_neff.png", dpi=150)


def plot_particle_cloud(pf: ParticleFilter, true_states: np.ndarray,
                         step: int = -1) -> None:
    """
    Scatter plot of particles at a given time step.
    Shows the full posterior distribution approximation.
    """
    if pf.particles.shape[1] < 2:
        print("  plot_particle_cloud: requires state_dim >= 2")
        return
    true = true_states[step]
    plt.figure(figsize=(7, 6))
    plt.scatter(pf.particles[:, 0], pf.particles[:, 1],
                s=5, alpha=0.4, c="steelblue", label="Particles")
    plt.scatter(*pf.estimates[step][:2], c="gold", s=100, marker="*", zorder=5,
                label="Estimate (mean)")
    plt.scatter(*true[:2], c="red", s=100, marker="x", zorder=5,
                label="True state")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(f"Particle Cloud at t={len(pf.estimates)+step}")
    plt.legend()
    plt.tight_layout()
    save_figure("pf_particles.png", dpi=150)


# ─────────────────────────────────────────────
# N_PARTICLES SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────

def n_particles_sweep(true_states: np.ndarray, observations: np.ndarray,
                       n_range: list = None) -> None:
    """
    Compare filter accuracy across different particle counts.
    Helps select minimum N for target RMSE.
    """
    if n_range is None:
        n_range = [50, 100, 250, 500, 1000, 2000]

    model  = StateSpaceModel()
    rmses  = []

    for N in n_range:
        cfg = {**PF_CONFIG, "n_particles": N}
        pf  = ParticleFilter(model, cfg)
        est = pf.run(observations)
        rmse = np.sqrt(np.mean((est - true_states)**2))
        rmses.append(rmse)
        print(f"  N={N:5d}  RMSE={rmse:.4f}")

    plt.figure(figsize=(8, 4))
    plt.semilogx(n_range, rmses, "o-", color="steelblue")
    plt.xlabel("Number of Particles (log scale)")
    plt.ylabel("RMSE")
    plt.title("Particle Filter — N Particles vs RMSE")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    save_figure("pf_n_sweep.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Particle Filter (Sequential Monte Carlo) ===")
    np.random.seed(42)

    model = StateSpaceModel(state_dim=2, obs_dim=1,
                             sigma_process=0.1, sigma_obs=0.5)

    # Placeholder: replace with real sensor observations
    true_states, observations = generate_tracking_data(n_steps=100, model=model)
    print(f"Sequence: {len(true_states)} steps | "
          f"State dim: {model.state_dim} | Obs dim: {model.obs_dim}")

    # Run particle filter
    print("\nRunning particle filter...")
    pf  = ParticleFilter(model, PF_CONFIG)
    est = pf.run(observations)

    print("\nEvaluation:")
    evaluate(est, true_states)

    # Particle count sensitivity
    print("\nParticle count sensitivity:")
    n_particles_sweep(true_states, observations)

    # Visualizations
    plot_tracking(est, true_states, observations, pf)
    plot_effective_sample_size(pf)
    plot_particle_cloud(pf, true_states)


if __name__ == "__main__":
    main()
