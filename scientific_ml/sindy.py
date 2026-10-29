"""
SINDy — Sparse Identification of Nonlinear Dynamics
======================================================

SINDy discovers governing equations of dynamical systems from data.
Given time-series measurements x(t), it finds a sparse set of basis
functions Θ(x) such that:

  ẋ = Θ(x) Ξ

where Θ(x) is a library of candidate functions and Ξ is a sparse
coefficient matrix (most entries zero).

Algorithm:
  1. Collect state data X ∈ ℝ^{T × n} and derivatives Ẋ ∈ ℝ^{T × n}
     (computed via finite differences or provided directly)
  2. Build feature library Θ(X): polynomial terms, trig functions, etc.
     e.g. [1, x₁, x₂, x₁², x₁x₂, x₂², sin(x₁), ...]
  3. Solve sparse regression:  Ẋ = Θ(X) Ξ
     - Sequential Thresholded Least Squares (STLSQ): iteratively zero
       out coefficients below threshold λ, then refit
     - LASSO: L1 regularization promotes sparsity
     - SR3 (Sparse Relaxed Regularized Regression): handles noisy data better

Why sparsity matters:
  Real physical equations are sparse in a rich function library.
  ẋ = σ(y-x) is Lorenz — only 2 terms in a library of 20+.

Derivative computation:
  - Finite differences (noisy data → use higher-order or Savitzky-Golay filter)
  - Provided analytically (clean integration of known ODE)
  - Total Variation Regularized Differentiator (TVdiff) for very noisy data

Feature library:
  PolynomialLibrary: 1, x, x², xy, x²y, ...  up to degree d
  FourierLibrary:    sin(kx), cos(kx) for k=1..n_freq
  CustomLibrary:     user-defined functions (lambda-based)

Fine-tuning recommendations:
  - poly_degree (#Placeholder: 3): higher = richer library, harder sparsification
  - threshold (#Placeholder: 0.05): STLSQ sparsification threshold; tune via validation
  - alpha (#Placeholder: 0.1): LASSO/ridge regularization strength
  - n_iterations (#Placeholder: 10): STLSQ refinement iterations
  - include_bias (#Placeholder: True): include constant term in library
  - normalize_columns (#Placeholder: True): important for numerical stability

References:
  Brunton et al. (2016) — Discovering Governing Equations from Data (original SINDy)
  Rudy et al. (2017) — Data-Driven Discovery of Partial Differential Equations
  Champion et al. (2019) — Data-Driven Discovery of Coordinates and Governing Equations
  de Silva et al. (2020) — PySINDy: A Python Package for SINDy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from itertools import combinations_with_replacement
from dataclasses import dataclass, field
from typing import Callable

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "poly_degree":         3,       # # Placeholder: polynomial library degree
    "include_bias":        True,    # # Placeholder: constant term in library
    "include_interaction": True,    # # Placeholder: cross terms (x*y)
    "stlsq_threshold":     0.05,    # # Placeholder: key parameter — larger = sparser
    "stlsq_n_iter":        10,      # # Placeholder:
    "lasso_alpha":         0.01,    # # Placeholder:
    "noise_std":           0.01,    # # Placeholder: measurement noise for experiments
    "dt":                  0.01,    # # Placeholder: time step
    "n_time_steps":        2000,    # # Placeholder:
    "sg_window":           5,       # # Placeholder: Savitzky-Golay derivative window
    "sg_poly":             3,       # # Placeholder: Savitzky-Golay polynomial order
    "seed":                42,
}


# ─────────────────────────────────────────────
# Feature Library
# ─────────────────────────────────────────────

class PolynomialLibrary:
    """
    Builds library of polynomial features up to degree d.
    For state x = [x₁, x₂]:
      degree=1: [1, x₁, x₂]
      degree=2: [1, x₁, x₂, x₁², x₁x₂, x₂²]
      degree=3: [1, x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³]
    """

    def __init__(self, degree: int = 2, include_bias: bool = True):
        self.degree = degree
        self.include_bias = include_bias
        self.feature_names_: list[str] = []
        self._powers: list[tuple] = []

    def fit(self, X: np.ndarray) -> "PolynomialLibrary":
        n_features = X.shape[1]
        idx = list(range(n_features))
        self._powers = []
        self.feature_names_ = []

        if self.include_bias:
            self._powers.append(tuple([0] * n_features))
            self.feature_names_.append("1")

        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(idx, d):
                power = [0] * n_features
                for i in combo:
                    power[i] += 1
                self._powers.append(tuple(power))
                name_parts = []
                for fi, p in enumerate(power):
                    if p == 1:
                        name_parts.append(f"x{fi+1}")
                    elif p > 1:
                        name_parts.append(f"x{fi+1}^{p}")
                self.feature_names_.append(" ".join(name_parts) if name_parts else "1")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rows = []
        for row in X:
            features = []
            for power in self._powers:
                term = np.prod([row[i] ** p for i, p in enumerate(power)])
                features.append(term)
            rows.append(features)
        return np.array(rows)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_features_out(self) -> int:
        return len(self._powers)


class CustomLibrary:
    """
    User-defined function library. Pass a list of (name, func) pairs.
    Each func takes the full state row x and returns a scalar.
    """

    def __init__(self, functions: list[tuple[str, Callable]]):
        self.functions = functions
        self.feature_names_ = [name for name, _ in functions]

    def fit(self, X):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.array([[f(row) for _, f in self.functions] for row in X])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_features_out(self) -> int:
        return len(self.functions)


class CombinedLibrary:
    """Horizontal concatenation of several feature libraries."""

    def __init__(self, libraries):
        self.libraries = libraries
        self.feature_names_ = []

    def fit(self, X):
        self.feature_names_ = []
        for lib in self.libraries:
            lib.fit(X)
            self.feature_names_.extend(lib.feature_names_)
        return self

    def transform(self, X):
        return np.hstack([lib.transform(X) for lib in self.libraries])

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def combine_libraries(*libs) -> CombinedLibrary:
    """Concatenate multiple libraries horizontally."""
    return CombinedLibrary(list(libs))


# ─────────────────────────────────────────────
# Derivative Computation
# ─────────────────────────────────────────────

def finite_differences(X: np.ndarray, dt: float, order: int = 2) -> np.ndarray:
    """
    Compute time derivative via finite differences.
    order=2: central differences (2nd-order accurate)
    order=4: 4th-order central differences (more accurate, less noise-robust)
    """
    T, n = X.shape
    Xdot = np.zeros_like(X)

    if order == 2:
        Xdot[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
        Xdot[0]    = (-3*X[0] + 4*X[1] - X[2]) / (2 * dt)
        Xdot[-1]   = (3*X[-1] - 4*X[-2] + X[-3]) / (2 * dt)
    elif order == 4:
        Xdot[2:-2] = (-X[4:] + 8*X[3:-1] - 8*X[1:-3] + X[:-4]) / (12 * dt)
        # Use 2nd order at edges
        Xdot[0]  = (-3*X[0] + 4*X[1] - X[2]) / (2 * dt)
        Xdot[1]  = (X[2] - X[0]) / (2 * dt)
        Xdot[-2] = (X[-1] - X[-3]) / (2 * dt)
        Xdot[-1] = (3*X[-1] - 4*X[-2] + X[-3]) / (2 * dt)

    return Xdot


def savitzky_golay_derivative(X: np.ndarray, dt: float,
                               window: int = 5, poly: int = 3) -> np.ndarray:
    """
    Savitzky-Golay smoothing derivative — better than raw finite diff on noisy data.
    Fits local polynomials and differentiates analytically.
    """
    Xdot = np.zeros_like(X)
    for j in range(X.shape[1]):
        Xdot[:, j] = savgol_filter(X[:, j], window_length=window,
                                    polyorder=poly, deriv=1, delta=dt)
    return Xdot


# ─────────────────────────────────────────────
# Sparse Regression
# ─────────────────────────────────────────────

def stlsq(
    Theta: np.ndarray,
    Xdot: np.ndarray,
    threshold: float = 0.05,
    n_iter: int = 10,
    alpha: float = 0.0,   # ridge regularization
) -> np.ndarray:
    """
    Sequential Thresholded Least Squares (STLSQ).
    Iteratively:
      1. Solve least squares: Ξ = (Θ^T Θ + αI)^{-1} Θ^T Ẋ
      2. Zero out |Ξ| < threshold (hard thresholding)
      3. Refit on remaining active terms only
    Returns coefficient matrix Ξ ∈ ℝ^{n_library × n_states}.
    """
    n_lib = Theta.shape[1]
    n_states = Xdot.shape[1]
    Xi = np.zeros((n_lib, n_states))

    for j in range(n_states):
        active = np.ones(n_lib, dtype=bool)
        xdot_j = Xdot[:, j]

        for _ in range(n_iter):
            Theta_active = Theta[:, active]
            A = Theta_active.T @ Theta_active + alpha * np.eye(active.sum())
            b = Theta_active.T @ xdot_j
            xi_active = np.linalg.solve(A, b)
            xi_full = np.zeros(n_lib)
            xi_full[active] = xi_active
            active = np.abs(xi_full) >= threshold
            if not np.any(active):
                break

        if np.any(active):
            Theta_active = Theta[:, active]
            A = Theta_active.T @ Theta_active + alpha * np.eye(active.sum())
            b = Theta_active.T @ xdot_j
            xi_full = np.zeros(n_lib)
            xi_full[active] = np.linalg.solve(A, b)
            Xi[:, j] = xi_full

    return Xi


def lasso_regression(Theta: np.ndarray, Xdot: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """L1-regularized regression (LASSO) for sparse recovery."""
    from sklearn.linear_model import Lasso
    Xi = np.zeros((Theta.shape[1], Xdot.shape[1]))
    for j in range(Xdot.shape[1]):
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        lasso.fit(Theta, Xdot[:, j])
        Xi[:, j] = lasso.coef_
    return Xi


# ─────────────────────────────────────────────
# SINDy Model
# ─────────────────────────────────────────────

@dataclass
class SINDy:
    """
    SINDy model: discovers ẋ = Θ(x) Ξ from trajectory data.
    """
    library: object
    threshold: float = 0.05       # # Placeholder: STLSQ threshold
    alpha:     float = 0.0        # # Placeholder: ridge regularization
    n_iter:    int   = 10         # # Placeholder: STLSQ iterations
    method:    str   = "stlsq"    # # Placeholder: "stlsq" | "lasso"

    Xi_: np.ndarray = field(default=None, init=False, repr=False)
    feature_names_: list = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, Xdot: np.ndarray) -> "SINDy":
        Theta = self.library.fit_transform(X)
        self.feature_names_ = self.library.feature_names_

        if self.method == "stlsq":
            self.Xi_ = stlsq(Theta, Xdot, self.threshold, self.n_iter, self.alpha)
        elif self.method == "lasso":
            self.Xi_ = lasso_regression(Theta, Xdot, self.alpha)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Theta = self.library.transform(X)
        return Theta @ self.Xi_

    def simulate(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Simulate discovered equations from initial condition x0."""
        def rhs(t_val, x):
            return self.predict(x[None, :])[0]
        sol = solve_ivp(rhs, [t[0], t[-1]], x0, t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y.T

    def print_equations(self):
        """Print discovered equations in human-readable form."""
        n_states = self.Xi_.shape[1]
        for j in range(n_states):
            terms = []
            for name, coef in zip(self.feature_names_, self.Xi_[:, j]):
                if abs(coef) > 1e-10:
                    terms.append(f"{coef:+.4f}·{name}")
            eq = " ".join(terms) if terms else "0"
            print(f"  ẋ{j+1} = {eq}")

    def score(self, X: np.ndarray, Xdot: np.ndarray) -> float:
        """R² score on derivative prediction."""
        Xdot_pred = self.predict(X)
        ss_res = np.sum((Xdot - Xdot_pred) ** 2)
        ss_tot = np.sum((Xdot - Xdot.mean(0)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-12))

    def sparsity(self) -> float:
        """Fraction of zero coefficients."""
        return float(np.mean(np.abs(self.Xi_) < 1e-10))


# ─────────────────────────────────────────────
# Benchmark Dynamical Systems
# ─────────────────────────────────────────────

def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8/3):
    """Classic Lorenz attractor — chaotic 3D system."""
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]


def vanderpol_system(t, state, mu=1.0):
    """Van der Pol oscillator — nonlinear oscillator with limit cycle."""
    x, y = state
    return [y, mu*(1 - x**2)*y - x]


def duffing_system(t, state, alpha=1.0, beta=-1.0, delta=0.3, gamma=0.5, omega=1.2):
    """Duffing oscillator — driven damped nonlinear oscillator."""
    x, y = state
    return [y, gamma*np.cos(omega*t) - delta*y - alpha*x - beta*x**3]


def cubic_ode_2d(t, state):
    """Simple 2D cubic ODE: ẋ = -x³ + y, ẏ = -x - y  (stable focus)."""
    x, y = state
    return [-x**3 + y, -x - y]


def generate_trajectory(
    system: Callable,
    x0: np.ndarray,
    t_span: tuple,
    dt: float,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ODE, optionally add measurement noise."""
    t = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(system, t_span, x0, t_eval=t, rtol=1e-10, atol=1e-12)
    X = sol.y.T
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        X = X + rng.normal(0, noise_std, X.shape)
    return X, t


# ─────────────────────────────────────────────
# PySINDy Wrapper
# ─────────────────────────────────────────────

def run_pysindy(X: np.ndarray, t: np.ndarray, degree: int = 3,
                threshold: float = 0.05) -> object:
    """
    Thin wrapper around PySINDy (de Silva et al., 2020) — production-grade SINDy.
    Install: pip install pysindy

    Advantages over this implementation:
      - SR3 optimizer (better noise handling)
      - Weak SINDy (integral formulation, bypasses derivative noise)
      - SINDy-PI (implicit equations)
      - PDELibrary for PDE discovery
    """
    try:
        import pysindy as ps
    except ImportError:
        print("PySINDy not installed: pip install pysindy")
        return None

    dt = t[1] - t[0]
    optimizer = ps.STLSQ(threshold=threshold)
    feature_lib = ps.PolynomialLibrary(degree=degree)
    model = ps.SINDy(optimizer=optimizer, feature_library=feature_lib)
    model.fit(X, t=dt)
    model.print()
    return model


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_sindy_results(
    model: SINDy,
    X_true: np.ndarray,
    t: np.ndarray,
    x0: np.ndarray,
    system_name: str = "system",
):
    """Compare true trajectory vs SINDy simulation."""
    try:
        X_sim = model.simulate(x0, t)
    except Exception as e:
        print(f"Simulation failed: {e}")
        X_sim = np.zeros_like(X_true)

    n_states = X_true.shape[1]
    state_labels = [f"x{i+1}" for i in range(n_states)]

    fig, axes = plt.subplots(1, min(n_states, 3), figsize=(5 * min(n_states, 3), 4))
    if n_states == 1:
        axes = [axes]
    fig.suptitle(f"SINDy — {system_name} (sparsity={model.sparsity():.1%})", fontsize=12)

    for i, ax in enumerate(axes):
        if i >= n_states:
            break
        ax.plot(t, X_true[:, i], "k-", lw=1.5, alpha=0.8, label="True")
        ax.plot(t, X_sim[:len(t), i], "r--", lw=1.5, label="SINDy sim")
        ax.set_xlabel("Time"); ax.set_ylabel(state_labels[i])
        ax.set_title(state_labels[i])
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"sindy_{system_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def plot_coefficient_heatmap(model: SINDy, state_names: list = None):
    """Heatmap of discovered coefficient matrix Ξ."""
    Xi = model.Xi_
    n_states = Xi.shape[1]
    labels = state_names or [f"ẋ{i+1}" for i in range(n_states)]

    fig, ax = plt.subplots(figsize=(max(6, n_states * 1.5), max(5, len(model.feature_names_) * 0.35)))
    im = ax.imshow(np.abs(Xi), aspect="auto", cmap="Blues")
    ax.set_xticks(range(n_states)); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(model.feature_names_)))
    ax.set_yticklabels(model.feature_names_, fontsize=8)
    plt.colorbar(im, ax=ax, label="|coefficient|")
    ax.set_title("SINDy Coefficient Matrix |Ξ|")
    plt.tight_layout()
    plt.savefig("sindy_coefficients.png", dpi=150)
    plt.show()
    print("Saved: sindy_coefficients.png")


def threshold_sweep(
    X: np.ndarray, Xdot: np.ndarray,
    library,
    thresholds: list = None,
) -> dict:
    """
    Sweep STLSQ threshold to show sparsity vs accuracy trade-off.
    Key diagnostic for threshold selection.
    """
    thresholds = thresholds or [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]  # # Placeholder:
    results = {"threshold": [], "sparsity": [], "r2": []}
    for thr in thresholds:
        Xi = stlsq(library.transform(X), Xdot, threshold=thr, n_iter=10)
        # Compute R²
        Xdot_pred = library.transform(X) @ Xi
        ss_res = np.sum((Xdot - Xdot_pred) ** 2)
        ss_tot = np.sum((Xdot - Xdot.mean(0)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        sparsity = np.mean(np.abs(Xi) < 1e-10)
        results["threshold"].append(thr)
        results["r2"].append(r2)
        results["sparsity"].append(sparsity)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogx(thresholds, results["r2"], marker="o", color="steelblue")
    axes[0].set_xlabel("Threshold λ"); axes[0].set_ylabel("R²")
    axes[0].set_title("Threshold vs R² (accuracy)")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(thresholds, results["sparsity"], marker="o", color="darkorange")
    axes[1].set_xlabel("Threshold λ"); axes[1].set_ylabel("Sparsity (frac. zeros)")
    axes[1].set_title("Threshold vs Sparsity")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sindy_threshold_sweep.png", dpi=150)
    plt.show()
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(CONFIG["seed"])
    print(f"Config: {CONFIG}\n")

    benchmarks = [
        ("Lorenz",        lorenz_system, np.array([-8.0, 7.0, 27.0]), (0, 10)),
        ("Van der Pol",   vanderpol_system, np.array([2.0, 0.0]), (0, 20)),
        ("Cubic ODE 2D",  cubic_ode_2d, np.array([2.0, 0.0]), (0, 10)),
    ]

    for name, system, x0, t_span in benchmarks:
        print(f"\n{'='*50}")
        print(f"System: {name}")
        print(f"{'='*50}")

        # Generate data
        X, t = generate_trajectory(system, x0, t_span, dt=CONFIG["dt"],
                                    noise_std=CONFIG["noise_std"])

        # Compute derivatives
        if CONFIG["noise_std"] > 0:
            Xdot = savitzky_golay_derivative(X, CONFIG["dt"],
                                              window=CONFIG["sg_window"],
                                              poly=CONFIG["sg_poly"])
        else:
            Xdot = finite_differences(X, CONFIG["dt"], order=4)

        # Build library and fit SINDy
        library = PolynomialLibrary(degree=CONFIG["poly_degree"],
                                     include_bias=CONFIG["include_bias"])
        model = SINDy(library=library, threshold=CONFIG["stlsq_threshold"],
                       n_iter=CONFIG["stlsq_n_iter"])
        model.fit(X, Xdot)

        print(f"\nDiscovered equations (R²={model.score(X, Xdot):.5f}, "
              f"sparsity={model.sparsity():.1%}):")
        model.print_equations()

        plot_sindy_results(model, X, t, x0, system_name=name)

    # Threshold sweep on Van der Pol
    print("\n=== Threshold Sweep (Van der Pol) ===")
    X, t = generate_trajectory(vanderpol_system, np.array([2.0, 0.0]), (0, 20),
                                 dt=CONFIG["dt"], noise_std=0.1)
    Xdot = savitzky_golay_derivative(X, CONFIG["dt"])
    lib = PolynomialLibrary(degree=CONFIG["poly_degree"])
    lib.fit(X)
    threshold_sweep(X, Xdot, lib)

    # Coefficient heatmap for Lorenz
    print("\n=== Lorenz Coefficient Heatmap ===")
    X, t = generate_trajectory(lorenz_system, np.array([-8.0, 7.0, 27.0]),
                                 (0, 10), dt=CONFIG["dt"])
    Xdot = finite_differences(X, CONFIG["dt"])
    lib = PolynomialLibrary(degree=2)
    model_lorenz = SINDy(library=lib, threshold=0.1)
    model_lorenz.fit(X, Xdot)
    plot_coefficient_heatmap(model_lorenz, state_names=["ẋ", "ẏ", "ż"])

    # Optional: PySINDy
    print("\n=== PySINDy (if installed) ===")
    run_pysindy(X, t)


if __name__ == "__main__":
    main()
