"""
Physics-Informed Neural Network (PINN)
========================================
A neural network that satisfies physical laws by embedding PDE residuals
directly into the training loss via automatic differentiation.

Rather than discretizing a domain onto a mesh, PINNs treat the network
output as a continuous solution u(x,t) and enforce the PDE at a set of
collocation points sampled from the interior and boundary of the domain.

Loss decomposition:
    L_total = λ_data · L_data + λ_pde · L_pde + λ_bc · L_bc + λ_ic · L_ic

    L_data:  MSE at observed measurement locations
    L_pde:   MSE of PDE residual at interior collocation points
             e.g., for heat equation: r = ∂u/∂t - α ∂²u/∂x²
    L_bc:    MSE at domain boundary
    L_ic:    MSE at initial condition (t=0)

    Derivatives are computed via torch.autograd.grad — the key
    ingredient that makes PINNs work without finite differences.

Two canonical PDEs implemented:
    1. Heat equation:    ∂u/∂t = α ∂²u/∂x²
       Parabolic PDE; models diffusion (thermal, chemical, financial)

    2. Burgers equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
       Nonlinear PDE; models viscous shock formation; standard PINN benchmark

Inverse problem mode:
    PINNs can identify PDE parameters (α, ν) from sparse data — make
    the parameter a learnable nn.Parameter alongside the network weights.

Use Cases:
    - Solving PDEs without meshes (fluid mechanics, heat transfer)
    - Inverse problems: identify diffusivity/viscosity from sensor data
    - Sparse-data regression constrained by known physics
    - Surrogate models for parametric PDE families
    - Trajectory estimation constrained by kinematic equations

Advantages:
    - Mesh-free: works on irregular domains
    - Handles inverse problems naturally
    - Encodes prior physics knowledge; regularizes solution

Limitations:
    - Ill-conditioned optimization; slow convergence for complex PDEs
    - Fails on sharp features (shocks, turbulent flows)
    - Loss weight tuning (λ_pde, λ_bc, λ_ic) is problem-specific
    - Requires retraining for each new set of boundary conditions

Fine-Tuning Recommendations:
    - Network width/depth: start with 4–8 layers × 20–50 neurons; tanh activation
      (smooth second derivatives required for PDE residuals).
    - Loss weights: λ_pde = 1.0 baseline. If boundary conditions are violated,
      increase λ_bc to 10–100. Data loss can be large — normalize first.
    - Collocation points: 1k–10k interior + 200–500 boundary points.
      Adaptive sampling: concentrate points where residual is large.
    - Optimizer: Adam first (1k–5k steps), then L-BFGS for fine convergence.
      L-BFGS is often critical for achieving low residuals.
    - Activation: tanh is standard. Swish (SiLU) also works.
    - Weight initialization: Xavier uniform is stable for tanh.
    - For inverse problems: start with known-parameter forward solve to verify
      network architecture before adding unknown parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import autograd

from deepdraft.core import fast_or, save_figure

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# Note: autograd.grad requires gradients through the computation graph.
# MPS (Apple Silicon) has known issues with higher-order grad; fall back to CPU.
if DEVICE == "mps":
    DEVICE = "cpu"


# ─────────────────────────────────────────────
# NETWORK ARCHITECTURE
# ─────────────────────────────────────────────

class PINN(nn.Module):
    """
    Fully-connected neural network serving as the PDE solution approximator.

    Input:  (x, t) — spatial and temporal coordinates
    Output: u(x, t) — scalar field value

    tanh activations are required because PDE residuals involve second
    derivatives of the network output — ReLU has zero second derivative.

    For multi-output PDEs (e.g., velocity + pressure in Navier-Stokes):
    increase output_dim accordingly.
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 1,
                 hidden_dim: int = 40, n_layers: int = 6):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: spatial coordinate (N, 1), requires_grad=True
            t: temporal coordinate (N, 1), requires_grad=True

        Returns:
            u: predicted field value (N, 1)
        """
        return self.net(torch.cat([x, t], dim=1))


# ─────────────────────────────────────────────
# AUTOMATIC DIFFERENTIATION UTILITIES
# ─────────────────────────────────────────────

def grad(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    First derivative of output w.r.t. inputs via autograd.

    create_graph=True is essential: allows differentiating through this
    gradient again to get second derivatives (needed for ∂²u/∂x²).
    """
    return autograd.grad(
        output, inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def grad2(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Second derivative via double application of autograd."""
    return grad(grad(output, inputs), inputs)


# ─────────────────────────────────────────────
# PDE RESIDUALS
# ─────────────────────────────────────────────

def heat_residual(model: PINN, x: torch.Tensor, t: torch.Tensor,
                   alpha: float = 0.1) -> torch.Tensor:
    """
    Residual for the 1D heat equation:
        ∂u/∂t - α ∂²u/∂x² = 0

    Alpha = thermal diffusivity (Placeholder: set to your physical value).
    """
    u     = model(x, t)
    u_t   = grad(u, t)
    u_xx  = grad2(u, x)
    return u_t - alpha * u_xx


def burgers_residual(model: PINN, x: torch.Tensor, t: torch.Tensor,
                      nu: float = 0.01 / np.pi) -> torch.Tensor:
    """
    Residual for the 1D Burgers equation:
        ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0

    Nu = kinematic viscosity (Placeholder: standard PINN benchmark uses 0.01/π).
    This nonlinear PDE develops a sharp shock front — a canonical PINN challenge.
    """
    u    = model(x, t)
    u_t  = grad(u, t)
    u_x  = grad(u, x)
    u_xx = grad2(u, x)
    return u_t + u * u_x - nu * u_xx


# ─────────────────────────────────────────────
# COLLOCATION POINT SAMPLING
# ─────────────────────────────────────────────

def sample_collocation(n_interior: int, n_boundary: int,
                        x_range: tuple = (-1.0, 1.0),
                        t_range: tuple = (0.0, 1.0)) -> dict:
    """
    Sample collocation points for PDE training.

    Returns dict with:
        x_f, t_f:  interior (PDE residual) points
        x_bc, t_bc, u_bc:  boundary condition points (Dirichlet: u=0)
        x_ic, t_ic, u_ic:  initial condition points
    """
    # Interior: Latin hypercube-like uniform sampling
    x_f = torch.FloatTensor(n_interior, 1).uniform_(*x_range).to(DEVICE).requires_grad_(True)
    t_f = torch.FloatTensor(n_interior, 1).uniform_(*t_range).to(DEVICE).requires_grad_(True)

    # Boundary: x = x_min and x = x_max, all t; Dirichlet u=0
    n_half = n_boundary // 2
    t_bc   = torch.FloatTensor(n_boundary, 1).uniform_(*t_range).to(DEVICE)
    x_bc   = torch.cat([
        torch.full((n_half, 1), x_range[0]),
        torch.full((n_boundary - n_half, 1), x_range[1]),
    ]).to(DEVICE)
    u_bc   = torch.zeros(n_boundary, 1).to(DEVICE)

    # Initial condition: t=0, u(x,0) = -sin(π x)
    x_ic = torch.FloatTensor(n_boundary, 1).uniform_(*x_range).to(DEVICE)
    t_ic = torch.zeros(n_boundary, 1).to(DEVICE)
    u_ic = -torch.sin(np.pi * x_ic)   # Burgers IC

    return {
        "x_f": x_f, "t_f": t_f,
        "x_bc": x_bc, "t_bc": t_bc, "u_bc": u_bc,
        "x_ic": x_ic, "t_ic": t_ic, "u_ic": u_ic,
    }


# ─────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune loss weights and architecture
PINN_CONFIG = {
    "pde":           "burgers",    # Placeholder: 'burgers' or 'heat'
    "hidden_dim":    40,           # Placeholder: 20–100 neurons per layer
    "n_layers":      6,            # Placeholder: 4–8 layers
    "n_interior":    fast_or(10_000, 500),  # Placeholder: 1k–50k interior collocation points
    "n_boundary":    200,          # Placeholder: 100–1000 boundary/IC points
    "lambda_pde":    1.0,          # Placeholder: PDE residual weight
    "lambda_bc":     10.0,         # Placeholder: increase if BC violation is large
    "lambda_ic":     10.0,         # Placeholder: increase if IC violation is large
    "lr_adam":       1e-3,
    "n_adam":        fast_or(5_000, 30),  # Adam iterations
    "n_lbfgs":       fast_or(500, 3),  # L-BFGS iterations (fine convergence)
    "nu":            0.01 / np.pi, # Burgers viscosity
    "alpha":         0.1,          # Heat diffusivity
}


def train_pinn(config: dict = None) -> tuple:
    """
    Train a PINN using Adam + L-BFGS optimizer schedule.

    Adam: fast initial progress.
    L-BFGS: second-order convergence to low residuals (critical for PINNs).

    Returns:
        model:   trained PINN
        history: loss history dict
    """
    if config is None:
        config = PINN_CONFIG

    model  = PINN(input_dim=2, output_dim=1,
                   hidden_dim=config["hidden_dim"],
                   n_layers=config["n_layers"]).to(DEVICE)
    pts    = sample_collocation(config["n_interior"], config["n_boundary"])
    lam_p  = config["lambda_pde"]
    lam_bc = config["lambda_bc"]
    lam_ic = config["lambda_ic"]

    pde_fn = burgers_residual if config["pde"] == "burgers" else heat_residual
    pde_kwargs = {"nu": config["nu"]} if config["pde"] == "burgers" else {"alpha": config["alpha"]}

    history = {"total": [], "pde": [], "bc": [], "ic": []}

    def compute_loss():
        # PDE residual loss
        r_pde  = pde_fn(model, pts["x_f"], pts["t_f"], **pde_kwargs)
        l_pde  = torch.mean(r_pde ** 2)

        # Boundary condition loss
        u_bc_pred = model(pts["x_bc"], pts["t_bc"])
        l_bc      = torch.mean((u_bc_pred - pts["u_bc"]) ** 2)

        # Initial condition loss
        u_ic_pred = model(pts["x_ic"], pts["t_ic"])
        l_ic      = torch.mean((u_ic_pred - pts["u_ic"]) ** 2)

        total = lam_p * l_pde + lam_bc * l_bc + lam_ic * l_ic
        return total, l_pde, l_bc, l_ic

    # Phase 1: Adam
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=config["lr_adam"])
    for step in range(1, config["n_adam"] + 1):
        optimizer_adam.zero_grad()
        total, l_pde, l_bc, l_ic = compute_loss()
        total.backward()
        optimizer_adam.step()

        if step % 500 == 0:
            history["total"].append(total.item())
            history["pde"].append(l_pde.item())
            history["bc"].append(l_bc.item())
            history["ic"].append(l_ic.item())
            print(f"  Adam {step:5d} | total: {total.item():.2e} | "
                  f"pde: {l_pde.item():.2e} | bc: {l_bc.item():.2e} | "
                  f"ic: {l_ic.item():.2e}")

    # Phase 2: L-BFGS
    print("\n  Switching to L-BFGS...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), max_iter=config["n_lbfgs"],
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=50, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        total, _, _, _ = compute_loss()
        total.backward()
        return total

    optimizer_lbfgs.step(closure)

    total, l_pde, l_bc, l_ic = compute_loss()
    history["total"].append(total.item())
    history["pde"].append(l_pde.item())
    history["bc"].append(l_bc.item())
    history["ic"].append(l_ic.item())
    print(f"  L-BFGS done | total: {total.item():.2e} | pde: {l_pde.item():.2e}")

    return model, history


# ─────────────────────────────────────────────
# INVERSE PROBLEM: IDENTIFY PDE PARAMETER
# ─────────────────────────────────────────────

class InversePINN(nn.Module):
    """
    PINN for inverse parameter identification.

    Adds viscosity (nu) as a trainable parameter alongside the network.
    Given sparse observations, the optimizer identifies both the solution
    field u(x,t) and the unknown PDE parameter simultaneously.
    """

    def __init__(self, hidden_dim: int = 40, n_layers: int = 6,
                 nu_init: float = 0.05):
        super().__init__()
        self.net = PINN(hidden_dim=hidden_dim, n_layers=n_layers)
        # Trainable PDE parameter (log-parameterized for positivity)
        self.log_nu = nn.Parameter(torch.tensor(np.log(nu_init)))

    @property
    def nu(self) -> torch.Tensor:
        return torch.exp(self.log_nu)

    def forward(self, x, t):
        return self.net(x, t)


def train_inverse_pinn(true_nu: float = 0.01 / np.pi,
                        n_obs: int = 200,
                        nu_init: float = 0.05) -> tuple:
    """
    Demonstrate inverse problem: recover ν from sparse Burgers observations.

    Generates 'observations' from the true solution, then trains the
    inverse PINN to simultaneously fit observations and satisfy the PDE,
    recovering the unknown viscosity ν.

    Returns:
        model, true_nu, recovered_nu
    """
    print(f"\n  Inverse problem: true ν = {true_nu:.6f}, initial guess = {nu_init:.4f}")

    # Generate synthetic 'measurements' from a forward solve
    forward_model, _ = train_pinn({**PINN_CONFIG, "nu": true_nu,
                                    "n_adam": fast_or(3000, 30), "n_lbfgs": fast_or(200, 3)})
    forward_model.eval()

    with torch.no_grad():
        x_obs = torch.FloatTensor(n_obs, 1).uniform_(-1, 1).to(DEVICE)
        t_obs = torch.FloatTensor(n_obs, 1).uniform_(0, 1).to(DEVICE)
        u_obs = forward_model(x_obs, t_obs)
        u_obs += 0.01 * torch.randn_like(u_obs)  # add measurement noise

    # Train inverse PINN
    inv_model = InversePINN(nu_init=nu_init).to(DEVICE)
    pts       = sample_collocation(PINN_CONFIG["n_interior"], PINN_CONFIG["n_boundary"])
    optimizer = torch.optim.Adam(inv_model.parameters(), lr=1e-3)

    for step in range(1, fast_or(3000, 30) + 1):
        optimizer.zero_grad()

        # Data loss on observations
        u_pred  = inv_model(x_obs, t_obs)
        l_data  = torch.mean((u_pred - u_obs) ** 2)

        # PDE residual with current ν estimate
        x_f = pts["x_f"]; t_f = pts["t_f"]
        u_f = inv_model(x_f, t_f)
        u_t = grad(u_f, t_f)
        u_x = grad(u_f, x_f)
        u_xx = grad2(u_f, x_f)
        r   = u_t + u_f * u_x - inv_model.nu * u_xx
        l_pde = torch.mean(r ** 2)

        loss = 100.0 * l_data + l_pde
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"  Inverse step {step:4d} | ν_est = {inv_model.nu.item():.6f} | "
                  f"l_data = {l_data.item():.2e} | l_pde = {l_pde.item():.2e}")

    recovered_nu = inv_model.nu.item()
    print(f"\n  True ν: {true_nu:.6f}  |  Recovered ν: {recovered_nu:.6f}  "
          f"|  Error: {abs(recovered_nu - true_nu)/true_nu*100:.2f}%")
    return inv_model, true_nu, recovered_nu


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_solution(model: PINN, config: dict = None, title: str = None) -> None:
    """
    2D heatmap of the PINN solution u(x,t) over the full domain.
    """
    if config is None:
        config = PINN_CONFIG
    model.eval()

    x_vals = np.linspace(-1, 1, 100)
    t_vals = np.linspace(0, 1, 100)
    X, T   = np.meshgrid(x_vals, t_vals)

    x_t = torch.tensor(X.ravel()[:, None], dtype=torch.float32).to(DEVICE)
    t_t = torch.tensor(T.ravel()[:, None], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        u_pred = model(x_t, t_t).cpu().numpy().reshape(100, 100)

    plt.figure(figsize=(9, 5))
    plt.contourf(x_vals, t_vals, u_pred, levels=50, cmap="RdBu_r")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title or f"PINN Solution — {config['pde'].capitalize()} Equation")
    plt.tight_layout()
    save_figure("pinn_solution.png", dpi=150)


def plot_loss_history(history: dict) -> None:
    """Plot training loss components over Adam iterations."""
    steps = range(len(history["total"]))
    plt.figure(figsize=(9, 4))
    plt.semilogy(steps, history["total"], label="Total", linewidth=2)
    plt.semilogy(steps, history["pde"],   label="PDE Residual", linewidth=1.5)
    plt.semilogy(steps, history["bc"],    label="Boundary", linewidth=1.5)
    plt.semilogy(steps, history["ic"],    label="Initial Cond.", linewidth=1.5)
    plt.xlabel("Checkpoint (every 500 Adam steps)")
    plt.ylabel("Loss (log scale)")
    plt.title("PINN — Training Loss Components")
    plt.legend()
    plt.tight_layout()
    save_figure("pinn_loss.png", dpi=150)


def plot_solution_slices(model: PINN, t_slices: list = None) -> None:
    """
    1D solution profiles at fixed time slices.
    Useful for comparing against analytical or reference solutions.
    """
    if t_slices is None:
        t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
    model.eval()

    x_vals = np.linspace(-1, 1, 200)
    x_t    = torch.tensor(x_vals[:, None], dtype=torch.float32).to(DEVICE)

    plt.figure(figsize=(10, 4))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(t_slices)))
    for t_val, col in zip(t_slices, cmap):
        t_t = torch.full((200, 1), t_val, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            u = model(x_t, t_t).cpu().numpy().ravel()
        plt.plot(x_vals, u, color=col, linewidth=1.5, label=f"t={t_val:.2f}")

    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("PINN — Solution Profiles at Fixed Times")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_figure("pinn_slices.png", dpi=150)


# ─────────────────────────────────────────────
# 2D POISSON EQUATION
# ─────────────────────────────────────────────

class PINN2D(nn.Module):
    """
    2D PINN: input (x, y), output u(x,y).
    tanh activation required for second-order spatial derivatives.
    """

    def __init__(self, hidden_dim: int = 50, n_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))


def poisson_residual(model: PINN2D,
                      x: torch.Tensor, y: torch.Tensor,
                      f_fn=None) -> torch.Tensor:
    """
    Residual for the 2D Poisson equation:
        ∇²u(x,y) = f(x,y)   on Ω = (0,1)²
        u(x,y) = 0            on ∂Ω  (Dirichlet, homogeneous)

    Manufactured solution: u(x,y) = sin(πx)sin(πy)
        → f(x,y) = -2π² sin(πx) sin(πy)  (from ∇²u = u_xx + u_yy)

    Placeholder: replace f_fn with your right-hand side function.
    """
    if f_fn is None:
        # Manufactured: f = -2π² sin(πx) sin(πy)
        f_fn = lambda xx, yy: -2.0 * (np.pi ** 2) * torch.sin(np.pi * xx) * torch.sin(np.pi * yy)

    u = model(x, y)
    u_xx = grad(grad(u, x), x)
    u_yy = grad(grad(u, y), y)
    f = f_fn(x, y)
    return u_xx + u_yy - f   # Laplacian - f = 0


def sample_collocation_2d(n_interior: int = 5000, n_boundary: int = 400):
    """
    Sample interior (PDE) and boundary (BC) collocation points on [0,1]².
    Boundary: 4 edges with Dirichlet u=0.
    """
    # Interior
    x_f = torch.rand(n_interior, 1, device=DEVICE).requires_grad_(True)
    y_f = torch.rand(n_interior, 1, device=DEVICE).requires_grad_(True)

    # Boundary (4 edges: bottom, top, left, right)
    n4 = n_boundary // 4
    t  = torch.rand(n4, 1)

    x_bc = torch.cat([t, t,
                       torch.zeros(n4, 1), torch.ones(n4, 1)], dim=0).to(DEVICE)
    y_bc = torch.cat([torch.zeros(n4, 1), torch.ones(n4, 1),
                       t, t], dim=0).to(DEVICE)
    u_bc = torch.zeros(4 * n4, 1, device=DEVICE)

    return {"x_f": x_f, "y_f": y_f, "x_bc": x_bc, "y_bc": y_bc, "u_bc": u_bc}


def poisson_exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytical solution for the manufactured Poisson problem."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def train_poisson_2d(
    n_interior: int = fast_or(5000, 500),  # Placeholder: 5k–20k interior points
    n_boundary: int = 400,         # Placeholder: 200–1000 boundary points
    hidden_dim: int = 50,          # Placeholder: 30–100 neurons
    n_layers: int = 5,             # Placeholder: 4–7 layers
    lambda_bc: float = 100.0,      # Placeholder: BC weight; high to enforce exact BCs
    lr: float = 1e-3,              # Placeholder: Adam learning rate
    n_adam: int = fast_or(5_000, 30),  # Placeholder: Adam steps
    n_lbfgs: int = fast_or(500, 3),  # Placeholder: L-BFGS steps
) -> PINN2D:
    """
    Train a PINN to solve ∇²u = f on [0,1]² with homogeneous Dirichlet BCs.

    Manufactured solution: u*(x,y) = sin(πx)sin(πy).
    After training, compares PINN solution against the analytical solution
    and reports relative L² error.
    """
    print(f"  Poisson 2D: {n_interior} interior, {n_boundary} boundary points")
    model = PINN2D(hidden_dim=hidden_dim, n_layers=n_layers).to(DEVICE)
    pts   = sample_collocation_2d(n_interior, n_boundary)

    def compute_loss():
        r    = poisson_residual(model, pts["x_f"], pts["y_f"])
        l_pde = torch.mean(r ** 2)

        u_bc_pred = model(pts["x_bc"], pts["y_bc"])
        l_bc      = torch.mean((u_bc_pred - pts["u_bc"]) ** 2)

        return l_pde + lambda_bc * l_bc, l_pde, l_bc

    # Adam phase
    opt_adam = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, n_adam + 1):
        opt_adam.zero_grad()
        total, l_pde, l_bc = compute_loss()
        total.backward()
        opt_adam.step()
        if step % 1000 == 0:
            print(f"  Adam {step:5d} | pde: {l_pde.item():.2e} | bc: {l_bc.item():.2e}")

    # L-BFGS phase
    print("  Switching to L-BFGS...")
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(), max_iter=n_lbfgs,
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=50, line_search_fn="strong_wolfe",
    )

    def closure():
        opt_lbfgs.zero_grad()
        total, _, _ = compute_loss()
        total.backward()
        return total

    opt_lbfgs.step(closure)

    # Evaluate relative L² error against manufactured solution
    model.eval()
    n_eval = 50
    xv = np.linspace(0, 1, n_eval)
    yv = np.linspace(0, 1, n_eval)
    Xg, Yg = np.meshgrid(xv, yv)
    x_t = torch.tensor(Xg.ravel()[:, None], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(Yg.ravel()[:, None], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_pred = model(x_t, y_t).cpu().numpy().reshape(n_eval, n_eval)
    u_exact = poisson_exact(Xg, Yg)
    rel_err = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f"  Poisson 2D relative L² error: {rel_err:.4e}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, data, ttl in zip(axes,
                                  [u_exact, u_pred, np.abs(u_pred - u_exact)],
                                  ["Exact", "PINN", "|Error|"]):
            cm = ax.contourf(Xg, Yg, data, levels=40, cmap="viridis")
            plt.colorbar(cm, ax=ax)
            ax.set_title(ttl)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.suptitle(f"Poisson 2D PINN  (rel L² err = {rel_err:.2e})")
        plt.tight_layout()
        save_figure("pinn_poisson2d.png", dpi=150)
        print("  Saved pinn_poisson2d.png")
    except ImportError:
        pass

    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"=== Physics-Informed Neural Network ===  (device: {DEVICE})")
    config = PINN_CONFIG
    print(f"PDE: {config['pde'].capitalize()}")
    print(f"Architecture: {config['n_layers']} layers × {config['hidden_dim']} neurons")
    print(f"Collocation: {config['n_interior']:,} interior + {config['n_boundary']} boundary pts")

    # Forward solve
    print("\nTraining forward PINN...")
    model, history = train_pinn(config)

    # Visualizations
    plot_loss_history(history)
    plot_solution(model, config)
    plot_solution_slices(model)

    # Inverse problem: identify viscosity from sparse observations
    print("\n--- Inverse Problem ---")
    train_inverse_pinn(true_nu=0.01 / np.pi, n_obs=200, nu_init=0.05)

    # 2D Poisson equation
    print("\n--- 2D Poisson Equation ---")
    train_poisson_2d()


if __name__ == "__main__":
    main()
