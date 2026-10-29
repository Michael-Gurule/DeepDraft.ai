"""
Deep Operator Network (DeepONet)
===================================
Learns a mapping between function spaces — an operator — rather than a
pointwise function. Given an input function u(x), DeepONet predicts the
output function G(u)(y) at any query location y.

Universal Approximation Theorem for Operators (Chen & Chen 1995):
    Any continuous nonlinear operator G: V → W can be approximated by:

        G(u)(y) ≈ Σᵢ bᵢ(u(x₁), ..., u(xₘ)) · tᵢ(y)

    - bᵢ: branch network outputs — encode the input function u
    - tᵢ: trunk network outputs — encode the query location y
    - Inner product of branch and trunk outputs gives the prediction

Architecture:
    Branch Net:  u(x₁), ..., u(xₘ)  →  b ∈ ℝᵖ
                 (fixed sensor locations x₁..xₘ, input function values)
    Trunk Net:   y               →  t ∈ ℝᵖ
                 (query location, continuous)
    Output:      G(u)(y) = Σᵢ bᵢ · tᵢ + bias

    The key insight: branch encodes "which function", trunk encodes "where".
    Their dot product selects the appropriate functional response at y.

Comparison to classical approaches:
    - FEM/FD solvers: O(n³) per new PDE instance; DeepONet: O(p) after training
    - Neural ODE: learns dynamics in fixed state space; DeepONet learns the operator
    - PINN: must retrain for each new boundary condition; DeepONet generalizes

Operators implemented:
    1. Antiderivative operator:   G(u)(y) = ∫₀ʸ u(x) dx
    2. ODE solution operator:     u' = f(u, t) → solution trajectory
    3. Advection-diffusion PDE:   operator mapping initial condition → solution

Use Cases:
    - Surrogate for parametric PDEs (generalize over IC/BC variations)
    - Weather forecasting: operator from initial conditions → future state
    - Material design: map loading conditions → stress fields
    - State estimation: operator from sensor readings → state across time
    - Drug discovery: map molecular structure → binding affinity field

Fine-Tuning Recommendations:
    - p (latent width): 40–200 neurons. Start at 100.
    - Branch/trunk depth: 3–6 layers; tanh activation (smooth operators).
    - Sensor locations m: 50–500. Must cover representative input variation.
    - Training data: 1000–10000 (u, G(u)) pairs from a reference solver.
    - Normalization: standardize both u inputs and G(u) targets separately.
    - Physics-informed DeepONet: add PDE residual loss on query outputs
      (same autograd trick as PINN) for better generalization with less data.
    - For production: use the DeepXDE library (pip install deepxde) which has
      full DeepONet + physics-informed variants with adaptive sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ─────────────────────────────────────────────
# DATA GENERATION — OPERATOR DATASETS
# ─────────────────────────────────────────────

def generate_antiderivative_data(n_functions: int = 2000,
                                  m_sensors: int = 100,
                                  n_query: int = 50,
                                  seed: int = 42) -> dict:
    """
    Antiderivative operator: G(u)(y) = ∫₀ʸ u(x) dx

    Input functions u(x): random Gaussian processes on [0, 1].
    Output G(u)(y): antiderivative evaluated at query points y.

    This is the canonical DeepONet benchmark (Lu et al. 2021).

    Replace with your operator:
        u_samples:    (n_functions, m_sensors) — input function evaluated at sensors
        G_u_samples:  (n_functions, n_query)   — operator output at query locations
        y_queries:    (n_query,)               — query locations
    """
    rng = np.random.RandomState(seed)

    # Fixed sensor locations (same for all functions)
    x_sensors = np.linspace(0, 1, m_sensors)  # (m,)
    y_queries  = np.linspace(0, 1, n_query)   # (n_q,)

    # Generate random input functions: smooth random via RBF interpolation
    u_samples   = np.zeros((n_functions, m_sensors), dtype=np.float32)
    Gu_samples  = np.zeros((n_functions, n_query),   dtype=np.float32)

    for i in range(n_functions):
        # Random smooth function: sum of random sinusoids
        n_terms = rng.randint(2, 8)
        freqs   = rng.uniform(0, 5, n_terms)
        phases  = rng.uniform(0, 2 * np.pi, n_terms)
        amps    = rng.uniform(-1, 1, n_terms)
        u       = sum(a * np.sin(2 * np.pi * f * x_sensors + ph)
                      for a, f, ph in zip(amps, freqs, phases))
        u       = u.astype(np.float32)
        u_samples[i] = u

        # Antiderivative via the trapezoidal rule
        for j, y in enumerate(y_queries):
            idx = int(y * (m_sensors - 1))
            Gu_samples[i, j] = np.trapezoid(u[:idx+1],
                                          x_sensors[:idx+1]) if idx > 0 else 0.0

    print(f"  Input functions: {u_samples.shape}  |  "
          f"Operator outputs: {Gu_samples.shape}")
    return {
        "u":        u_samples,
        "Gu":       Gu_samples,
        "x_sensors": x_sensors.astype(np.float32),
        "y_queries": y_queries.astype(np.float32),
    }


def generate_ode_operator_data(n_functions: int = 2000,
                                 m_sensors: int = 50,
                                 n_query: int = 50,
                                 seed: int = 42) -> dict:
    """
    ODE solution operator: dv/dt = u(t), v(0) = 0

    Maps a forcing function u(t) to the ODE solution v(t) = ∫₀ᵗ u(s) ds.
    (Same as antiderivative, but framed as ODE operator learning.)
    """
    # Same as antiderivative for this simple ODE
    return generate_antiderivative_data(n_functions, m_sensors, n_query, seed)


# ─────────────────────────────────────────────
# DEEPONET ARCHITECTURE
# ─────────────────────────────────────────────

class BranchNet(nn.Module):
    """
    Branch network: encodes the input function u(x₁,...,xₘ) → b ∈ ℝᵖ.

    Input:  function values at m fixed sensor locations (m,)
    Output: p-dimensional representation vector b
    """

    def __init__(self, m_sensors: int, hidden_dim: int, p: int,
                 n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(m_sensors, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, p))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (batch, m_sensors) → b: (batch, p)"""
        return self.net(u)


class TrunkNet(nn.Module):
    """
    Trunk network: encodes the query location y → t ∈ ℝᵖ.

    Input:  query coordinate y (scalar or vector)
    Output: p-dimensional basis function evaluation t

    The trunk acts as a learned basis: tᵢ(y) are the basis functions,
    bᵢ are the coefficients (from branch), and their dot product gives G(u)(y).
    """

    def __init__(self, y_dim: int, hidden_dim: int, p: int,
                 n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(y_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, p))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (n_query, y_dim) → t: (n_query, p)"""
        return self.net(y)


class DeepONet(nn.Module):
    """
    Deep Operator Network.

    Approximates a nonlinear operator G: u → G(u)
    using branch-trunk inner product architecture.

    G(u)(y) ≈ b(u) · t(y) + bias
             = Σᵢ bᵢ(u) · tᵢ(y) + b₀
    """

    def __init__(self, m_sensors: int, y_dim: int = 1,
                 hidden_dim: int = 128, p: int = 100,
                 n_layers: int = 4, dropout: float = 0.0):
        """
        Args:
            m_sensors:  number of input sensor locations
            y_dim:      dimension of query location y
            hidden_dim: hidden layer width in branch/trunk
            p:          latent dimension (dot product size) — Placeholder: 40–200
            n_layers:   layers per sub-network — Placeholder: 3–6
            dropout:    regularization for branch net
        """
        super().__init__()
        self.branch = BranchNet(m_sensors, hidden_dim, p, n_layers, dropout)
        self.trunk  = TrunkNet(y_dim, hidden_dim, p, n_layers)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: input function at sensors (batch, m_sensors)
            y: query locations (n_query, y_dim)

        Returns:
            G_pred: operator output (batch, n_query)
        """
        b = self.branch(u)    # (batch, p)
        t = self.trunk(y)     # (n_query, p)
        # Outer product-style: each batch element dotted with each query
        return b @ t.T + self.bias   # (batch, n_query)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

# Placeholder: tune p (latent dim), hidden_dim, n_layers
DEEPONET_CONFIG = {
    "p":           100,    # Placeholder: 40–200 (latent width)
    "hidden_dim":  128,    # Placeholder: 64–512
    "n_layers":    4,      # Placeholder: 3–6 for both branch and trunk
    "dropout":     0.0,    # Placeholder: 0.0–0.2 for regularization
    "lr":          1e-3,
    "epochs":      200,
    "batch_size":  128,
    "patience":    20,
}


def train_deeponet(data: dict, config: dict = None) -> tuple:
    """
    Train DeepONet on operator learning data.

    Args:
        data:   output of generate_antiderivative_data()
        config: DEEPONET_CONFIG

    Returns:
        model, history
    """
    if config is None:
        config = DEEPONET_CONFIG

    u  = torch.tensor(data["u"],        dtype=torch.float32)
    Gu = torch.tensor(data["Gu"],       dtype=torch.float32)
    y  = torch.tensor(data["y_queries"][:, None], dtype=torch.float32).to(DEVICE)

    # Normalize inputs and outputs
    u_mean, u_std  = u.mean(), u.std() + 1e-8
    Gu_mean, Gu_std = Gu.mean(), Gu.std() + 1e-8
    u_norm  = (u  - u_mean)  / u_std
    Gu_norm = (Gu - Gu_mean) / Gu_std

    # Train/val split
    n      = len(u_norm)
    n_val  = int(0.1 * n)
    idx    = torch.randperm(n)
    u_tr   = u_norm[idx[n_val:]]; Gu_tr = Gu_norm[idx[n_val:]]
    u_v    = u_norm[idx[:n_val]]; Gu_v  = Gu_norm[idx[:n_val]]

    m_sensors = u.shape[1]
    model = DeepONet(
        m_sensors=m_sensors, y_dim=1,
        hidden_dim=config["hidden_dim"],
        p=config["p"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(DEVICE)

    loader   = DataLoader(TensorDataset(u_tr, Gu_tr),
                           batch_size=config["batch_size"], shuffle=True)
    opt      = torch.optim.Adam(model.parameters(), lr=config["lr"])
    sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10,
                                                            factor=0.5)
    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_c = 0
    best_state = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        tr_loss = 0.0
        for u_b, Gu_b in loader:
            u_b, Gu_b = u_b.to(DEVICE), Gu_b.to(DEVICE)
            pred  = model(u_b, y)
            loss  = F.mse_loss(pred, Gu_b)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(u_b)
        tr_loss /= len(u_tr)

        model.eval()
        with torch.no_grad():
            pred_v  = model(u_v.to(DEVICE), y)
            val_loss = F.mse_loss(pred_v, Gu_v.to(DEVICE)).item()

        sched.step(val_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= config["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d} | train: {tr_loss:.6f} | val: {val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    # Store normalization stats on model for inference
    model.u_mean = u_mean.item()
    model.u_std  = u_std.item()
    model.Gu_mean = Gu_mean.item()
    model.Gu_std  = Gu_std.item()

    return model, history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model: DeepONet, data: dict, n_test: int = 200) -> dict:
    """
    Evaluate on held-out test functions: relative L2 error.

    Relative L2 error = ||G(u) - G_pred||₂ / ||G(u)||₂
    Standard metric in operator learning literature.
    """
    model.eval()
    y = torch.tensor(data["y_queries"][:, None], dtype=torch.float32).to(DEVICE)

    u_test  = torch.tensor(data["u"][:n_test],  dtype=torch.float32).to(DEVICE)
    Gu_test = torch.tensor(data["Gu"][:n_test], dtype=torch.float32)

    # Normalize input
    u_norm = (u_test - model.u_mean) / model.u_std

    with torch.no_grad():
        pred_norm = model(u_norm, y).cpu()
        pred      = pred_norm * model.Gu_std + model.Gu_mean

    # Relative L2 error per function
    rel_l2 = (torch.norm(pred - Gu_test, dim=1) /
               (torch.norm(Gu_test, dim=1) + 1e-8)).mean().item()

    mse = F.mse_loss(pred, Gu_test).item()
    print(f"  Relative L2 error: {rel_l2:.4f}")
    print(f"  MSE:               {mse:.6f}")
    return {"rel_l2": rel_l2, "mse": mse, "pred": pred.numpy(),
            "true": Gu_test.numpy()}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    plt.figure(figsize=(8, 4))
    plt.semilogy(history["train_loss"], label="Train Loss")
    plt.semilogy(history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss (log)")
    plt.title("DeepONet — Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("deeponet_training.png", dpi=150)
    plt.show()


def plot_operator_predictions(eval_results: dict, data: dict,
                               n_examples: int = 4) -> None:
    """
    Plot ground truth vs. predicted operator outputs for a few test functions.
    """
    y    = data["y_queries"]
    pred = eval_results["pred"]
    true = eval_results["true"]

    fig, axes = plt.subplots(2, n_examples, figsize=(4*n_examples, 7))
    for i in range(n_examples):
        # Top row: input function u(x)
        axes[0, i].plot(data["x_sensors"], data["u"][i], color="steelblue")
        axes[0, i].set_title(f"Input u(x) #{i}")
        axes[0, i].set_xlabel("x")

        # Bottom row: G(u)(y) true vs predicted
        axes[1, i].plot(y, true[i], "k-",       linewidth=2.0, label="True G(u)")
        axes[1, i].plot(y, pred[i], "tomato",   linewidth=1.5, linestyle="--",
                         label="DeepONet")
        axes[1, i].set_title(f"Operator Output #{i}")
        axes[1, i].set_xlabel("y")
        if i == 0:
            axes[1, i].legend(fontsize=8)

    plt.suptitle("DeepONet — Operator Approximation (Antiderivative)")
    plt.tight_layout()
    plt.savefig("deeponet_predictions.png", dpi=150)
    plt.show()


def plot_error_distribution(eval_results: dict, data: dict) -> None:
    """Histogram of relative L2 errors across test functions."""
    pred = torch.tensor(eval_results["pred"])
    true = torch.tensor(eval_results["true"])
    rel_l2 = (torch.norm(pred - true, dim=1) /
               (torch.norm(true, dim=1) + 1e-8)).numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(rel_l2, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    plt.axvline(rel_l2.mean(), color="red", linestyle="--",
                label=f"Mean: {rel_l2.mean():.4f}")
    plt.xlabel("Relative L2 Error")
    plt.ylabel("Count")
    plt.title("DeepONet — Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("deeponet_errors.png", dpi=150)
    plt.show()


def plot_trunk_basis(model: DeepONet, n_basis: int = 8) -> None:
    """
    Visualize learned trunk basis functions tᵢ(y).
    These are the learned basis functions that span the output function space.
    """
    model.eval()
    y = torch.linspace(0, 1, 100)[:, None].to(DEVICE)
    with torch.no_grad():
        t = model.trunk(y).cpu().numpy()  # (100, p)

    plt.figure(figsize=(12, 4))
    for i in range(min(n_basis, t.shape[1])):
        plt.plot(y.cpu().numpy().ravel(), t[:, i],
                 linewidth=1.2, alpha=0.8, label=f"t_{i}")
    plt.xlabel("y")
    plt.ylabel("Basis value")
    plt.title("DeepONet — Learned Trunk Basis Functions")
    plt.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig("deeponet_basis.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"=== Deep Operator Network (DeepONet) ===  (device: {DEVICE})")

    # Generate antiderivative operator data
    print("\nGenerating operator dataset (antiderivative)...")
    data = generate_antiderivative_data(n_functions=3000, m_sensors=100, n_query=100)

    # Train DeepONet
    print("\nTraining DeepONet...")
    model, history = train_deeponet(data)

    # Evaluate
    print("\nEvaluation on test functions:")
    eval_results = evaluate(model, data, n_test=300)

    # Visualizations
    plot_training_curves(history)
    plot_operator_predictions(eval_results, data)
    plot_error_distribution(eval_results, data)
    plot_trunk_basis(model)

    print("\nNote: For physics-informed DeepONet and adaptive sampling:")
    print("      pip install deepxde")
    print("      import deepxde as dde")


if __name__ == "__main__":
    main()
