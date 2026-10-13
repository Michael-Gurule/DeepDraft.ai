"""
Fourier Neural Operator (FNO)
==============================

FNO learns mappings between infinite-dimensional function spaces by
parameterizing the integral kernel operator in Fourier space.

Operator learning problem:
  Given input function a(x) and output function u(x) related by G: a → u,
  learn G from (a_i, u_i) pairs without solving the underlying PDE.

FNO architecture (per layer):
  v_{t+1}(x) = σ( W v_t(x) + (K(a; φ) v_t)(x) )
  where K is the Fourier integral operator:
    (K v)(x) = F^{-1}[ R_φ(k) · F[v](k) ]
  R_φ(k) ∈ ℂ^{d_v × d_v} are learned complex weights (only low modes k < k_max)

Key ideas:
  1. Global kernel: captures long-range dependencies in one layer (vs CNN's local)
  2. Resolution invariance: train at one resolution, evaluate at another
  3. Efficient: only learn n_modes low-frequency Fourier modes, discard high-freq
  4. Linear complexity in resolution: O(n log n) via FFT

Comparison with DeepONet:
  ┌──────────────────────────────────────────────────────┐
  │ Feature          │ FNO               │ DeepONet      │
  ├──────────────────────────────────────────────────────┤
  │ Architecture     │ Fourier layers    │ Branch/Trunk  │
  │ Resolution       │ Can vary          │ Fixed sensors │
  │ Periodic PDEs    │ Natural fit       │ Less natural  │
  │ Complexity       │ O(n log n)        │ O(n·p)        │
  │ 2D/3D support    │ Straightforward   │ More complex  │
  └──────────────────────────────────────────────────────┘

Fine-tuning recommendations:
  - n_modes (#Placeholder: 16): Fourier truncation modes per dimension (< n//2)
  - width (#Placeholder: 64): channel width (hidden dimension of v_t)
  - n_layers (#Placeholder: 4): FNO blocks; 4 is standard
  - learning_rate (#Placeholder: 1e-3): use cosine decay
  - batch_size (#Placeholder: 20): can be small (expensive operators)
  - weight_decay (#Placeholder: 1e-4):
  - modes2d (#Placeholder: 12): modes in each spatial dimension for 2D FNO

References:
  Li et al. (2021) — Fourier Neural Operator for Parametric PDEs
  Li et al. (2022) — Geo-FNO, U-FNO, F-FNO extensions
  Kovachki et al. (2023) — Neural Operator: Learning Maps Between Function Spaces
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CONFIG = {
    "n_modes":      16,      # # Placeholder: Fourier modes to keep (< n_grid//2)
    "width":        64,      # # Placeholder: channel width
    "n_layers":     4,       # # Placeholder: FNO blocks
    "lr":           1e-3,    # # Placeholder:
    "weight_decay": 1e-4,    # # Placeholder:
    "batch_size":   20,      # # Placeholder:
    "n_epochs":     100,     # # Placeholder:
    "n_train":      800,     # # Placeholder:
    "n_test":       200,     # # Placeholder:
    "n_grid":       64,      # # Placeholder: spatial resolution
    "seed":         42,
}


# ─────────────────────────────────────────────
# Fourier Layer (1D)
# ─────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    """
    Fourier integral operator in 1D:
      (K v)(x) = F^{-1}[ R(k) · F[v](k) ]
    where R ∈ ℂ^{in_channels × out_channels × n_modes} are learned weights.
    Only the lowest n_modes Fourier coefficients are retained.
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes      = n_modes

        # Complex weights: stored as real/imag pairs via torch.cfloat
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, n_modes, dtype=torch.cfloat)
        )

    def _compl_mul1d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Complex-valued einsum: (batch, in_ch, modes) × (in_ch, out_ch, modes)."""
        return torch.einsum("bim,iom->bom", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, n_grid) → (batch, out_channels, n_grid)
        """
        batchsize = x.shape[0]
        n = x.shape[-1]

        # FFT along spatial dimension
        x_ft = torch.fft.rfft(x, norm="ortho")                   # (B, C, n//2+1)

        # Multiply low Fourier modes by learned weights
        out_ft = torch.zeros(batchsize, self.out_channels, n // 2 + 1,
                              dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.n_modes] = self._compl_mul1d(
            x_ft[:, :, :self.n_modes], self.weights
        )

        # Inverse FFT back to spatial domain
        return torch.fft.irfft(out_ft, n=n, norm="ortho")         # (B, C, n)


class FNOBlock1d(nn.Module):
    """
    One FNO layer:
      v_{t+1} = σ( SpectralConv(v_t) + W·v_t )
    SpectralConv captures global information via Fourier modes.
    W·v_t is a local pointwise linear map (skip connection in Fourier space).
    """

    def __init__(self, width: int, n_modes: int):
        super().__init__()
        self.conv = SpectralConv1d(width, width, n_modes)
        self.w    = nn.Conv1d(width, width, kernel_size=1)   # local linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x) + self.w(x))


# ─────────────────────────────────────────────
# Fourier Layer (2D)
# ─────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """2D Fourier integral operator."""

    def __init__(self, in_channels: int, out_channels: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, n_modes1, n_modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, n_modes1, n_modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")   # (B, C, H, W//2+1)

        out_ft = torch.zeros(B, x_ft.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        # Low-frequency modes in both quadrants
        out_ft[:, :, :self.n_modes1, :self.n_modes2] = \
            self._compl_mul2d(x_ft[:, :, :self.n_modes1, :self.n_modes2], self.weights1)
        out_ft[:, :, -self.n_modes1:, :self.n_modes2] = \
            self._compl_mul2d(x_ft[:, :, -self.n_modes1:, :self.n_modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, n_modes1: int, n_modes2: int):
        super().__init__()
        self.conv = SpectralConv2d(width, width, n_modes1, n_modes2)
        self.w    = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.w(x))


# ─────────────────────────────────────────────
# Full FNO Models
# ─────────────────────────────────────────────

class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator.
    Maps a(x) → u(x) for 1D PDEs.

    Input:  (batch, n_grid, in_channels)  — function values on grid + coordinates
    Output: (batch, n_grid, out_channels) — solution values on same grid
    """

    def __init__(self, n_modes: int, width: int, n_layers: int,
                 in_channels: int = 2, out_channels: int = 1):
        super().__init__()
        self.fc0  = nn.Linear(in_channels, width)    # lift to channel dim
        self.blocks = nn.ModuleList([FNOBlock1d(width, n_modes) for _ in range(n_layers)])
        self.fc1  = nn.Linear(width, 128)
        self.fc2  = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_grid, in_channels)"""
        x = self.fc0(x)                               # (B, n, width)
        x = x.permute(0, 2, 1)                        # (B, width, n) for Conv1d
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)                        # (B, n, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x                                       # (B, n, out_channels)


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.
    Maps a(x,y) → u(x,y) for 2D PDEs (e.g., Darcy flow, Navier-Stokes).

    Input:  (batch, H, W, in_channels)
    Output: (batch, H, W, out_channels)
    """

    def __init__(self, n_modes1: int, n_modes2: int, width: int, n_layers: int,
                 in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([
            FNOBlock2d(width, n_modes1, n_modes2) for _ in range(n_layers)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, H, W, in_channels)"""
        x = self.fc0(x)                                # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)                     # (B, width, H, W)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1)                     # (B, H, W, width)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)                             # (B, H, W, out_channels)


# ─────────────────────────────────────────────
# Data Generation
# ─────────────────────────────────────────────

def generate_1d_antiderivative_data(
    n_train: int,
    n_test: int,
    n_grid: int = 64,
    seed: int = 42,
) -> dict:
    """
    1D antiderivative operator: G(a)(x) = ∫₀ˣ a(s) ds.
    Input a: random superposition of sinusoids.
    Compare with DeepONet for same task.
    Replace with real PDE data for actual operator learning.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_grid, dtype=np.float32)
    n_total = n_train + n_test

    a_data = np.zeros((n_total, n_grid), dtype=np.float32)
    u_data = np.zeros((n_total, n_grid), dtype=np.float32)

    for i in range(n_total):
        n_terms = rng.integers(2, 8)
        freqs   = rng.uniform(1, 6, n_terms)
        phases  = rng.uniform(0, 2 * np.pi, n_terms)
        amps    = rng.normal(0, 1, n_terms)
        a = sum(a * np.sin(f * np.pi * x + p) for a, f, p in zip(amps, freqs, phases))
        a_data[i] = a.astype(np.float32)
        # Antiderivative via cumulative trapezoid
        u_data[i] = np.cumsum(a) * (1.0 / n_grid)

    # Normalize targets
    u_mean = u_data.mean()
    u_std  = u_data.std() + 1e-8
    u_data = (u_data - u_mean) / u_std

    # Build FNO input: [a(x), x] as 2-channel input
    x_coord = np.tile(x[None, :], (n_total, 1))
    inputs  = np.stack([a_data, x_coord], axis=-1)  # (N, n, 2)
    outputs = u_data[:, :, None]                     # (N, n, 1)

    return {
        "x_train": torch.tensor(inputs[:n_train]),
        "y_train": torch.tensor(outputs[:n_train]),
        "x_test":  torch.tensor(inputs[n_train:]),
        "y_test":  torch.tensor(outputs[n_train:]),
        "grid":    x, "u_mean": u_mean, "u_std": u_std,
    }


def generate_1d_heat_data(
    n_train: int, n_test: int, n_grid: int = 64,
    n_t: int = 100, alpha: float = 0.01, seed: int = 42,
) -> dict:
    """
    1D heat equation operator: a(x) → u(x, T).
    Maps initial condition to solution at time T.
    ∂u/∂t = α ∂²u/∂x²  with zero Dirichlet BCs.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_grid)
    t = np.linspace(0, 0.1, n_t)
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    r = alpha * dt / dx ** 2
    n_total = n_train + n_test

    a_all, u_all = [], []
    for _ in range(n_total):
        a = rng.normal(0, 1, n_grid) * np.sin(np.pi * x * rng.integers(1, 4))
        u = a.copy()
        for _ in range(n_t):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            u_new[0] = u_new[-1] = 0.0
            u = u_new
        a_all.append(a)
        u_all.append(u)

    a_all = np.array(a_all, dtype=np.float32)
    u_all = np.array(u_all, dtype=np.float32)
    x_coord = np.tile(x[None, :], (n_total, 1)).astype(np.float32)
    inputs  = np.stack([a_all, x_coord], axis=-1)
    outputs = u_all[:, :, None]

    return {
        "x_train": torch.tensor(inputs[:n_train]),
        "y_train": torch.tensor(outputs[:n_train]),
        "x_test":  torch.tensor(inputs[n_train:]),
        "y_test":  torch.tensor(outputs[n_train:]),
        "grid": x,
    }


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Standard FNO metric: ||u - û||₂ / ||u||₂."""
    diff = (pred - target).reshape(pred.shape[0], -1)
    true = target.reshape(target.shape[0], -1)
    return float((diff.norm(dim=1) / (true.norm(dim=1) + 1e-8)).mean())


def train_fno(data: dict, cfg: dict) -> tuple[FNO1d, dict]:
    torch.manual_seed(cfg["seed"])

    train_ds = TensorDataset(data["x_train"], data["y_train"])
    test_ds  = TensorDataset(data["x_test"],  data["y_test"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False)

    model = FNO1d(
        n_modes=cfg["n_modes"], width=cfg["width"],
        n_layers=cfg["n_layers"],
        in_channels=data["x_train"].shape[-1],
        out_channels=data["y_train"].shape[-1],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["n_epochs"])

    train_losses, test_errors = [], []

    for epoch in range(cfg["n_epochs"]):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        # Test evaluation
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb.to(DEVICE))
                all_pred.append(pred.cpu()); all_true.append(yb)
        err = relative_l2_error(torch.cat(all_pred), torch.cat(all_true))
        test_errors.append(err)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['n_epochs']} | "
                  f"Train MSE: {train_losses[-1]:.5f} | Test Rel-L2: {err:.5f}")

    return model, {"train_losses": train_losses, "test_errors": test_errors}


# ─────────────────────────────────────────────
# n_modes Ablation
# ─────────────────────────────────────────────

def modes_ablation(data: dict, cfg: dict, modes_list: list = None) -> dict:
    """
    Sweep n_modes to show trade-off between expressivity and efficiency.
    FNO with too few modes misses fine structure; too many adds noise.
    """
    modes_list = modes_list or [4, 8, 16, 32]   # # Placeholder:
    results = {}
    for nm in modes_list:
        print(f"\n  n_modes={nm}")
        _, history = train_fno(data, {**cfg, "n_modes": nm, "n_epochs": 50})
        results[nm] = history["test_errors"][-1]
        print(f"    Final test Rel-L2: {results[nm]:.5f}")
    return results


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_fno_results(model: FNO1d, data: dict, n_examples: int = 4):
    """Plot predicted vs true operator outputs for a few test examples."""
    model.eval()
    x_test = data["x_test"][:n_examples].to(DEVICE)
    with torch.no_grad():
        pred = model(x_test).cpu().numpy()
    true = data["y_test"][:n_examples].numpy()
    grid = data["grid"]

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
    fig.suptitle("FNO — Predicted vs True Operator Output", fontsize=12)
    for i, ax in enumerate(axes):
        ax.plot(grid, true[i, :, 0], "k-", lw=2, label="True")
        ax.plot(grid, pred[i, :, 0], "r--", lw=1.5, label="FNO")
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("x")
        if i == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fno_predictions.png", dpi=150)
    plt.show()
    print("Saved: fno_predictions.png")


def plot_training_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("FNO Training", fontsize=12)
    axes[0].semilogy(history["train_losses"], color="steelblue")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train MSE (log)")
    axes[0].set_title("Training Loss"); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(history["test_errors"], color="darkorange")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Relative L2 (log)")
    axes[1].set_title("Test Relative L2 Error"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fno_training.png", dpi=150)
    plt.show()
    print("Saved: fno_training.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}\n")

    # ── 1D Antiderivative operator ─────────────────────────────────
    print("=== FNO — 1D Antiderivative Operator ===")
    data = generate_1d_antiderivative_data(
        n_train=CONFIG["n_train"], n_test=CONFIG["n_test"],
        n_grid=CONFIG["n_grid"], seed=CONFIG["seed"],
    )
    print(f"Train: {data['x_train'].shape} | Test: {data['x_test'].shape}")

    model, history = train_fno(data, CONFIG)
    plot_training_history(history)
    plot_fno_results(model, data)

    # ── 1D Heat equation operator ──────────────────────────────────
    print("\n=== FNO — 1D Heat Equation Operator ===")
    heat_data = generate_1d_heat_data(
        n_train=CONFIG["n_train"], n_test=CONFIG["n_test"],
        n_grid=CONFIG["n_grid"], seed=CONFIG["seed"],
    )
    heat_model, heat_history = train_fno(heat_data, CONFIG)
    plot_fno_results(heat_model, heat_data, n_examples=4)

    # ── n_modes ablation ───────────────────────────────────────────
    print("\n=== n_modes Ablation ===")
    ablation = modes_ablation(data, CONFIG, modes_list=[4, 8, 16, 32])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(k) for k in ablation.keys()], list(ablation.values()),
           color="steelblue", edgecolor="white")
    ax.set_xlabel("n_modes"); ax.set_ylabel("Test Relative L2")
    ax.set_title("FNO — n_modes Ablation")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("fno_modes_ablation.png", dpi=150)
    plt.show()

    final_err = history["test_errors"][-1]
    print(f"\nFinal test Relative L2 error: {final_err:.5f}")


if __name__ == "__main__":
    main()
