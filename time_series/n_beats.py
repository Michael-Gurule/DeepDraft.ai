"""
N-BEATS and N-HiTS: Neural Basis Expansion for Time Series Forecasting
======================================================================

N-BEATS (Oreshkin et al., 2020) decomposes forecasting into stacked blocks,
each producing a backcast (what the block "explains") and a forecast via
learned or structured basis expansions.

Architecture:
    - Block: FC stack → θ_b (backcast coeffs) + θ_f (forecast coeffs)
    - Basis: g_b(θ_b) → backcast, g_f(θ_f) → forecast
    - Stack: multiple blocks of same type with residual (doubly-residual) connections
    - N-BEATS: generic stacks (fully learned) OR trend + seasonality stacks (interpretable)

Basis types:
    - Generic:     g(θ) = Vθ, where V is a learned projection matrix
    - Trend:       polynomial basis T(t) = [1, t/H, (t/H)², ..., (t/H)^p]
    - Seasonality: Fourier basis S(t) = [1, cos(2πt/H), sin(2πt/H), ..., cos(2πKt/H), sin(2πKt/H)]

Loss:
    - sMAPE: 2|y-ŷ| / (|y| + |ŷ| + ε)  (scale-invariant)
    - MASE:  MAE(ŷ) / MAE(naïve seasonal)

N-HiTS (Challu et al., 2022) extends N-BEATS with:
    - Multi-rate sampling: each stack samples input at different resolution
    - Hierarchical interpolation: forecast upsampled via linear interp
    - Expressivity vs. computation tradeoff via pooling

Fine-tuning recommendations:
    # Placeholder: lookback_multiplier=5 (try 2–7×forecast_horizon)
    # Placeholder: n_blocks_per_stack=3 (try 1–5)
    # Placeholder: hidden_units=512 (try 128–1024)
    # Placeholder: n_layers=4 (try 3–6)
    # Placeholder: dropout=0.0 (try 0.1–0.3 for noisy series)
    # Placeholder: trend_poly_degree=3 (try 2–5)
    # Placeholder: seasonality_harmonics=None (defaults to H//2; reduce for speed)
    # Placeholder: nhits_pool_sizes=[8,4,1] (coarse to fine; adjust per horizon)
    # Placeholder: batch_size=256 (try 64–1024)
    # Placeholder: lr=1e-3 (try 5e-4 to 2e-3; cosine annealing helps)
    # Placeholder: max_epochs=100 (use early stopping on validation sMAPE)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# CONFIG — replace with your values
# ---------------------------------------------------------------------------
CONFIG = {
    "lookback": 168,          # Placeholder: input window length (e.g. 7×24 for hourly)
    "horizon": 24,            # Placeholder: forecast steps ahead
    "n_stacks": 2,            # Placeholder: number of stacks
    "n_blocks": 3,            # Placeholder: blocks per stack
    "hidden": 512,            # Placeholder: FC hidden units
    "n_layers": 4,            # Placeholder: FC layers per block
    "dropout": 0.0,           # Placeholder: dropout probability
    "poly_degree": 3,         # Placeholder: trend polynomial degree
    "n_harmonics": 12,        # Placeholder: seasonality Fourier harmonics
    "lr": 1e-3,               # Placeholder: learning rate
    "batch_size": 256,        # Placeholder: mini-batch size
    "max_epochs": 50,         # Placeholder: training epochs
    "patience": 10,           # Placeholder: early stopping patience
    "val_fraction": 0.1,      # Placeholder: fraction of series for validation
    "nhits_pool_sizes": [8, 4, 1],  # Placeholder: N-HiTS pooling per stack
    "device": "cpu",          # Placeholder: "cuda" if GPU available
}

# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

class GenericBasis(nn.Module):
    """Fully-learned linear projection for backcast and forecast."""

    def __init__(self, theta_dim: int, backcast_len: int, forecast_len: int):
        super().__init__()
        self.backcast_basis = nn.Linear(theta_dim, backcast_len, bias=False)
        self.forecast_basis = nn.Linear(theta_dim, forecast_len, bias=False)

    def forward(self, theta_b: torch.Tensor, theta_f: torch.Tensor):
        return self.backcast_basis(theta_b), self.forecast_basis(theta_f)


class TrendBasis(nn.Module):
    """
    Polynomial trend basis.
    g(θ, t) = Σ_{p=0}^{P} θ_p * (t/T)^p
    V[i,p] = (i/T)^p,  i = 0..T-1
    """

    def __init__(self, theta_dim: int, backcast_len: int, forecast_len: int,
                 poly_degree: int = 3):
        super().__init__()
        self.poly_degree = poly_degree
        self.theta_dim = theta_dim

        # Fixed polynomial bases — registered as buffers (not trained)
        t_b = torch.arange(backcast_len, dtype=torch.float32) / backcast_len
        t_f = torch.arange(forecast_len, dtype=torch.float32) / forecast_len
        B_b = torch.stack([t_b ** p for p in range(poly_degree + 1)], dim=0)  # (P+1, L)
        B_f = torch.stack([t_f ** p for p in range(poly_degree + 1)], dim=0)  # (P+1, H)
        self.register_buffer("B_b", B_b)
        self.register_buffer("B_f", B_f)

        # theta_dim must equal poly_degree+1 for trend
        assert theta_dim == poly_degree + 1, (
            f"TrendBasis requires theta_dim == poly_degree+1, got {theta_dim} vs {poly_degree+1}"
        )

    def forward(self, theta_b: torch.Tensor, theta_f: torch.Tensor):
        # theta: (batch, P+1)  B: (P+1, T)  → (batch, T)
        backcast = theta_b @ self.B_b
        forecast = theta_f @ self.B_f
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Fourier seasonality basis.
    g(θ, t) = θ_0 + Σ_{k=1}^{K} [θ_{2k-1} cos(2πkt/H) + θ_{2k} sin(2πkt/H)]
    theta_dim = 2K + 1
    """

    def __init__(self, theta_dim: int, backcast_len: int, forecast_len: int):
        super().__init__()
        assert theta_dim % 2 == 1, "SeasonalityBasis theta_dim must be odd (2K+1)"
        n_harmonics = (theta_dim - 1) // 2

        t_b = torch.arange(backcast_len, dtype=torch.float32)
        t_f = torch.arange(forecast_len, dtype=torch.float32)

        def fourier_matrix(t, K):
            cols = [torch.ones(len(t))]
            for k in range(1, K + 1):
                cols.append(torch.cos(2 * math.pi * k * t / forecast_len))
                cols.append(torch.sin(2 * math.pi * k * t / forecast_len))
            return torch.stack(cols, dim=0)  # (2K+1, T)

        self.register_buffer("B_b", fourier_matrix(t_b, n_harmonics))
        self.register_buffer("B_f", fourier_matrix(t_f, n_harmonics))

    def forward(self, theta_b: torch.Tensor, theta_f: torch.Tensor):
        backcast = theta_b @ self.B_b
        forecast = theta_f @ self.B_f
        return backcast, forecast


# ---------------------------------------------------------------------------
# N-BEATS block
# ---------------------------------------------------------------------------

class NBeatsBlock(nn.Module):
    """
    Single N-BEATS block.

    FC stack → split → θ_b (backcast) + θ_f (forecast) → basis expansion.
    """

    def __init__(
        self,
        backcast_len: int,
        forecast_len: int,
        hidden: int,
        n_layers: int,
        basis: nn.Module,
        theta_b_dim: int,
        theta_f_dim: int,
        dropout: float = 0.0,
        share_weights: bool = False,
    ):
        super().__init__()
        self.basis = basis

        layers: List[nn.Module] = []
        in_dim = backcast_len
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.fc_stack = nn.Sequential(*layers)

        self.theta_b_proj = nn.Linear(hidden, theta_b_dim)
        if share_weights:
            self.theta_f_proj = self.theta_b_proj
        else:
            self.theta_f_proj = nn.Linear(hidden, theta_f_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc_stack(x)
        theta_b = self.theta_b_proj(h)
        theta_f = self.theta_f_proj(h)
        backcast, forecast = self.basis(theta_b, theta_f)
        return backcast, forecast


# ---------------------------------------------------------------------------
# N-BEATS stack (multiple blocks of same type)
# ---------------------------------------------------------------------------

class NBeatsStack(nn.Module):
    """
    Stack of N-BEATS blocks of the same type.
    Doubly-residual connections: each block receives (input - sum_of_backcasts).
    Forecasts are summed across blocks.
    """

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        stack_forecast = torch.zeros(x.shape[0], self.blocks[0].theta_f_proj.out_features
                                     if hasattr(self.blocks[0], '_forecast_len')
                                     else _infer_forecast_len(self.blocks[0]),
                                     device=x.device)
        # Recompute from basis output shapes
        forecast_accum = None
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            if forecast_accum is None:
                forecast_accum = forecast
            else:
                forecast_accum = forecast_accum + forecast
        return residual, forecast_accum


def _infer_forecast_len(block: NBeatsBlock) -> int:
    """Infer forecast length from block's basis output."""
    # Inspect basis buffer shapes
    if hasattr(block.basis, 'B_f'):
        return block.basis.B_f.shape[-1]
    return block.theta_f_proj.out_features


# ---------------------------------------------------------------------------
# N-BEATS model (interpretable and generic)
# ---------------------------------------------------------------------------

class NBeats(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series.

    mode='interpretable': trend stack + seasonality stack with fixed bases.
    mode='generic':       all stacks use learned generic bases.
    """

    def __init__(
        self,
        backcast_len: int,
        forecast_len: int,
        mode: str = "interpretable",
        n_stacks: int = 2,
        n_blocks: int = 3,
        hidden: int = 512,
        n_layers: int = 4,
        poly_degree: int = 3,
        n_harmonics: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert mode in ("interpretable", "generic")
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len

        if n_harmonics is None:
            n_harmonics = forecast_len // 2  # Placeholder: Nyquist default

        if mode == "interpretable":
            stacks = self._build_interpretable_stacks(
                backcast_len, forecast_len, n_blocks, hidden, n_layers,
                poly_degree, n_harmonics, dropout
            )
        else:
            stacks = self._build_generic_stacks(
                backcast_len, forecast_len, n_stacks, n_blocks, hidden,
                n_layers, dropout
            )

        self.stacks = nn.ModuleList(stacks)

    @staticmethod
    def _build_interpretable_stacks(
        L, H, n_blocks, hidden, n_layers, poly_degree, n_harmonics, dropout
    ) -> List[NBeatsStack]:
        theta_trend = poly_degree + 1
        theta_season = 2 * n_harmonics + 1

        trend_blocks = nn.ModuleList([
            NBeatsBlock(
                L, H, hidden, n_layers,
                TrendBasis(theta_trend, L, H, poly_degree),
                theta_trend, theta_trend, dropout
            )
            for _ in range(n_blocks)
        ])
        season_blocks = nn.ModuleList([
            NBeatsBlock(
                L, H, hidden, n_layers,
                SeasonalityBasis(theta_season, L, H),
                theta_season, theta_season, dropout
            )
            for _ in range(n_blocks)
        ])
        return [NBeatsStack(trend_blocks), NBeatsStack(season_blocks)]

    @staticmethod
    def _build_generic_stacks(
        L, H, n_stacks, n_blocks, hidden, n_layers, dropout
    ) -> List[NBeatsStack]:
        theta_dim = max(H, L // 2)  # Placeholder: generic theta size
        stacks = []
        for _ in range(n_stacks):
            blocks = nn.ModuleList([
                NBeatsBlock(
                    L, H, hidden, n_layers,
                    GenericBasis(theta_dim, L, H),
                    theta_dim, theta_dim, dropout
                )
                for _ in range(n_blocks)
            ])
            stacks.append(NBeatsStack(blocks))
        return stacks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, backcast_len)
        residual = x
        forecast_total = None
        for stack in self.stacks:
            residual, forecast = stack(residual)
            if forecast_total is None:
                forecast_total = forecast
            else:
                forecast_total = forecast_total + forecast
        return forecast_total  # (batch, forecast_len)

    def forward_decomposed(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return per-stack forecasts (trend, seasonality, ...) for interpretability."""
        residual = x
        components = []
        for stack in self.stacks:
            residual, forecast = stack(residual)
            components.append(forecast)
        return components


# ---------------------------------------------------------------------------
# N-HiTS: Hierarchical Interpolation for Time Series Forecasting
# ---------------------------------------------------------------------------

class NHiTSBlock(nn.Module):
    """
    N-HiTS block with multi-rate pooling.

    pool_size controls temporal resolution:
      pool_size=1: full resolution (like N-BEATS)
      pool_size=k: AvgPool1d with stride k (coarser representation)

    Output forecast is upsampled back to horizon length via linear interpolation.
    """

    def __init__(
        self,
        backcast_len: int,
        forecast_len: int,
        hidden: int,
        n_layers: int,
        pool_size: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.pool_size = pool_size

        pooled_len = math.ceil(backcast_len / pool_size)
        expr_ratio = math.ceil(forecast_len / pool_size)

        layers: List[nn.Module] = []
        in_dim = pooled_len
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.fc_stack = nn.Sequential(*layers)

        # backcast output at original resolution
        self.backcast_proj = nn.Linear(hidden, backcast_len)
        # forecast at coarser resolution (upsampled after)
        self.forecast_proj = nn.Linear(hidden, expr_ratio)

        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, backcast_len)
        x_pooled = self.pool(x.unsqueeze(1)).squeeze(1)  # (batch, pooled_len)
        h = self.fc_stack(x_pooled)
        backcast = self.backcast_proj(h)
        forecast_coarse = self.forecast_proj(h)  # (batch, expr_ratio)
        # Upsample to forecast_len via linear interpolation
        forecast = F.interpolate(
            forecast_coarse.unsqueeze(1),
            size=self.forecast_len,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return backcast, forecast


class NHiTS(nn.Module):
    """
    N-HiTS model.

    pool_sizes: one per stack, ordered coarse-to-fine (e.g. [8, 4, 1]).
    Larger pool_size = coarser temporal scale = captures long-range patterns.
    """

    def __init__(
        self,
        backcast_len: int,
        forecast_len: int,
        pool_sizes: List[int] = None,
        n_blocks: int = 3,
        hidden: int = 512,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if pool_sizes is None:
            pool_sizes = [8, 4, 1]  # Placeholder: coarse, medium, fine
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len

        self.stacks = nn.ModuleList()
        for ps in pool_sizes:
            blocks = nn.ModuleList([
                NHiTSBlock(backcast_len, forecast_len, hidden, n_layers, ps, dropout)
                for _ in range(n_blocks)
            ])
            self.stacks.append(NBeatsStack(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        forecast_total = None
        for stack in self.stacks:
            residual, forecast = stack(residual)
            if forecast_total is None:
                forecast_total = forecast
            else:
                forecast_total = forecast_total + forecast
        return forecast_total


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def smape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric MAPE: 2|y-ŷ| / (|y| + |ŷ| + ε) ∈ [0, 2].
    Scale-invariant; good for series with varying magnitudes.
    """
    return torch.mean(2 * (pred - target).abs() / (target.abs() + pred.abs() + eps))


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Sliding-window dataset.

    Generates (lookback_window, forecast_window) pairs from a 1D series.
    """

    def __init__(self, series: np.ndarray, lookback: int, horizon: int):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.n = len(series) - lookback - horizon + 1
        assert self.n > 0, f"Series too short: {len(series)} < {lookback + horizon}"

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.lookback]
        y = self.series[idx + self.lookback: idx + self.lookback + self.horizon]
        return x, y


def create_dataloaders(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    val_fraction: float = 0.1,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader]:
    """Split series into train/val windows (temporal split, no shuffle on val)."""
    split = int(len(series) * (1 - val_fraction))
    train_ds = WindowDataset(series[:split], lookback, horizon)
    val_ds = WindowDataset(series[split - lookback:], lookback, horizon)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_windows(x: torch.Tensor, eps: float = 1e-8):
    """Instance normalisation: subtract last value, divide by std. Returns (x_norm, last, std)."""
    last = x[:, -1:].clone()       # (batch, 1)
    std = x.std(dim=1, keepdim=True).clamp(min=eps)
    return (x - last) / std, last, std


def denormalize(pred: torch.Tensor, last: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pred * std + last


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_nbeats(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    lr: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 10,
    use_smape: bool = True,
    device: str = "cpu",
    use_instance_norm: bool = True,
) -> dict:
    """
    Train N-BEATS or N-HiTS with optional instance normalisation.

    Returns dict with 'train_losses', 'val_losses', 'best_epoch'.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Placeholder: cosine annealing with warm restarts often helps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = smape_loss if use_smape else mse_loss

    best_val = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        # --- train ---
        model.train()
        ep_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if use_instance_norm:
                x_norm, last, std = normalize_windows(x)
                pred_norm = model(x_norm)
                pred = denormalize(pred_norm, last, std)
            else:
                pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Placeholder: grad clip
            optimizer.step()
            ep_loss += loss.item()
        train_losses.append(ep_loss / len(train_dl))
        scheduler.step()

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                if use_instance_norm:
                    x_norm, last, std = normalize_windows(x)
                    pred_norm = model(x_norm)
                    pred = denormalize(pred_norm, last, std)
                else:
                    pred = model(x)
                val_loss += loss_fn(pred, y).item()
        val_losses.append(val_loss / len(val_dl))

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_epoch = int(np.argmin(val_losses))
    return {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    val_dl: DataLoader,
    device: str = "cpu",
    use_instance_norm: bool = True,
) -> dict:
    """Compute MAE, RMSE, sMAPE on validation loader."""
    model.eval().to(device)
    preds, targets = [], []
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            if use_instance_norm:
                x_norm, last, std = normalize_windows(x)
                pred_norm = model(x_norm)
                pred = denormalize(pred_norm, last, std)
            else:
                pred = model(x)
            preds.append(pred.cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    mae = F.l1_loss(preds, targets).item()
    rmse = math.sqrt(F.mse_loss(preds, targets).item())
    smape = smape_loss(preds, targets).item() * 100  # percent

    print(f"\nEvaluation metrics:")
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  sMAPE: {smape:.2f}%")
    return {"mae": mae, "rmse": rmse, "smape": smape, "preds": preds, "targets": targets}


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_synthetic_series(
    n: int = 2000,
    trend_slope: float = 0.01,
    seasonality_period: int = 24,
    noise_std: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates a synthetic series with trend + seasonality + noise.

    Replace with your actual time series:
        series = pd.read_csv("data.csv")["value"].values
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = trend_slope * t
    season = np.sin(2 * math.pi * t / seasonality_period)
    noise = rng.normal(0, noise_std, n)
    return (trend + season + noise).astype(np.float32)


def load_m4_hourly_demo() -> np.ndarray:
    """
    Load M4 hourly demo series.
    Replace with actual M4 data from: https://github.com/Mcompetitions/M4-methods
        import pandas as pd
        df = pd.read_csv("Hourly-train.csv")
        series = df.iloc[0, 1:].dropna().values.astype(np.float32)
    """
    return generate_synthetic_series(n=3000, seasonality_period=24, seed=0)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_forecast(
    series: np.ndarray,
    pred: np.ndarray,
    lookback: int,
    title: str = "N-BEATS Forecast",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    n = len(series)
    horizon = len(pred)
    t_hist = np.arange(n)
    t_pred = np.arange(n - horizon, n)

    plt.figure(figsize=(12, 4))
    plt.plot(t_hist, series, label="Actual", color="steelblue")
    plt.plot(t_pred, pred, label="Forecast", color="orange", linewidth=2)
    plt.axvline(n - horizon, color="gray", linestyle="--", alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig("nbeats_forecast.png", dpi=150)
    plt.show()
    print("Saved nbeats_forecast.png")


def plot_decomposition(model: NBeats, x: torch.Tensor) -> None:
    """Plot trend and seasonality components (interpretable mode only)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not hasattr(model, "forward_decomposed"):
        print("Decomposition only available for NBeats.")
        return

    model.eval()
    with torch.no_grad():
        x_norm, last, std = normalize_windows(x[:1])
        components = model.forward_decomposed(x_norm)
        components = [denormalize(c, last, std).squeeze().numpy() for c in components]

    labels = ["Trend", "Seasonality"] + [f"Stack {i+3}" for i in range(len(components) - 2)]
    fig, axes = plt.subplots(len(components) + 1, 1, figsize=(12, 3 * (len(components) + 1)))

    total = sum(components)
    axes[0].plot(total, color="orange", label="Total forecast")
    axes[0].set_title("Total Forecast")
    axes[0].legend()

    for ax, comp, lbl in zip(axes[1:], components, labels):
        ax.plot(comp, label=lbl)
        ax.set_title(lbl)
        ax.legend()

    plt.tight_layout()
    plt.savefig("nbeats_decomposition.png", dpi=150)
    plt.show()
    print("Saved nbeats_decomposition.png")


def plot_training_curves(history: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(history["train_losses"], label="Train")
    plt.plot(history["val_losses"], label="Val")
    plt.axvline(history["best_epoch"], color="gray", linestyle="--", label="Best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("sMAPE")
    plt.legend()
    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig("nbeats_training.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Benchmark: N-BEATS vs N-HiTS vs baseline
# ---------------------------------------------------------------------------

def naive_seasonal_forecast(series: np.ndarray, horizon: int, period: int = 24) -> np.ndarray:
    """Seasonal naïve: repeat the last full period."""
    last_period = series[-period:]
    reps = math.ceil(horizon / period)
    return np.tile(last_period, reps)[:horizon]


def benchmark(
    series: np.ndarray,
    lookback: int = 168,
    horizon: int = 24,
    n_blocks: int = 3,
    hidden: int = 256,
    n_layers: int = 4,
    max_epochs: int = 30,
    device: str = "cpu",
) -> None:
    """
    Train and compare N-BEATS (interpretable), N-BEATS (generic), and N-HiTS.
    Also computes seasonal naïve baseline.
    """
    print("=" * 60)
    print(f"Benchmark: lookback={lookback}, horizon={horizon}")
    print("=" * 60)

    train_dl, val_dl = create_dataloaders(series, lookback, horizon, val_fraction=0.1)

    models = {
        "NBeats-Interp": NBeats(lookback, horizon, mode="interpretable",
                                n_blocks=n_blocks, hidden=hidden, n_layers=n_layers),
        "NBeats-Generic": NBeats(lookback, horizon, mode="generic",
                                 n_stacks=2, n_blocks=n_blocks, hidden=hidden, n_layers=n_layers),
        "NHiTS": NHiTS(lookback, horizon, pool_sizes=[8, 4, 1],
                       n_blocks=n_blocks, hidden=hidden, n_layers=n_layers),
    }

    results = {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- {name} ({n_params:,} params) ---")
        history = train_nbeats(model, train_dl, val_dl, max_epochs=max_epochs,
                               patience=5, device=device)
        metrics = evaluate(model, val_dl, device=device)
        results[name] = metrics

    # Seasonal naïve
    val_start = int(len(series) * 0.9)
    test_series = series[val_start - lookback:]
    naive_pred = naive_seasonal_forecast(test_series[:lookback], horizon)
    naive_target = test_series[lookback: lookback + horizon]
    naive_smape = float(smape_loss(
        torch.tensor(naive_pred), torch.tensor(naive_target)
    )) * 100
    results["Seasonal-Naïve"] = {"smape": naive_smape}

    print("\n" + "=" * 40)
    print(f"{'Model':<20} {'sMAPE%':>8}")
    print("-" * 30)
    for name, m in results.items():
        print(f"{name:<20} {m['smape']:>8.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = CONFIG
    device = cfg["device"]

    print("Generating synthetic series...")
    series = generate_synthetic_series(
        n=3000,
        trend_slope=0.005,
        seasonality_period=24,
        noise_std=0.15,
    )
    # Replace above with: series = load_m4_hourly_demo()

    lookback = cfg["lookback"]
    horizon = cfg["horizon"]

    train_dl, val_dl = create_dataloaders(
        series, lookback, horizon,
        val_fraction=cfg["val_fraction"],
        batch_size=cfg["batch_size"],
    )

    print(f"\nTraining N-BEATS (interpretable)...")
    model = NBeats(
        backcast_len=lookback,
        forecast_len=horizon,
        mode="interpretable",
        n_blocks=cfg["n_blocks"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
        poly_degree=cfg["poly_degree"],
        n_harmonics=cfg["n_harmonics"],
        dropout=cfg["dropout"],
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_nbeats(
        model, train_dl, val_dl,
        lr=cfg["lr"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        device=device,
    )

    metrics = evaluate(model, val_dl, device=device)
    plot_training_curves(history)

    # Show decomposed components on one window
    sample_x = torch.tensor(series[:lookback]).unsqueeze(0)
    plot_decomposition(model, sample_x)

    # Plot forecast on last window
    last_window = series[-(lookback + horizon): -horizon]
    actual_tail = series[-horizon:]
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(last_window).unsqueeze(0)
        x_norm, last_v, std = normalize_windows(x_t)
        pred_norm = model(x_norm)
        pred = denormalize(pred_norm, last_v, std).squeeze().numpy()
    plot_forecast(series[-(lookback + horizon):], pred, lookback,
                  title="N-BEATS Interpretable — Final Window Forecast")

    print("\n--- N-HiTS ---")
    nhits = NHiTS(
        backcast_len=lookback,
        forecast_len=horizon,
        pool_sizes=cfg["nhits_pool_sizes"],
        n_blocks=cfg["n_blocks"],
        hidden=cfg["hidden"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    print(f"N-HiTS parameters: {sum(p.numel() for p in nhits.parameters()):,}")
    history_h = train_nbeats(nhits, train_dl, val_dl, lr=cfg["lr"],
                              max_epochs=cfg["max_epochs"], patience=cfg["patience"],
                              device=device)
    evaluate(nhits, val_dl, device=device)

    print("\n--- Benchmark (quick, 20 epochs) ---")
    benchmark(series, lookback=lookback, horizon=horizon,
              hidden=128, max_epochs=20, device=device)


if __name__ == "__main__":
    main()
