"""
Temporal Fusion Transformer (TFT)
===================================
A state-of-the-art deep learning model for multi-horizon time series forecasting.
Designed by Google (Lim et al. 2019) to handle:
    - Multiple input types: static metadata, known future inputs, observed past inputs
    - Interpretability: variable importance, attention weights, quantile forecasts

Architecture components:
    1. Input embedding layers (static, observed, known future covariates)
    2. Variable Selection Networks (VSN) — learns which inputs matter
    3. Static Enrichment — conditions dynamic features on static context
    4. LSTM encoder-decoder — processes temporal sequences
    5. Multi-head attention (temporal self-attention) — captures long-range dependencies
    6. Quantile regression output head — produces calibrated prediction intervals

Output: quantile forecasts (e.g., q10, q50, q90) over the forecast horizon.

Input Types:
    - Static (time-invariant): entity metadata (e.g., region, category)
    - Known future (exogenous, known ahead of time): day-of-week, promotions
    - Observed (historical, unknown in future): past target values, past covariates

Use Cases:
    - Multi-entity demand forecasting (many time series)
    - Marketing mix modeling + channel budget allocation
    - Financial forecasting with multiple instruments
    - Energy grid load forecasting with weather + time features
    - Retail and supply-chain forecasting with promotions as known future inputs

Why TFT vs LSTM vs Transformer:
    - Vanilla Transformer: no temporal structure inductive bias
    - LSTM: good sequences but poor at long-range dependencies + no attention interpret.
    - TFT: designed specifically for time series, handles heterogeneous inputs,
      produces quantile outputs, variable importance scores

Fine-Tuning Recommendations:
    - hidden_size: 16–256. Most impactful. Start at 64.
    - attention_head_size: 1–4 heads. Start with 1.
    - dropout: 0.1–0.3 for regularization.
    - hidden_continuous_size: 8–32 (continuous variable embedding).
    - learning_rate: 1e-3 (default). Use reduce_on_plateau callback.
    - context_length (encoder steps): how much history to look back.
    - prediction_length: your forecast horizon.
    - max_encoder_length: typically 2–5× prediction_length.
    - gradient_clip_val: 0.1 (TFT is sensitive to exploding gradients).
    - For production: use PyTorch Forecasting library which has full TFT
      implementation with TimeSeriesDataSet, interpretability tools,
      and automated hyperparameter search with Optuna.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from deepdraft.core import fast_or, save_figure

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_multivariate_series(
    n_entities:    int = 5,
    n_timesteps:   int = 500,
    n_covariates:  int = 3,
    seed:          int = 42,
) -> pd.DataFrame:
    """
    Generate multi-entity time series with static metadata and known covariates.

    Replace with your dataset using PyTorch Forecasting's TimeSeriesDataSet:
        from pytorch_forecasting import TimeSeriesDataSet
        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["entity_id"],
            static_categoricals=["category"],
            time_varying_known_reals=["month", "day_of_week"],
            time_varying_unknown_reals=["target", "covariate1"],
            max_encoder_length=60,
            max_prediction_length=30,
        )

    Returns:
        DataFrame with columns: entity_id, time_idx, target, [covariates], month, dow
    """
    rng = np.random.RandomState(seed)
    rows = []
    for eid in range(n_entities):
        base_level = rng.uniform(50, 150)
        trend      = rng.uniform(0.02, 0.1)
        amplitude  = rng.uniform(5, 20)

        for t in range(n_timesteps):
            target = (base_level + trend * t
                      + amplitude * np.sin(2 * np.pi * t / 52)
                      + rng.normal(0, 3))
            covs = rng.randn(n_covariates).tolist()
            date = pd.Timestamp("2020-01-01") + pd.Timedelta(weeks=t)
            rows.append({
                "entity_id": eid,
                "time_idx":  t,
                "date":      date,
                "target":    float(target),
                "month":     float(date.month),
                "dow":       float(date.dayofweek),
                **{f"cov_{i}": covs[i] for i in range(n_covariates)},
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────

class TimeSeriesWindowDataset(Dataset):
    """
    Sliding window dataset for multi-step forecasting.

    Each sample:
        X: (encoder_len, n_features) — past window
        y: (decoder_len,)            — future target values

    For production TFT: replace this with PyTorch Forecasting's TimeSeriesDataSet
    which handles multiple entities, static features, and known future covariates
    properly.
    """

    def __init__(self, series: np.ndarray, encoder_len: int, decoder_len: int):
        self.series      = series.astype(np.float32)
        self.enc_len     = encoder_len
        self.dec_len     = decoder_len
        self.n_features  = series.shape[1]

    def __len__(self):
        # Number of complete (encoder, decoder) windows; never negative.
        return max(0, len(self.series) - self.enc_len - self.dec_len + 1)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.enc_len]
        y = self.series[idx + self.enc_len: idx + self.enc_len + self.dec_len, 0]
        return torch.tensor(x), torch.tensor(y)


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

# Placeholder: tune hidden_size, n_heads, dropout, encoder_len, decoder_len
TFT_CONFIG = {
    "encoder_len":         52,    # Placeholder: look-back window (timesteps)
    "decoder_len":         12,    # Placeholder: forecast horizon
    "hidden_size":         64,    # Placeholder: 16–256
    "n_heads":             4,     # Placeholder: 1–4 attention heads
    "dropout":             0.1,   # Placeholder: 0.0–0.3
    "n_quantiles":         3,     # q10, q50, q90
    "lr":                  1e-3,  # Placeholder: 1e-4 to 1e-2
    "batch_size":          32,    # Placeholder: 16–128
    "epochs":              fast_or(50, 2),  # Placeholder: 20–200
    "patience":            10,
    "quantiles":           [0.1, 0.5, 0.9],
}


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) — core TFT building block.

    Applies a dense layer with gating mechanism:
        output = LayerNorm(x + GLU(ELU(W1 x + b1), W2 x + b2))

    Used throughout TFT for variable selection and feature processing.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int = None, dropout: float = 0.0):
        super().__init__()
        if output_size is None:
            output_size = hidden_size

        self.fc1      = nn.Linear(input_size, hidden_size)
        self.fc2      = nn.Linear(hidden_size, output_size)
        self.gate_fc  = nn.Linear(input_size, output_size)
        self.norm     = nn.LayerNorm(output_size)
        self.dropout  = nn.Dropout(dropout)

        # Skip connection if dims differ
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h        = F.elu(self.fc1(x))
        h        = self.dropout(h)
        h        = self.fc2(h)
        gate     = torch.sigmoid(self.gate_fc(x))
        h        = h * gate
        return self.norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network — learns which input variables matter.

    Produces softmax weights over input features and applies GRN to each.
    Returns weighted sum of per-variable transformed representations.
    """

    def __init__(self, n_features: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.grn_flat = GatedResidualNetwork(n_features * hidden_size, hidden_size,
                                              n_features, dropout)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(n_features)
        ])
        self.fc_embed = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(n_features)
        ])
        self.n_features = n_features
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, n_features) or (batch, time, n_features)

        Returns:
            out: (batch, hidden_size) or (batch, time, hidden_size) weighted sum
            weights: (batch, n_features) attention weights
        """
        if x.dim() == 3:
            B, T, F = x.shape
            # Embed each feature individually
            xi   = [self.var_grns[i](x[:, :, i:i+1].reshape(B*T, 1)).reshape(B, T, -1)
                    for i in range(F)]
            flat = torch.cat([e.reshape(B*T, -1) for e in xi], dim=-1)
            w    = torch.softmax(self.grn_flat(flat), dim=-1).reshape(B, T, F)
            out  = sum(w[:, :, i:i+1] * xi[i] for i in range(F))
        else:
            B, F = x.shape
            xi   = [self.var_grns[i](x[:, i:i+1]) for i in range(F)]
            flat = torch.cat(xi, dim=-1)
            w    = torch.softmax(self.grn_flat(flat), dim=-1)
            out  = sum(w[:, i:i+1] * xi[i] for i in range(F))
        return out, w


class SimplifiedTFT(nn.Module):
    """
    Simplified Temporal Fusion Transformer for multi-step forecasting.

    This implements the core TFT architecture components:
        - Variable selection for encoder inputs
        - LSTM encoder + decoder
        - Multi-head self-attention (temporal)
        - Quantile output head

    For full production TFT with all features (static enrichment, known future
    inputs, interpretability outputs), use the PyTorch Forecasting library:
        pip install pytorch-forecasting
        from pytorch_forecasting import TemporalFusionTransformer

    Args:
        n_features:   number of input features (including target)
        encoder_len:  encoder time steps
        decoder_len:  forecast horizon
        hidden_size:  model width
        n_heads:      attention heads
        n_quantiles:  output quantiles
        dropout:      dropout rate
    """

    def __init__(self, n_features: int, encoder_len: int, decoder_len: int,
                 hidden_size: int = 64, n_heads: int = 4, n_quantiles: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.hidden_size = hidden_size
        self.n_quantiles = n_quantiles

        # Variable selection
        self.vsn_encoder = VariableSelectionNetwork(n_features, hidden_size, dropout)

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Temporal self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # GRN post-attention
        self.post_attn_grn = GatedResidualNetwork(hidden_size, hidden_size,
                                                    hidden_size, dropout)

        # Quantile output: decoder_len × n_quantiles
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, decoder_len * n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, encoder_len, n_features)

        Returns:
            quantiles: (batch, decoder_len, n_quantiles)
        """
        # Variable selection
        vsn_out, var_weights = self.vsn_encoder(x)  # (B, T, hidden)

        # LSTM encoder
        lstm_out, (h_n, c_n) = self.lstm_encoder(vsn_out)  # (B, T, hidden)

        # Temporal self-attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_norm(attn_out + lstm_out)  # residual + norm

        # GRN post-attention
        enriched = self.post_attn_grn(attn_out)  # (B, T, hidden)

        # Pool over time → decode
        context = enriched.mean(dim=1)           # (B, hidden)
        out     = self.output_head(context)       # (B, dec_len * n_quantiles)
        return out.view(-1, self.decoder_len, self.n_quantiles)


# ─────────────────────────────────────────────
# LOSS: QUANTILE REGRESSION
# ─────────────────────────────────────────────

def quantile_loss(preds: torch.Tensor, targets: torch.Tensor,
                   quantiles: list) -> torch.Tensor:
    """
    Pinball / quantile regression loss.

    QL(q, y, ŷ) = q · max(y - ŷ, 0) + (1-q) · max(ŷ - y, 0)

    Minimized by the q-th quantile of y's conditional distribution.
    """
    loss = torch.tensor(0.0, device=preds.device)
    for i, q in enumerate(quantiles):
        e    = targets - preds[:, :, i]
        loss += (q * e.clamp(min=0) + (1 - q) * (-e).clamp(min=0)).mean()
    return loss / len(quantiles)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_tft(df: pd.DataFrame, entity_id: int = 0,
               config: dict = None) -> tuple:
    """
    Train TFT on a single entity's time series.

    For multi-entity training, concatenate entity DataLoaders or use
    PyTorch Forecasting's TimeSeriesDataSet.

    Args:
        df:        DataFrame with entity_id, time_idx, target, covariates
        entity_id: which entity to train on
        config:    TFT_CONFIG dict

    Returns:
        model, history
    """
    if config is None:
        config = TFT_CONFIG

    # Extract single entity
    entity_df = df[df["entity_id"] == entity_id].sort_values("time_idx")
    feature_cols = ["target", "month", "dow"] + [c for c in entity_df.columns
                                                    if c.startswith("cov_")]
    series = entity_df[feature_cols].values.astype(np.float32)

    # Normalize
    mean, std = series.mean(axis=0), series.std(axis=0) + 1e-8
    series_norm = (series - mean) / std

    # Split
    enc, dec = config["encoder_len"], config["decoder_len"]
    split    = int(0.8 * len(series_norm))
    train_ds = TimeSeriesWindowDataset(series_norm[:split], enc, dec)
    # Start the validation slice enc steps early so its first window can use
    # training history as context; every validation target lies after split.
    val_ds   = TimeSeriesWindowDataset(series_norm[split - enc:], enc, dec)

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=config["batch_size"])

    n_features = series_norm.shape[1]
    model = SimplifiedTFT(
        n_features=n_features,
        encoder_len=enc,
        decoder_len=dec,
        hidden_size=config["hidden_size"],
        n_heads=config["n_heads"],
        n_quantiles=config["n_quantiles"],
        dropout=config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    quantiles = config["quantiles"]

    history    = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    patience_c = 0
    best_state = None

    for epoch in range(1, config["epochs"] + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            preds     = model(X_b)
            loss      = quantile_loss(preds, y_b, quantiles)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_dl), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                preds     = model(X_b)
                val_loss += quantile_loss(preds, y_b, quantiles).item()
        val_loss /= max(len(val_dl), 1)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
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

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, history, mean, std, feature_cols


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_forecasts(model: nn.Module, df: pd.DataFrame,
                        entity_id: int, mean: np.ndarray,
                        std: np.ndarray, feature_cols: list,
                        config: dict = None) -> dict:
    """
    Evaluate quantile forecasts: RMSE on median (q50), coverage of q10-q90 interval.
    """
    if config is None:
        config = TFT_CONFIG

    entity_df  = df[df["entity_id"] == entity_id].sort_values("time_idx")
    series     = entity_df[feature_cols].values.astype(np.float32)
    series_n   = (series - mean) / std

    enc, dec = config["encoder_len"], config["decoder_len"]
    ds   = TimeSeriesWindowDataset(series_n, enc, dec)
    dl   = DataLoader(ds, batch_size=64)

    preds_q  = []
    true_q   = []
    model.eval()
    with torch.no_grad():
        for X_b, y_b in dl:
            p = model(X_b.to(DEVICE)).cpu().numpy()
            preds_q.append(p)
            true_q.append(y_b.numpy())

    preds_q = np.concatenate(preds_q, axis=0)   # (n, dec, 3)
    true_q  = np.concatenate(true_q,  axis=0)   # (n, dec)

    # Denormalize target
    target_mean = mean[0]
    target_std  = std[0]
    pred_median = preds_q[:, :, 1] * target_std + target_mean
    true_denorm = true_q           * target_std + target_mean
    lower       = preds_q[:, :, 0] * target_std + target_mean
    upper       = preds_q[:, :, 2] * target_std + target_mean

    rmse     = np.sqrt(np.mean((pred_median - true_denorm) ** 2))
    mae      = np.mean(np.abs(pred_median - true_denorm))
    coverage = np.mean((true_denorm >= lower) & (true_denorm <= upper))

    print(f"  RMSE (q50):      {rmse:.4f}")
    print(f"  MAE (q50):       {mae:.4f}")
    print(f"  Q10-Q90 Coverage:{coverage*100:.1f}%  (target: ~80%)")
    return {"rmse": rmse, "mae": mae, "coverage": coverage}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_curve(history: dict) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train (Quantile Loss)")
    plt.plot(history["val_loss"],   label="Val (Quantile Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Quantile Loss")
    plt.title("TFT — Training Curve")
    plt.legend()
    plt.tight_layout()
    save_figure("tft_training.png", dpi=150)


def plot_forecast_quantiles(model: nn.Module, df: pd.DataFrame,
                             entity_id: int, mean: np.ndarray,
                             std: np.ndarray, feature_cols: list,
                             config: dict = None, n_examples: int = 3) -> None:
    """
    Plot a few example forecasts with quantile uncertainty bands.
    """
    if config is None:
        config = TFT_CONFIG

    entity_df = df[df["entity_id"] == entity_id].sort_values("time_idx")
    series    = entity_df[feature_cols].values.astype(np.float32)
    series_n  = (series - mean) / std

    enc, dec = config["encoder_len"], config["decoder_len"]
    ds = TimeSeriesWindowDataset(series_n, enc, dec)

    model.eval()
    fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 4), sharey=True)

    indices = np.linspace(len(ds) // 4, 3 * len(ds) // 4, n_examples, dtype=int)
    for ax, idx in zip(axes, indices):
        X, y_true = ds[idx]
        with torch.no_grad():
            preds = model(X.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]  # (dec, 3)

        t_std  = std[0]
        t_mean = mean[0]
        y_true_d = y_true.numpy() * t_std + t_mean
        p_lo     = preds[:, 0] * t_std + t_mean
        p_med    = preds[:, 1] * t_std + t_mean
        p_hi     = preds[:, 2] * t_std + t_mean

        past = (X[:, 0].numpy() * t_std + t_mean)
        ax.plot(range(-enc, 0), past, color="steelblue", linewidth=1.0, label="History")
        ax.plot(range(dec),     y_true_d, "k--", linewidth=1.0, label="True")
        ax.plot(range(dec),     p_med,    "tomato", linewidth=1.5, label="q50")
        ax.fill_between(range(dec), p_lo, p_hi, alpha=0.3, color="tomato", label="q10–q90")
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Sample {idx}")
        ax.set_xlabel("Steps")
        if ax == axes[0]:
            ax.legend(fontsize=7)

    plt.suptitle(f"TFT — Quantile Forecasts (Entity {entity_id})")
    plt.tight_layout()
    save_figure("tft_forecasts.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"=== Temporal Fusion Transformer ===  (device: {DEVICE})")

    # Placeholder: replace with your multi-entity time series dataset
    df = generate_multivariate_series(n_entities=5, n_timesteps=300, n_covariates=3)
    print(f"Dataset: {df['entity_id'].nunique()} entities × {df['time_idx'].max()+1} timesteps")

    # Train on entity 0
    print("\nTraining on entity 0...")
    model, history, mean, std, feature_cols = train_tft(df, entity_id=0)

    print("\nEvaluation:")
    evaluate_forecasts(model, df, entity_id=0, mean=mean, std=std,
                        feature_cols=feature_cols)

    # Visualizations
    plot_training_curve(history)
    plot_forecast_quantiles(model, df, entity_id=0, mean=mean,
                             std=std, feature_cols=feature_cols)

    print("\nNote: For production multi-entity TFT with full interpretability,")
    print("      use the PyTorch Forecasting library:")
    print("      pip install pytorch-forecasting")
    print("      from pytorch_forecasting import TemporalFusionTransformer")


if __name__ == "__main__":
    main()
