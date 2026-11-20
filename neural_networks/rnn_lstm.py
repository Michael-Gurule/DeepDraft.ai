"""
Recurrent Neural Networks (RNN) & Long Short-Term Memory (LSTM)
================================================================
Architectures designed for sequential data. An RNN processes inputs
one step at a time, passing a hidden state forward that encodes "memory"
of past inputs. The LSTM extends this with a gating mechanism that controls
what information is remembered, forgotten, and output — solving the vanishing
gradient problem that hampers plain RNNs.

RNN hidden state update:
    h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)

LSTM gates (at each timestep t):
    Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   ← how much to forget
    Input gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   ← how much new info to store
    Cell gate:   g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) ← new candidate info
    Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   ← what to output
    Cell state:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
    Hidden state:h_t = o_t ⊙ tanh(C_t)

Use Cases:
    - Time series classification and forecasting
    - Natural language processing (sequence labeling, text classification)
    - Speech recognition (before Transformers)
    - Anomaly detection in multivariate sequences
    - Music generation

Limitations:
    - Sequential processing: cannot be parallelized across time steps (slow)
    - Long sequences (>500 steps) still struggle with information retention
    - GRU is a lighter alternative with comparable performance to LSTM
    - For most NLP tasks, Transformers have replaced LSTMs

Fine-Tuning Recommendations:
    - Hidden size: 64–512 per layer. Start with 128; increase if underfitting.
    - Num layers: 1–3 stacked LSTM layers. Add bidirectionality first before
      adding more layers.
    - Bidirectional: doubles hidden size by processing sequence in both directions.
      Critical for tasks where future context helps (classification, NER).
      Do NOT use for autoregressive generation (future is not yet known).
    - Dropout: recurrent_dropout on hidden-to-hidden connections is more
      effective than input dropout for LSTMs. In PyTorch, `dropout` param
      applies between LSTM layers (not between timesteps).
    - Sequence length: trim/pad to a consistent length. Very long sequences
      may need gradient checkpointing or Transformer architecture instead.
    - Learning rate: 1e-3 to 1e-4. LSTM training can be unstable — use
      gradient clipping (max_norm=5.0) to prevent exploding gradients.
    - Batch-first: use batch_first=True for compatibility with PyTorch conventions.
    - For time-series forecasting: normalize each feature to zero mean, unit std.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DEVICE CONFIGURATION
# ─────────────────────────────────────────────

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_classification_sequences(
    n_samples: int = 1000,
    seq_len: int = 50,
    n_features: int = 3,
    n_classes: int = 2,
):
    """
    Generate synthetic multivariate time series for classification.

    Each class is generated from a distinct combination of sinusoids with
    added noise. Mimics real-world scenarios like activity recognition or
    fault detection.

    Replace with your own data loading:
        # For CSV-based sequences (e.g., each row = one timestep):
        import pandas as pd
        df = pd.read_csv("your_sequences.csv")
        # Reshape into (n_samples, seq_len, n_features)

    Args:
        n_samples:  number of sequences
        seq_len:    length of each sequence (timesteps)
        n_features: number of features per timestep
        n_classes:  number of classes

    Returns:
        X: float32 array of shape (n_samples, seq_len, n_features)
        y: int64 class labels of shape (n_samples,)
    """
    X, y = [], []
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, seq_len)

    for _ in range(n_samples):
        cls = np.random.randint(0, n_classes)
        # Different frequency patterns per class (easily separable)
        signal = np.stack([
            (cls + 1) * np.sin(t * (f + 1)) + np.random.randn(seq_len) * 0.5
            for f in range(n_features)
        ], axis=1)
        X.append(signal)
        y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_regression_sequences(
    n_samples: int = 800,
    seq_len: int = 30,
    n_features: int = 2,
    forecast_horizon: int = 1,
):
    """
    Generate synthetic multivariate time series for sequence-to-value regression
    (e.g., predicting the next value from a window of past values).

    This is the standard "sliding window" setup for time series forecasting.

    Replace with your own data:
        # Sliding window from a real time series:
        def make_windows(series, seq_len, horizon):
            X, y = [], []
            for i in range(len(series) - seq_len - horizon + 1):
                X.append(series[i:i+seq_len])
                y.append(series[i+seq_len:i+seq_len+horizon])
            return np.array(X), np.array(y)

    Args:
        n_samples:        number of windows
        seq_len:          length of the input window (timesteps)
        n_features:       number of input features per timestep
        forecast_horizon: number of future steps to predict

    Returns:
        X: float32 (n_samples, seq_len, n_features)
        y: float32 (n_samples, forecast_horizon) targets
    """
    np.random.seed(42)
    t = np.linspace(0, 8 * np.pi, n_samples + seq_len)
    signal = np.sin(t) + 0.1 * np.cumsum(np.random.randn(len(t)))
    signal = (signal - signal.mean()) / signal.std()

    X, y = [], []
    for i in range(n_samples):
        window = np.stack([signal[i:i + seq_len]] * n_features, axis=1)
        X.append(window)
        y.append(signal[i + seq_len:i + seq_len + forecast_horizon])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    LSTM-based sequence classifier.

    Architecture:
        Input (seq_len, n_features) →
        LSTM (stacked, optional bidirectional) →
        Take last hidden state →
        Dropout →
        Linear → n_classes

    For bidirectional: last hidden state from both directions are concatenated,
    so the effective hidden size doubles.

    Args:
        input_size:    number of features per timestep
        hidden_size:   LSTM hidden state size per direction
        num_layers:    number of stacked LSTM layers
        n_classes:     number of output classes
        dropout:       dropout between LSTM layers (and on final projection)
        bidirectional: process sequence in both directions
        rnn_type:      'lstm', 'gru', or 'rnn' (choose cell type)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,        # Placeholder: try 64, 128, 256, 512
        num_layers: int = 2,           # Placeholder: 1–3 layers
        n_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,    # Placeholder: set False for causal (generative) tasks
        rnn_type: str = "lstm",        # Placeholder: 'lstm', 'gru', 'rnn'
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.rnn_type      = rnn_type

        # Select RNN cell type
        # LSTM: best for long-range dependencies (complex gating)
        # GRU:  slightly faster, fewer parameters, comparable performance
        # RNN:  simplest; struggles with long sequences (vanishing gradients)
        RNN_CLASSES = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
        rnn_class = RNN_CLASSES[rnn_type]

        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # (batch, seq, features) — PyTorch default is False
            dropout=dropout if num_layers > 1 else 0.0,  # dropout only between layers
            bidirectional=bidirectional,
        )

        # Effective hidden size: doubles if bidirectional
        effective_hidden = hidden_size * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(effective_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_size) — input sequences

        Returns:
            logits: (batch, n_classes)
        """
        # rnn_out: (batch, seq_len, hidden_size * num_directions)
        # For LSTM: (rnn_out, (h_n, c_n)); for GRU/RNN: (rnn_out, h_n)
        if self.rnn_type == "lstm":
            rnn_out, (h_n, _) = self.rnn(x)
        else:
            rnn_out, h_n = self.rnn(x)

        # Strategy: use the last hidden state from the last layer.
        # Alternative: mean-pool over all timesteps (use rnn_out.mean(dim=1))
        # — better for very long sequences where the last state forgets early info.
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_forward  = h_n[-2]  # last layer, forward direction
            h_backward = h_n[-1]  # last layer, backward direction
            last_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            last_hidden = h_n[-1]  # last layer hidden state

        return self.head(last_hidden)


class LSTMForecaster(nn.Module):
    """
    LSTM-based time series forecaster (sequence-to-value regression).

    Unidirectional LSTM (causal): only uses past context to predict future.
    Bidirectional would "cheat" by seeing future timesteps in the input window.

    Args:
        input_size:       number of input features
        hidden_size:      LSTM hidden state size
        num_layers:       stacked LSTM layers
        forecast_horizon: number of future steps to predict
        dropout:          dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        forecast_horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # causal: cannot see the future
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            forecasts: (batch, forecast_horizon)
        """
        rnn_out, _ = self.lstm(x)
        last_out   = rnn_out[:, -1, :]  # take output at last timestep
        return self.head(last_out)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "batch_size":     64,
    "learning_rate":  1e-3,
    "weight_decay":   1e-4,
    "epochs":         60,
    "patience":       15,
    "clip_grad_norm": 5.0,    # gradient clipping: critical for RNNs
}


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
           task: str = "classification", epochs: int = 60,
           lr: float = 1e-3, weight_decay: float = 1e-4,
           patience: int = 15, clip_norm: float = 5.0) -> dict:
    """
    Training loop for sequence models.

    Key difference from MLP training:
        - Gradient clipping (clip_grad_norm_) is critical for RNNs/LSTMs.
          Exploding gradients are common in recurrent networks.
          max_norm=5.0 is standard for LSTMs.

    Args:
        model:        LSTM model
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        task:         'classification' or 'regression'
        epochs:       maximum training epochs
        lr:           initial learning rate
        weight_decay: L2 regularization
        patience:     early stopping patience
        clip_norm:    max gradient norm for clipping

    Returns:
        training history dict
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            if task == "regression":
                output = output.squeeze(-1) if output.dim() > 1 and output.shape[-1] == 1 else output
            loss = criterion(output, y_batch)
            loss.backward()
            # Gradient clipping: prevents exploding gradients (critical for LSTM)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                output = model(X_batch)
                if task == "regression":
                    output = output.squeeze(-1) if output.dim() > 1 and output.shape[-1] == 1 else output
                val_loss += criterion(output, y_batch).item() * len(X_batch)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model: nn.Module, X_test: np.ndarray,
                          y_test: np.ndarray) -> dict:
    """Evaluate sequence classifier."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    y_pred = logits.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"accuracy": acc}


def evaluate_forecaster(model: nn.Module, X_test: np.ndarray,
                         y_test: np.ndarray) -> dict:
    """Evaluate time series forecaster."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_test.flatten(), preds.flatten()))
    print(f"  RMSE: {rmse:.4f}")
    return {"rmse": rmse}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_loss_curves(history: dict, title: str = "LSTM") -> None:
    """Plot training and validation loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"{title} — Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.lower()}_loss.png", dpi=150)
    plt.show()


def plot_forecast(model: nn.Module, X_test: np.ndarray,
                   y_test: np.ndarray, n_samples: int = 100) -> None:
    """
    Plot actual vs. predicted time series values.

    Fine-tuning tip: if predicted values are consistently lagged
    (predict the previous value), the model is learning a trivial solution.
    Add differencing or longer context windows.
    """
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test[:n_samples]).to(DEVICE)).cpu().numpy().flatten()
    actuals = y_test[:n_samples].flatten()

    plt.figure(figsize=(12, 4))
    plt.plot(actuals, label="Actual",    alpha=0.8)
    plt.plot(preds,   label="Predicted", alpha=0.8, linestyle="--")
    plt.xlabel("Sample"); plt.ylabel("Value")
    plt.title("LSTM Forecaster — Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lstm_forecast.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    from sklearn.model_selection import train_test_split

    # ── Sequence Classification ──────────────────────────────────────
    print("=== LSTM Sequence Classification ===")
    X_cls, y_cls = generate_classification_sequences(n_samples=1000, seq_len=50,
                                                       n_features=3, n_classes=3)

    # Normalize each feature across all timesteps
    scaler = StandardScaler()
    shape = X_cls.shape
    X_cls = scaler.fit_transform(X_cls.reshape(-1, shape[-1])).reshape(shape)
    X_cls = X_cls.astype(np.float32)

    X_tr, X_test, y_tr, y_test = train_test_split(X_cls, y_cls, test_size=0.2,
                                                    random_state=42, stratify=y_cls)
    X_tr, X_val, y_tr, y_val   = train_test_split(X_tr, y_tr, test_size=0.15,
                                                    random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                               batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                               batch_size=64)

    # Build classifier
    clf = LSTMClassifier(
        input_size=3,
        hidden_size=128,
        num_layers=2,
        n_classes=3,
        dropout=0.3,
        bidirectional=True,
        rnn_type="lstm",
    )
    print(f"Parameters: {sum(p.numel() for p in clf.parameters()):,}")

    history = train(clf, train_loader, val_loader, task="classification",
                     epochs=TRAINING_CONFIG["epochs"],
                     patience=TRAINING_CONFIG["patience"])

    print("\nTest Metrics:")
    evaluate_classifier(clf, X_test, y_test)
    plot_loss_curves(history, title="LSTM Classifier")

    # ── Time Series Forecasting ──────────────────────────────────────
    print("\n=== LSTM Time Series Forecasting ===")
    X_reg, y_reg = generate_regression_sequences(n_samples=800, seq_len=30,
                                                   n_features=1, forecast_horizon=1)

    Xr_tr, Xr_test, yr_tr, yr_test = train_test_split(X_reg, y_reg, test_size=0.2,
                                                        random_state=42)
    Xr_tr, Xr_val, yr_tr, yr_val   = train_test_split(Xr_tr, yr_tr, test_size=0.15,
                                                        random_state=42)

    reg_train = DataLoader(TensorDataset(torch.tensor(Xr_tr), torch.tensor(yr_tr)),
                            batch_size=32, shuffle=True)
    reg_val   = DataLoader(TensorDataset(torch.tensor(Xr_val), torch.tensor(yr_val)),
                            batch_size=32)

    forecaster = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2,
                                 forecast_horizon=1, dropout=0.2)

    reg_history = train(forecaster, reg_train, reg_val, task="regression",
                         epochs=60, patience=15)

    print("\nForecasting Metrics:")
    evaluate_forecaster(forecaster, Xr_test, yr_test)
    plot_forecast(forecaster, Xr_test, yr_test)
    plot_loss_curves(reg_history, title="LSTM Forecaster")


if __name__ == "__main__":
    main()
