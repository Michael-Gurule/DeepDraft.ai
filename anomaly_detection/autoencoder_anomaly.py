"""
Autoencoder Anomaly Detection
==============================
Detects anomalies by learning a compressed representation of normal data.
An autoencoder trained exclusively on normal samples learns to reconstruct
normal patterns well. Anomalies, being outside the learned distribution,
produce high reconstruction error — that error becomes the anomaly score.

Architecture:
    Input → Encoder (compresses) → Bottleneck (latent z) → Decoder (reconstructs) → Output
    Loss = MSE(input, reconstruction)

    At inference:
        anomaly_score(x) = ||x - decoder(encoder(x))||²

Key insight: the model is trained unsupervised on normal data only. It never
sees anomalies during training, so it cannot reconstruct them well at test time.
Reconstruction error separates normal from anomalous samples.

Relationship to VAE:
    VAE adds a KL-divergence term to regularize the latent space and enable
    generative sampling. For anomaly detection, a plain AE is often sufficient
    and easier to threshold; VAE reconstruction probability can also serve as
    an anomaly score (see frontier/vae.py).

Use Cases:
    - Sensor fault detection (Hyperion sensor health monitoring)
    - Network intrusion detection (traffic patterns)
    - Log anomaly detection (embedding log sequences)
    - Medical imaging (detecting abnormal scans)
    - Manufacturing quality control

Advantages:
    - Learns nonlinear normal manifold (unlike PCA/One-Class SVM)
    - Scales to high-dimensional inputs
    - Anomaly score is interpretable (reconstruction error per feature)
    - Can be extended to VAE for probabilistic scoring

Fine-Tuning Recommendations:
    - Bottleneck size: start at 10–20% of input dim; too small → underfitting;
      too large → AE memorizes anomalies without generalizing.
    - Depth: 2–3 encoder/decoder layers. Deeper → captures more complex patterns.
    - Threshold: use contamination percentile or maximize F1 on a small labeled set.
    - Training data: use *only* normal samples. Mixed training degrades separation.
    - Epochs: 50–200 with early stopping on validation reconstruction loss.
    - Dropout: 0.1–0.3 helps regularize; prevents overfitting on small datasets.
    - Activation: ReLU for hidden layers; no activation (linear) on final decoder
      layer when input is standardized (can go negative).
    - For sequences or images: replace FC layers with LSTM/Conv AE backbone.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_anomaly_data(
    n_normal: int = 1000,
    n_anomalies: int = 50,
    n_features: int = 20,
):
    """
    Generate synthetic normal data with injected anomalies.

    Replace with your own data:
        import pandas as pd
        df = pd.read_csv("sensor_data.csv")
        X = df.drop(columns=["timestamp", "label"]).values
        y = df["label"].values  # 1=anomaly, 0=normal

    Args:
        n_normal:    number of normal samples
        n_anomalies: number of anomaly samples
        n_features:  feature dimensionality

    Returns:
        X:       full dataset (normal + anomalies)
        y:       labels (0=normal, 1=anomaly)
        X_train: normal-only training set (unlabeled, for AE training)
    """
    X_normal, _ = make_blobs(n_samples=n_normal, n_features=n_features,
                              centers=3, cluster_std=0.8, random_state=42)
    rng = np.random.RandomState(0)
    X_anomalies = rng.uniform(
        low=X_normal.min(axis=0) * 3,
        high=X_normal.max(axis=0) * 3,
        size=(n_anomalies, n_features)
    )
    X = np.vstack([X_normal, X_anomalies]).astype(np.float32)
    y = np.array([0] * n_normal + [1] * n_anomalies)
    X_train = X_normal.astype(np.float32)
    return X, y, X_train


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

# Placeholder: tune bottleneck_dim (latent size) and hidden_dims depth
AE_CONFIG = {
    "hidden_dims":    [64, 32],   # encoder layer sizes; decoder mirrors in reverse
    "bottleneck_dim": 8,          # Placeholder: ~10–20% of input_dim
    "dropout":        0.1,        # Placeholder: 0.0–0.3
    "lr":             1e-3,       # Placeholder: 1e-4 to 1e-2
    "batch_size":     64,         # Placeholder: 32–256
    "epochs":         100,        # Placeholder: 50–200
    "patience":       15,         # early stopping patience
}


class Autoencoder(nn.Module):
    """
    Symmetric fully-connected autoencoder.

    Encoder: input_dim → hidden_dims → bottleneck_dim
    Decoder: bottleneck_dim → reversed(hidden_dims) → input_dim

    The decoder output is linear (no activation) — suitable for standardized
    continuous inputs. For bounded inputs (e.g., images [0,1]) use Sigmoid.
    """

    def __init__(self, input_dim: int, hidden_dims: list, bottleneck_dim: int,
                 dropout: float = 0.1):
        super().__init__()

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirrors encoder in reverse)
        dec_layers = []
        prev = bottleneck_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))  # linear output
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_autoencoder(
    X_train: np.ndarray,
    config: dict = None,
) -> tuple:
    """
    Train the autoencoder on normal data only.

    Uses MSE reconstruction loss with a validation split for early stopping.

    Args:
        X_train: normal-only training data (before scaling)
        config:  hyperparameter dict; defaults to AE_CONFIG

    Returns:
        model:  trained Autoencoder (on DEVICE)
        scaler: fitted StandardScaler
        history: dict with train_loss and val_loss lists
    """
    if config is None:
        config = AE_CONFIG

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train / validation split (80/20 of normal data)
    n_val   = max(1, int(0.2 * len(X_scaled)))
    idx     = np.random.RandomState(42).permutation(len(X_scaled))
    X_val_t = torch.tensor(X_scaled[idx[:n_val]],  dtype=torch.float32)
    X_tr_t  = torch.tensor(X_scaled[idx[n_val:]], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr_t), batch_size=config["batch_size"],
                               shuffle=True, drop_last=False)

    input_dim = X_scaled.shape[1]
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        bottleneck_dim=config["bottleneck_dim"],
        dropout=config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    patience_c = 0
    best_state = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(X_tr_t)

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_t.to(DEVICE))
            val_loss  = criterion(val_recon, X_val_t.to(DEVICE)).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= config["patience"]:
                print(f"  Early stopping at epoch {epoch} (best val: {best_val:.6f})")
                break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | train: {epoch_loss:.6f} | val: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, history


# ─────────────────────────────────────────────
# SCORING & EVALUATION
# ─────────────────────────────────────────────

def reconstruction_scores(model: nn.Module, scaler: StandardScaler,
                           X: np.ndarray) -> np.ndarray:
    """
    Compute per-sample reconstruction error (MSE).
    Higher score → more anomalous.
    """
    model.eval()
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        recon = model(X_t).cpu().numpy()
    errors = np.mean((X_scaled - recon) ** 2, axis=1)
    return errors


def find_threshold(scores: np.ndarray, contamination: float = 0.05) -> float:
    """
    Set anomaly threshold at the (1-contamination) percentile of scores.

    Placeholder: adjust contamination to match expected anomaly rate.
    Alternatively, use find_optimal_threshold() with labeled data.
    """
    return float(np.percentile(scores, 100 * (1 - contamination)))


def find_optimal_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """
    Find threshold that maximizes F1 (requires labeled data).
    Use this for calibration when some labeled anomalies are available.
    """
    precisions, recalls, thresholds = precision_recall_curve(y, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx])


def evaluate(model: nn.Module, scaler: StandardScaler,
             X: np.ndarray, y: np.ndarray,
             contamination: float = 0.047) -> dict:
    """
    Evaluate anomaly detection performance.

    Args:
        model:         trained Autoencoder
        scaler:        fitted StandardScaler
        X:             all samples (normal + anomalies)
        y:             true labels (0=normal, 1=anomaly)
        contamination: expected fraction of anomalies

    Returns:
        dict of metrics
    """
    scores  = reconstruction_scores(model, scaler, X)
    thresh  = find_threshold(scores, contamination)
    y_pred  = (scores > thresh).astype(int)

    auc = roc_auc_score(y, scores)
    ap  = average_precision_score(y, scores)
    cm  = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return {"roc_auc": auc, "avg_precision": ap,
            "precision": precision, "recall": recall}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_curve(history: dict) -> None:
    """Plot train/validation reconstruction loss over epochs."""
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Reconstruction Loss")
    plt.title("Autoencoder — Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ae_training_curve.png", dpi=150)
    plt.show()


def plot_reconstruction_errors(model: nn.Module, scaler: StandardScaler,
                                X: np.ndarray, y: np.ndarray,
                                contamination: float = 0.047) -> None:
    """
    Histogram of reconstruction errors for normal vs. anomalous samples.
    Clear separation → model is discriminating well.
    """
    scores = reconstruction_scores(model, scaler, X)
    thresh = find_threshold(scores, contamination)

    plt.figure(figsize=(9, 4))
    plt.hist(scores[y == 0], bins=50, alpha=0.6, label="Normal",  color="steelblue")
    plt.hist(scores[y == 1], bins=50, alpha=0.6, label="Anomaly", color="tomato")
    plt.axvline(thresh, color="black", linestyle="--",
                label=f"Threshold ({thresh:.4f})")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Count")
    plt.title("Autoencoder — Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ae_reconstruction_errors.png", dpi=150)
    plt.show()


def plot_feature_reconstruction(model: nn.Module, scaler: StandardScaler,
                                 X: np.ndarray, y: np.ndarray,
                                 n_samples: int = 5) -> None:
    """
    Bar chart: mean per-feature reconstruction error for normal vs. anomalous.
    Reveals which features drive the anomaly signal.
    """
    model.eval()
    X_scaled = scaler.transform(X)
    X_t      = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        recon = model(X_t).cpu().numpy()
    per_feature_err = (X_scaled - recon) ** 2  # (n, d)

    mean_normal  = per_feature_err[y == 0].mean(axis=0)
    mean_anomaly = per_feature_err[y == 1].mean(axis=0)
    features = np.arange(X.shape[1])

    plt.figure(figsize=(12, 4))
    w = 0.35
    plt.bar(features - w/2, mean_normal,  w, label="Normal",  color="steelblue", alpha=0.8)
    plt.bar(features + w/2, mean_anomaly, w, label="Anomaly", color="tomato",    alpha=0.8)
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Squared Error")
    plt.title("Autoencoder — Per-Feature Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ae_feature_errors.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model: nn.Module, scaler: StandardScaler,
                           X: np.ndarray, y: np.ndarray,
                           contamination: float = 0.047) -> None:
    """Display confusion matrix at the contamination threshold."""
    scores = reconstruction_scores(model, scaler, X)
    thresh = find_threshold(scores, contamination)
    y_pred = (scores > thresh).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y, y_pred),
                           display_labels=["Normal", "Anomaly"]).plot(cmap="Oranges")
    plt.title("Autoencoder — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("ae_confusion.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# CONVOLUTIONAL AUTOENCODER VARIANT
# ─────────────────────────────────────────────

# Placeholder: use CONV_AE_CONFIG for image or multichannel time-series inputs.
# For 1D tabular data, the FC Autoencoder above is preferred.
CONV_AE_CONFIG = {
    "in_channels":   1,          # Placeholder: 1 (grayscale), 3 (RGB), n_sensors (multivariate TS)
    "base_channels": 16,         # Placeholder: 16–64; doubles at each encoder stage
    "n_stages":      3,          # Placeholder: 2–4; each stage halves spatial resolution
    "bottleneck_channels": 8,    # Placeholder: 4–32; compressed representation channels
    "input_h":       28,         # Placeholder: image height (or 1 for 1D conv)
    "input_w":       28,         # Placeholder: image width (or sequence_len for 1D)
    "lr":            1e-3,
    "batch_size":    64,
    "epochs":        50,
    "patience":      10,
    "use_1d":        False,      # Placeholder: True for 1D signals (sensor streams)
}


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for image-like or multichannel time-series inputs.

    Architecture: Conv2d (stride=2, halves H×W) × n_stages → flatten → linear bottleneck.
    Each conv stage doubles the channel count (base → 2*base → 4*base → …).

    For 1D signals (e.g., multivariate sensor streams), set use_1d=True to switch
    Conv2d → Conv1d throughout.

    Args:
        in_channels:          input channels (1=grayscale, 3=RGB, n=sensors)
        base_channels:        channels at first conv layer
        n_stages:             number of downsampling stages
        bottleneck_channels:  latent space channels (spatial size shrinks with n_stages)
        input_h, input_w:     spatial input dimensions (for fc layer sizing)
        use_1d:               True → use Conv1d instead of Conv2d
    """

    def __init__(self, in_channels: int, base_channels: int, n_stages: int,
                 bottleneck_channels: int, input_h: int, input_w: int,
                 use_1d: bool = False):
        super().__init__()
        self.use_1d = use_1d
        conv_cls = nn.Conv1d if use_1d else nn.Conv2d
        norm_cls = nn.BatchNorm1d if use_1d else nn.BatchNorm2d

        layers = []
        c_in = in_channels
        c_out = base_channels
        for _ in range(n_stages):
            layers += [
                conv_cls(c_in, c_out, kernel_size=3, stride=2, padding=1),
                norm_cls(c_out),
                nn.LeakyReLU(0.2),
            ]
            c_in, c_out = c_out, c_out * 2
        layers += [conv_cls(c_in, bottleneck_channels, kernel_size=1)]
        self.conv_layers = nn.Sequential(*layers)

        # Compute flattened spatial size after all downsampling
        factor = 2 ** n_stages
        if use_1d:
            self._flat_size = bottleneck_channels * (input_w // factor)
        else:
            self._flat_size = bottleneck_channels * (input_h // factor) * (input_w // factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x).flatten(1)

    @property
    def flat_size(self) -> int:
        return self._flat_size


class ConvDecoder(nn.Module):
    """
    Convolutional decoder (mirrors ConvEncoder).

    Reconstructs the original spatial dimensions via ConvTranspose2d (stride=2)
    upsampling stages.  The final layer uses Sigmoid for image outputs [0,1] or
    no activation for standardized inputs.
    """

    def __init__(self, in_channels: int, base_channels: int, n_stages: int,
                 bottleneck_channels: int, input_h: int, input_w: int,
                 use_1d: bool = False, output_activation: str = "sigmoid"):
        super().__init__()
        self.use_1d = use_1d
        self.n_stages = n_stages
        factor = 2 ** n_stages

        if use_1d:
            self._spatial_h = 1
            self._spatial_w = input_w // factor
        else:
            self._spatial_h = input_h // factor
            self._spatial_w = input_w // factor
        self._bottleneck_ch = bottleneck_channels

        conv_t_cls = nn.ConvTranspose1d if use_1d else nn.ConvTranspose2d
        norm_cls   = nn.BatchNorm1d if use_1d else nn.BatchNorm2d

        # Build reversed channel list
        channel_list = [base_channels * (2 ** i) for i in range(n_stages)]
        channel_list = list(reversed(channel_list))  # wide→narrow for decoder

        layers = [conv_t_cls(bottleneck_channels, channel_list[0], kernel_size=1)]
        for i in range(n_stages - 1):
            layers += [
                conv_t_cls(channel_list[i], channel_list[i + 1],
                            kernel_size=4, stride=2, padding=1),
                norm_cls(channel_list[i + 1]),
                nn.ReLU(),
            ]
        layers += [
            conv_t_cls(channel_list[-1], in_channels, kernel_size=4, stride=2, padding=1),
        ]
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        if self.use_1d:
            z = z_flat.view(z_flat.size(0), self._bottleneck_ch, self._spatial_w)
        else:
            z = z_flat.view(z_flat.size(0), self._bottleneck_ch,
                            self._spatial_h, self._spatial_w)
        return self.layers(z)


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for image or multichannel time-series anomaly detection.

    When to use over the FC Autoencoder:
        - Input is an image (e.g., spectrogram, thermal map, X-ray)
        - Input is a multichannel time-series where local correlations matter
        - Input dimension is large (>200 features) — conv is more parameter-efficient
        - You want equivariance to local shifts (translation invariance)

    When to stick with the FC Autoencoder:
        - Input is a flat feature vector with no spatial/temporal structure
        - Dataset is small (<5k samples) — conv overfits more easily
        - Interpretability per feature index is needed

    Args:
        config: CONV_AE_CONFIG dict (or compatible custom dict)
    """

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or CONV_AE_CONFIG
        self.encoder = ConvEncoder(
            in_channels=cfg["in_channels"],
            base_channels=cfg["base_channels"],
            n_stages=cfg["n_stages"],
            bottleneck_channels=cfg["bottleneck_channels"],
            input_h=cfg["input_h"],
            input_w=cfg["input_w"],
            use_1d=cfg["use_1d"],
        )
        self.decoder = ConvDecoder(
            in_channels=cfg["in_channels"],
            base_channels=cfg["base_channels"],
            n_stages=cfg["n_stages"],
            bottleneck_channels=cfg["bottleneck_channels"],
            input_h=cfg["input_h"],
            input_w=cfg["input_w"],
            use_1d=cfg["use_1d"],
            output_activation="sigmoid",  # Placeholder: "none" for standardized inputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error."""
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=list(range(1, x.ndim)))


def train_conv_autoencoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict = None,
    device: str = DEVICE,
) -> tuple:
    """
    Train a ConvAutoencoder on normal-only data.

    Args:
        train_loader: DataLoader of normal training images/signals, (B, C, H, W)
        val_loader:   DataLoader for validation (normal only)
        config:       CONV_AE_CONFIG dict
        device:       'cpu', 'cuda', or 'mps'

    Returns:
        model, history dict
    """
    cfg = config or CONV_AE_CONFIG
    model = ConvAutoencoder(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()

    best_val, patience_c, best_state = float("inf"), 0, None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss += loss.item() * x.size(0)
        ep_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                val_loss += criterion(model(x), x).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(ep_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= cfg["patience"]:
                print(f"  Conv-AE early stop at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train: {ep_loss:.6f} | val: {val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history


def demo_conv_ae_on_mnist() -> None:
    """
    Demonstrate ConvAutoencoder anomaly detection on MNIST.

    Normal class: digit '1'. Anomaly: any other digit.
    Replace this with your own image dataset.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("torchvision not available; skipping Conv-AE MNIST demo.")
        return

    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)

    # Normal = digit 1; anomaly = digit 7
    normal_idx   = (mnist.targets == 1).nonzero(as_tuple=True)[0]
    anomaly_idx  = (mnist.targets == 7).nonzero(as_tuple=True)[0][:200]
    normal_data  = mnist.data[normal_idx].float().unsqueeze(1) / 255.0
    anomaly_data = mnist.data[anomaly_idx].float().unsqueeze(1) / 255.0

    # Train on normal only
    n_val = 200
    train_ds = TensorDataset(normal_data[:-n_val])
    val_ds   = TensorDataset(normal_data[-n_val:])
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=64)

    cfg = {**CONV_AE_CONFIG, "in_channels": 1, "input_h": 28, "input_w": 28,
           "n_stages": 2, "base_channels": 16, "bottleneck_channels": 4,
           "epochs": 20, "lr": 1e-3, "patience": 5, "use_1d": False}

    print("\nTraining Conv-AE on MNIST (digit=1 as normal)...")
    model, history = train_conv_autoencoder(train_dl, val_dl, config=cfg)

    # Compute scores on normal + anomaly
    model.eval()
    all_images = torch.cat([normal_data[-n_val:], anomaly_data])
    y_true     = np.array([0] * n_val + [1] * len(anomaly_data))
    with torch.no_grad():
        scores = model.anomaly_score(all_images.to(DEVICE)).cpu().numpy()

    auc = roc_auc_score(y_true, scores)
    print(f"  Conv-AE MNIST anomaly ROC-AUC: {auc:.4f}")

    plt.figure(figsize=(8, 4))
    plt.hist(scores[y_true == 0], bins=40, alpha=0.6, label="Normal (1)", color="steelblue")
    plt.hist(scores[y_true == 1], bins=40, alpha=0.6, label="Anomaly (7)", color="tomato")
    plt.xlabel("Reconstruction Error"); plt.ylabel("Count")
    plt.title(f"Conv-AE MNIST — AUC={auc:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("conv_ae_mnist.png", dpi=150)
    plt.show()
    print("  Saved conv_ae_mnist.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"=== Autoencoder Anomaly Detection ===  (device: {DEVICE})")

    # Placeholder: replace with your dataset
    X, y, X_train = generate_anomaly_data(n_normal=1000, n_anomalies=50, n_features=20)
    print(f"Dataset: {X.shape[0]} samples ({y.sum()} anomalies) | {X.shape[1]} features")

    contamination = y.sum() / len(y)  # Placeholder: set to known or estimated rate

    print("\nTraining autoencoder on normal data only...")
    model, scaler, history = train_autoencoder(X_train)

    print("\nEvaluation (all samples):")
    evaluate(model, scaler, X, y, contamination=contamination)

    # Threshold using labeled data (if available)
    scores      = reconstruction_scores(model, scaler, X)
    opt_thresh  = find_optimal_threshold(scores, y)
    y_pred_opt  = (scores > opt_thresh).astype(int)
    from sklearn.metrics import f1_score
    print(f"\nOptimal threshold: {opt_thresh:.4f}  (F1: {f1_score(y, y_pred_opt):.4f})")

    plot_training_curve(history)
    plot_reconstruction_errors(model, scaler, X, y, contamination=contamination)
    plot_feature_reconstruction(model, scaler, X, y)
    plot_confusion_matrix(model, scaler, X, y, contamination=contamination)

    # Conv-AE on MNIST (image inputs)
    # Placeholder: replace demo_conv_ae_on_mnist() with your own image data
    print("\n--- Convolutional AE (MNIST demo) ---")
    demo_conv_ae_on_mnist()


if __name__ == "__main__":
    main()
