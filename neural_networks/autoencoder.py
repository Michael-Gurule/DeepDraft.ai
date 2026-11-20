"""
Autoencoder (AE) — General-Purpose Template
=============================================

An autoencoder learns a compressed representation z = f_enc(x) ∈ ℝ^d_latent
and reconstruction x̂ = f_dec(z), trained to minimize ||x - x̂||².

Variants covered:
  FC-AE:         Fully-connected autoencoder for tabular/vector data
  Conv-AE:       Convolutional encoder/decoder for image data
  Denoising AE:  Add noise at input, train to reconstruct clean signal
                 Better representations: learns to extract signal from noise
  Sparse AE:     L1 penalty on activations → KL divergence on ρ-hat vs ρ
                 Encourages each neuron to be active sparsely
  Contractive AE:||∂h/∂x||²_F penalty → local invariance to input perturbations
  Deep AE:       Many layers with residual connections

Relationship to other models:
  VAE:    adds probabilistic latent space z ~ N(μ,σ²); enables sampling
  EBM:    reconstruction error ≈ energy function
  PCA:    linear AE with tied weights = PCA (Baldi & Hornik 1989)
  t-SNE/UMAP: AE pre-training provides better initial embedding for viz

Applications:
  - Feature extraction / representation learning
  - Data compression
  - Anomaly detection (high reconstruction error = anomaly; see autoencoder_anomaly.py)
  - Denoising (audio, image, sensor data)
  - Pre-training for classification (semi-supervised)

Fine-tuning recommendations:
  - latent_dim (#Placeholder: 32): bottleneck size — try 2D for visualization
  - hidden_dims (#Placeholder: [256, 128, 64]): encoder channel progression
  - dropout (#Placeholder: 0.1):
  - noise_std (#Placeholder: 0.1): for denoising AE; match expected noise level
  - sparsity_rho (#Placeholder: 0.05): target activation sparsity
  - sparsity_weight (#Placeholder: 0.5): KL weight for sparse AE
  - contractive_weight (#Placeholder: 1e-4): Jacobian penalty

References:
  Rumelhart et al. (1986) — Learning Representations by Backpropagating Errors
  Vincent et al. (2010) — Stacked Denoising Autoencoders
  Ng (2011) — Sparse Autoencoder (lecture notes)
  Rifai et al. (2011) — Contractive Autoencoders
  Baldi & Hornik (1989) — Neural Networks and PCA
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CONFIG = {
    "latent_dim":           32,          # # Placeholder: bottleneck dimension
    "hidden_dims":          [256, 128],  # # Placeholder: encoder progression
    "dropout":              0.0,         # # Placeholder:
    "lr":                   1e-3,        # # Placeholder:
    "weight_decay":         1e-5,        # # Placeholder:
    "batch_size":           128,         # # Placeholder:
    "n_epochs":             100,         # # Placeholder:
    "noise_std":            0.2,         # # Placeholder: for denoising AE
    "sparsity_rho":         0.05,        # # Placeholder: target sparsity
    "sparsity_weight":      0.5,         # # Placeholder: KL penalty weight
    "contractive_weight":   1e-4,        # # Placeholder: Jacobian norm penalty
    "patience":             15,          # # Placeholder: early stopping
    "seed":                 42,
}


# ─────────────────────────────────────────────
# Fully-Connected Autoencoder
# ─────────────────────────────────────────────

class FCEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]),
                       nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int,
                 output_activation: str = "none"):
        super().__init__()
        dims = [latent_dim] + list(reversed(hidden_dims)) + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    """Standard FC autoencoder with configurable depth and latent dim."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int,
                 dropout: float = 0.0, output_activation: str = "none"):
        super().__init__()
        self.encoder = FCEncoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = FCDecoder(latent_dim, hidden_dims, input_dim, output_activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)


# ─────────────────────────────────────────────
# Convolutional Autoencoder
# ─────────────────────────────────────────────

class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, latent_dim: int,
                 n_levels: int = 3):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_levels):
            out_ch = base_channels * (2 ** i)
            layers += [nn.Conv2d(ch, out_ch, 3, stride=2, padding=1),
                       nn.BatchNorm2d(out_ch), nn.ReLU()]
            ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.flatten_dim = ch  # track for decoder
        self.fc = nn.Linear(ch, latent_dim)  # global avg pool → linear
        self._ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.mean(dim=(2, 3))   # global average pooling
        return self.fc(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, base_channels: int, out_channels: int,
                 spatial_size: int = 7, n_levels: int = 3):
        super().__init__()
        top_ch = base_channels * (2 ** (n_levels - 1))
        self.fc = nn.Linear(latent_dim, top_ch * spatial_size * spatial_size)
        self.spatial = spatial_size
        self.top_ch = top_ch

        layers = []
        ch = top_ch
        for i in range(n_levels - 1, 0, -1):
            out_ch = base_channels * (2 ** (i - 1))
            layers += [nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                       nn.BatchNorm2d(out_ch), nn.ReLU()]
            ch = out_ch
        layers += [nn.ConvTranspose2d(ch, out_channels, 4, stride=2, padding=1),
                   nn.Sigmoid()]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(z))
        h = h.view(-1, self.top_ch, self.spatial, self.spatial)
        return self.deconv(h)


class ConvAutoencoder(nn.Module):
    """Convolutional AE for image data (MNIST, CIFAR, etc.)."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 latent_dim: int = 64, n_levels: int = 3, spatial_size: int = 4):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, base_channels, latent_dim, n_levels)
        self.decoder = ConvDecoder(latent_dim, base_channels, in_channels, spatial_size, n_levels)

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))


# ─────────────────────────────────────────────
# Training Objectives
# ─────────────────────────────────────────────

def denoising_loss(model: Autoencoder, x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Reconstruct clean x from noisy x̃ = x + ε, ε ~ N(0, σ²I)."""
    noise = torch.randn_like(x) * noise_std
    x_noisy = x + noise
    x_hat = model(x_noisy)
    return F.mse_loss(x_hat, x)


def sparse_ae_loss(
    model: Autoencoder, x: torch.Tensor,
    rho: float = 0.05,
    sparse_weight: float = 0.5,
) -> torch.Tensor:
    """
    MSE + KL divergence sparsity penalty on hidden activations.
    KL(ρ || ρ̂) = ρ log(ρ/ρ̂) + (1-ρ) log((1-ρ)/(1-ρ̂))
    """
    z = model.encode(x)
    x_hat = model.decode(z)
    recon = F.mse_loss(x_hat, x)

    # Mean activation per neuron
    rho_hat = torch.sigmoid(z).mean(dim=0).clamp(1e-6, 1 - 1e-6)
    kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return recon + sparse_weight * kl.sum()


def contractive_loss(
    model: Autoencoder, x: torch.Tensor, weight: float = 1e-4,
) -> torch.Tensor:
    """
    MSE + Frobenius norm of Jacobian ∂h/∂x.
    Computed analytically for FC layer: ||W||²_F · ρ'(Wx)²
    """
    x = x.requires_grad_(True)
    z = model.encode(x)
    x_hat = model.decode(z)
    recon = F.mse_loss(x_hat, x)

    # Jacobian via autograd
    jacob_rows = []
    for i in range(z.shape[1]):
        grad = torch.autograd.grad(z[:, i].sum(), x,
                                    create_graph=True, retain_graph=True)[0]
        jacob_rows.append(grad)
    J = torch.stack(jacob_rows, dim=1)   # (batch, latent, input)
    contractive = (J ** 2).sum(dim=(1, 2)).mean()
    return recon + weight * contractive


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: dict,
    mode: str = "standard",   # # Placeholder: "standard" | "denoising" | "sparse" | "contractive"
) -> tuple[Autoencoder, dict]:
    """Train FC autoencoder. Returns model and loss history."""
    X_tr  = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    input_dim = X_tr.shape[1]

    loader = DataLoader(TensorDataset(X_tr), batch_size=cfg["batch_size"], shuffle=True)

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(cfg["n_epochs"]):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            if mode == "standard":
                loss = F.mse_loss(model(xb), xb)
            elif mode == "denoising":
                loss = denoising_loss(model, xb, cfg["noise_std"])
            elif mode == "sparse":
                loss = sparse_ae_loss(model, xb, cfg["sparsity_rho"], cfg["sparsity_weight"])
            elif mode == "contractive":
                loss = contractive_loss(model, xb, cfg["contractive_weight"])
            else:
                raise ValueError(mode)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(loader)
        train_losses.append(avg_train)

        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(X_val_t.to(DEVICE)), X_val_t.to(DEVICE)).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {avg_train:.5f} | Val: {val_loss:.5f}")

    model.load_state_dict(best_state)
    return model, {"train": train_losses, "val": val_losses}


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_reconstructions(model: Autoencoder, X: np.ndarray, n: int = 8,
                          mode: str = "standard"):
    """Show original vs reconstructed samples."""
    X_t = torch.tensor(X[:n], dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        if mode == "denoising":
            X_noisy = X_t + torch.randn_like(X_t) * CONFIG["noise_std"]
            X_hat = model(X_noisy).cpu().numpy()
        else:
            X_hat = model(X_t).cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    fig.suptitle(f"Autoencoder ({mode}) — Reconstructions", fontsize=11)
    for i in range(n):
        for row, data in enumerate([X[:n], X_hat]):
            if data.shape[1] in [64, 784, 1024]:
                side = int(np.sqrt(data.shape[1]))
                img = data[i].reshape(side, side)
                axes[row, i].imshow(img, cmap="gray", interpolation="nearest")
            else:
                axes[row, i].bar(range(min(20, data.shape[1])), data[i, :20])
            axes[row, i].axis("off")
    axes[0, 0].set_ylabel("Original"); axes[1, 0].set_ylabel("Reconstructed")
    plt.tight_layout()
    plt.savefig(f"ae_{mode}_reconstructions.png", dpi=150)
    plt.show()
    print(f"Saved: ae_{mode}_reconstructions.png")


def plot_latent_space(model: Autoencoder, X: np.ndarray, y: np.ndarray = None,
                       mode: str = "standard"):
    """Visualize latent space (via t-SNE if >2D)."""
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        Z = model.encode(X_t).cpu().numpy()

    if Z.shape[1] > 2:
        Z2d = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(Z)
        method = "t-SNE"
    else:
        Z2d = Z
        method = "direct"

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z2d[:, 0], Z2d[:, 1], c=y, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax)
    ax.set_title(f"Latent Space ({method}) — {mode} AE")
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"ae_{mode}_latent.png", dpi=150)
    plt.show()
    print(f"Saved: ae_{mode}_latent.png")


def compare_ae_modes(X_train, X_val, X_test, y_test, cfg):
    """Train all AE modes and compare reconstruction error."""
    modes = ["standard", "denoising", "sparse"]
    results = {}
    for mode in modes:
        print(f"\n=== {mode.capitalize()} Autoencoder ===")
        model, history = train_autoencoder(X_train, X_val, cfg, mode=mode)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        model.eval()
        with torch.no_grad():
            test_loss = F.mse_loss(model(X_test_t), X_test_t).item()
        print(f"  Test MSE: {test_loss:.5f}")
        results[mode] = {"model": model, "history": history, "test_mse": test_loss}
        plot_reconstructions(model, X_test, mode=mode)
        plot_latent_space(model, X_test, y_test, mode=mode)

    # Bar comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(results.keys()), [v["test_mse"] for v in results.values()],
           color=["steelblue", "darkorange", "forestgreen"], edgecolor="white")
    ax.set_ylabel("Test MSE"); ax.set_title("Autoencoder Mode Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("ae_mode_comparison.png", dpi=150)
    plt.show()
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}\n")

    # Load Digits (8x8, 64 features, 10 classes)
    digits = load_digits()
    X, y = digits.data.astype(np.float32), digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    split1 = int(0.7 * len(X))
    split2 = int(0.85 * len(X))
    X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
    y_test = y[split2:]

    # Training curve for standard AE
    print("=== Standard Autoencoder ===")
    model, history = train_autoencoder(X_train, X_val, CONFIG, mode="standard")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train"], label="Train"); axes[0].plot(history["val"], label="Val")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("(See reconstructions plot)")
    plt.tight_layout(); plt.savefig("ae_training.png", dpi=150); plt.show()

    # Compare all modes
    compare_ae_modes(X_train, X_val, X_test, y_test, CONFIG)


if __name__ == "__main__":
    main()
