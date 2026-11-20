"""
Variational Autoencoder (VAE)
==============================
A generative model that learns a structured, continuous latent space by encoding
inputs into probability distributions (rather than point estimates). Uses the
"reparameterization trick" to enable backpropagation through sampling.

How it works:
    Encoder: q_φ(z|x) → maps input x to distribution parameters (μ, σ)
    Sampling: z = μ + σ * ε, where ε ~ N(0,I)  (reparameterization trick)
    Decoder: p_θ(x|z) → maps latent sample z back to reconstruction x̂

Loss function:
    ELBO = E[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
         = Reconstruction Loss + KL Divergence
         ↑ maximize this (lower bound on log p(x))

    Where:
        Reconstruction: how well we reconstruct x from z (MSE or BCE)
        KL divergence:  how close the learned posterior is to the prior N(0,I)
                        KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

Key concepts:
    - Latent space is regularized by KL divergence → smooth and continuous
    - Neighboring latent codes decode to similar outputs (interpolation works)
    - Can sample z ~ N(0,I) to generate new data points

Use Cases:
    - Image generation and style interpolation
    - Anomaly detection (high reconstruction loss → anomaly)
    - Semi-supervised learning (latent representations as features)
    - Drug discovery (molecule generation)
    - Data augmentation

Fine-Tuning Recommendations:
    - beta-VAE: multiply KL term by beta > 1 for more disentangled representations.
      Higher beta → better disentanglement, worse reconstruction quality.
    - Latent dimension: 2–4 for visualization; 64–256 for high-quality generation.
    - KL annealing: start beta at 0, linearly increase to target value over the
      first N epochs. Prevents posterior collapse (all z mapped to prior).
    - Reconstruction loss: BCE for binary/image data (pixel values in [0,1]);
      MSE for continuous features. For images, perceptual loss often works better.
    - Decoder capacity: should be >= encoder capacity to prevent information bottleneck.
    - Learning rate: 1e-3 to 3e-4. Adam is standard.
    - Posterior collapse: if KL goes to 0 early, the model ignores the latent code.
      Fix: KL annealing, free bits, or stronger decoder constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
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

def load_mnist(data_root: str = "./data", batch_size: int = 128):
    """
    Load MNIST for VAE training.

    Note: for VAE, we use the raw pixel values normalized to [0,1].
    No labels are needed — VAE is unsupervised.
    We keep labels for latent space visualization only.

    Replace with your own data:
        - For continuous/tabular data: use MSE as reconstruction loss
        - For images: ensure pixel values are in [0, 1] (for BCE)

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten 28×28 → 784
    ])
    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# VAE MODEL
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    VAE Encoder: maps input x to distribution parameters (μ, log σ²).

    Output is NOT a single point but a Gaussian distribution in latent space.
    Using log(σ²) instead of σ² for numerical stability (can take any real value).

    Architecture: Input → FC → ReLU → [FC → μ, FC → log σ²]

    Args:
        input_dim:   dimension of input (e.g., 784 for MNIST)
        hidden_dims: list of hidden layer sizes
        latent_dim:  dimension of the latent space z
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],   # Placeholder: e.g., [512, 256]
        latent_dim: int,           # Placeholder: 2 for visualization; 64+ for quality
    ):
        super().__init__()

        # Shared trunk
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        # Two separate heads for μ and log σ²
        self.fc_mu     = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h  = self.trunk(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    VAE Decoder: maps latent sample z back to reconstructed input x̂.

    For binary/image data (pixel values in [0,1]): output with Sigmoid activation.
    For continuous data: output raw linear values, use MSE reconstruction loss.

    Architecture: z → FC → ReLU → FC → Sigmoid

    Args:
        latent_dim:   dimension of the latent space
        hidden_dims:  list of hidden layer sizes (reversed encoder dims)
        output_dim:   dimension of the reconstruction output
        output_activation: 'sigmoid' for image data, 'none' for continuous
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        output_activation: str = "sigmoid",  # Placeholder: 'sigmoid' or 'none'
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class VAE(nn.Module):
    """
    Variational Autoencoder.

    The key innovation is the reparameterization trick:
        Instead of sampling z ~ q(z|x) directly (non-differentiable),
        we sample ε ~ N(0,I) and compute z = μ + σ * ε.
        This makes the sampling step differentiable w.r.t. μ and σ.

    Args:
        input_dim:   dimension of input data
        hidden_dims: hidden layer sizes for encoder/decoder
        latent_dim:  latent space dimension
        beta:        KL weight for β-VAE (1.0 = standard VAE)
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] = None,  # Placeholder: [512, 256]
        latent_dim: int = 16,           # Placeholder: 2 for viz, 64–256 for quality
        beta: float = 1.0,              # Placeholder: beta > 1 for disentanglement
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        self.latent_dim = latent_dim
        self.beta       = beta

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)),
                                input_dim, output_activation="sigmoid")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, ε ~ N(0, I)

        During training: ε is random (enables stochastic latent sampling)
        During evaluation: typically use μ directly (deterministic reconstruction)

        Args:
            mu:     mean vector (batch, latent_dim)
            logvar: log variance vector (batch, latent_dim)

        Returns:
            z: latent sample (batch, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log σ²)
            eps = torch.randn_like(std)     # ε ~ N(0, I)
            return mu + eps * std
        else:
            return mu  # deterministic at eval time

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass: encode → sample → decode.

        Returns:
            x_recon: reconstructed input
            mu:      latent mean
            logvar:  latent log variance
        """
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decoder(z)
        return x_recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent mean (deterministic)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def sample(self, n: int) -> torch.Tensor:
        """
        Generate n new samples by sampling from the prior z ~ N(0, I).

        This is the core generation capability of the VAE.
        The KL regularization ensures the prior is a meaningful sample distribution.
        """
        z = torch.randn(n, self.latent_dim).to(next(self.parameters()).device)
        return self.decode(z)


# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────

def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE ELBO loss: Reconstruction Loss + β * KL Divergence.

    Reconstruction loss (BCE for binary/image data):
        -Σ [x * log(x̂) + (1-x) * log(1-x̂)]
        Note: BCE treats each pixel independently (ignores correlations)

    KL divergence (closed form for Gaussian q vs. N(0,I) prior):
        KL = -0.5 * Σ(1 + log σ² - μ² - σ²)

    β-VAE: multiplying KL by β > 1 encourages disentanglement at the cost
    of reconstruction quality. Start at 1.0 and increase if interpretable
    latent dimensions are needed.

    Args:
        x_recon:   reconstructed input (batch, input_dim)
        x:         original input (batch, input_dim)
        mu:        latent mean (batch, latent_dim)
        logvar:    latent log variance (batch, latent_dim)
        beta:      KL weight (1.0 = standard VAE)
        reduction: 'mean' or 'sum'

    Returns:
        total_loss, recon_loss, kl_loss (all tensors)
    """
    # Reconstruction: binary cross-entropy (pixels in [0,1])
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction=reduction)

    # KL divergence: -0.5 * sum(1 + log σ² - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "latent_dim":    16,
    "hidden_dims":   [512, 256],
    "beta":          1.0,          # Placeholder: increase to 4.0 for beta-VAE
    "epochs":        30,
    "learning_rate": 1e-3,
    "batch_size":    128,
    "kl_warmup_epochs": 10,        # Placeholder: epochs to linearly increase beta from 0
}


def train(model: VAE, train_loader: DataLoader,
           epochs: int = 30, lr: float = 1e-3,
           kl_warmup: int = 10) -> dict:
    """
    VAE training loop with optional KL annealing.

    KL annealing: gradually increase the KL weight from 0 to beta.
    Prevents posterior collapse — a failure mode where the model ignores
    the latent code entirely (KL → 0) because the decoder is too powerful.

    Args:
        model:         VAE instance
        train_loader:  training DataLoader
        epochs:        number of training epochs
        lr:            learning rate
        kl_warmup:     number of epochs for KL annealing (0 = no annealing)

    Returns:
        training history
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"total_loss": [], "recon_loss": [], "kl_loss": [], "beta_used": []}

    for epoch in range(1, epochs + 1):
        model.train()

        # KL annealing: linearly increase beta from 0 to model.beta over kl_warmup epochs
        # Placeholder: disable by setting kl_warmup=0 or increase for stronger annealing
        if kl_warmup > 0:
            beta = model.beta * min(1.0, epoch / kl_warmup)
        else:
            beta = model.beta

        total_loss_sum, recon_sum, kl_sum, n = 0.0, 0.0, 0.0, 0

        for X_batch, _ in train_loader:   # ignore labels (unsupervised)
            X_batch = X_batch.to(DEVICE)
            optimizer.zero_grad()

            x_recon, mu, logvar = model(X_batch)
            loss, recon, kl     = vae_loss(x_recon, X_batch, mu, logvar, beta=beta)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_sum += loss.item() * len(X_batch)
            recon_sum      += recon.item() * len(X_batch)
            kl_sum         += kl.item()    * len(X_batch)
            n              += len(X_batch)

        history["total_loss"].append(total_loss_sum / n)
        history["recon_loss"].append(recon_sum / n)
        history["kl_loss"].append(kl_sum / n)
        history["beta_used"].append(beta)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss_sum/n:.4f} "
                  f"| Recon: {recon_sum/n:.4f} | KL: {kl_sum/n:.4f} | β: {beta:.3f}")

    return history


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_history(history: dict) -> None:
    """Plot VAE training losses (total, reconstruction, KL)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["total_loss"]) + 1)

    axes[0].plot(epochs, history["total_loss"]);  axes[0].set_title("Total Loss")
    axes[1].plot(epochs, history["recon_loss"]);  axes[1].set_title("Reconstruction Loss")
    axes[2].plot(epochs, history["kl_loss"]);     axes[2].set_title("KL Divergence")

    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")

    plt.suptitle("VAE — Training Losses")
    plt.tight_layout()
    plt.savefig("vae_training_losses.png", dpi=150)
    plt.show()


def plot_reconstructions(model: VAE, test_loader: DataLoader, n: int = 10) -> None:
    """
    Side-by-side: original images vs. VAE reconstructions.

    Good reconstruction = the model has learned a useful latent representation.
    Blurry reconstructions are common with VAE (a known limitation of MSE/BCE loss).

    Fine-tuning tip: if reconstructions are very blurry, increase decoder capacity
    or switch to a perceptual/VGG loss function.
    """
    model.eval()
    X_batch, _ = next(iter(test_loader))
    X_sample   = X_batch[:n].to(DEVICE)

    with torch.no_grad():
        x_recon, _, _ = model(X_sample)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(X_sample[i].cpu().reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_recon[i].cpu().reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=8)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=8)
    plt.suptitle("VAE — Reconstructions")
    plt.tight_layout()
    plt.savefig("vae_reconstructions.png", dpi=150)
    plt.show()


def plot_generated_samples(model: VAE, n: int = 64) -> None:
    """
    Generate new samples by sampling z ~ N(0, I) and decoding.

    Quality of samples indicates the quality of the learned latent space.
    Poor samples suggest: KL underweight, posterior collapse, or low latent dim.
    """
    model.eval()
    with torch.no_grad():
        samples = model.sample(n).cpu().reshape(n, 28, 28).numpy()

    cols = int(np.sqrt(n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(samples[i], cmap="gray")
        ax.axis("off")

    plt.suptitle("VAE — Generated Samples (z ~ N(0,I))")
    plt.tight_layout()
    plt.savefig("vae_generated.png", dpi=150)
    plt.show()


def plot_latent_space_2d(model: VAE, test_loader: DataLoader,
                          latent_dim: int = 16) -> None:
    """
    Visualize the latent space using t-SNE (or directly if latent_dim=2).

    A well-structured latent space shows: distinct clusters per class with
    smooth transitions between them.

    Fine-tuning insight: overlapping clusters → insufficient KL regularization.
    Completely separated clusters without smooth transitions → over-regularized.
    """
    model.eval()
    zs, labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            z = model.encode(X_batch.to(DEVICE)).cpu().numpy()
            zs.append(z)
            labels.extend(y_batch.numpy())

    z_all = np.concatenate(zs, axis=0)
    y_all = np.array(labels)

    if latent_dim == 2:
        z_2d = z_all
    else:
        print("  Reducing latent space with t-SNE (may take a moment)...")
        z_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z_all)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_all, cmap="tab10",
                           s=4, alpha=0.6)
    plt.colorbar(scatter, label="Digit Class")
    plt.title(f"VAE — Latent Space ({'2D' if latent_dim == 2 else 't-SNE'})")
    plt.tight_layout()
    plt.savefig("vae_latent_space.png", dpi=150)
    plt.show()


def plot_latent_interpolation(model: VAE, test_loader: DataLoader,
                               n_steps: int = 10) -> None:
    """
    Linear interpolation between two latent codes.

    Smooth transitions between decoded images demonstrate the continuity of the
    VAE latent space — a key advantage over regular autoencoders.

    Fine-tuning tip: if the middle interpolations are blurry or nonsensical,
    the latent space is not smooth — increase beta or model capacity.
    """
    model.eval()
    X_batch, _ = next(iter(test_loader))
    z1 = model.encode(X_batch[0:1].to(DEVICE))
    z2 = model.encode(X_batch[1:2].to(DEVICE))

    alphas = np.linspace(0, 1, n_steps)
    interps = []
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            decoded  = model.decode(z_interp).cpu().reshape(28, 28).numpy()
            interps.append(decoded)

    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))
    for i, (ax, img) in enumerate(zip(axes, interps)):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"α={alphas[i]:.1f}", fontsize=7)

    plt.suptitle("VAE — Latent Space Interpolation")
    plt.tight_layout()
    plt.savefig("vae_interpolation.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Variational Autoencoder (VAE) ===")

    train_loader, test_loader = load_mnist(batch_size=TRAINING_CONFIG["batch_size"])

    model = VAE(
        input_dim=784,
        hidden_dims=TRAINING_CONFIG["hidden_dims"],
        latent_dim=TRAINING_CONFIG["latent_dim"],
        beta=TRAINING_CONFIG["beta"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    history = train(
        model, train_loader,
        epochs=TRAINING_CONFIG["epochs"],
        lr=TRAINING_CONFIG["learning_rate"],
        kl_warmup=TRAINING_CONFIG["kl_warmup_epochs"],
    )

    plot_training_history(history)
    plot_reconstructions(model, test_loader)
    plot_generated_samples(model)
    plot_latent_space_2d(model, test_loader, latent_dim=TRAINING_CONFIG["latent_dim"])
    plot_latent_interpolation(model, test_loader)


if __name__ == "__main__":
    main()
