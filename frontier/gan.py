"""
Generative Adversarial Network (GAN)
======================================
A framework where two networks are trained adversarially:
    Generator (G):     learns to produce realistic fake samples from noise
    Discriminator (D): learns to distinguish real from fake samples

Training dynamics (minimax game):
    D maximizes: E[log D(x)] + E[log(1 - D(G(z)))]
    G minimizes: E[log(1 - D(G(z)))]  ← or maximizes E[log D(G(z))] (non-saturating)

The non-saturating generator loss is used in practice:
    G loss = -E[log D(G(z))]  ← better gradients early in training

Equilibrium: D cannot distinguish real from fake (D(x) = 0.5 everywhere).

This implementation includes:
    - DCGAN: deep convolutional GAN (standard image GAN)
    - Wasserstein GAN with gradient penalty (WGAN-GP): more stable training
    - Training diagnostics (FID not included but noted)

Use Cases:
    - Image synthesis and super-resolution
    - Data augmentation (generate rare/minority class samples)
    - Style transfer
    - Image-to-image translation (pix2pix, CycleGAN)
    - Video generation
    - Anomaly detection (reconstruction error from G)

Known Challenges:
    - Mode collapse: G generates limited variety (ignores parts of the data distribution)
    - Training instability: G and D loss oscillate without converging
    - Evaluation: no single loss metric indicates quality — requires FID, IS, or human eval

Fine-Tuning Recommendations:
    - Architecture: use DCGAN (batch norm in G, instance norm in D or no norm)
    - Learning rates: D typically trains faster; use 2e-4 for G, 2e-4 for D (DCGAN)
    - Update ratio: train D multiple times per G update (e.g., n_critic=5 for WGAN-GP)
    - WGAN-GP: more stable than vanilla GAN; use when training is unstable
    - Noise dimension (latent_dim): 100–256. Larger = more capacity but slower.
    - Label smoothing for D: use 0.9 instead of 1.0 for real labels (reduces overconfidence)
    - Add noise to D inputs: adds robustness to D's training signal
    - Spectral normalization on D: stabilizes GAN training (alternative to WGAN-GP)
    - Progressive growing (ProGAN): train at low resolution first, then increase
    - Mode collapse fix: use minibatch discrimination or feature matching loss
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DEVICE
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
    Load MNIST normalized to [-1, 1].

    GAN generators typically use Tanh output activation, which outputs in [-1, 1].
    Matching the data range to [-1, 1] is therefore important for training stability.

    Placeholder: replace with any image dataset using torchvision.datasets.ImageFolder.

    Returns:
        DataLoader for training
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # normalize to [-1, 1]
    ])
    dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


# ─────────────────────────────────────────────
# DCGAN ARCHITECTURE
# ─────────────────────────────────────────────

class DCGANGenerator(nn.Module):
    """
    DCGAN Generator for MNIST (1×28×28).

    Architecture: z → FC → Reshape → [ConvTranspose → BN → ReLU] × N → Tanh

    ConvTranspose2d (transposed convolution / "deconvolution"):
        Upsamples feature maps by inserting learned values between pixels.
        Alternative: nn.Upsample + nn.Conv2d (avoids checkerboard artifacts).

    Placeholder: adjust latent_dim and channel sizes for other image resolutions.
    For 64×64 images: add more upsampling blocks. For 256×256: even more.

    Args:
        latent_dim:     dimension of the input noise vector z
        base_channels:  base number of channels (doubled each layer going forward)
        img_channels:   number of output image channels (1=grayscale, 3=RGB)
    """

    def __init__(
        self,
        latent_dim: int = 100,
        base_channels: int = 64,    # Placeholder: increase for larger images
        img_channels: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Map z to initial feature map
        self.project = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 4 * 7 * 7),
            nn.ReLU(True),
        )

        # Upsample: 7×7 → 14×14 → 28×28
        self.main = nn.Sequential(
            # (B, base*4, 7, 7)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            # (B, base*2, 14, 14)
            nn.ConvTranspose2d(base_channels * 2, img_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),  # output in [-1, 1]
            # (B, 1, 28, 28)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """DCGAN paper recommends N(0, 0.02) for all weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate images from noise vectors.

        Args:
            z: (batch, latent_dim) noise vectors

        Returns:
            images: (batch, img_channels, H, W)
        """
        h = self.project(z).view(z.size(0), -1, 7, 7)
        return self.main(h)

    def sample_noise(self, n: int) -> torch.Tensor:
        """Sample n noise vectors z ~ N(0, I)."""
        return torch.randn(n, self.latent_dim, device=next(self.parameters()).device)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator for MNIST.

    Architecture: Image → [Conv → LeakyReLU → (optional BN)] × N → FC → Sigmoid

    Key differences from encoder:
        - LeakyReLU (not ReLU): prevents dead neurons, improves gradient flow
        - No BN on first layer (original DCGAN recommendation)
        - No pooling: use strided convolutions for downsampling
        - Output: scalar probability of being real

    Placeholder: for WGAN-GP, remove the final Sigmoid and
    return raw scores (logits) instead of probabilities.

    Args:
        base_channels:  base number of channels
        img_channels:   number of input image channels (1=grayscale, 3=RGB)
        use_spectral_norm: apply spectral normalization for training stability
    """

    def __init__(
        self,
        base_channels: int = 64,
        img_channels: int = 1,
        use_spectral_norm: bool = False,  # Placeholder: True for more stable training
    ):
        super().__init__()

        def conv_block(in_c, out_c, stride=2, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)]
            if use_spectral_norm:
                layers[0] = nn.utils.spectral_norm(layers[0])
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.main = nn.Sequential(
            # (B, 1, 28, 28)
            *conv_block(img_channels, base_channels, bn=False),     # no BN on input
            # (B, base, 14, 14)
            *conv_block(base_channels, base_channels * 2),
            # (B, base*2, 7, 7)
            nn.Flatten(),
            nn.Linear(base_channels * 2 * 7 * 7, 1),
            nn.Sigmoid(),    # Placeholder: remove for WGAN-GP (use raw logits)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify images as real (→1) or fake (→0).

        Args:
            x: (batch, channels, H, W) image tensor

        Returns:
            scores: (batch, 1) values in [0, 1]
        """
        return self.main(x)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "latent_dim":        100,
    "base_channels":     64,
    "epochs":            30,
    "lr_g":              2e-4,   # generator learning rate
    "lr_d":              2e-4,   # discriminator learning rate
    "beta1":             0.5,    # Adam beta1 (0.5 recommended for GANs, not 0.9)
    "beta2":             0.999,
    "n_critic":          1,      # Placeholder: set to 5 for WGAN-GP
    "label_smoothing":   0.9,    # Placeholder: 0.9 for real labels reduces D overconfidence
    "batch_size":        128,
}


def train_gan(
    G: DCGANGenerator,
    D: DCGANDiscriminator,
    train_loader: DataLoader,
    epochs: int = 30,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    beta1: float = 0.5,
    latent_dim: int = 100,
    label_smoothing: float = 0.9,
    n_critic: int = 1,
) -> dict:
    """
    GAN training loop (DCGAN with non-saturating generator loss).

    Training procedure per batch:
        1. Train D on real + fake data
        2. Train G to fool D (n_critic times per G update for WGAN)

    Non-saturating G loss: maximize log D(G(z)) instead of minimizing log(1-D(G(z)))
    → Better gradient signal early in training when D is strong.

    Fine-tuning tips:
        - If D loss → 0 quickly: D is dominating → lower lr_d or use n_critic < 1
        - If G loss doesn't decrease: G can't fool D → increase G capacity
        - Monitor D(real) ≈ 0.5 and D(fake) ≈ 0.5 at convergence

    Args:
        G, D:            Generator and Discriminator
        train_loader:    DataLoader of real images
        epochs:          training epochs
        lr_g, lr_d:      learning rates
        beta1:           Adam beta1 (0.5 is standard for GANs)
        latent_dim:      noise vector size
        label_smoothing: smooth real labels to this value (0.9 is common)
        n_critic:        D updates per G update

    Returns:
        training history
    """
    G = G.to(DEVICE)
    D = D.to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, 0.999))

    criterion = nn.BCELoss()

    # Fixed noise for consistent sample visualization across epochs
    fixed_z = torch.randn(64, latent_dim, device=DEVICE)

    history = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}

    for epoch in range(1, epochs + 1):
        d_loss_sum, g_loss_sum, d_real_sum, d_fake_sum, n = 0, 0, 0, 0, 0

        for X_real, _ in train_loader:
            X_real = X_real.to(DEVICE)
            batch  = X_real.size(0)

            real_labels = torch.full((batch, 1), label_smoothing, device=DEVICE)  # soft real
            fake_labels = torch.zeros(batch, 1, device=DEVICE)

            # ── Train Discriminator ──────────────────────────────────
            for _ in range(n_critic):
                D.zero_grad()
                d_real  = D(X_real)
                loss_real = criterion(d_real, real_labels)

                z       = G.sample_noise(batch)
                X_fake  = G(z).detach()  # detach: don't backprop into G here
                d_fake  = D(X_fake)
                loss_fake = criterion(d_fake, fake_labels)

                d_loss = loss_real + loss_fake
                d_loss.backward()
                opt_D.step()

            # ── Train Generator ──────────────────────────────────────
            G.zero_grad()
            z      = G.sample_noise(batch)
            X_fake = G(z)
            d_on_fake = D(X_fake)

            # Non-saturating loss: G tries to maximize D(G(z)), i.e., fool D
            g_loss = criterion(d_on_fake, torch.ones(batch, 1, device=DEVICE))
            g_loss.backward()
            opt_G.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            d_real_sum += d_real.mean().item()
            d_fake_sum += d_fake.mean().item()
            n += 1

        history["d_loss"].append(d_loss_sum / n)
        history["g_loss"].append(g_loss_sum / n)
        history["d_real"].append(d_real_sum / n)
        history["d_fake"].append(d_fake_sum / n)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | D_loss: {d_loss_sum/n:.4f} | G_loss: {g_loss_sum/n:.4f} "
                  f"| D(real): {d_real_sum/n:.3f} | D(fake): {d_fake_sum/n:.3f}")

    return history, fixed_z


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    """
    Plot GAN training losses and discriminator outputs over time.

    Interpretation:
        - D_loss and G_loss should oscillate but not diverge
        - D(real) should be near 0.5–0.9 throughout
        - D(fake) should start near 0 and approach 0.5 as G improves
        - If D(real) → 1 and D(fake) → 0 consistently: G is failing (mode collapse or collapse)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["d_loss"]) + 1)

    ax1.plot(epochs, history["d_loss"], label="D Loss")
    ax1.plot(epochs, history["g_loss"], label="G Loss")
    ax1.set_title("GAN Losses"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, history["d_real"], label="D(real)")
    ax2.plot(epochs, history["d_fake"], label="D(fake)")
    ax2.axhline(0.5, color="gray", linestyle="--", label="0.5 (equilibrium)")
    ax2.set_title("Discriminator Outputs"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.suptitle("GAN — Training Diagnostics")
    plt.tight_layout()
    plt.savefig("gan_training_curves.png", dpi=150)
    plt.show()


def plot_generated_samples(G: DCGANGenerator, fixed_z: torch.Tensor,
                            epoch: int = -1) -> None:
    """
    Display a grid of GAN-generated samples.

    Using fixed_z across epochs allows visual comparison of generation quality
    over time — the same latent codes produce increasingly realistic images.
    """
    G.eval()
    with torch.no_grad():
        fakes = G(fixed_z).cpu()

    # Rescale from [-1, 1] to [0, 1] for display
    fakes = (fakes + 1) / 2.0
    grid  = make_grid(fakes, nrow=8, padding=2, normalize=False)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    title = f"GAN — Generated Samples" + (f" (epoch {epoch})" if epoch > 0 else "")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"gan_samples_epoch{epoch if epoch > 0 else 'final'}.png", dpi=150)
    plt.show()
    G.train()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== DCGAN on MNIST ===")

    train_loader = load_mnist(batch_size=TRAINING_CONFIG["batch_size"])

    G = DCGANGenerator(
        latent_dim=TRAINING_CONFIG["latent_dim"],
        base_channels=TRAINING_CONFIG["base_channels"],
    )
    D = DCGANDiscriminator(
        base_channels=TRAINING_CONFIG["base_channels"],
    )

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"G parameters: {g_params:,}")
    print(f"D parameters: {d_params:,}")

    history, fixed_z = train_gan(
        G, D, train_loader,
        epochs=TRAINING_CONFIG["epochs"],
        lr_g=TRAINING_CONFIG["lr_g"],
        lr_d=TRAINING_CONFIG["lr_d"],
        beta1=TRAINING_CONFIG["beta1"],
        latent_dim=TRAINING_CONFIG["latent_dim"],
        label_smoothing=TRAINING_CONFIG["label_smoothing"],
    )

    plot_training_curves(history)
    plot_generated_samples(G, fixed_z, epoch=TRAINING_CONFIG["epochs"])

    # Generate a few individual samples
    with torch.no_grad():
        z_new = G.sample_noise(16)
        new_samples = G(z_new).cpu()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    new_samples = (new_samples + 1) / 2.0
    for i, ax in enumerate(axes.flat):
        ax.imshow(new_samples[i].squeeze().numpy(), cmap="gray")
        ax.axis("off")
    plt.suptitle("GAN — New Samples (fresh noise)")
    plt.tight_layout()
    plt.savefig("gan_new_samples.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
