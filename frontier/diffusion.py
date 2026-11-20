"""
Denoising Diffusion Probabilistic Model (DDPM)
================================================
The class of generative models underlying Stable Diffusion, DALL-E 2, and Imagen.
Learns to reverse a gradual noising process, turning Gaussian noise back into data.

Forward process (adding noise):
    q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t)*x_{t-1}, β_t*I)
    After T steps, x_T ≈ N(0, I) regardless of the original x_0.

Reverse process (learned denoising):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_t * I)
    A neural network (typically U-Net) predicts the noise ε at each step.

Training objective (simplified):
    L = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]
    → Train the network to predict the noise that was added at step t.

Sampling (inference):
    Start with x_T ~ N(0, I), then iteratively denoise:
    x_{t-1} = (1/sqrt(α_t)) * (x_t - (1-α_t)/sqrt(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z

Key insight: instead of learning to generate data directly, the model learns
to remove noise — a much simpler per-step regression problem.

Use Cases:
    - High-quality image synthesis (state-of-the-art)
    - Text-to-image generation (DALL-E, Stable Diffusion)
    - Audio generation (DiffWave, WaveGrad)
    - Molecular design / protein structure prediction
    - Image inpainting and super-resolution

Advantages:
    - Higher sample quality and diversity than GANs
    - Stable training (no mode collapse, no adversarial dynamics)
    - Controllable generation via conditioning (text, class labels)

Limitations:
    - Slow inference: T=1000 denoising steps (DDIM reduces this to ~50 steps)
    - High GPU memory requirements for large models
    - Training requires significant compute

Fine-Tuning Recommendations:
    - T (timesteps): 1000 (training) standard. DDIM sampling can use 50–200 at inference.
    - Beta schedule:
        'linear': β linearly from 1e-4 to 0.02 (original DDPM)
        'cosine': more uniform noise schedule; better for small/simple datasets
    - U-Net architecture: the denoising backbone. Wider channels = more capacity.
    - Time embedding: sinusoidal + MLP, similar to positional encoding. Must be
      injected into every level of the U-Net.
    - Conditioning: inject class/text embeddings via cross-attention or AdaLN.
    - Loss weighting: 'simple' (MSE on noise prediction) or 'vlb' (full ELBO).
      'simple' works best in practice for sample quality.
    - Classifier-free guidance: train with 10% unconditional; at inference use
      combined score: ε = ε_uncond + w * (ε_cond - ε_uncond). Higher w → sharper, less diverse.
    - For large image resolutions (>64px): use latent diffusion (Stable Diffusion)
      — operate in the compressed latent space of a VAE.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# NOISE SCHEDULE
# ─────────────────────────────────────────────

class NoiseSchedule:
    """
    Precomputes all noise schedule quantities for the diffusion process.

    Linear schedule (original DDPM):
        β_t linearly from β_start to β_end over T steps.

    Cosine schedule (Nichol & Dhariwal, 2021):
        α̅_t = cos((t/T + s)/(1+s) * π/2)²
        Better for small images — the linear schedule destroys too much
        signal too quickly.

    Precomputed quantities (all of shape (T,)):
        betas:         β_t (noise variance added at each step)
        alphas:        α_t = 1 - β_t
        alphas_cumprod:ᾱ_t = Π_{s=1}^t α_s
        sqrt_alphas_cumprod:      used in forward process: x_t = sqrt(ᾱ_t)*x_0 + ...
        sqrt_one_minus_alphas_cumprod: ... + sqrt(1-ᾱ_t)*ε
        sqrt_recip_alphas:        used in reverse process
        posterior_variance:       σ²_t used in sampling

    Args:
        T:          number of diffusion timesteps
        schedule:   'linear' or 'cosine'
        beta_start: starting β value (linear schedule only)
        beta_end:   ending β value (linear schedule only)
    """

    def __init__(
        self,
        T: int = 1000,
        schedule: str = "cosine",      # Placeholder: 'linear' or 'cosine'
        beta_start: float = 1e-4,      # linear schedule params
        beta_end: float = 0.02,
    ):
        self.T = T

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            # Cosine schedule: ᾱ_t follows a cosine curve
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999).float()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas           = 1.0 - betas
        alphas_cumprod   = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Quantities used in forward process q(x_t | x_0)
        self.betas                     = betas
        self.alphas_cumprod            = alphas_cumprod
        self.sqrt_alphas_cumprod       = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        # Quantities used in reverse process (sampling)
        self.sqrt_recip_alphas         = torch.sqrt(1.0 / alphas)
        self.posterior_variance        = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor | None = None) -> tuple:
        """
        Forward diffusion: sample x_t from x_0 at timestep t.

        Closed form: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        This allows sampling any timestep t directly (no iterative noise addition).

        Args:
            x_0:   original data (batch, channels, H, W)
            t:     timestep indices (batch,) — one per sample
            noise: optional pre-sampled noise ε ~ N(0,I)

        Returns:
            x_t: noisy version of x_0 at timestep t
            noise: the noise that was added (target for training)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract batch-indexed schedule values
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_0.device)
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_0.device)

        x_t = sqrt_acp * x_0 + sqrt_omc * noise
        return x_t, noise

    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single reverse step: sample x_{t-1} from x_t using the trained model.

        x_{t-1} = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z

        Args:
            model: trained denoising network
            x_t:   current noisy sample (batch, channels, H, W)
            t:     current timestep integer

        Returns:
            x_{t-1}: less noisy sample
        """
        t_batch = torch.full((x_t.size(0),), t, dtype=torch.long, device=x_t.device)

        with torch.no_grad():
            eps_pred = model(x_t, t_batch)   # predict noise

        beta_t    = self.betas[t].to(x_t.device)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].to(x_t.device)
        sqrt_omc  = self.sqrt_one_minus_alphas_cumprod[t].to(x_t.device)

        # Predicted mean of x_{t-1}
        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_omc * eps_pred)

        if t == 0:
            return mean  # no noise at final step
        else:
            noise    = torch.randn_like(x_t)
            variance = self.posterior_variance[t].to(x_t.device)
            return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: tuple) -> torch.Tensor:
        """
        Full reverse process: generate samples from x_T ~ N(0, I).

        Iterates from t=T-1 to t=0, applying p_sample at each step.
        This is the inference algorithm — generates new images.

        For faster inference, use ddim_sample_loop (50 steps ≈ same quality).

        Args:
            model: trained denoising network
            shape: (batch, channels, H, W) output shape

        Returns:
            x_0: generated samples
        """
        x = torch.randn(shape, device=DEVICE)

        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)

        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        n_steps: int = 50,          # Placeholder: 20–100; 50 is the standard sweet spot
        eta: float = 0.0,           # Placeholder: 0.0 = deterministic DDIM; 1.0 = DDPM
    ) -> torch.Tensor:
        """
        DDIM deterministic sampling (Song et al., 2020).

        Uses a subsequence of T timesteps instead of all T, giving ~20× speedup
        with comparable sample quality. When eta=0 the process is fully
        deterministic: the same x_T always produces the same x_0.

        DDIM update (from x_t to x_{t-1} via predicted x_0):
            x̂_0 = (x_t - sqrt(1-ᾱ_t) * ε_θ) / sqrt(ᾱ_t)       [predicted x_0]
            σ_t  = eta * sqrt((1-ᾱ_{t-1}) / (1-ᾱ_t)) * sqrt(1-ᾱ_t/ᾱ_{t-1})
            dir  = sqrt(1 - ᾱ_{t-1} - σ_t²) * ε_θ                [direction]
            x_{t-1} = sqrt(ᾱ_{t-1}) * x̂_0 + dir + σ_t * z        [step]

        When eta=1 and the subsequence equals the full T steps, this reduces
        exactly to DDPM sampling.

        Args:
            model:   trained denoising network
            shape:   (batch, channels, H, W) output shape
            n_steps: number of inference steps (<<T, e.g. 50)
            eta:     stochasticity: 0.0 = deterministic, 1.0 = DDPM-equivalent

        Returns:
            x_0: generated samples
        """
        # Build evenly-spaced subsequence [T-1, ..., 0] of length n_steps
        step_size = self.T // n_steps
        timesteps = list(reversed(range(0, self.T, step_size)))[:n_steps]

        x = torch.randn(shape, device=DEVICE)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            t_batch = torch.full((shape[0],), t, dtype=torch.long, device=DEVICE)
            eps = model(x, t_batch)

            acp_t = self.alphas_cumprod[t].to(DEVICE)
            acp_prev = self.alphas_cumprod[t_prev].to(DEVICE) if t_prev >= 0 else torch.tensor(1.0, device=DEVICE)

            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - acp_t) * eps) / torch.sqrt(acp_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)  # Placeholder: clipping stabilises quality

            # Compute sigma (DDPM-style noise when eta>0)
            sigma = eta * torch.sqrt(
                (1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev)
            )

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - acp_prev - sigma ** 2) * eps

            # DDIM step
            noise = torch.randn_like(x) if (eta > 0 and t > 0) else torch.zeros_like(x)
            x = torch.sqrt(acp_prev) * x0_pred + dir_xt + sigma * noise

        return x


# ─────────────────────────────────────────────
# DENOISING U-NET
# ─────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding: encodes the timestep t as a vector.

    Critical for the denoising network — the model must know which noise
    level it's operating at to correctly predict the noise.

    Same formulation as Transformer positional encodings, but for time.

    Args:
        dim: output embedding dimension (must be even)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) integer timesteps

        Returns:
            (batch, dim) sinusoidal embeddings
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        ).float()
        emb = t.float()[:, None] * freqs[None, :]  # (batch, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for DDPM on MNIST (28×28 grayscale).

    U-Net architecture:
        Encoder: downsampling path (conv + pool)
        Bottleneck: deepest feature representation
        Decoder: upsampling path (transposed conv + skip connections)
        Skip connections: pass encoder features directly to the decoder

    Skip connections are critical: they allow the decoder to use both
    the high-level bottleneck features AND the low-level encoder features,
    enabling precise spatial localization.

    Time conditioning: at every level, add the time embedding to the
    feature maps so the model knows which timestep it's denoising.

    Placeholder: for real use cases (64×64+), use a full U-Net with:
        - Residual blocks instead of plain conv blocks
        - Self-attention at spatial resolution 16×16 and 8×8
        - GroupNorm instead of BatchNorm (works better for small batches)

    Args:
        in_channels:  input image channels (1 = grayscale)
        base_channels: width of the network at the first level
        time_emb_dim:  time embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,    # Placeholder: 128–256 for higher quality
        time_emb_dim: int = 128,
    ):
        super().__init__()
        c = base_channels

        # Time embedding MLP: sinusoidal → linear → SiLU → linear
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        # Project time embedding to match each encoder/decoder level
        self.t_proj1 = nn.Linear(time_emb_dim, c)
        self.t_proj2 = nn.Linear(time_emb_dim, c * 2)
        self.t_proj3 = nn.Linear(time_emb_dim, c * 4)

        # ── Encoder ────────────────────────────────────────────────
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GroupNorm(8, c*2), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(c*2, c * 4, 3, stride=2, padding=1), nn.GroupNorm(8, c*4), nn.SiLU())

        # ── Bottleneck ─────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c*4, c*4, 3, padding=1), nn.GroupNorm(8, c*4), nn.SiLU(),
            nn.Conv2d(c*4, c*4, 3, padding=1), nn.GroupNorm(8, c*4), nn.SiLU(),
        )

        # ── Decoder ─────────────────────────────────────────────────
        # Skip connections: concat encoder output → double input channels
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(c*8, c*2, 2, stride=2), nn.GroupNorm(8, c*2), nn.SiLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(c*4, c,   2, stride=2), nn.GroupNorm(8, c),   nn.SiLU())
        self.dec1 = nn.Sequential(nn.Conv2d(c*2, c, 3, padding=1), nn.SiLU())

        # Output: predict noise ε (same shape as input)
        self.output = nn.Conv2d(c, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise ε given noisy input x_t and timestep t.

        Args:
            x: (batch, channels, H, W) noisy input at timestep t
            t: (batch,) integer timestep indices

        Returns:
            noise_pred: (batch, channels, H, W) predicted noise
        """
        t_emb = self.time_mlp(t)  # (batch, time_emb_dim)

        # ── Encode + inject time ────────────────────────────────────
        e1 = self.enc1(x)
        # Add time embedding: broadcast (batch, c) → (batch, c, H, W)
        e1 = e1 + self.t_proj1(t_emb)[:, :, None, None]

        e2 = self.enc2(e1)
        e2 = e2 + self.t_proj2(t_emb)[:, :, None, None]

        e3 = self.enc3(e2)
        e3 = e3 + self.t_proj3(t_emb)[:, :, None, None]

        # ── Bottleneck ──────────────────────────────────────────────
        b = self.bottleneck(e3)

        # ── Decode with skip connections ────────────────────────────
        d3 = self.dec3(torch.cat([b, e3], dim=1))   # skip from enc3
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # skip from enc2
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # skip from enc1

        return self.output(d1)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "T":             1000,
    "schedule":      "cosine",
    "epochs":        15,              # Placeholder: increase to 100+ for full MNIST quality
    "learning_rate": 2e-4,
    "batch_size":    128,
    "base_channels": 32,              # Placeholder: 64–128 for better quality
}


def load_mnist(data_root: str = "./data", batch_size: int = 128):
    """Load MNIST normalized to [-1, 1] for diffusion training."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] → [-1,1]
    ])
    ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


def train(
    model: nn.Module,
    schedule: NoiseSchedule,
    train_loader: DataLoader,
    epochs: int = 15,
    lr: float = 2e-4,
) -> dict:
    """
    DDPM training loop.

    Training procedure (per batch):
        1. Sample random timesteps t ~ Uniform(0, T-1)
        2. Sample noise ε ~ N(0, I)
        3. Compute noisy image x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε
        4. Predict ε̂ = model(x_t, t)
        5. Loss = ||ε - ε̂||²  (MSE between true and predicted noise)

    This is the "noise prediction" (ε-parameterization) formulation.
    Alternative: predict x_0 directly or the score function.

    Args:
        model:        U-Net denoising network
        schedule:     NoiseSchedule instance
        train_loader: DataLoader of real images
        epochs:       number of training epochs
        lr:           learning rate

    Returns:
        training history
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0

        for X_batch, _ in train_loader:
            X_batch = X_batch.to(DEVICE)
            batch   = X_batch.size(0)

            # Sample random timesteps (uniform over [0, T-1])
            t = torch.randint(0, schedule.T, (batch,), device=DEVICE)

            # Forward diffusion: add noise
            noise   = torch.randn_like(X_batch)
            x_t, _  = schedule.q_sample(X_batch, t, noise)

            # Predict the noise
            optimizer.zero_grad()
            eps_pred = model(x_t, t)

            # MSE loss between true noise and predicted noise
            loss = F.mse_loss(eps_pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_sum += loss.item() * batch
            n        += batch

        avg_loss = loss_sum / n
        history["loss"].append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.5f}")

    return history


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_forward_process(schedule: NoiseSchedule, sample: torch.Tensor,
                          timesteps: list = None) -> None:
    """
    Visualize the forward noising process at different timesteps.

    Shows how the original image is gradually destroyed by Gaussian noise.
    x_T should look nearly indistinguishable from random noise.
    """
    timesteps = timesteps or [0, 100, 250, 500, 750, 999]
    x_0 = sample.unsqueeze(0)  # add batch dim

    fig, axes = plt.subplots(1, len(timesteps), figsize=(2.5 * len(timesteps), 3))
    for ax, t_val in zip(axes, timesteps):
        t = torch.tensor([t_val])
        x_t, _ = schedule.q_sample(x_0, t)
        img = (x_t.squeeze().cpu() + 1) / 2.0  # [-1,1] → [0,1]
        ax.imshow(img.clamp(0, 1), cmap="gray")
        ax.set_title(f"t={t_val}")
        ax.axis("off")

    plt.suptitle("DDPM — Forward Process (Noising)")
    plt.tight_layout()
    plt.savefig("ddpm_forward_process.png", dpi=150)
    plt.show()


def plot_reverse_process(model: nn.Module, schedule: NoiseSchedule,
                          n_samples: int = 16, use_ddim: bool = True,
                          ddim_steps: int = 50, eta: float = 0.0) -> None:
    """
    Generate samples via DDIM (fast, default) or full DDPM reverse process.

    DDIM with 50 steps is ~20× faster than DDPM with 1000 steps and produces
    comparable quality. Set use_ddim=False to use the full DDPM process.

    Args:
        use_ddim:    True = DDIM (fast); False = DDPM (slow, full T steps)
        ddim_steps:  number of DDIM inference steps (ignored if use_ddim=False)
        eta:         0.0 = deterministic DDIM; 1.0 = DDPM-equivalent stochastic
    """
    model.eval()
    shape = (n_samples, 1, 28, 28)

    if use_ddim:
        print(f"  DDIM sampling {n_samples} images ({ddim_steps} steps, eta={eta})...")
        samples = schedule.ddim_sample_loop(model, shape, n_steps=ddim_steps, eta=eta)
        title = f"DDIM Generated Samples ({ddim_steps} steps, η={eta})"
        fname = "ddim_generated_samples.png"
    else:
        print(f"  DDPM sampling {n_samples} images (T={schedule.T} steps)...")
        samples = schedule.p_sample_loop(model, shape)
        title = "DDPM — Generated Samples"
        fname = "ddpm_generated_samples.png"

    samples = (samples.cpu() + 1) / 2.0  # [-1,1] → [0,1]
    grid    = make_grid(samples.clamp(0, 1), nrow=4, padding=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved {fname}")


def plot_loss_curve(history: dict) -> None:
    """Plot training loss over epochs."""
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"])
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("DDPM — Training Loss")
    plt.tight_layout()
    plt.savefig("ddpm_training_loss.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Denoising Diffusion Probabilistic Model (DDPM) — MNIST ===")

    # 1. Data
    train_loader = load_mnist(batch_size=TRAINING_CONFIG["batch_size"])

    # 2. Noise schedule
    schedule = NoiseSchedule(T=TRAINING_CONFIG["T"], schedule=TRAINING_CONFIG["schedule"])

    # 3. Model
    model = SimpleUNet(in_channels=1, base_channels=TRAINING_CONFIG["base_channels"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {total_params:,}")

    # 4. Visualize forward process
    X_demo, _ = next(iter(train_loader))
    plot_forward_process(schedule, X_demo[0])

    # 5. Train
    history = train(
        model, schedule, train_loader,
        epochs=TRAINING_CONFIG["epochs"],
        lr=TRAINING_CONFIG["learning_rate"],
    )
    plot_loss_curve(history)

    # 6. Generate samples via DDIM (fast) and compare with DDPM
    print("\nGenerating samples via DDIM (50 steps, deterministic)...")
    plot_reverse_process(model, schedule, n_samples=16, use_ddim=True,
                         ddim_steps=50, eta=0.0)

    # Stochastic DDIM (eta=1.0 ≈ DDPM at matching steps)
    print("\nGenerating samples via DDIM (50 steps, stochastic, eta=1.0)...")
    plot_reverse_process(model, schedule, n_samples=16, use_ddim=True,
                         ddim_steps=50, eta=1.0)

    # Full DDPM (slow — uncomment for quality comparison)
    # print("\nGenerating samples via full DDPM (T=1000 steps)...")
    # plot_reverse_process(model, schedule, n_samples=16, use_ddim=False)


if __name__ == "__main__":
    main()
