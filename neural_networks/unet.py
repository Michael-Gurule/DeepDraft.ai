"""
U-Net
======

U-Net (Ronneberger et al. 2015) is an encoder-decoder architecture with
skip connections that concatenate encoder feature maps to the corresponding
decoder levels.

Architecture:
  Encoder (contracting path): Conv blocks + MaxPool → progressively downsample
  Bottleneck:                 Deepest Conv block at lowest resolution
  Decoder (expanding path):  UpConv + skip concat + Conv blocks → upsample
  Output:                     1×1 Conv → logits (segmentation map)

Skip connections:
  Each encoder level i produces features F_i; decoder level i receives
  upsampled F_{i+1} concatenated with F_i:
    D_i = ConvBlock( UpConv(D_{i+1}) ⊕ F_i )
  This allows the network to combine high-resolution (early) and
  semantic (deep) features simultaneously.

Variants implemented:
  U-Net (2D):     Original for biomedical image segmentation
  Residual U-Net: Replace ConvBlocks with ResidualBlocks (better gradient flow)
  Attention U-Net: Add attention gates on skip connections (focus on relevant regions)
  Lightweight U-Net: Reduced channels for embedding in diffusion models

Applications:
  - Image segmentation (medical, satellite)
  - Image-to-image translation
  - Denoising backbone (DDPM, score-based diffusion)
  - Depth estimation
  - Super-resolution

Fine-tuning recommendations:
  - base_channels (#Placeholder: 64): scale up for better performance, down for speed
  - n_levels (#Placeholder: 4): depth of the U (resolution halved per level)
  - bilinear (#Placeholder: True): bilinear upsampling + conv vs transposed conv
  - use_residual (#Placeholder: False): ResNet blocks improve deep architectures
  - use_attention (#Placeholder: False): attention gates for non-uniform objects
  - dropout (#Placeholder: 0.1): add at bottleneck and deep decoder levels
  - time_embed_dim (#Placeholder: 256): for diffusion model conditioning

References:
  Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
  He et al. (2016) — Deep Residual Learning (ResNet blocks)
  Oktay et al. (2018) — Attention U-Net
  Ho et al. (2020) — DDPM (uses U-Net as denoising backbone)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CONFIG = {
    "base_channels":    32,     # # Placeholder: initial feature channels
    "n_levels":         4,      # # Placeholder: encoder/decoder depth
    "bilinear":         True,   # # Placeholder: bilinear vs ConvTranspose2d
    "use_residual":     False,  # # Placeholder: residual conv blocks
    "use_attention":    False,  # # Placeholder: attention gates on skips
    "dropout":          0.1,    # # Placeholder:
    "n_classes":        2,      # # Placeholder: output segmentation classes
    "in_channels":      1,      # # Placeholder: input image channels
    "lr":               1e-3,   # # Placeholder:
    "batch_size":       8,      # # Placeholder:
    "n_epochs":         30,     # # Placeholder:
    "img_size":         64,     # # Placeholder: spatial resolution
    "seed":             42,
}


# ─────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Standard U-Net double conv block:
    Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualConvBlock(nn.Module):
    """ResNet-style block: skip connection across double conv."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class AttentionGate(nn.Module):
    """
    Attention gate: reweights skip features using gating signal from decoder.
    α = σ(W_g·g + W_x·x + b)
    Filters out irrelevant skip-connection features.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """g: gating (from decoder), x: skip features (from encoder)."""
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        attn = self.psi(F.relu(self.W_g(g_up) + self.W_x(x)))
        return x * attn


class UpBlock(nn.Module):
    """Decoder upsampling block: upsample + concat skip + ConvBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 bilinear: bool = True, use_residual: bool = False,
                 use_attention: bool = False, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, 1),
            )
            concat_ch = in_ch // 2 + skip_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            concat_ch = in_ch // 2 + skip_ch

        BlockCls = ResidualConvBlock if use_residual else ConvBlock
        self.conv = BlockCls(concat_ch, out_ch, dropout)

        self.attention = AttentionGate(in_ch // 2, skip_ch, skip_ch // 2) if use_attention else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.attention is not None:
            skip = self.attention(x, skip)
        # Handle size mismatch (if input not divisible by 2^n_levels)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    Configurable U-Net with optional residual blocks and attention gates.

    Input:  (B, in_channels, H, W)
    Output: (B, n_classes, H, W)  — logits (apply softmax/sigmoid for probabilities)
    """

    def __init__(
        self,
        in_channels:  int = 1,
        n_classes:    int = 2,
        base_channels: int = 64,
        n_levels:     int = 4,
        bilinear:     bool = True,
        use_residual: bool = False,
        use_attention: bool = False,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.n_levels = n_levels
        BlockCls = ResidualConvBlock if use_residual else ConvBlock

        # Encoder
        enc_channels = [in_channels] + [base_channels * (2 ** i) for i in range(n_levels)]
        self.encoders = nn.ModuleList([
            BlockCls(enc_channels[i], enc_channels[i+1], dropout if i > 0 else 0.0)
            for i in range(n_levels)
        ])
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(n_levels)])

        # Bottleneck
        self.bottleneck = BlockCls(enc_channels[-1], enc_channels[-1] * 2, dropout)

        # Decoder
        dec_in = [enc_channels[-1] * 2] + [enc_channels[-(i+1)] for i in range(1, n_levels)]
        dec_skip = [enc_channels[-(i+1)] for i in range(1, n_levels + 1)]
        dec_out = [enc_channels[-(i+1)] for i in range(1, n_levels + 1)]

        self.decoders = nn.ModuleList([
            UpBlock(dec_in[i], dec_skip[i], dec_out[i],
                    bilinear, use_residual, use_attention, dropout)
            for i in range(n_levels)
        ])

        self.out_conv = nn.Conv2d(base_channels, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.out_conv(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Time-Conditioned U-Net (for Diffusion Models)
# ─────────────────────────────────────────────

class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding used in DDPM / score models."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        emb = t[:, None] * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimedConvBlock(nn.Module):
    """ConvBlock conditioned on timestep embedding."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, out_ch), nn.SiLU())
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(nn.Dropout2d(dropout),
                                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, out_ch), nn.SiLU())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x) + self.time_proj(t_emb)[:, :, None, None]
        return self.conv2(h)


class DiffusionUNet(nn.Module):
    """
    Lightweight time-conditioned U-Net for diffusion model denoising.
    Equivalent to the ε_θ(x_t, t) network in DDPM.
    For full diffusion training, see frontier/diffusion.py.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 n_levels: int = 3, time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.n_levels = n_levels
        ch = [base_channels * (2 ** i) for i in range(n_levels + 1)]

        self.init_conv = nn.Conv2d(in_channels, ch[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([
            TimedConvBlock(ch[i], ch[i+1], time_dim) for i in range(n_levels)
        ])
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(n_levels)])
        self.bottleneck = TimedConvBlock(ch[-1], ch[-1], time_dim)
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(ch[i+1], ch[i+1] // 2, 2, stride=2)
            for i in range(n_levels - 1, -1, -1)
        ])
        self.up_blocks = nn.ModuleList([
            TimedConvBlock(ch[n_levels - i] + ch[n_levels - i] // 2,
                           ch[n_levels - i - 1] if i < n_levels else ch[0], time_dim)
            for i in range(n_levels)
        ])
        self.out_conv = nn.Conv2d(ch[0], in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise ε(x_t, t). t: (batch,) integer or float time steps."""
        t_emb = self.time_embed(t.float())
        x = self.init_conv(x)
        skips = []
        for down, pool in zip(self.down_blocks, self.pools):
            x = down(x, t_emb); skips.append(x); x = pool(x)
        x = self.bottleneck(x, t_emb)
        for up_conv, up_block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            x = up_conv(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, t_emb)
        return self.out_conv(x)


# ─────────────────────────────────────────────
# Synthetic Segmentation Data
# ─────────────────────────────────────────────

class SyntheticCircleDataset(Dataset):
    """
    Binary segmentation: segment circular blobs from background.
    Replace with your real segmentation dataset (e.g., PASCAL VOC, COCO, custom).
    """
    def __init__(self, n: int, img_size: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.images = []
        self.masks  = []
        for _ in range(n):
            img  = rng.normal(0.0, 0.1, (1, img_size, img_size)).astype(np.float32)
            mask = np.zeros((img_size, img_size), dtype=np.int64)
            n_circles = rng.integers(1, 4)
            for _ in range(n_circles):
                cx, cy = rng.integers(10, img_size - 10, 2)
                r = rng.integers(5, 15)
                ys, xs = np.ogrid[:img_size, :img_size]
                circle = (xs - cx)**2 + (ys - cy)**2 < r**2
                img[0, circle] += rng.normal(0.8, 0.1)
                mask[circle] = 1
            self.images.append(torch.tensor(img))
            self.masks.append(torch.tensor(mask))

    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.masks[idx]


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for segmentation:
    L = 1 - (2 |P ∩ T| + ε) / (|P| + |T| + ε)
    Better than cross-entropy for imbalanced foreground/background.
    """
    pred_probs = torch.softmax(pred, dim=1)[:, 1]   # foreground probability
    target_float = target.float()
    intersection = (pred_probs * target_float).sum()
    dice = (2 * intersection + eps) / (pred_probs.sum() + target_float.sum() + eps)
    return 1 - dice


def iou_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Intersection over Union (Jaccard Index)."""
    pred_class = pred.argmax(dim=1)
    intersection = ((pred_class == 1) & (target == 1)).sum().float()
    union = ((pred_class == 1) | (target == 1)).sum().float()
    return float(intersection / (union + 1e-6))


def train_unet(cfg: dict) -> tuple[UNet, dict]:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_ds = SyntheticCircleDataset(400, cfg["img_size"], cfg["seed"])
    val_ds   = SyntheticCircleDataset(100, cfg["img_size"], cfg["seed"] + 1)
    test_ds  = SyntheticCircleDataset(100, cfg["img_size"], cfg["seed"] + 2)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"])
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"])

    model = UNet(
        in_channels=cfg["in_channels"], n_classes=cfg["n_classes"],
        base_channels=cfg["base_channels"], n_levels=cfg["n_levels"],
        bilinear=cfg["bilinear"], use_residual=cfg["use_residual"],
        use_attention=cfg["use_attention"], dropout=cfg["dropout"],
    ).to(DEVICE)
    print(f"U-Net parameters: {model.n_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["n_epochs"])
    train_losses, val_ious = [], []

    for epoch in range(cfg["n_epochs"]):
        model.train()
        ep_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            pred = model(imgs)
            loss = F.cross_entropy(pred, masks) + dice_loss(pred, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        train_losses.append(ep_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            iou = np.mean([iou_score(model(i.to(DEVICE)), m.to(DEVICE))
                           for i, m in val_loader])
        val_ious.append(iou)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['n_epochs']} | Loss: {train_losses[-1]:.4f} | Val IoU: {iou:.4f}")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_iou = np.mean([iou_score(model(i.to(DEVICE)), m.to(DEVICE))
                             for i, m in test_loader])
    print(f"\nTest IoU: {test_iou:.4f}")

    return model, {"train_losses": train_losses, "val_ious": val_ious, "test_iou": test_iou}


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_segmentation(model: UNet, dataset, n: int = 4):
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 8))
    fig.suptitle("U-Net Segmentation — Image / True Mask / Predicted", fontsize=11)
    for i in range(n):
        img, mask = dataset[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(DEVICE)).cpu()
        pred_mask = pred.argmax(dim=1)[0].numpy()
        axes[0, i].imshow(img[0].numpy(), cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(mask.numpy(), cmap="gray");   axes[1, i].axis("off")
        axes[2, i].imshow(pred_mask, cmap="gray");       axes[2, i].axis("off")
    axes[0, 0].set_title("Image"); axes[1, 0].set_title("True"); axes[2, 0].set_title("Pred")
    plt.tight_layout()
    plt.savefig("unet_segmentation.png", dpi=150)
    plt.show()
    print("Saved: unet_segmentation.png")


def plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("U-Net Training", fontsize=12)
    axes[0].plot(history["train_losses"], color="steelblue")
    axes[0].set_title("Training Loss (CE + Dice)"); axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["val_ious"], color="darkorange")
    axes[1].set_title("Validation IoU"); axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("unet_training.png", dpi=150)
    plt.show()
    print("Saved: unet_training.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG["seed"])
    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}\n")

    # ── Standard U-Net segmentation ─────────────────────────────────
    print("=== U-Net — Synthetic Circle Segmentation ===")
    model, history = train_unet(CONFIG)
    plot_training_curves(history)
    test_ds = SyntheticCircleDataset(20, CONFIG["img_size"], seed=99)
    plot_segmentation(model, test_ds)

    # ── DiffusionUNet parameter count ───────────────────────────────
    diff_unet = DiffusionUNet(in_channels=1, base_channels=32, n_levels=3, time_dim=128)
    n_params = sum(p.numel() for p in diff_unet.parameters() if p.requires_grad)
    print(f"\nDiffusion U-Net parameters: {n_params:,}")

    # ── Attention U-Net comparison ──────────────────────────────────
    print("\n=== Attention U-Net ===")
    attn_model, attn_history = train_unet({**CONFIG, "use_attention": True, "n_epochs": 20})
    print(f"Attention U-Net Test IoU: {attn_history['test_iou']:.4f}")
    print(f"Standard U-Net Test IoU:  {history['test_iou']:.4f}")


if __name__ == "__main__":
    main()
