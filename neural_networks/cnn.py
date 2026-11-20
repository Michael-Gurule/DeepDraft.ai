"""
Convolutional Neural Network (CNN)
====================================
A specialized neural network architecture for grid-like data (images, audio
spectrograms, 1D signals). Uses convolutional layers to learn spatial feature
hierarchies — from low-level edges/textures to high-level semantic features.

Core Operations:
    Convolution:   applies learnable filters across the input, detecting features
                   regardless of their spatial position (translation equivariance)
    Pooling:       reduces spatial dimensions; provides translation invariance
                   and reduces computational cost
    Batch Norm:    normalizes activations; stabilizes and accelerates training
    Global Avg Pool: collapses spatial dimensions before the classifier head;
                     more parameter-efficient than fully-connected flattening

Architecture (generic CNN for image classification):
    Input → [Conv → BN → ReLU → MaxPool] × N → GlobalAvgPool → [FC → Dropout] → Output

Use Cases:
    - Image classification (MNIST, CIFAR, ImageNet)
    - Object detection (backbone networks)
    - Medical image analysis
    - Audio classification (mel-spectrogram CNNs)
    - Time-series classification (1D CNN)
    - Document classification (character/word CNNs)

Fine-Tuning Recommendations:
    - Depth: start with 3–4 conv blocks; add more for harder tasks.
    - Filters: typical progression is 32→64→128→256.
      Wider (more filters) = more features detected per layer.
    - Kernel size: 3×3 is universal best practice (deep vs. wide filter study).
      Use 5×5 or 7×7 only in the first layer for large images.
    - Activation: ReLU after each conv; GELU in transformer-hybrid models.
    - Pooling: MaxPool2d(2,2) halves spatial dims. Replace with stride=2 conv
      for learnable downsampling.
    - Data augmentation: horizontal flips, crops, color jitter, cutout, mixup.
      Critical for small datasets. See torchvision.transforms.
    - Learning rate: start at 1e-3, decay with cosine annealing or step schedule.
    - Transfer learning: for small datasets (<10k images), use pretrained
      ResNet/EfficientNet as feature extractor and only fine-tune the head.
      Fine-tune all layers only with > 10k samples.
    - Batch size: 64–256 for images; larger is more stable but needs higher LR.
    - Regularization: weight decay 1e-4, dropout 0.3–0.5 before final FC layers.
    - Normalization: always normalize inputs to [0,1] then apply
      channel-wise mean/std normalization (e.g., ImageNet stats for transfer learning).
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

def load_mnist(data_root: str = "./data", batch_size: int = 64):
    """
    Load the MNIST handwritten digit dataset.

    MNIST: 60,000 training, 10,000 test 28×28 grayscale images, 10 classes.
    Used here as a standard benchmark. Replace with your own dataset using
    torchvision.datasets.ImageFolder for custom image directories:

        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder("path/to/train", transform=train_transform)

    Placeholder: replace these transforms with your own augmentation pipeline.
    For CIFAR-10 / larger images, add RandomCrop, ColorJitter, etc.

    Args:
        data_root:  directory to download/load MNIST from
        batch_size: samples per mini-batch

    Returns:
        train_loader, val_loader, test_loader, n_classes
    """
    # Training transforms: normalize + light augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # data augmentation
        transforms.ToTensor(),
        # Normalize to zero mean, unit std (MNIST-specific stats)
        # Placeholder: change to ImageNet stats for transfer learning:
        #   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Test transforms: only normalization (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_full = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
    test_ds    = datasets.MNIST(data_root, train=False, download=True, transform=test_transform)

    # Split training into train (90%) and validation (10%)
    n_val    = int(0.1 * len(train_full))
    n_train  = len(train_full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, 10


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Reusable convolutional block: Conv → BatchNorm → Activation → (optional MaxPool).

    This modular design makes it easy to add/remove layers and swap components.

    Args:
        in_channels:  number of input feature maps
        out_channels: number of output feature maps (filters)
        kernel_size:  size of the convolving kernel (usually 3 or 5)
        pool:         whether to apply MaxPool2d(2,2) after activation
        activation:   activation function class
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # 'same' padding: output has same spatial size
                bias=False,  # bias=False is standard when using BatchNorm
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    Generic CNN for image classification.

    Architecture:
        ConvBlock(32) → ConvBlock(64) → ConvBlock(128)
        → GlobalAvgPool → FC(256) → Dropout → FC(n_classes)

    Placeholder: adjust channels, depth, and FC size based on task complexity.
    - Simpler tasks (MNIST): 2 conv blocks, 32–64 channels
    - Complex tasks (ImageNet): ResNet/EfficientNet or 5+ blocks with residuals

    Args:
        in_channels: number of input channels (1=grayscale, 3=RGB)
        n_classes:   number of output classes
        dropout:     dropout probability before the output layer
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes:   int = 10,
        dropout:     float = 0.3,
    ):
        super().__init__()

        # ── Feature Extractor ──────────────────────────────────────
        # Progressively learn more abstract features:
        #   Block 1: 32 filters → low-level features (edges, corners)
        #   Block 2: 64 filters → mid-level features (shapes, textures)
        #   Block 3: 128 filters → high-level features (object parts)
        # Placeholder: add/remove blocks or change channel sizes here
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, pool=True),
            ConvBlock(32, 64, kernel_size=3, pool=True),
            ConvBlock(64, 128, kernel_size=3, pool=False),  # no pooling on last conv
        )

        # ── Global Average Pooling ─────────────────────────────────
        # Collapses (B, C, H, W) → (B, C) by averaging over spatial dims.
        # More parameter-efficient than nn.Flatten() + nn.Linear(C*H*W, ...).
        # Also provides implicit regularization (reduces overfitting).
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Classifier Head ────────────────────────────────────────
        # The head maps the feature vector to class probabilities.
        # Placeholder: replace with a single linear layer for simple tasks.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "epochs":         20,              # Placeholder: increase for harder tasks
    "learning_rate":  1e-3,
    "weight_decay":   1e-4,
    "patience":       10,              # early stopping patience
    "lr_milestones":  [10, 15],        # epoch numbers to reduce LR by lr_gamma
    "lr_gamma":       0.1,             # LR multiplier at each milestone
}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
) -> dict:
    """
    Training loop with step LR scheduling and early stopping.

    Fine-tuning tip: for image tasks, cosine annealing LR schedule often
    outperforms step decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Args:
        model:        CNN instance
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        epochs:       maximum training epochs
        lr:           initial learning rate
        weight_decay: L2 regularization
        patience:     early stopping patience

    Returns:
        dict with training history
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # label_smoothing=0.1: replaces hard labels with soft distribution.
    # Reduces overconfidence and improves calibration.
    # Placeholder: set to 0.0 to disable.

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Step LR: multiply LR by gamma at each milestone epoch
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=TRAINING_CONFIG["lr_milestones"],
        gamma=TRAINING_CONFIG["lr_gamma"]
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    for epoch in range(1, epochs + 1):
        # ── Train ───────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(X_batch)

        train_loss /= total
        train_acc   = correct / total

        # ── Validate ────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits  = model(X_batch)
                loss    = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                correct  += (logits.argmax(1) == y_batch).sum().item()
                total    += len(X_batch)

        val_loss /= total
        val_acc   = correct / total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train: loss={train_loss:.4f} acc={train_acc:.4f} "
                  f"| Val: loss={val_loss:.4f} acc={val_acc:.4f} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ── Early Stopping ──────────────────────────────────────────
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

def evaluate(model: nn.Module, test_loader: DataLoader) -> dict:
    """
    Evaluate the CNN on the test set.

    Args:
        model:       trained CNN
        test_loader: test DataLoader

    Returns:
        dict of metrics
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(DEVICE)).cpu()
            all_preds.append(logits.argmax(1).numpy())
            all_labels.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = (y_pred == y_true).mean()
    print(f"  Test Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"accuracy": acc}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    """Plot loss and accuracy curves during training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.suptitle("CNN — Training Curves")
    plt.tight_layout()
    plt.savefig("cnn_training_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model: nn.Module, test_loader: DataLoader,
                           class_names: list | None = None) -> None:
    """Display confusion matrix for the test set."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("CNN — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("cnn_confusion.png", dpi=150)
    plt.show()


def visualize_filters(model: nn.Module) -> None:
    """
    Visualize the first-layer convolutional filters.

    First-layer filters are directly interpretable because they operate on
    raw pixels. They typically learn Gabor-like edge and orientation detectors.

    Fine-tuning insight: if the first-layer filters look random/noisy after
    training, the model may not have converged or the learning rate was too high.
    """
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break

    if first_conv is None or first_conv.weight.shape[0] < 4:
        print("Not enough filters to visualize. Skipping.")
        return

    filters = first_conv.weight.data.cpu()
    n = min(32, filters.shape[0])
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i, ax in enumerate(axes.flat):
        if i < n:
            f = filters[i, 0].numpy()  # grayscale: channel 0
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f, cmap="gray")
        ax.axis("off")

    plt.suptitle("CNN — First Layer Filters")
    plt.tight_layout()
    plt.savefig("cnn_filters.png", dpi=150)
    plt.show()


def visualize_feature_maps(model: nn.Module, sample_image: torch.Tensor) -> None:
    """
    Visualize feature map activations for a sample image.

    Shows what each filter "sees" / activates on in the first conv layer.
    High activation = the corresponding pattern is present at that location.

    Fine-tuning insight: dead feature maps (all zeros) indicate dead neurons
    — possibly from too-high learning rate or poor initialization.

    Args:
        model:        trained CNN
        sample_image: (1, C, H, W) tensor of a single image
    """
    model.eval()
    activations = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach().cpu()

    # Register a forward hook on the first conv layer
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    hook = first_conv.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(sample_image.to(DEVICE))
    hook.remove()

    acts = activations[0]  # (C, H, W)
    n    = min(32, acts.shape[0])
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i, ax in enumerate(axes.flat):
        if i < n:
            a = acts[i].numpy()
            ax.imshow(a, cmap="viridis")
        ax.axis("off")

    plt.suptitle("CNN — First Conv Layer Feature Maps")
    plt.tight_layout()
    plt.savefig("cnn_feature_maps.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MODEL SUMMARY
# ─────────────────────────────────────────────

def print_model_summary(model: nn.Module, input_shape: tuple) -> None:
    """Print parameter count and layer info."""
    print("\nCNN Architecture:")
    print(model)
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nInput shape:          {input_shape}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== CNN — MNIST Image Classification ===")

    # 1. Load data
    train_loader, val_loader, test_loader, n_classes = load_mnist()

    # 2. Build model
    model = SimpleCNN(in_channels=1, n_classes=n_classes, dropout=0.3)
    print_model_summary(model, input_shape=(1, 1, 28, 28))

    # 3. Train
    history = train(
        model, train_loader, val_loader,
        epochs=TRAINING_CONFIG["epochs"],
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        patience=TRAINING_CONFIG["patience"],
    )

    # 4. Evaluate
    print("\nTest Set Metrics:")
    evaluate(model, test_loader)

    # 5. Visualize
    plot_training_curves(history)
    plot_confusion_matrix(model, test_loader, class_names=[str(i) for i in range(10)])
    visualize_filters(model)

    # Visualize feature maps for one sample
    sample_img, _ = next(iter(test_loader))
    visualize_feature_maps(model, sample_img[:1])


if __name__ == "__main__":
    main()
