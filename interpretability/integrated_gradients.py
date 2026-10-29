"""
Integrated Gradients — Attribution for Deep Learning Models
=============================================================
An axiomatic attribution method that explains a neural network's prediction
by assigning each input feature a contribution score — how much that feature
moved the output from a baseline prediction to the actual prediction.

Core idea (Sundararajan et al., 2017):
    Integrate gradients along a straight-line path from a baseline x' to input x:

    IG_i(x) = (x_i - x'_i) × ∫_0^1  ∂F(x' + α(x - x')) / ∂x_i  dα

    The integral is approximated by a Riemann sum over m steps.

Two key axioms satisfied by Integrated Gradients (uniquely so):
    1. Completeness: Σ_i IG_i(x) = F(x) - F(x')
       The attributions sum exactly to the prediction gap from baseline.
    2. Sensitivity: if F(x) ≠ F(x') and x differs from x' in exactly one feature,
       that feature receives a non-zero attribution.

Baseline choice:
    The baseline x' is the "neutral reference" — the point of no information.
    Common choices:
        - Zero vector: appropriate for standardized inputs
        - Black image (zeros): for image models
        - Mean of training set: for tabular data
        - Random noise averaged: reduces sensitivity to baseline choice

Completeness check:
    ||Σ_i IG_i(x) - (F(x) - F(x'))|| should be < ε (typically 1e-4).
    If it fails, increase n_steps.

Use Cases:
    - Debugging neural networks (which input regions drive a prediction?)
    - Medical imaging (highlight lesion regions driving a diagnosis)
    - NLP (which tokens contributed to a sentiment score?)
    - Model auditing for bias (which demographic features are being used?)

Advantages:
    - Theoretically grounded (two axioms guarantee uniqueness)
    - No model architecture modifications required
    - Works for any differentiable model
    - Completeness enables "attribution conservation" analysis

Limitations:
    - Requires a baseline choice — results are relative to x'
    - Gradients can be noisy; smooth gradients via SmoothGrad or averaging
    - Only applicable to differentiable models (not trees, rule-based)
    - Integration along a straight path may not reflect true attribution geometry

Fine-Tuning Recommendations:
    1. n_steps: 50 (default). Use 100–300 for smoother attributions.
       Check completeness error ≤ 1e-3 to confirm sufficient steps.
    2. Baseline: use zeros for standardized inputs; use training mean for
       non-standardized or sparse inputs (text embeddings).
    3. Absolute attributions: use |IG| for importance ranking; signed IG for
       understanding direction of effect.
    4. Smoothing: average IG over multiple baselines (IntGrad × N baselines)
       or combine with SmoothGrad (add noise to input, average).
    5. For images: visualize as a heatmap overlaid on the original image;
       apply threshold to show only top-k% attributions.
"""

from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and split breast cancer classification dataset.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    data = load_breast_cancer()
    scaler = StandardScaler()
    X = scaler.fit_transform(data.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data.target.astype(np.float32), test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, list(data.feature_names)


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class MLP(nn.Module):
    """
    Feedforward MLP for binary classification.

    Architecture: Input → [Linear → BatchNorm → ReLU → Dropout] × n_layers → Linear(1)

    Args:
        input_dim:    number of input features
        hidden_dims:  list of hidden layer sizes
        dropout:      dropout probability between layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,  # Placeholder: [128, 64] → [256, 128, 64] for richer models
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            logits: (batch,) unnormalized scores
        """
        return self.network(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probability for class 1."""
        return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "hidden_dims": [128, 64],
    "dropout":     0.2,
    "epochs":      50,
    "batch_size":  64,
    "lr":          1e-3,
    "n_steps_ig":  100,             # Placeholder: 50–300 for Integrated Gradients
}


def train(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict:
    """
    Train the MLP classifier.

    Args:
        model:      MLP instance
        X_train:    (n, p) training features
        y_train:    (n,) binary labels
        epochs:     number of training epochs
        batch_size: mini-batch size
        lr:         learning rate

    Returns:
        training history
    """
    model = model.to(DEVICE)
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    dataset    = TensorDataset(X_t, y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history   = {"loss": [], "acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0

        for X_b, y_b in dataloader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = F.binary_cross_entropy_with_logits(logits, y_b)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y_b)
            correct  += ((logits > 0).float() == y_b).sum().item()
            n        += len(y_b)

        scheduler.step()
        history["loss"].append(loss_sum / n)
        history["acc"].append(correct / n)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss_sum/n:.4f} | Acc: {correct/n:.3f}")

    return history


# ─────────────────────────────────────────────
# INTEGRATED GRADIENTS
# ─────────────────────────────────────────────

class IntegratedGradients:
    """
    Integrated Gradients attribution for differentiable PyTorch models.

    Approximates the path integral using n_steps uniformly spaced interpolation
    points between baseline x' and input x.

    Riemann sum approximation:
        IG_i(x) ≈ (x_i - x'_i) × (1/m) Σ_{k=1}^{m} ∂F(x' + (k/m)(x-x')) / ∂x_i

    Args:
        model:    fitted PyTorch model with a differentiable forward pass
        n_steps:  number of integration steps (higher = lower approximation error)
        baseline: 'zero' (zero vector) or 'mean' (training data mean)
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 100,         # Placeholder: 50 (fast), 300 (accurate)
        baseline: str = "zero",     # Placeholder: 'zero' or 'mean'
    ):
        self.model   = model.eval()
        self.n_steps = n_steps
        self.baseline_type = baseline
        self._baseline: torch.Tensor | None = None

    def set_baseline(self, X_ref: np.ndarray | None = None) -> None:
        """
        Set the baseline tensor.

        Args:
            X_ref: reference dataset (used for 'mean' baseline)
        """
        if self.baseline_type == "mean" and X_ref is not None:
            self._baseline = torch.from_numpy(X_ref.mean(axis=0, keepdims=True)).float()
        else:
            # Default: zeros (appropriate for standardized inputs)
            self._baseline = None

    def _get_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """Return baseline tensor shaped to match x."""
        if self._baseline is not None:
            return self._baseline.to(x.device).expand_as(x)
        return torch.zeros_like(x)

    def attribute(
        self,
        x: torch.Tensor,
        target_fn: Callable | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Compute Integrated Gradients for input x.

        Args:
            x:         (1, n_features) input tensor (single sample)
            target_fn: function to compute the scalar prediction from model output;
                       defaults to sigmoid(logit) for binary classification

        Returns:
            attributions: (1, n_features) IG attribution values
            completeness_error: |Σ IG - (F(x) - F(x'))|  (should be near 0)
        """
        if target_fn is None:
            target_fn = torch.sigmoid

        x = x.to(DEVICE).requires_grad_(False)
        baseline = self._get_baseline(x)

        # Interpolations: x' + α(x - x') for α in [0, 1]
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=DEVICE)   # (m+1,)
        interpolated = baseline + alphas[:, None] * (x - baseline)        # (m+1, n_features)
        interpolated.requires_grad_(True)

        # Forward pass on all interpolation points
        with torch.enable_grad():
            logits = self.model(interpolated)                              # (m+1,)
            outputs = target_fn(logits)                                   # (m+1,)
            # Sum outputs and backpropagate to get gradients simultaneously
            grads = torch.autograd.grad(outputs.sum(), interpolated)[0]   # (m+1, n_features)

        # Riemann sum: average gradients over interpolation steps (trapezoid rule)
        avg_grads = (grads[:-1] + grads[1:]).mean(dim=0, keepdim=True)   # (1, n_features)
        attributions = (x - baseline) * avg_grads                         # (1, n_features)

        # Completeness check: Σ IG ≈ F(x) - F(x')
        with torch.no_grad():
            f_x       = target_fn(self.model(x))
            f_baseline = target_fn(self.model(baseline))
        completeness_error = float(abs(attributions.sum().item() - (f_x - f_baseline).item()))

        return attributions.detach().cpu(), completeness_error

    def attribute_batch(
        self,
        X: np.ndarray,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute IG attributions for a batch of samples.

        Args:
            X:       (n_samples, n_features) numpy array
            verbose: if True, report completeness errors

        Returns:
            attributions: (n_samples, n_features)
            errors:       (n_samples,) completeness errors
        """
        all_attr, all_err = [], []
        X_t = torch.from_numpy(X).float()

        for i in range(len(X)):
            attr, err = self.attribute(X_t[i:i+1])
            all_attr.append(attr.numpy())
            all_err.append(err)

        if verbose:
            print(f"  Mean completeness error: {np.mean(all_err):.5f} "
                  f"(max: {np.max(all_err):.5f})")

        return np.concatenate(all_attr, axis=0), np.array(all_err)


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_global_attribution(
    attributions: np.ndarray,
    feature_names: list[str],
    title: str = "Integrated Gradients — Global Attribution",
    filename: str = "ig_global.png",
    top_k: int = 15,
) -> None:
    """
    Bar plot of mean |IG| — global feature importance via IG.

    Uses absolute attributions averaged over all explained samples.
    Compare with SHAP's global importance (should correlate well).

    Args:
        attributions: (n_samples, n_features) IG attributions
        top_k:        number of top features to display
    """
    mean_abs = np.abs(attributions).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]

    plt.figure(figsize=(8, max(4, top_k * 0.35)))
    plt.barh(range(top_k), mean_abs[order][::-1], color="steelblue")
    plt.yticks(range(top_k), [feature_names[i] for i in order[::-1]], fontsize=9)
    plt.xlabel("Mean |Integrated Gradient|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_sample_attribution(
    attributions: np.ndarray,
    feature_names: list[str],
    sample_idx: int = 0,
    prediction: float | None = None,
    filename: str = "ig_sample.png",
    top_k: int = 15,
) -> None:
    """
    Horizontal bar chart of IG attributions for a single sample.

    Red = positive attribution (pushes prediction toward class 1).
    Blue = negative attribution (pushes prediction toward class 0).

    Args:
        attributions:  (n_samples, n_features) IG values
        sample_idx:    which sample to visualize
        prediction:    model output probability (optional, shown in title)
        top_k:         show top K features by |IG|
    """
    ig = attributions[sample_idx]
    order = np.argsort(np.abs(ig))[::-1][:top_k]

    ig_top  = ig[order]
    fn_top  = [feature_names[i] for i in order]
    colors  = ["tomato" if v > 0 else "steelblue" for v in ig_top]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.4)))
    ax.barh(range(top_k), ig_top[::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(fn_top[::-1], fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Integrated Gradient (attribution)")
    title = f"IG Attribution — Sample {sample_idx}"
    if prediction is not None:
        title += f" (p(y=1)={prediction:.3f})"
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_completeness_check(errors: np.ndarray, filename: str = "ig_completeness.png") -> None:
    """
    Histogram of completeness errors across all explained samples.

    Completeness error = |Σ IG_i - (F(x) - F(x'))|
    Should be near zero for all samples (validates the approximation quality).
    Large errors indicate n_steps is too low.
    """
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=30, color="steelblue", edgecolor="white")
    plt.axvline(errors.mean(), color="darkorange", linestyle="--",
                label=f"Mean = {errors.mean():.5f}")
    plt.xlabel("|Σ IG - (F(x) - F(x'))|")
    plt.ylabel("Count")
    plt.title("Integrated Gradients — Completeness Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_attribution_heatmap(
    attributions: np.ndarray,
    feature_names: list[str],
    n_samples: int = 30,
    filename: str = "ig_heatmap.png",
    top_k: int = 15,
) -> None:
    """
    Heatmap of IG attributions across multiple samples.

    Each row is a sample; each column is a feature.
    Red = positive (toward class 1), blue = negative (toward class 0).
    Reveals subgroup patterns — clusters of similar attribution profiles.

    Args:
        n_samples: number of samples to display (rows)
        top_k:     number of top features to display (columns)
    """
    mean_abs = np.abs(attributions).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    data = attributions[:n_samples, :][:, top_idx]
    vmax = np.abs(data).max()

    fig, ax = plt.subplots(figsize=(top_k * 0.8, max(5, n_samples * 0.25)))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="IG Attribution")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sample index")
    ax.set_title(f"IG Attribution Heatmap (top {top_k} features, {n_samples} samples)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_training_history(history: dict, filename: str = "ig_training.png") -> None:
    """Plot training loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"]);  axes[0].set_title("Loss");     axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, history["acc"]);   axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    plt.suptitle("MLP Training History")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Integrated Gradients — Neural Network Attribution ===")

    # 1. Data
    X_train, X_test, y_train, y_test, feat_names = load_data()
    print(f"Data: train={X_train.shape}, test={X_test.shape}, features={len(feat_names)}")

    # 2. Train MLP
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dims=TRAINING_CONFIG["hidden_dims"],
        dropout=TRAINING_CONFIG["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    history = train(
        model, X_train, y_train,
        epochs=TRAINING_CONFIG["epochs"],
        batch_size=TRAINING_CONFIG["batch_size"],
        lr=TRAINING_CONFIG["lr"],
    )
    test_acc = ((torch.sigmoid(model(torch.from_numpy(X_test).float().to(DEVICE))) > 0.5)
                .cpu().numpy() == y_test).mean()
    print(f"Test accuracy: {test_acc:.3f}")
    plot_training_history(history)

    # 3. Integrated Gradients
    print("\nComputing Integrated Gradients...")
    ig = IntegratedGradients(model, n_steps=TRAINING_CONFIG["n_steps_ig"], baseline="zero")
    ig.set_baseline(X_train)

    attributions, errors = ig.attribute_batch(X_test[:100], verbose=True)

    # 4. Predictions for individual sample annotation
    with torch.no_grad():
        probs = torch.sigmoid(model(
            torch.from_numpy(X_test[:100]).float().to(DEVICE)
        )).cpu().numpy()

    # 5. Visualize
    plot_completeness_check(errors)
    plot_global_attribution(attributions, feat_names)
    plot_sample_attribution(attributions, feat_names, sample_idx=0, prediction=probs[0])
    plot_attribution_heatmap(attributions, feat_names, n_samples=30)

    print("\nTop 5 features by mean |IG|:")
    mean_abs = np.abs(attributions).mean(axis=0)
    for i in np.argsort(mean_abs)[::-1][:5]:
        print(f"  {feat_names[i]}: {mean_abs[i]:.4f}")


if __name__ == "__main__":
    main()
