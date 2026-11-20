"""
LoRA — Low-Rank Adaptation for Large Language Models
======================================================
A parameter-efficient fine-tuning (PEFT) technique that adapts a pre-trained
model by injecting trainable low-rank matrices into its weight layers, while
keeping the original weights frozen.

Core idea:
    A full weight update ΔW ∈ ℝ^{d×k} is expensive.
    LoRA decomposes it as: ΔW = B A,  where A ∈ ℝ^{r×k}, B ∈ ℝ^{d×r}, r ≪ min(d,k)
    During fine-tuning only A and B are updated; the original W is frozen.
    At inference: W' = W + (α/r) * B A   (can be merged for zero overhead)

Forward pass with LoRA:
    h = x W^T + x A^T B^T * (α / r)

Why it works:
    The hypothesis is that weight updates during fine-tuning live in a low-rank
    manifold. Empirically r=4–16 recovers full fine-tuning quality with ~0.1–1%
    of the parameter count.

Parameter savings:
    Original: d × k parameters
    LoRA:     r × (d + k) parameters,  r ≪ d,k → 100–10,000× fewer trainable params

Use Cases:
    - Fine-tuning LLMs (GPT, LLaMA, Mistral) on domain-specific data
    - Adapting vision models (ViT) to new tasks
    - Multi-task: train multiple cheap LoRA adapters, swap at inference
    - QLoRA: combine with 4-bit quantization for consumer-GPU fine-tuning

Advantages:
    - No inference latency: merge ΔW back into W before deployment
    - Multiple adapters can be hot-swapped on the same frozen base model
    - Works well at very low rank (r=4) — parameter count is tiny
    - Compatible with quantization (QLoRA)

Limitations:
    - Rank is a hard cap on adapter expressiveness — very distribution-shifted tasks
      may need r=64+ or full fine-tuning
    - Does not help with memory during forward passes (base model still runs in full)
    - Choosing which layers to adapt (Q/K/V vs. all projections) requires search

Fine-Tuning Recommendations:
    1. rank (r): start with 8. Try 4 (cheapest) and 16 (stronger). >64 rarely helps.
    2. alpha (α): set equal to rank as default. Higher α/r ratio → larger effective LR.
    3. Which layers: target attention projections (q_proj, v_proj) first.
       Adding k_proj and out_proj gives marginal gains; ff layers sometimes help.
    4. Dropout: 0.05–0.1 on the LoRA matrices to prevent overfitting on small datasets.
    5. Learning rate: 2e-4 to 1e-3 (higher than full fine-tuning — fewer params to update).
    6. Merge before deployment: call lora_linear.merge_weights() to eliminate inference cost.
    7. QLoRA: load the base model in 4-bit (bitsandbytes NF4), add LoRA — fits 65B on 48 GB.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
# LORA PRIMITIVES
# ─────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a LoRA adapter.

    Computes: h = x W^T + (α/r) * x A^T B^T

    The original weight W is frozen (requires_grad=False).
    Only A and B are trained. After training, call merge_weights()
    to fold ΔW into W for zero-overhead inference.

    Args:
        in_features:  input dimension (matches original Linear)
        out_features: output dimension (matches original Linear)
        rank:         LoRA rank r — controls adapter expressiveness
        alpha:        LoRA scaling α — effective update scale = α/r
        dropout:      dropout on x before LoRA path (regularization)
        bias:         whether the original linear has a bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,              # Placeholder: 4 (minimal), 8 (default), 16–64 (strong)
        alpha: float = 16.0,        # Placeholder: set to rank for neutral scaling
        dropout: float = 0.0,       # Placeholder: 0.05–0.1 for small datasets
        bias: bool = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank  = rank
        self.scale = alpha / rank   # effective scaling factor

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias_param = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

        # Trainable LoRA matrices
        # A is initialized with kaiming uniform (as in the paper)
        # B is initialized to zero so ΔW = BA = 0 at the start of fine-tuning
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._merged = False

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """
        Wrap an existing nn.Linear with a LoRA adapter.

        The original weight is copied and frozen. This is the typical usage
        when adapting a pre-trained model: replace target layers with LoRALinear.

        Args:
            linear: the pre-trained nn.Linear to wrap
            rank:   LoRA rank
            alpha:  LoRA alpha scaling

        Returns:
            LoRALinear with frozen base weights and trainable A, B matrices
        """
        has_bias = linear.bias is not None
        lora_layer = cls(
            linear.in_features, linear.out_features,
            rank=rank, alpha=alpha, dropout=dropout, bias=has_bias,
        )
        # Copy and freeze the pre-trained weights
        lora_layer.weight.data.copy_(linear.weight.data)
        if has_bias:
            lora_layer.bias_param.data.copy_(linear.bias.data)
        return lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x W^T + bias + (α/r) * (dropout(x) A^T) B^T

        Args:
            x: (..., in_features)

        Returns:
            (..., out_features)
        """
        # Base (frozen) computation
        base_out = F.linear(x, self.weight, self.bias_param)

        if self._merged:
            return base_out

        # LoRA path: (α/r) * x A^T B^T
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scale * lora_out

    def merge_weights(self) -> None:
        """
        Fold LoRA matrices into the base weight for inference: W ← W + (α/r) B A

        After merging, forward() is identical to a plain nn.Linear — zero overhead.
        Irreversible unless you keep a copy of the original W.
        """
        if not self._merged:
            self.weight.data += self.scale * (self.lora_B @ self.lora_A)
            self._merged = True

    def unmerge_weights(self) -> None:
        """Undo merge — restores adapter for continued fine-tuning."""
        if self._merged:
            self.weight.data -= self.scale * (self.lora_B @ self.lora_A)
            self._merged = False

    def lora_parameter_count(self) -> int:
        """Number of trainable LoRA parameters (A + B)."""
        return self.rank * (self.in_features + self.out_features)


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Replace named Linear layers in a model with LoRALinear adapters in-place.

    Typical usage: target_modules=['q_proj', 'v_proj'] for attention adapters.
    All other parameters remain frozen; only LoRA A and B are trainable.

    Args:
        model:          pre-trained nn.Module to adapt
        target_modules: list of module name suffixes to replace (e.g. ['q_proj'])
        rank:           LoRA rank
        alpha:          LoRA alpha
        dropout:        LoRA dropout rate

    Returns:
        model with LoRA adapters injected (same object, modified in-place)
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Replace target Linear layers with LoRALinear
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                lora_layer = LoRALinear.from_linear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                # Make LoRA params trainable
                lora_layer.lora_A.requires_grad = True
                lora_layer.lora_B.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


# ─────────────────────────────────────────────
# DEMO MODEL — Small Transformer
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with multi-head self-attention.

    Used as the base model that LoRA will be applied to.
    In practice, replace with a pre-trained LLM or ViT.

    Args:
        d_model:  model dimension
        n_heads:  number of attention heads
        ff_dim:   feed-forward hidden dimension
        dropout:  attention and FF dropout
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Attention projections — these are the LoRA targets
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # Self-attention
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, L, H, Dh).transpose(1, 2)

        scale = math.sqrt(Dh)
        attn = F.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)

        x = x + self.dropout(out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class SmallTransformer(nn.Module):
    """
    Small Transformer encoder for text classification.

    Serves as the pre-trained base model that will be adapted with LoRA.
    In a real workflow, load weights from HuggingFace instead of training from scratch.

    Args:
        vocab_size:  vocabulary size
        d_model:     embedding / model dimension
        n_layers:    number of Transformer blocks
        n_heads:     attention heads per block
        n_classes:   number of output classes
        max_seq_len: maximum sequence length
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_classes: int = 4,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads=n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer token ids

        Returns:
            logits: (batch, n_classes)
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h).mean(dim=1)
        return self.head(h)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_token_data(
    n_samples: int = 3000,
    seq_len: int = 64,
    vocab_size: int = 1000,
    n_classes: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic token sequences for classification.

    Each class has a set of "signal tokens" inserted at random positions.
    The model must learn to identify the class from the signal tokens amid noise.

    Args:
        n_samples:  number of sequences
        seq_len:    sequence length
        vocab_size: vocabulary size
        n_classes:  number of classes

    Returns:
        X: (n_samples, seq_len) integer token sequences
        y: (n_samples,) class labels
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(50, vocab_size, size=(n_samples, seq_len)).astype(np.int64)
    y = rng.integers(0, n_classes, size=n_samples).astype(np.int64)

    # Class-specific signal tokens (1–9 per class, low token ids)
    signal_tokens = {c: list(range(c * 10 + 1, c * 10 + 10)) for c in range(n_classes)}
    for i in range(n_samples):
        c = y[i]
        n_signals = rng.integers(3, 7)
        positions = rng.choice(seq_len, n_signals, replace=False)
        X[i, positions] = rng.choice(signal_tokens[c], n_signals)

    return torch.from_numpy(X), torch.from_numpy(y)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "vocab_size":   1000,
    "d_model":      128,
    "n_layers":     4,
    "n_heads":      4,
    "n_classes":    4,
    "seq_len":      64,
    "pretrain_epochs": 10,
    "finetune_epochs": 10,
    "batch_size":   64,
    "pretrain_lr":  3e-4,
    "finetune_lr":  2e-4,           # Placeholder: LoRA typically uses higher LR than full FT
    "lora_rank":    8,              # Placeholder: 4 (tiny), 8 (default), 16 (strong)
    "lora_alpha":   16.0,           # Placeholder: set to rank for neutral; higher→ larger Δ
    "lora_dropout": 0.05,           # Placeholder: 0.0–0.1
    "lora_targets": ["q_proj", "v_proj"],  # Placeholder: add 'k_proj', 'out_proj' for more capacity
}


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    label: str = "",
) -> dict:
    """
    Generic training loop for classification.

    Only optimizes parameters with requires_grad=True, so LoRA training
    naturally skips the frozen base weights.

    Args:
        model:        any nn.Module
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        epochs:       number of training epochs
        lr:           learning rate
        label:        label for print output

    Returns:
        history dict with train_loss and val_acc
    """
    model = model.to(DEVICE)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * len(y_batch)
            n += len(y_batch)

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                correct += (model(X_val).argmax(1) == y_val).sum().item()
                total += len(y_val)

        avg_loss = loss_sum / n
        val_acc  = correct / total
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            params = count_parameters(model)
            print(f"  [{label}] Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.3f} | Trainable: {params['trainable']:,}")

    return history


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_comparison(
    history_full: dict,
    history_lora: dict,
) -> None:
    """
    Compare full fine-tuning vs. LoRA fine-tuning training curves.

    Demonstrates that LoRA achieves competitive accuracy with far fewer
    trainable parameters.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for history, label, color in [
        (history_full, "Full Fine-Tune", "steelblue"),
        (history_lora, "LoRA Fine-Tune", "darkorange"),
    ]:
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=label, color=color)
        axes[1].plot(epochs, history["val_acc"],    label=label, color=color)

    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.suptitle("LoRA vs. Full Fine-Tuning")
    plt.tight_layout()
    plt.savefig("lora_comparison.png", dpi=150)
    plt.show()


def plot_lora_weight_norms(model: nn.Module) -> None:
    """
    Visualize the L2 norm of each LoRA adapter's delta weight (B @ A).

    Larger norms indicate layers where the adapter made bigger adjustments —
    useful for understanding which layers drove adaptation.
    """
    names, norms = [], []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            delta_norm = (module.lora_B @ module.lora_A).norm().item()
            names.append(name)
            norms.append(delta_norm)

    if not names:
        print("No LoRA layers found.")
        return

    plt.figure(figsize=(max(6, len(names) * 1.2), 4))
    plt.bar(range(len(names)), norms)
    plt.xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("||B @ A||₂  (LoRA delta norm)")
    plt.title("LoRA Adapter Weight Norms by Layer")
    plt.tight_layout()
    plt.savefig("lora_weight_norms.png", dpi=150)
    plt.show()


def plot_parameter_efficiency(counts_full: dict, counts_lora: dict) -> None:
    """
    Bar chart comparing trainable parameters: full fine-tune vs. LoRA.

    Shows the compression ratio — the key selling point of LoRA.
    """
    labels = ["Full Fine-Tune", "LoRA"]
    trainable = [counts_full["trainable"], counts_lora["trainable"]]
    frozen    = [counts_full["frozen"],    counts_lora["frozen"]]

    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    bars1 = plt.bar(x, trainable, label="Trainable", color="steelblue")
    bars2 = plt.bar(x, frozen, bottom=trainable, label="Frozen", color="lightgray")
    plt.xticks(x, labels)
    plt.ylabel("Parameters")
    plt.title("Parameter Efficiency: LoRA vs. Full Fine-Tuning")
    plt.legend()

    for bar, val in zip(bars1, trainable):
        plt.text(bar.get_x() + bar.get_width() / 2, val / 2,
                 f"{val:,}", ha="center", va="center", fontsize=8, color="white")

    plt.tight_layout()
    plt.savefig("lora_parameter_efficiency.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== LoRA — Low-Rank Adaptation ===")

    cfg = TRAINING_CONFIG

    # 1. Data
    X, y = generate_token_data(
        n_samples=3000, seq_len=cfg["seq_len"],
        vocab_size=cfg["vocab_size"], n_classes=cfg["n_classes"],
    )
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds   = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"])

    # 2. Pre-train base model
    print("\n--- Pre-training base Transformer ---")
    base_model = SmallTransformer(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        n_classes=cfg["n_classes"], max_seq_len=cfg["seq_len"],
    )
    params_pretrain = count_parameters(base_model)
    print(f"Total parameters: {params_pretrain['total']:,}")

    history_pretrain = run_training_loop(
        base_model, train_loader, val_loader,
        epochs=cfg["pretrain_epochs"], lr=cfg["pretrain_lr"], label="pretrain",
    )

    # 3. Full fine-tune (all params unfrozen — baseline)
    print("\n--- Full Fine-Tuning (all params) ---")
    import copy
    full_ft_model = copy.deepcopy(base_model)
    for p in full_ft_model.parameters():
        p.requires_grad = True
    counts_full = count_parameters(full_ft_model)
    print(f"Trainable: {counts_full['trainable']:,} / {counts_full['total']:,}")

    history_full = run_training_loop(
        full_ft_model, train_loader, val_loader,
        epochs=cfg["finetune_epochs"], lr=cfg["finetune_lr"], label="full-ft",
    )

    # 4. LoRA fine-tune
    print(f"\n--- LoRA Fine-Tuning (rank={cfg['lora_rank']}, targets={cfg['lora_targets']}) ---")
    lora_model = copy.deepcopy(base_model)
    lora_model = inject_lora(
        lora_model,
        target_modules=cfg["lora_targets"],
        rank=cfg["lora_rank"],
        alpha=cfg["lora_alpha"],
        dropout=cfg["lora_dropout"],
    )
    counts_lora = count_parameters(lora_model)
    compression = counts_lora["trainable"] / counts_full["trainable"]
    print(f"Trainable: {counts_lora['trainable']:,} / {counts_lora['total']:,}  "
          f"({compression:.2%} of full fine-tune)")

    history_lora = run_training_loop(
        lora_model, train_loader, val_loader,
        epochs=cfg["finetune_epochs"], lr=cfg["finetune_lr"], label="lora",
    )

    # 5. Merge LoRA weights (inference: zero overhead)
    print("\nMerging LoRA weights into base model for inference...")
    for module in lora_model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    print("Weights merged. Forward pass is now identical to a plain Linear.")

    # 6. Visualizations
    plot_training_comparison(history_full, history_lora)
    plot_lora_weight_norms(lora_model)
    plot_parameter_efficiency(counts_full, counts_lora)

    print(f"\nLoRA used {compression:.2%} of trainable parameters vs full fine-tuning.")


if __name__ == "__main__":
    main()
