"""
Mamba State Space Model (SSM)
==============================
A linear-time sequence model that rivals Transformers on language and long-range
tasks by replacing attention with a selective state space mechanism.

Core idea — Selective State Spaces (S6):
    Classical SSMs (S4) use fixed (A, B, C) matrices for all tokens.
    Mamba makes B, C, and Δ (step size) *input-dependent*, so the model can
    selectively remember or forget context based on content.

Continuous-time SSM:
    h'(t) = A h(t) + B x(t)      [state transition]
    y(t)  = C h(t)                [output projection]

Discretized (ZOH):
    h_t = Ā h_{t-1} + B̄ x_t    where Ā = exp(Δ A), B̄ = (Δ A)⁻¹(Ā - I) Δ B
    y_t = C h_t

Selective mechanism (the key innovation):
    Δ, B, C = linear_projection(x_t)   # all depend on input content
    This lets the model learn to forget irrelevant context and focus on signals.

Mamba Block:
    Input → LayerNorm → [SSM branch] × [gate branch (SiLU)] → output
    The SSM branch: Linear → conv1d → SSM → Linear
    The gate branch: Linear → SiLU  (multiplicative gating, like GLU)

Complexity:
    Transformers:  O(L²) attention per layer
    Mamba:         O(L)  recurrent scan — scales to very long sequences

Use Cases:
    - Long-context language modeling (>100K tokens)
    - Time-series and signal processing
    - Genomics / DNA sequence modeling
    - Audio generation (long waveforms)
    - Any domain where Transformers hit memory limits

Advantages:
    - Linear time and memory in sequence length
    - Competitive with Transformers at scale (Mamba-3B ≈ GPT-3)
    - Efficient hardware-aware parallel scan (CUDA)
    - Handles very long sequences without positional encodings

Limitations:
    - Simplified SSM here (no hardware-aware scan); full speed requires custom CUDA
    - Shorter in-context retrieval vs. Transformers (recall degrades at very long range)
    - Less mature ecosystem than Transformers

Fine-Tuning Recommendations:
    1. d_model: 256–2048. Width is the main capacity knob.
    2. d_state (N): 16 is the default; 32–64 for longer-range dependencies.
    3. d_conv: 4 (local depthwise conv width). Rarely needs changing.
    4. expand (E): inner projection = E * d_model. Default 2. Increase for more capacity.
    5. n_layers: 24–48 for large models; 4–8 for experiments.
    6. Learning rate: 3e-4 with cosine decay + warmup (same as Transformers).
    7. For classification tasks: pool the SSM hidden states (mean or last-token).
    8. dt_rank: controls rank of Δ projection. 'auto' = ceil(d_model / 16).
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
# DATA
# ─────────────────────────────────────────────

def generate_sequence_data(
    n_samples: int = 2000,
    seq_len: int = 128,
    n_classes: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Synthetic long-range dependency task.

    Each sequence has a "trigger" token embedded early; the label depends on
    which trigger was seen — requiring the model to remember across the sequence.
    This mimics the kind of long-range recall that distinguishes SSMs from RNNs.

    Args:
        n_samples: number of sequences
        seq_len:   length of each sequence
        n_classes: number of class labels

    Returns:
        X: (n_samples, seq_len, 1) float sequences
        y: (n_samples,) long class labels
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, seq_len, 1)).astype(np.float32)
    trigger_pos = seq_len // 8  # trigger placed in the first 1/8 of the sequence
    triggers = rng.integers(0, n_classes, size=n_samples)
    for i, t in enumerate(triggers):
        X[i, trigger_pos, 0] += (t + 1) * 3.0  # class-specific impulse
    y = triggers.astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


# ─────────────────────────────────────────────
# SSM CORE
# ─────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Simplified selective state space layer (the "S6" inner loop of Mamba).

    Makes B, C, and Δ input-dependent via small linear projections.
    Applies ZOH discretization and runs the recurrence sequentially.

    Note: the production Mamba implementation uses a hardware-aware parallel
    associative scan in CUDA. This version is functionally equivalent but
    O(L) sequential — correct for research and small-scale use.

    State space:
        h_t = Ā h_{t-1} + B̄ x_t    (B̄ = B * Δ for simplified version)
        y_t = C h_t + D x_t          (D is a skip connection / residual)

    Args:
        d_model: input/output feature dimension
        d_state: SSM state dimension N (depth of memory)
        dt_rank: rank of the Δ (step-size) projection
        dt_min:  minimum allowed step size
        dt_max:  maximum allowed step size
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,          # Placeholder: 32–64 for longer-range tasks
        dt_rank: int | None = None, # Placeholder: None → auto = ceil(d_model/16)
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        dt_rank = dt_rank or math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # Learnable A: initialized to -[1..N] (diagonal, stable)
        # Stored as log(-A) so exp gives positive values → A stays negative
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection (direct x → y path)
        self.D = nn.Parameter(torch.ones(d_model))

        # Input-selective projections: B, C, Δ all depend on x_t
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model)

        # Initialize dt_proj so Δ starts in [dt_min, dt_max]
        nn.init.uniform_(self.dt_proj.bias, math.log(dt_min), math.log(dt_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM forward pass.

        Args:
            x: (batch, seq_len, d_model) input sequence

        Returns:
            y: (batch, seq_len, d_model) output sequence
        """
        B, L, D = x.shape
        N = self.d_state

        # A: (d_model, d_state) — fixed structured matrix, stays negative for stability
        A = -torch.exp(self.A_log)

        # Project x to get selective Δ, B, C
        x_proj = self.x_proj(x)                            # (B, L, dt_rank + 2N)
        dt_raw, B_sel, C_sel = x_proj.split([self.dt_rank, N, N], dim=-1)

        # Δ: log-space softplus ensures positivity
        dt = F.softplus(self.dt_proj(dt_raw))               # (B, L, D)
        B_sel = B_sel                                        # (B, L, N)
        C_sel = C_sel                                        # (B, L, N)

        # ZOH discretization per-token:  Ā = exp(Δ * A),  B̄ = Δ * B
        # A: (D, N), dt: (B, L, D) → Ā: (B, L, D, N)
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)   # (B, L, D, N)
        dA = torch.exp(dt_A)                                     # (B, L, D, N)
        dB = dt.unsqueeze(-1) * B_sel.unsqueeze(2)               # (B, L, D, N)

        # Sequential recurrence over L steps
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # h: (B, D, N), dA: (B, D, N), dB: (B, D, N), x: (B, D)
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # y_t = C h_t: (B, D, N) * (B, N) → (B, D)
            y_t = (h * C_sel[:, t].unsqueeze(1)).sum(-1)         # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                               # (B, L, D)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)            # skip connection
        return y


class MambaBlock(nn.Module):
    """
    One Mamba block: LayerNorm → SSM-gated MLP → residual.

    Architecture:
        Input → LayerNorm
              → Linear(D → E*D) → conv1d → SSM    [SSM branch]
              × Linear(D → E*D) → SiLU             [gate branch]
              → Linear(E*D → D)
        → + residual

    The gating (element-wise product of SSM and SiLU branches) is inspired by
    Gated Linear Units (GLU) — it lets the model control information flow.

    Args:
        d_model: model dimension D
        d_state: SSM state size N
        expand:  expansion factor E; inner dim = E * d_model
        d_conv:  depthwise conv kernel size (local context)
        dt_rank: rank of Δ projection
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,            # Placeholder: 2 is standard; 4 for more capacity
        d_conv: int = 4,            # Placeholder: local conv window; rarely needs changing
        dt_rank: int | None = None,
    ):
        super().__init__()
        self.d_inner = expand * d_model
        self.norm = nn.LayerNorm(d_model)

        # In-projection: expand x into two parallel branches
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Short depthwise conv for local context before SSM
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state, dt_rank=dt_rank)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split into SSM and gate branches
        xz = self.in_proj(x)                           # (B, L, 2 * d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        # Local conv (transpose for Conv1d: B, D, L)
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # Selective SSM
        y = self.ssm(x_ssm)

        # Gating: multiply SSM output with SiLU gate
        y = y * F.silu(z)

        # Project back to d_model + residual
        return self.out_proj(y) + residual


class MambaModel(nn.Module):
    """
    Stacked Mamba blocks for sequence classification.

    Architecture:
        Linear embedding → N × MambaBlock → mean pool → Linear head

    Args:
        input_dim:  input feature size per timestep
        d_model:    model dimension D
        n_layers:   number of Mamba blocks
        n_classes:  number of output classes
        d_state:    SSM state dimension
        expand:     inner expansion factor
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,         # Placeholder: 256–2048 for large models
        n_layers: int = 4,          # Placeholder: 24–48 for production scale
        n_classes: int = 4,
        d_state: int = 16,          # Placeholder: 32–64 for long-range tasks
        expand: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            logits: (batch, n_classes)
        """
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)           # mean pool over sequence
        return self.head(x)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "d_model":    128,
    "n_layers":   4,
    "d_state":    16,
    "expand":     2,
    "n_classes":  4,
    "seq_len":    128,
    "epochs":     30,
    "batch_size": 64,
    "lr":         3e-4,             # Placeholder: cosine decay from 3e-4 works well
}


def train(
    model: MambaModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 3e-4,
) -> dict:
    """
    Training loop with cosine LR schedule.

    Args:
        model:        MambaModel instance
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        epochs:       number of epochs
        lr:           peak learning rate

    Returns:
        training history dict
    """
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * len(y_batch)
            n += len(y_batch)

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                preds = model(X_val).argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += len(y_val)

        avg_loss = loss_sum / n
        val_acc = correct / total
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.3f}")

    return history


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_history(history: dict) -> None:
    """Plot train loss and validation accuracy side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy")

    axes[1].plot(epochs, history["val_acc"])
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")

    plt.suptitle("Mamba SSM — Training History")
    plt.tight_layout()
    plt.savefig("mamba_training.png", dpi=150)
    plt.show()


def plot_ssm_states(model: MambaModel, X: torch.Tensor, title: str = "SSM Hidden States") -> None:
    """
    Visualize SSM hidden state magnitudes across the sequence for one sample.

    Reveals where the model's "memory" is most active — useful for interpreting
    which timesteps the SSM attends to most strongly.

    Args:
        X: (1, seq_len, input_dim) single input sequence
    """
    model.eval()
    activations = []

    # Hook to capture output of the first SSM block
    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu())

    handle = model.blocks[0].ssm.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(X.to(DEVICE))
    handle.remove()

    if not activations:
        print("No activations captured.")
        return

    states = activations[0][0]          # (seq_len, d_inner)
    plt.figure(figsize=(12, 4))
    plt.imshow(states.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    plt.colorbar(label="Activation")
    plt.xlabel("Sequence position"); plt.ylabel("Hidden dimension")
    plt.title(f"Mamba — {title}")
    plt.tight_layout()
    plt.savefig("mamba_ssm_states.png", dpi=150)
    plt.show()


def plot_selective_dt(model: MambaModel, X: torch.Tensor) -> None:
    """
    Visualize the selective step-size Δ (dt) across the sequence.

    Large Δ at position t means the SSM takes a big step — effectively
    "paying more attention" to that token. This is the core selectivity mechanism.

    Args:
        X: (1, seq_len, input_dim)
    """
    model.eval()
    dts = []

    ssm = model.blocks[0].ssm

    def hook_fn(module, inp, out):
        x_in = inp[0]
        x_proj = module.x_proj(x_in)
        dt_raw = x_proj[..., :module.dt_rank]
        dt = F.softplus(module.dt_proj(dt_raw)).detach().cpu()
        dts.append(dt)

    handle = ssm.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(X.to(DEVICE))
    handle.remove()

    if not dts:
        return

    dt_vals = dts[0][0].mean(-1).numpy()   # (seq_len,)
    plt.figure(figsize=(10, 3))
    plt.plot(dt_vals)
    plt.xlabel("Sequence position"); plt.ylabel("Mean Δ (step size)")
    plt.title("Mamba — Selective Step Size Δ (large = more attended)")
    plt.tight_layout()
    plt.savefig("mamba_delta.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Mamba State Space Model ===")

    # 1. Data
    X, y = generate_sequence_data(
        n_samples=2000,
        seq_len=TRAINING_CONFIG["seq_len"],
        n_classes=TRAINING_CONFIG["n_classes"],
    )
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds   = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=TRAINING_CONFIG["batch_size"])

    # 2. Model
    model = MambaModel(
        input_dim=1,
        d_model=TRAINING_CONFIG["d_model"],
        n_layers=TRAINING_CONFIG["n_layers"],
        n_classes=TRAINING_CONFIG["n_classes"],
        d_state=TRAINING_CONFIG["d_state"],
        expand=TRAINING_CONFIG["expand"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 3. Train
    history = train(
        model, train_loader, val_loader,
        epochs=TRAINING_CONFIG["epochs"],
        lr=TRAINING_CONFIG["lr"],
    )
    plot_training_history(history)

    # 4. Visualize SSM internals on one sample
    X_sample = X[:1]   # single sequence
    plot_ssm_states(model, X_sample)
    plot_selective_dt(model, X_sample)

    print("\nDone. Inspect mamba_ssm_states.png and mamba_delta.png to see selectivity.")


if __name__ == "__main__":
    main()
