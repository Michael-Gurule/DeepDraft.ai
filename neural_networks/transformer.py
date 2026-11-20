"""
Transformer (Encoder)
======================
The Transformer architecture replaced RNNs as the dominant sequence model.
Rather than processing tokens sequentially, it computes attention over all
positions simultaneously — enabling parallelism and capturing long-range
dependencies directly without vanishing gradients.

Core Components:
    Multi-Head Self-Attention (MHSA):
        Each token attends to every other token in the sequence.
        Q = XW_Q, K = XW_K, V = XW_V
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        Multiple "heads" learn different types of relationships in parallel.

    Positional Encoding:
        Since attention is position-agnostic, position information is injected
        via sinusoidal encodings (original paper) or learned embeddings.

    Feed-Forward Network (FFN):
        Applied per-position after attention:
        FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
        Typically width = 4 × d_model.

    Layer Normalization:
        Applied before (Pre-LN, more stable) or after (Post-LN, original)
        each sub-layer.

    Residual Connections:
        x = x + Sublayer(LayerNorm(x))  — enables gradients to flow through
        deep networks without vanishing.

Use Cases (encoder-only Transformer):
    - Sequence classification (sentiment, intent detection)
    - Token classification (NER, POS tagging)
    - Time series classification
    - Anomaly detection
    - Feature extraction (BERT-style models)

Fine-Tuning Recommendations:
    - d_model: 64–512 for small tasks; 768–1024 for large-scale pretraining.
    - n_heads: must divide d_model evenly. 4–16 heads typical.
    - n_layers: 2–6 for small tasks; 12+ for large models.
    - d_ff: usually 4 × d_model.
    - Dropout: 0.1 is the standard starting point. Increase to 0.2–0.3 for
      small datasets to prevent overfitting.
    - Learning rate: use a warmup schedule (linear warmup + decay).
      Peak LR: 1e-4 to 3e-4. Transformers are very sensitive to LR.
    - Warmup steps: ~10% of total training steps prevents early divergence.
    - Batch size: larger is better (256–2048). Use gradient accumulation if
      GPU memory is limited.
    - Pre-LN vs Post-LN: Pre-LN (LayerNorm before attention) is more stable
      and the modern standard. This implementation uses Pre-LN.
    - For NLP: use tokenization + embedding layer instead of raw feature projection.
    - For transfer learning: start from HuggingFace pretrained models
      (BERT, RoBERTa, etc.) rather than training from scratch.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
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

def generate_sequence_classification_data(
    n_samples: int = 1000,
    seq_len: int = 64,
    n_features: int = 8,
    n_classes: int = 3,
):
    """
    Generate synthetic multivariate sequence classification data.

    Each class is generated with a distinct frequency pattern.

    Replace with real data. For NLP tasks, use:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # Then use an embedding layer to map token IDs to dense vectors.

    Returns:
        X: float32 (n_samples, seq_len, n_features)
        y: int64 class labels
    """
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, seq_len)
    X, y = [], []

    for _ in range(n_samples):
        cls = np.random.randint(0, n_classes)
        signal = np.stack([
            (cls + 1) * np.sin(t * (f + 1) + cls * np.pi / 3) + np.random.randn(seq_len) * 0.4
            for f in range(n_features)
        ], axis=1)
        X.append(signal)
        y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ─────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need" (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sinusoidal?
        - Works for any sequence length (even unseen lengths at test time)
        - Each position gets a unique encoding
        - Relative positions can be expressed via linear combinations

    Alternative: use nn.Embedding(max_seq_len, d_model) for learned positional
    embeddings — slightly better empirically on fixed-length sequences.

    Args:
        d_model:  embedding dimension (must match model d_model)
        max_len:  maximum sequence length to pre-compute
        dropout:  dropout applied after adding position encoding
    """

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encoding matrix (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer: not a parameter, but saved with model state
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model) input embeddings

        Returns:
            x + positional encoding, with dropout applied
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block with Pre-LN residual connections.

    Pre-LN (this implementation):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Post-LN (original paper):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))

    Pre-LN is more stable and generally preferred for training from scratch.

    Args:
        d_model:     embedding dimension
        n_heads:     number of attention heads (must divide d_model)
        d_ff:        dimension of the feed-forward inner layer (usually 4 × d_model)
        dropout:     dropout rate for attention weights and FFN activations
        activation:  FFN activation function ('gelu' or 'relu')
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",  # GELU is standard in BERT; ReLU in original paper
    ):
        super().__init__()

        # ── Multi-Head Self-Attention ──────────────────────────────
        # PyTorch's nn.MultiheadAttention implements scaled dot-product attention.
        # batch_first=True makes it compatible with (batch, seq, features) tensors.
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # ── Position-wise Feed-Forward Network ────────────────────
        # Applied independently at each position (no cross-position interactions here).
        # Width expansion (d_model → d_ff → d_model) increases model capacity.
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pre-LN Transformer block forward pass.

        Args:
            x:                     (batch, seq_len, d_model) input tensor
            src_key_padding_mask:  (batch, seq_len) bool mask for padded positions
                                   True = ignore this position (masked out)

        Returns:
            (batch, seq_len, d_model) output tensor
        """
        # ── Self-Attention + Residual (Pre-LN) ─────────────────────
        residual = x
        x_norm   = self.norm1(x)
        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=src_key_padding_mask,
        )
        x = residual + attn_out

        # ── FFN + Residual (Pre-LN) ─────────────────────────────────
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


# ─────────────────────────────────────────────
# FULL TRANSFORMER CLASSIFIER
# ─────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    Full Transformer encoder for sequence classification.

    Architecture:
        Input projection → Positional Encoding →
        [TransformerEncoderBlock] × n_layers →
        CLS token pooling (or mean pooling) →
        LayerNorm → Dropout → Linear → n_classes

    Two pooling strategies (Placeholder: choose one):
        1. CLS token: prepend a learnable [CLS] token; use its final state as
           the sequence representation (BERT-style).
        2. Mean pooling: average all token states. Simpler, often comparable.

    Args:
        input_size:  number of input features per timestep
        d_model:     transformer embedding dimension
        n_heads:     number of attention heads
        n_layers:    number of stacked encoder blocks
        d_ff:        feed-forward inner dimension (usually 4 × d_model)
        n_classes:   number of output classes
        dropout:     dropout rate throughout the model
        max_seq_len: maximum sequence length for positional encoding
        pool:        'cls' (CLS token) or 'mean' (average pooling)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,           # Placeholder: 64–512 depending on task complexity
        n_heads: int = 4,             # Placeholder: must divide d_model evenly
        n_layers: int = 3,            # Placeholder: 2–6 for small tasks
        d_ff: int | None = None,      # None → auto-set to 4 × d_model
        n_classes: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pool: str = "cls",            # Placeholder: 'cls' or 'mean'
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.pool    = pool
        self.d_model = d_model

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)

        # Learnable CLS token: prepended to every sequence for 'cls' pooling
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len,
                                                     dropout=dropout)

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, n_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear layers with Xavier uniform.
        Transformer models are sensitive to initialization.
        Xavier uniform works well with GELU/ReLU activations.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:            (batch, seq_len, input_size) input sequences
            padding_mask: (batch, seq_len) bool tensor; True = padded/ignore

        Returns:
            logits: (batch, n_classes)
        """
        batch_size = x.size(0)

        # Project input features to transformer dimension
        x = self.input_projection(x)   # (B, T, d_model)

        # Prepend CLS token (if using CLS pooling)
        if self.pool == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)
            if padding_mask is not None:
                # Prepend False (don't mask the CLS token)
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Add positional encoding
        x = self.pos_enc(x)

        # Pass through transformer encoder layers
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = self.norm(x)

        # Pooling: extract sequence representation
        if self.pool == "cls":
            x = x[:, 0, :]   # CLS token representation
        else:
            x = x.mean(dim=1)   # mean over all positions

        return self.classifier(x)


# ─────────────────────────────────────────────
# LEARNING RATE SCHEDULER (Warmup + Decay)
# ─────────────────────────────────────────────

def get_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup followed by cosine decay learning rate schedule.

    Why warmup?
        Transformers are sensitive to early-training learning rate.
        Large LR at step 0 can cause loss explosion before parameters settle.
        Warmup ramps LR from 0 to peak_lr over `warmup_steps` steps.

    After warmup: cosine annealing decays LR smoothly to near-zero.

    Fine-tuning tip: set warmup_steps = 10% of total training steps.

    Args:
        optimizer:     PyTorch optimizer
        warmup_steps:  number of warmup steps
        total_steps:   total training steps

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "d_model":       128,
    "n_heads":       4,
    "n_layers":      3,
    "dropout":       0.1,
    "batch_size":    64,
    "learning_rate": 3e-4,     # Transformer-specific: lower than MLP/CNN
    "weight_decay":  1e-4,
    "epochs":        50,
    "warmup_ratio":  0.1,      # fraction of total steps to use for warmup
    "patience":      15,
}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_ratio: float = 0.1,
    patience: int = 15,
) -> dict:
    """
    Training loop with warmup + cosine LR schedule and early stopping.

    Key differences from MLP training:
        - Uses warmup LR schedule (critical for Transformers)
        - Label smoothing in CrossEntropyLoss (reduces overconfidence)
        - Gradient clipping still recommended

    Args:
        model:         TransformerClassifier
        train_loader:  training DataLoader
        val_loader:    validation DataLoader
        epochs:        maximum epochs
        lr:            peak learning rate (after warmup)
        weight_decay:  L2 regularization
        warmup_ratio:  fraction of steps to use for warmup
        patience:      early stopping patience

    Returns:
        training history dict
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                             betas=(0.9, 0.98), eps=1e-9)
    # AdamW with betas=(0.9, 0.98): original Transformer optimizer settings

    total_steps  = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_warmup_scheduler(optimizer, warmup_steps, total_steps)

    history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state    = None
    wait          = 0
    global_step   = 0

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
            scheduler.step()
            global_step  += 1
            train_loss   += loss.item() * len(X_batch)
            correct      += (logits.argmax(1) == y_batch).sum().item()
            total        += len(X_batch)

        train_loss /= total
        train_acc   = correct / total

        # ── Validate ────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits  = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * len(X_batch)
                correct  += (logits.argmax(1) == y_batch).sum().item()
                total    += len(X_batch)

        val_loss /= total
        val_acc   = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | Train: loss={train_loss:.4f} acc={train_acc:.4f} "
                  f"| Val: loss={val_loss:.4f} acc={val_acc:.4f} | LR: {lr_now:.6f}")

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
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    """Plot loss and accuracy training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves"); ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves"); ax2.legend()

    plt.suptitle("Transformer — Training Curves")
    plt.tight_layout()
    plt.savefig("transformer_training_curves.png", dpi=150)
    plt.show()


def plot_attention_weights(model: nn.Module, x: torch.Tensor,
                            layer_idx: int = 0) -> None:
    """
    Visualize attention weights for a single input sequence.

    The attention heatmap shows which positions attend to which.
    High value at (i, j) = token i attends strongly to token j.

    Interpretability insight: attention heads often specialize:
        - Syntactic heads (adjacent token attention)
        - Long-range dependency heads
        - [CLS] token aggregation heads

    Args:
        model:     trained TransformerClassifier
        x:         (1, seq_len, input_size) single input sequence
        layer_idx: which transformer layer to visualize
    """
    model.eval()
    attn_weights = None

    def hook(module, input, output):
        nonlocal attn_weights
        # nn.MultiheadAttention returns (output, weights); hook output[1] = weights
        attn_weights = output[1].detach().cpu()

    # Register hook on the target layer's attention module
    handle = model.encoder[layer_idx].attn.register_forward_hook(hook)

    with torch.no_grad():
        model(x.to(DEVICE))
    handle.remove()

    if attn_weights is None:
        print("Could not capture attention weights (need_weights=True required). Skipping.")
        return

    # attn_weights: (batch, seq_len, seq_len); take first sample, average over heads
    weights = attn_weights[0].numpy()  # (seq_len, seq_len)

    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(f"Transformer — Attention Weights (Layer {layer_idx})")
    plt.xlabel("Key Position"); plt.ylabel("Query Position")
    plt.tight_layout()
    plt.savefig(f"transformer_attention_layer{layer_idx}.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Transformer Sequence Classification ===")

    # 1. Generate data
    X, y = generate_sequence_classification_data(
        n_samples=1200, seq_len=64, n_features=8, n_classes=3
    )

    # Normalize each feature
    shape = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=TRAINING_CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=TRAINING_CONFIG["batch_size"]
    )

    # 2. Build model
    model = TransformerClassifier(
        input_size=X.shape[2],
        d_model=TRAINING_CONFIG["d_model"],
        n_heads=TRAINING_CONFIG["n_heads"],
        n_layers=TRAINING_CONFIG["n_layers"],
        n_classes=3,
        dropout=TRAINING_CONFIG["dropout"],
        max_seq_len=X.shape[1] + 1,   # +1 for CLS token
        pool="cls",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 3. Train
    history = train(
        model, train_loader, val_loader,
        epochs=TRAINING_CONFIG["epochs"],
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        patience=TRAINING_CONFIG["patience"],
    )

    # 4. Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
    y_pred = logits.argmax(axis=1)
    print(f"\n  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 5. Visualize
    plot_training_curves(history)

    # Attention visualization on a single test sample
    sample = torch.tensor(X_test[:1])
    plot_attention_weights(model, sample, layer_idx=0)


if __name__ == "__main__":
    main()
