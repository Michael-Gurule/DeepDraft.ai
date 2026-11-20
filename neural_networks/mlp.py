"""
Multi-Layer Perceptron (MLP) / Feedforward Neural Network
==========================================================
A fully-connected neural network composed of an input layer, one or more
hidden layers with non-linear activation functions, and an output layer.
Trained via backpropagation and stochastic gradient descent (or its variants).

Architecture:
    Input → [Dense + Activation + Dropout] × N → Output

Use Cases:
    - Tabular data classification and regression
    - Multi-output prediction
    - Learning abstract feature representations
    - Function approximation

Advantages:
    - Universal approximator (can learn any continuous function given enough neurons)
    - Handles complex non-linear relationships
    - PyTorch and TensorFlow/Keras implementations allow full customization
    - Flexible output (multi-class, multi-label, regression, mixed)

Limitations:
    - Requires hyperparameter tuning (architecture, LR, regularization)
    - Does not naturally capture spatial (CNN) or sequential (RNN) structure
    - Can overfit with small datasets without regularization
    - Less interpretable than tree-based models

Fine-Tuning Recommendations:
    - Architecture: start shallow (2–3 layers), scale up if underfitting.
      Width: 64–512 neurons per layer for tabular data.
      Rule of thumb: decrease width from input to output (funnel).
    - Activation: ReLU is default; LeakyReLU/GELU for deeper networks;
      Sigmoid/Softmax only in output layer.
    - Batch normalization: add after each Dense layer (before activation)
      to stabilize training and allow higher learning rates.
    - Dropout: 0.2–0.5 between hidden layers for regularization.
      Higher dropout → more regularization. Don't use on output layer.
    - Learning rate: 1e-3 (default Adam) is a good start. Use LR scheduling
      (ReduceLROnPlateau or CosineAnnealingLR) for better convergence.
    - Batch size: 32–256. Larger batches → faster training but may generalize worse.
    - Weight initialization: PyTorch default (Kaiming uniform for ReLU) is good.
    - Optimizer: Adam is the default; SGD with momentum can generalize better
      for large datasets; AdamW adds weight decay directly.
    - Early stopping: monitor validation loss; stop if no improvement for ~20 epochs.
    - Weight decay (L2): 1e-4 to 1e-2 in optimizer for regularization.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error, r2_score
)
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DEVICE CONFIGURATION
# ─────────────────────────────────────────────

# Automatically use GPU (CUDA or Apple Silicon MPS) if available
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_classification_data(n_samples: int = 1000, n_features: int = 20):
    """
    Generate synthetic multi-class classification data.

    Replace with your own loading:
        import pandas as pd
        df = pd.read_csv("your_data.csv")
        X = df.drop(columns=["label"]).values.astype(np.float32)
        y = df["label"].values.astype(np.int64)

    Returns:
        X: float32 feature matrix
        y: int64 class labels
        n_classes: number of classes
    """
    n_classes = 3
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int64), n_classes


def load_regression_data(n_samples: int = 600, n_features: int = 10):
    """
    Generate synthetic regression data.

    Returns:
        X: float32 feature matrix
        y: float32 continuous target
    """
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, n_informative=7,
        noise=10.0, random_state=42
    )
    return X.astype(np.float32), y.astype(np.float32)


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size: int = 64):
    """
    Wrap numpy arrays in PyTorch DataLoaders.

    DataLoader handles batching, shuffling, and (optionally) multiprocess loading.

    Args:
        X_train, y_train: training data (numpy arrays)
        X_val, y_val:     validation data (numpy arrays)
        batch_size:       samples per mini-batch

    Returns:
        train_loader, val_loader (PyTorch DataLoader objects)
    """
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train)
    X_vl_t = torch.tensor(X_val,   dtype=torch.float32)
    y_vl_t = torch.tensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(X_vl_t, y_vl_t),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron.

    Architecture:
        Input → [Linear → BatchNorm → Activation → Dropout] × N → Output

    Placeholder: modify hidden_sizes, dropout_rate, and activation to change architecture.

    Args:
        input_size:   number of input features
        hidden_sizes: list of integers defining the width of each hidden layer
                      e.g., [256, 128, 64] = 3 hidden layers
        output_size:  number of output neurons
                      (n_classes for classification, 1 for regression)
        dropout_rate: dropout probability (0 = disabled, 0.3–0.5 = typical)
        activation:   activation function class (nn.ReLU, nn.GELU, nn.LeakyReLU)
        use_batchnorm: whether to apply batch normalization after each layer
        task:         'classification' or 'regression' (affects output activation)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],   # Placeholder: e.g., [256, 128, 64]
        output_size: int,
        dropout_rate: float = 0.3,
        activation: type = nn.ReLU,
        use_batchnorm: bool = True,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task

        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))

            if use_batchnorm:
                # Batch normalization: normalizes activations across the batch.
                # Reduces sensitivity to initialization, allows higher LR,
                # and acts as mild regularization.
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(activation())

            if dropout_rate > 0:
                # Dropout: randomly zeros neurons during training.
                # Forces the network to learn redundant representations.
                layers.append(nn.Dropout(p=dropout_rate))

            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))

        # For regression, no output activation (raw linear output).
        # For classification, CrossEntropyLoss handles softmax internally.
        if task == "regression":
            pass  # no output activation

        self.network = nn.Sequential(*layers)

        # Weight initialization: Kaiming uniform for ReLU (default in PyTorch).
        # He initialization prevents vanishing/exploding gradients with ReLU.
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming uniform (He initialization)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: adjust these hyperparameters
CLASSIFIER_ARCH = {
    "hidden_sizes":   [256, 128, 64],  # layer widths; try [512, 256, 128] for more capacity
    "dropout_rate":   0.3,             # 0 = no dropout; 0.5 = heavy regularization
    "activation":     nn.ReLU,         # nn.ReLU, nn.GELU, nn.LeakyReLU
    "use_batchnorm":  True,            # recommended; disable if batch size < 8
}

TRAINING_CONFIG = {
    "batch_size":     64,
    "learning_rate":  1e-3,             # Adam default; try 1e-4 for fine-tuning
    "weight_decay":   1e-4,             # L2 regularization in optimizer
    "epochs":         100,
    "patience":       20,               # early stopping: epochs without improvement
}


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task: str = "classification",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 20,
) -> dict:
    """
    Full training loop with early stopping and LR scheduling.

    Components:
        - Optimizer: AdamW (Adam + weight decay decoupled from gradient scale)
        - Loss: CrossEntropyLoss (classification) or MSELoss (regression)
        - LR Scheduler: ReduceLROnPlateau — halves LR when validation loss plateaus
        - Early Stopping: stops training if val_loss doesn't improve for `patience` epochs

    Fine-tuning tip: monitor train_loss vs val_loss curves.
        - Both decreasing: model is learning normally
        - val_loss rising while train_loss drops: overfitting → more dropout, weight decay, or data
        - Both high / not decreasing: underfitting → more neurons, layers, or lower LR

    Args:
        model:        MLP instance
        train_loader: training DataLoader
        val_loader:   validation DataLoader
        task:         'classification' or 'regression'
        lr:           initial learning rate
        weight_decay: L2 regularization strength
        epochs:       maximum training epochs
        patience:     early stopping patience

    Returns:
        dict with training history (train_loss, val_loss per epoch)
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    # AdamW: like Adam but weight decay is applied directly to weights, not gradients.
    # Generally better generalization than Adam on large models.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ReduceLROnPlateau: halves learning rate when val_loss hasn't improved
    # for `patience//2` epochs. Helps escape plateaus.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, verbose=False
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    for epoch in range(1, epochs + 1):
        # ── Training ────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            if task == "regression":
                output = output.squeeze(1)
            loss = criterion(output, y_batch)
            loss.backward()
            # Gradient clipping: prevents exploding gradients in deep networks
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_loader.dataset)

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                output = model(X_batch)
                if task == "regression":
                    output = output.squeeze(1)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * len(X_batch)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {lr_now:.6f}")

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

    # Restore the best model weights
    if best_state:
        model.load_state_dict(best_state)

    return history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_classifier(model: nn.Module, X_test: np.ndarray,
                          y_test: np.ndarray) -> dict:
    """
    Evaluate an MLP classifier on the test set.

    Args:
        model:  trained MLP
        X_test: test features (numpy float32)
        y_test: true integer labels

    Returns:
        dict of metrics
    """
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()

    y_pred = logits.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"accuracy": acc}


def evaluate_regressor(model: nn.Module, X_test: np.ndarray,
                        y_test: np.ndarray) -> dict:
    """
    Evaluate an MLP regressor.

    Args:
        model:  trained MLP
        X_test: test features (numpy float32)
        y_test: true float targets

    Returns:
        dict of metrics
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test).to(DEVICE)).squeeze(1).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    return {"rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_loss_curves(history: dict) -> None:
    """
    Plot training and validation loss over epochs.

    Diagnostic patterns:
        Both curves decreasing    → healthy training
        Val > Train (gap growing) → overfitting (more dropout, weight decay, data)
        Both curves flat/high     → underfitting (more capacity, lower LR)
        Val < Train               → unusual; check for data leakage
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP — Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mlp_loss_curves.png", dpi=150)
    plt.show()


def plot_architecture(model: nn.Module) -> None:
    """Print a summary of the MLP architecture layers."""
    print("\nMLP Architecture:")
    for i, layer in enumerate(model.network):
        print(f"  Layer {i}: {layer}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Classification ──────────────────────────────────────────────
    print("=== MLP Classification ===")
    X, y, n_classes = load_classification_data()

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    train_loader, val_loader = make_dataloaders(X_tr, y_tr, X_val, y_val,
                                                 batch_size=TRAINING_CONFIG["batch_size"])

    clf = MLP(
        input_size=X.shape[1],
        output_size=n_classes,
        task="classification",
        **CLASSIFIER_ARCH,
    )
    plot_architecture(clf)

    history = train(
        clf, train_loader, val_loader,
        task="classification",
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        epochs=TRAINING_CONFIG["epochs"],
        patience=TRAINING_CONFIG["patience"],
    )

    print("\nTest Metrics:")
    evaluate_classifier(clf, X_test, y_test)
    plot_loss_curves(history)

    # ── Regression ──────────────────────────────────────────────────
    print("\n=== MLP Regression ===")
    Xr, yr = load_regression_data()
    scaler_r = StandardScaler()
    Xr = scaler_r.fit_transform(Xr).astype(np.float32)
    yr = (yr - yr.mean()) / yr.std()  # normalize target for regression

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    Xr_tr, Xr_val, yr_tr, yr_val = train_test_split(
        Xr_train, yr_train, test_size=0.15, random_state=42
    )

    tr_loader, vl_loader = make_dataloaders(
        Xr_tr, yr_tr.astype(np.float32),
        Xr_val, yr_val.astype(np.float32),
    )

    reg = MLP(
        input_size=Xr.shape[1],
        hidden_sizes=[128, 64],
        output_size=1,
        dropout_rate=0.2,
        use_batchnorm=True,
        task="regression",
    )

    reg_history = train(reg, tr_loader, vl_loader, task="regression")
    print("\nRegression Test Metrics:")
    evaluate_regressor(reg, Xr_test, yr_test.astype(np.float32))
    plot_loss_curves(reg_history)


if __name__ == "__main__":
    main()
