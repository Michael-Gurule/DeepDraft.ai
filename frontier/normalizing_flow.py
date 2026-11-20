"""
Normalizing Flow
=================
A generative model that transforms a simple base distribution (e.g., Gaussian)
into a complex target distribution through a sequence of invertible, differentiable
transformations. Unlike VAEs or GANs, normalizing flows provide an exact log-likelihood.

Change-of-variables formula:
    If z ~ p_z(z) and x = f(z) where f is invertible and differentiable:

        log p_x(x) = log p_z(f⁻¹(x)) + log |det J_{f⁻¹}(x)|

    where J_{f⁻¹} is the Jacobian of the inverse transformation.

    Key: we need both f and f⁻¹ to be efficient to compute, and the
    Jacobian determinant must be tractable (not O(d³)).

Flow types implemented:
    1. RealNVP (Dinh et al. 2017) — coupling layers:
       Split x into (x₁, x₂); transform x₂ using scale/shift from x₁.
       Jacobian is triangular → det is product of diagonals → O(d).

    2. Masked Autoregressive Flow (MAF) — autoregressive coupling:
       Each dimension is transformed conditioned on all previous.
       Fast density estimation; slow sampling (sequential).

    3. 1D Flow for density estimation demonstration.

RealNVP coupling layer:
    Forward (z→x):
        x₁ = z₁                         (passthrough)
        x₂ = z₂ · exp(s(z₁)) + t(z₁)   (scale+shift by z₁)
    Inverse (x→z):
        z₁ = x₁
        z₂ = (x₂ - t(x₁)) · exp(-s(x₁))
    Log |det J| = Σ s(z₁)    (sum of log-scales)

Comparison to other generative models:
    VAE:    approximate posterior (ELBO); no exact likelihood; fast
    GAN:    no likelihood; adversarial training; sharp samples
    Diffusion: exact score; slow sampling; best quality
    Flow:   exact likelihood; invertible; fast sampling; limited expressivity

Use Cases:
    - Exact density estimation (anomaly detection via log-likelihood)
    - Probabilistic Bayesian inference (posterior approximation)
    - Lossless data compression (bits-back coding)
    - Generative modeling with guaranteed coverage
    - Latent space normalization for downstream tasks
    - Continuous normalizing flows for Neural ODEs

Fine-Tuning Recommendations:
    - n_flows: 8–16 coupling layers. More flows → more expressive.
    - hidden_dim: 256–512 per coupling network.
    - Alternating masks: alternate which half is transformed each layer.
    - Batch norm between flows: improves training stability.
    - Learning rate: 1e-4 with cosine annealing or ReduceLROnPlateau.
    - For images: use multi-scale architecture (Glow); each level uses
      1×1 convolutions + squeeze + coupling layers.
    - For tabular data: RealNVP or MAF works well out of the box.
    - normflows library: pip install normflows — includes many flow types.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles
import warnings
warnings.filterwarnings("ignore")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_target_distribution(dist: str = "moons", n: int = 5000,
                                   seed: int = 42) -> torch.Tensor:
    """
    Generate samples from a complex 2D distribution.

    Replace with your own dataset:
        X = pd.read_csv("data.csv")[["feature1", "feature2"]].values
        X = (X - X.mean(0)) / X.std(0)  # standardize

    Available: 'moons', 'circles', 'checkerboard', 'gaussian_mixture'
    """
    rng = np.random.RandomState(seed)

    if dist == "moons":
        X, _ = make_moons(n_samples=n, noise=0.05, random_state=seed)
    elif dist == "circles":
        X, _ = make_circles(n_samples=n, noise=0.05, factor=0.5, random_state=seed)
    elif dist == "checkerboard":
        x = rng.uniform(-2, 2, n)
        y = rng.uniform(-2, 2, n)
        keep = ((np.floor(x) + np.floor(y)) % 2 == 0)
        x, y = x[keep], y[keep]
        X = np.c_[x[:n//2*2:], y[:n//2*2:]][:n]
    elif dist == "gaussian_mixture":
        centers = np.array([[0, 0], [3, 0], [-3, 0], [0, 3], [0, -3]])
        idx     = rng.choice(5, n)
        X       = centers[idx] + rng.randn(n, 2) * 0.5
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return torch.tensor(X, dtype=torch.float32)


# ─────────────────────────────────────────────
# COUPLING NETWORK (shared by all flows)
# ─────────────────────────────────────────────

class CouplingNet(nn.Module):
    """
    Small MLP that outputs scale s and shift t for an affine coupling layer.
    Input: x₁ (first half of features)
    Output: (s, t) each of same size as x₂
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net_s = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim), nn.Tanh(),  # bounded scale
        )
        self.net_t = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x1: torch.Tensor) -> tuple:
        """Returns scale (s) and translation (t) from x1."""
        return self.net_s(x1), self.net_t(x1)


# ─────────────────────────────────────────────
# REALNVP COUPLING LAYER
# ─────────────────────────────────────────────

class AffineCouplingLayer(nn.Module):
    """
    RealNVP Affine Coupling Layer.

    Splits input into (x₁, x₂) using a binary mask.
    Transforms x₂ using scale/shift predicted from x₁.
    Jacobian determinant = exp(Σ s) — diagonal, O(d).
    """

    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 256):
        """
        Args:
            dim:        input dimension
            mask:       binary mask (1 = passthrough, 0 = transformed)
            hidden_dim: coupling network width
        """
        super().__init__()
        self.register_buffer("mask", mask)
        in_dim  = mask.sum().int().item()
        out_dim = dim - in_dim
        self.coupling = CouplingNet(in_dim, out_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass: x → z (data → latent space)

        Returns:
            z:       transformed sample
            log_det: log |det J| accumulated log-likelihood contribution
        """
        x1 = x[:, self.mask.bool()]     # passthrough part
        x2 = x[:, ~self.mask.bool()]    # transformed part
        s, t = self.coupling(x1)
        z2   = x2 * torch.exp(s) + t
        z    = x.clone()
        z[:, ~self.mask.bool()] = z2
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: z → x (latent → data space, for sampling)
        """
        z1 = z[:, self.mask.bool()]
        z2 = z[:, ~self.mask.bool()]
        s, t = self.coupling(z1)
        x2   = (z2 - t) * torch.exp(-s)
        x    = z.clone()
        x[:, ~self.mask.bool()] = x2
        return x


# ─────────────────────────────────────────────
# REALNVP MODEL
# ─────────────────────────────────────────────

class RealNVP(nn.Module):
    """
    RealNVP normalizing flow: composition of affine coupling layers
    with alternating masks.

    log p_x(x) = log p_z(f(x)) + Σ_k log |det J_k(x)|

    where f = f_K ∘ ... ∘ f_1 is the composition of coupling layers.
    """

    def __init__(self, dim: int = 2, n_flows: int = 8,
                 hidden_dim: int = 256):
        """
        Args:
            dim:        data dimensionality
            n_flows:    number of coupling layers — Placeholder: 8–16
            hidden_dim: coupling network width — Placeholder: 128–512
        """
        super().__init__()
        self.dim = dim

        # Alternate masks: even layers mask first half, odd layers mask second half
        flows = []
        for k in range(n_flows):
            mask = torch.zeros(dim)
            if k % 2 == 0:
                mask[:dim // 2] = 1
            else:
                mask[dim // 2:] = 1
            flows.append(AffineCouplingLayer(dim, mask, hidden_dim))

        self.flows = nn.ModuleList(flows)

        # Standard Gaussian base distribution
        self.register_buffer("base_mean", torch.zeros(dim))
        self.register_buffer("base_std",  torch.ones(dim))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact log-likelihood: log p_x(x).

        Applies forward (x→z) through all coupling layers and accumulates
        log |det J|.

        Args:
            x: data samples (batch, dim)

        Returns:
            log_p: exact log-likelihood (batch,)
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det

        # Log-likelihood under base distribution
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        return log_pz + log_det_total

    def sample(self, n: int) -> torch.Tensor:
        """
        Generate samples: z ~ N(0,I) → x via inverse flows.
        """
        z = torch.randn(n, self.dim, device=self.base_mean.device)
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns negative log-likelihood (loss)."""
        return -self.log_prob(x).mean()


# ─────────────────────────────────────────────
# MASKED AUTOREGRESSIVE LAYER
# ─────────────────────────────────────────────

class MaskedLinear(nn.Linear):
    """
    Linear layer with a binary mask applied to the weight matrix.
    Used in MADE (Masked Autoencoder for Distribution Estimation) for MAF.
    """

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__(in_features, out_features, bias=True)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class MAFLayer(nn.Module):
    """
    Single Masked Autoregressive Flow (MAF) layer.

    Each dimension i is transformed conditioned on all previous dimensions:
        xᵢ = zᵢ · exp(αᵢ(x_{<i})) + μᵢ(x_{<i})

    Uses MADE (masked MLP) to enforce autoregressive structure.
    Fast density estimation (single forward pass); slow sampling (sequential).
    """

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        # Build MADE masks
        d_in = dim; d_h = hidden_dim; d_out = 2 * dim  # (μ, log_α)

        # Input → hidden: m(k) = k+1 % dim, m_in = range(1, dim+1)
        m_in = torch.arange(1, dim + 1)
        m_h  = torch.arange(hidden_dim) % (dim - 1) + 1
        m_out = torch.arange(dim)  # for μ and log_α (each dim, conditioned on <i)

        mask_ih = (m_h.unsqueeze(1) >= m_in.unsqueeze(0)).float()
        mask_ho = (m_out.unsqueeze(1) > m_h.unsqueeze(0)).float()
        mask_ho = mask_ho.repeat(2, 1)  # for both μ and log_α

        self.lin1 = MaskedLinear(dim, hidden_dim, mask_ih)
        self.lin2 = MaskedLinear(hidden_dim, 2 * dim, mask_ho)
        self.act  = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple:
        """x → z, log_det"""
        h = self.act(self.lin1(x))
        params   = self.lin2(h)
        mu       = params[:, :self.dim]
        log_alpha = params[:, self.dim:]
        z        = (x - mu) * torch.exp(-log_alpha)
        log_det  = -log_alpha.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """z → x (must be done sequentially, O(dim))"""
        x = torch.zeros_like(z)
        for i in range(self.dim):
            h = self.act(self.lin1(x))
            params   = self.lin2(h)
            mu_i     = params[:, i]
            log_a_i  = params[:, self.dim + i]
            x[:, i]  = z[:, i] * torch.exp(log_a_i) + mu_i
        return x


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

# Placeholder: tune n_flows, hidden_dim
FLOW_CONFIG = {
    "model":       "realnvp",  # Placeholder: 'realnvp' or 'maf'
    "n_flows":     8,          # Placeholder: 8–16
    "hidden_dim":  256,        # Placeholder: 128–512
    "lr":          1e-4,       # Placeholder: 1e-4 to 1e-3
    "epochs":      100,
    "batch_size":  256,
    "patience":    15,
}


def train_flow(X: torch.Tensor, config: dict = None) -> tuple:
    """
    Train a normalizing flow by maximizing log-likelihood (MLE).

    Loss = -E[log p_x(x)] = negative log-likelihood.
    """
    if config is None:
        config = FLOW_CONFIG

    dim = X.shape[1]
    if config["model"] == "realnvp":
        model = RealNVP(dim=dim, n_flows=config["n_flows"],
                         hidden_dim=config["hidden_dim"]).to(DEVICE)
    else:
        raise NotImplementedError("MAF training: use RealNVP wrapper with MAF layers")

    # Train/val split
    n     = len(X)
    n_val = int(0.1 * n)
    idx   = torch.randperm(n)
    X_tr  = X[idx[n_val:]]; X_v = X[idx[:n_val]]

    loader = DataLoader(TensorDataset(X_tr), batch_size=config["batch_size"],
                         shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=config["lr"],
                               weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8,
                                                          factor=0.5)

    history    = {"train_nll": [], "val_nll": []}
    best_val   = float("inf")
    patience_c = 0
    best_state = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        tr_nll = 0.0
        for (xb,) in loader:
            xb    = xb.to(DEVICE)
            loss  = model(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_nll += loss.item() * len(xb)
        tr_nll /= len(X_tr)

        model.eval()
        with torch.no_grad():
            val_nll = model(X_v.to(DEVICE)).item()

        sched.step(val_nll)
        history["train_nll"].append(tr_nll)
        history["val_nll"].append(val_nll)

        if val_nll < best_val:
            best_val   = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= config["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | train NLL: {tr_nll:.4f} | val NLL: {val_nll:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_density(model: RealNVP, X: torch.Tensor, title: str = "RealNVP") -> None:
    """
    2D heatmap of the learned density p_x(x) over the data domain.
    """
    model.eval()
    x_range = X[:, 0].min().item() - 0.5, X[:, 0].max().item() + 0.5
    y_range = X[:, 1].min().item() - 0.5, X[:, 1].max().item() + 0.5

    xx, yy = np.meshgrid(np.linspace(*x_range, 200),
                          np.linspace(*y_range, 200))
    grid   = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                           dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        log_p = model.log_prob(grid).cpu().numpy()

    density = np.exp(log_p).reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Learned density
    ax1.contourf(xx, yy, density, levels=40, cmap="Blues")
    ax1.set_title(f"{title} — Learned Density")
    ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂")

    # Samples vs training data
    with torch.no_grad():
        samples = model.sample(2000).cpu().numpy()
    ax2.scatter(X[:, 0].numpy(), X[:, 1].numpy(), s=4, alpha=0.3,
                c="steelblue", label="Training data")
    ax2.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.3,
                c="tomato", label="Flow samples")
    ax2.legend(fontsize=8)
    ax2.set_title(f"{title} — Generated Samples")

    plt.suptitle(f"Normalizing Flow — {title}")
    plt.tight_layout()
    plt.savefig("flow_density.png", dpi=150)
    plt.show()


def plot_latent_space(model: RealNVP, X: torch.Tensor) -> None:
    """
    Map data to latent space — should be approximately N(0, I).
    Good calibration = circular, Gaussian point cloud.
    """
    model.eval()
    with torch.no_grad():
        z_total = 0.0
        log_det = torch.zeros(len(X))
        z = X.to(DEVICE)
        for flow in model.flows:
            z, ld = flow(z)
            log_det += ld.cpu()
    z_np = z.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(z_np[:, 0], z_np[:, 1], s=4, alpha=0.3, c="steelblue")
    # Reference Gaussian contours
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [1, 2, 3]:
        plt.plot(r*np.cos(theta), r*np.sin(theta), "k--", linewidth=0.5, alpha=0.5)
    plt.title("Normalizing Flow — Latent Space (should be N(0,I))")
    plt.xlabel("z₁"); plt.ylabel("z₂")
    plt.tight_layout()
    plt.savefig("flow_latent.png", dpi=150)
    plt.show()


def plot_training_curve(history: dict) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_nll"], label="Train NLL")
    plt.plot(history["val_nll"],   label="Val NLL")
    plt.xlabel("Epoch"); plt.ylabel("Negative Log-Likelihood")
    plt.title("Normalizing Flow — Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("flow_training.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# ANOMALY DETECTION VIA LOG-LIKELIHOOD
# ─────────────────────────────────────────────

def anomaly_detection_demo(model: RealNVP, X_normal: torch.Tensor,
                             X_anomaly: torch.Tensor) -> None:
    """
    Use flow log-likelihood as anomaly score.
    Normal samples: high log p(x). Anomalies: low log p(x).
    """
    model.eval()
    with torch.no_grad():
        ll_normal  = model.log_prob(X_normal.to(DEVICE)).cpu().numpy()
        ll_anomaly = model.log_prob(X_anomaly.to(DEVICE)).cpu().numpy()

    plt.figure(figsize=(9, 4))
    plt.hist(ll_normal,  bins=50, alpha=0.6, color="steelblue", label="Normal",  density=True)
    plt.hist(ll_anomaly, bins=50, alpha=0.6, color="tomato",    label="Anomaly", density=True)
    plt.xlabel("Log p(x)"); plt.ylabel("Density")
    plt.title("Normalizing Flow — Log-Likelihood Anomaly Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig("flow_anomaly.png", dpi=150)
    plt.show()

    from sklearn.metrics import roc_auc_score
    y_true  = np.array([0]*len(ll_normal)  + [1]*len(ll_anomaly))
    scores  = np.concatenate([-ll_normal, -ll_anomaly])  # higher = more anomalous
    auc     = roc_auc_score(y_true, scores)
    print(f"  Anomaly detection ROC-AUC: {auc:.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"=== Normalizing Flow (RealNVP) ===  (device: {DEVICE})")

    # Placeholder: replace with your target distribution
    dist = "moons"  # Placeholder: 'moons', 'circles', 'checkerboard', 'gaussian_mixture'
    X    = generate_target_distribution(dist=dist, n=5000)
    print(f"Data: {X.shape}  |  Distribution: {dist}")

    # Train RealNVP
    print("\nTraining RealNVP...")
    model, history = train_flow(X)

    # Evaluation
    model.eval()
    with torch.no_grad():
        nll = model(X.to(DEVICE)).item()
    print(f"\nFinal NLL: {nll:.4f}  (lower = better density estimate)")

    # Anomaly detection demo
    rng = np.random.RandomState(0)
    X_anomaly = torch.tensor(rng.uniform(-3, 3, (500, 2)), dtype=torch.float32)
    print("\nAnomaly detection via log-likelihood:")
    anomaly_detection_demo(model, X, X_anomaly)

    # Visualizations
    plot_training_curve(history)
    plot_density(model, X, title="RealNVP")
    plot_latent_space(model, X[:1000])

    print("\nNote: For more flow types (Glow, Neural Spline Flows, etc.):")
    print("      pip install normflows")


if __name__ == "__main__":
    main()
