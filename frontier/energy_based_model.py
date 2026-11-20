"""
Energy-Based Model (EBM)
========================

Framework: p_θ(x) = exp(-E_θ(x)) / Z_θ
where Z_θ = ∫ exp(-E_θ(x)) dx is the intractable partition function.

Training objectives:
  1. Contrastive Divergence (CD-k):
       ∇_θ log p = -E_x[∇_θ E(x)] + E_{x~p}[∇_θ E(x)]
       Approximate the model expectation via short MCMC chains (k steps).

  2. Score Matching (SM / DSM):
       J_SM = E_x[ ½||s_θ(x)||² + tr(∇_x s_θ(x)) ]
       where s_θ(x) = -∇_x E_θ(x) is the score function.
       Denoising SM (Vincent 2011): perturb data → predict score of clean data.

  3. Noise Contrastive Estimation (NCE):
       Train a binary classifier: real data vs noise distribution q(x).
       logit = log p_θ(x) - log q(x), optimize binary cross-entropy.

Sampling (Langevin MCMC):
  x_{t+1} = x_t - (α/2) ∇_x E_θ(x_t) + √α · ε,  ε ~ N(0, I)
  Unadjusted Langevin Algorithm (ULA) — no Metropolis correction.
  Stochastic Gradient Langevin Dynamics (SGLD) for large datasets.

Connection to Diffusion Models:
  Score-based diffusion = EBM where score s_θ(x, σ) is learned at multiple
  noise levels σ via DSM. Sampling uses annealed Langevin dynamics.

Fine-tuning recommendations:
  - energy_hidden_dim (#Placeholder: 256): larger = more expressive energy landscape
  - energy_depth (#Placeholder: 4): depth of energy network (>=3 for multimodal)
  - mcmc_step_size (#Placeholder: 0.01): Langevin step α (tune for mixing vs stability)
  - mcmc_steps_train (#Placeholder: 60): CD-k steps for training (higher = better but slower)
  - mcmc_steps_sample (#Placeholder: 1000): steps for final sample generation
  - n_noise_levels (#Placeholder: 10): for DSM / score-based training
  - replay_buffer_size (#Placeholder: 10000): persistent chain replay buffer for CD
  - dsm_sigma (#Placeholder: 0.3): noise std for denoising score matching
  - nce_noise_ratio (#Placeholder: 1): noise samples per real sample in NCE

References:
  LeCun et al. (2006) — A Tutorial on Energy-Based Learning
  Du & Mordatch (2019) — Implicit Generation and Modeling with EBMs
  Song & Ermon (2019) — Generative Modeling by Estimating Gradients (NCSN)
  Vincent (2011) — A Connection Between Score Matching and Denoising Autoencoders
  Gutmann & Hyvärinen (2010) — Noise-Contrastive Estimation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# MPS note: Langevin sampling requires grad w.r.t. inputs (not parameters),
# which works on MPS. No higher-order autograd, so MPS is safe here.
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ─────────────────────────────────────────────
# CONFIG  (all # Placeholder: values are tunable)
# ─────────────────────────────────────────────
CONFIG = {
    "energy_hidden_dim":    256,        # # Placeholder: hidden units per layer
    "energy_depth":         4,          # # Placeholder: number of hidden layers
    "mcmc_step_size":       0.01,       # # Placeholder: Langevin α
    "mcmc_steps_train":     60,         # # Placeholder: CD-k during training
    "mcmc_steps_sample":    1000,       # # Placeholder: steps for generation
    "replay_buffer_size":   10_000,     # # Placeholder: persistent chain buffer
    "replay_prob":          0.95,       # # Placeholder: prob of using replay vs noise init
    "batch_size":           256,        # # Placeholder:
    "n_epochs":             100,        # # Placeholder:
    "lr":                   1e-4,       # # Placeholder:
    "weight_decay":         1e-4,       # # Placeholder: L2 regularization
    "l2_reg_coef":          0.1,        # # Placeholder: energy regularization coefficient
    "dsm_sigma":            0.3,        # # Placeholder: noise std for denoising score matching
    "n_noise_levels":       10,         # # Placeholder: for multi-scale score matching
    "nce_noise_ratio":      1,          # # Placeholder: noise samples per real sample
    "clip_grad":            0.1,        # # Placeholder: gradient clip norm
    "data_dim":             2,          # # Placeholder: replace with actual input dim
    "n_samples":            2000,       # # Placeholder:
    "seed":                 42,
}


# ─────────────────────────────────────────────
# Energy Network
# ─────────────────────────────────────────────

class EnergyNet(nn.Module):
    """
    Scalar energy function E_θ: ℝ^d → ℝ.
    Lower energy = higher probability.
    Architecture: MLP with Swish activations (smooth, unbounded).
    """

    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns E_θ(x), shape (batch,)."""
        return self.net(x).squeeze(-1)


class ScoreNet(nn.Module):
    """
    Score network s_θ(x, σ): ℝ^d × ℝ → ℝ^d.
    Used for noise-conditioned score matching (NCSN).
    Concatenates log(σ) as conditioning input.
    """

    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        # +1 for log_sigma conditioning
        layers = [nn.Linear(input_dim + 1, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d)
            sigma: (batch,) noise level
        Returns:
            score: (batch, d)  — estimated ∇_x log p_σ(x)
        """
        log_sigma = sigma.log().unsqueeze(1)          # (batch, 1)
        inp = torch.cat([x, log_sigma], dim=1)
        return self.net(inp)


# ─────────────────────────────────────────────
# Langevin MCMC Sampler
# ─────────────────────────────────────────────

def langevin_dynamics(
    energy_fn: EnergyNet,
    x_init: torch.Tensor,
    n_steps: int,
    step_size: float,
    noise_scale: float = 1.0,
    clamp_range: tuple | None = (-3.0, 3.0),
) -> torch.Tensor:
    """
    Unadjusted Langevin Algorithm (ULA):
      x_{t+1} = x_t - (α/2) ∇_x E(x_t) + √α · ε

    Args:
        energy_fn: EBM energy network (parameters frozen during sampling)
        x_init:    initial samples (batch, d) — detached from graph
        n_steps:   number of Langevin steps
        step_size: α (learning rate for chain)
        noise_scale: scale factor on injected noise (set 0 for MAP estimate)
        clamp_range: keep samples in range to prevent explosion

    Returns:
        x_final: (batch, d) — samples after n_steps
    """
    x = x_init.detach().clone().to(DEVICE)
    x.requires_grad_(True)

    for _ in range(n_steps):
        energy = energy_fn(x).sum()
        grad = torch.autograd.grad(energy, x)[0]

        noise = torch.randn_like(x) * noise_scale
        x = x - (step_size / 2.0) * grad + (step_size ** 0.5) * noise
        x = x.detach()

        if clamp_range is not None:
            x = x.clamp(*clamp_range)

        x.requires_grad_(True)

    return x.detach()


def annealed_langevin(
    score_fn: ScoreNet,
    x_init: torch.Tensor,
    sigmas: torch.Tensor,
    n_steps_per_sigma: int,
    step_size_factor: float = 1.0,
) -> torch.Tensor:
    """
    Annealed Langevin dynamics (Song & Ermon 2019):
    For each noise level σ_i (high → low), run n_steps_per_sigma Langevin steps
    using score s_θ(x, σ_i) as the gradient signal.

    α_i = ε · (σ_i / σ_min)²   where ε = step_size_factor
    """
    x = x_init.detach().clone().to(DEVICE)
    sigma_min = sigmas[-1].item()

    for sigma in sigmas:
        alpha = step_size_factor * (sigma.item() / sigma_min) ** 2
        sigma_vec = sigma.expand(x.shape[0]).to(DEVICE)

        for _ in range(n_steps_per_sigma):
            x = x.detach()
            x.requires_grad_(False)
            with torch.no_grad():
                score = score_fn(x, sigma_vec)
            noise = torch.randn_like(x)
            x = x + (alpha / 2.0) * score + (alpha ** 0.5) * noise

    return x.detach()


# ─────────────────────────────────────────────
# Replay Buffer for Persistent Contrastive Divergence
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Persistent Markov chain replay buffer for CD training.
    Stores past MCMC samples; with probability p use stored init,
    else start from uniform noise. This prevents mode collapse.
    """

    def __init__(self, buffer_size: int, data_dim: int):
        self.buffer = torch.randn(buffer_size, data_dim)
        self.buffer_size = buffer_size

    def sample(self, n: int, prob_replay: float = 0.95) -> torch.Tensor:
        idx = torch.randint(0, self.buffer_size, (n,))
        stored = self.buffer[idx].clone()
        # Mix: replay vs fresh noise init
        use_replay = torch.rand(n) < prob_replay
        fresh = torch.randn(n, stored.shape[1])
        init = torch.where(use_replay.unsqueeze(1), stored, fresh)
        return init

    def update(self, samples: torch.Tensor):
        n = samples.shape[0]
        idx = torch.randint(0, self.buffer_size, (n,))
        self.buffer[idx] = samples.cpu().detach()


# ─────────────────────────────────────────────
# Training Objectives
# ─────────────────────────────────────────────

def contrastive_divergence_loss(
    energy_fn: EnergyNet,
    x_real: torch.Tensor,
    replay_buffer: ReplayBuffer,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CD-k loss with persistent replay buffer (PCD):
      L_CD = E_data[E(x)] - E_model[E(x̃)]
      + λ · (E_data[E(x)²] + E_model[E(x̃)²])  ← energy regularization

    Returns:
        loss: scalar
        x_fake: generated samples (for visualization / buffer update)
    """
    batch_size = x_real.shape[0]

    # Sample init from replay buffer
    x_init = replay_buffer.sample(batch_size, cfg["replay_prob"]).to(DEVICE)

    # Run Langevin MCMC
    x_fake = langevin_dynamics(
        energy_fn, x_init,
        n_steps=cfg["mcmc_steps_train"],
        step_size=cfg["mcmc_step_size"],
    )

    # Energy at real and fake samples
    e_real = energy_fn(x_real)       # (batch,)
    e_fake = energy_fn(x_fake)       # (batch,)

    # CD objective: minimize E(real) - E(fake)
    # (we want real to have lower energy than fake)
    cd_loss = e_real.mean() - e_fake.mean()

    # Energy regularization: prevents energy from collapsing to ±∞
    reg = cfg["l2_reg_coef"] * (e_real.pow(2).mean() + e_fake.pow(2).mean())

    loss = cd_loss + reg
    return loss, x_fake


def denoising_score_matching_loss(
    score_fn: ScoreNet,
    x: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """
    Denoising Score Matching (Vincent 2011):
      J_DSM = E_{σ, x, ñ}[ σ² · ||s_θ(x̃, σ) - ñ/σ||² ]
    where x̃ = x + σ·ñ, ñ ~ N(0, I).

    Equivalent to score matching but avoids tr(∇s) via the denoising identity.
    The σ² weighting makes loss scale-invariant across noise levels.
    """
    batch_size = x.shape[0]

    # Sample random noise levels per sample
    sigma_idx = torch.randint(0, len(sigmas), (batch_size,))
    sigma = sigmas[sigma_idx].to(DEVICE)            # (batch,)

    # Perturb data
    noise = torch.randn_like(x)
    x_noisy = x + sigma.unsqueeze(1) * noise        # (batch, d)

    # Predicted score
    score_pred = score_fn(x_noisy, sigma)            # (batch, d)

    # Target score: -noise/sigma (∇_{x̃} log p_σ(x̃|x))
    target = -noise / sigma.unsqueeze(1)

    # Weighted MSE: σ² · ||s - target||²
    loss = (sigma.unsqueeze(1) ** 2 * (score_pred - target).pow(2)).mean()
    return loss


def nce_loss(
    energy_fn: EnergyNet,
    x_real: torch.Tensor,
    noise_dist_std: float = 1.5,
) -> torch.Tensor:
    """
    Noise Contrastive Estimation (Gutmann & Hyvärinen 2010):
    Binary classification: real data vs noise q(x) = N(0, σ²I).

    logit(x) = log p_θ(x) - log q(x)
             = -E_θ(x) - log Z_θ + const   (treat log Z as learnable scalar)

    Here we approximate log Z = 0 for simplicity (valid up to a constant shift).
    The noise log-prob is log q(x) = -d/2 log(2π σ²) - ||x||²/(2σ²).
    """
    batch_size = x_real.shape[0]
    d = x_real.shape[1]

    # Sample noise
    x_noise = torch.randn(batch_size, d, device=DEVICE) * noise_dist_std

    # Log-probability of noise distribution
    log_var = 2 * np.log(noise_dist_std)
    log_q_real = -0.5 * (d * np.log(2 * np.pi) + d * log_var
                          + (x_real.pow(2).sum(1) / noise_dist_std ** 2))
    log_q_noise = -0.5 * (d * np.log(2 * np.pi) + d * log_var
                           + (x_noise.pow(2).sum(1) / noise_dist_std ** 2))

    # Model log-prob (unnormalized): -E_θ(x)
    log_p_real = -energy_fn(x_real)
    log_p_noise = -energy_fn(x_noise)

    # Logistic regression: logit = log p_model - log q
    logit_real = log_p_real - log_q_real
    logit_noise = log_p_noise - log_q_noise

    # Binary cross-entropy: real=1, noise=0
    loss_real = torch.nn.functional.softplus(-logit_real).mean()
    loss_noise = torch.nn.functional.softplus(logit_noise).mean()

    return loss_real + loss_noise


# ─────────────────────────────────────────────
# EBM Trainer (CD / NCE)
# ─────────────────────────────────────────────

class EBMTrainer:
    """
    Wraps EnergyNet with Contrastive Divergence or NCE training.
    Uses a replay buffer for persistent CD chains.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = EnergyNet(
            input_dim=cfg["data_dim"],
            hidden_dim=cfg["energy_hidden_dim"],
            depth=cfg["energy_depth"],
        ).to(DEVICE)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        self.replay_buffer = ReplayBuffer(
            buffer_size=cfg["replay_buffer_size"],
            data_dim=cfg["data_dim"],
        )
        self.train_losses: list[float] = []

    def train(
        self,
        X_train: np.ndarray,
        mode: str = "cd",
        verbose: bool = True,
    ):
        """
        Args:
            X_train: (n, d) training data (numpy)
            mode: "cd" (contrastive divergence) or "nce"
        """
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=True)

        for epoch in range(self.cfg["n_epochs"]):
            epoch_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(DEVICE)
                self.optimizer.zero_grad()

                if mode == "cd":
                    loss, x_fake = contrastive_divergence_loss(
                        self.model, x_batch, self.replay_buffer, self.cfg
                    )
                    self.replay_buffer.update(x_fake)
                elif mode == "nce":
                    loss = nce_loss(self.model, x_batch)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip_grad"])
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.train_losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.cfg['n_epochs']} | Loss: {avg_loss:.4f}")

    def sample(self, n: int) -> np.ndarray:
        """Generate n samples via Langevin dynamics from noise init."""
        x_init = torch.randn(n, self.cfg["data_dim"], device=DEVICE)
        samples = langevin_dynamics(
            self.model, x_init,
            n_steps=self.cfg["mcmc_steps_sample"],
            step_size=self.cfg["mcmc_step_size"],
        )
        return samples.cpu().numpy()

    def energy(self, X: np.ndarray) -> np.ndarray:
        x = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            e = self.model(x)
        return e.cpu().numpy()


# ─────────────────────────────────────────────
# Score-Based Model Trainer (DSM / NCSN)
# ─────────────────────────────────────────────

class NCSNTrainer:
    """
    Noise-Conditioned Score Network (Song & Ermon 2019).
    Trains s_θ(x, σ) via denoising score matching across multiple noise levels.
    Samples via annealed Langevin dynamics (σ_max → σ_min).
    """

    def __init__(self, cfg: dict, sigma_min: float = 0.01, sigma_max: float = 1.0):
        self.cfg = cfg
        self.sigmas = torch.exp(
            torch.linspace(np.log(sigma_max), np.log(sigma_min), cfg["n_noise_levels"])
        ).to(DEVICE)
        self.model = ScoreNet(
            input_dim=cfg["data_dim"],
            hidden_dim=cfg["energy_hidden_dim"],
            depth=cfg["energy_depth"],
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["lr"])
        self.train_losses: list[float] = []

    def train(self, X_train: np.ndarray, verbose: bool = True):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=True)

        for epoch in range(self.cfg["n_epochs"]):
            epoch_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(DEVICE)
                self.optimizer.zero_grad()
                loss = denoising_score_matching_loss(self.model, x_batch, self.sigmas)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["clip_grad"])
                self.optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            self.train_losses.append(avg)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.cfg['n_epochs']} | DSM Loss: {avg:.4f}")

    def sample(self, n: int, n_steps_per_sigma: int = 100) -> np.ndarray:
        x_init = torch.randn(n, self.cfg["data_dim"], device=DEVICE) * self.sigmas[0].item()
        samples = annealed_langevin(
            self.model, x_init, self.sigmas,
            n_steps_per_sigma=n_steps_per_sigma,
            step_size_factor=self.cfg["mcmc_step_size"],
        )
        return samples.cpu().numpy()


# ─────────────────────────────────────────────
# Data Generation
# ─────────────────────────────────────────────

def generate_data(
    dataset: str = "moons",
    n_samples: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D toy datasets.
    Replace with your own data loader for real applications.

    Supported: "moons", "circles", "checkerboard", "gaussian_mixture", "pinwheel"
    Returns: (X_train, X_test) — standardized to zero mean, unit variance
    """
    rng = np.random.default_rng(seed)

    if dataset == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    elif dataset == "circles":
        X, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=seed)
    elif dataset == "checkerboard":
        # 4x4 checkerboard: alternate squares
        n_sq = n_samples // 8
        pieces = []
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    pts = rng.uniform(size=(n_sq, 2))
                    pts[:, 0] = pts[:, 0] + i
                    pts[:, 1] = pts[:, 1] + j
                    pieces.append(pts)
        X = np.vstack(pieces)
        X = (X - X.mean(0)) / X.std(0)
    elif dataset == "gaussian_mixture":
        centers = [(0, 0), (3, 0), (-3, 0), (1.5, 2.6), (-1.5, 2.6),
                   (1.5, -2.6), (-1.5, -2.6)]
        pieces = []
        n_per = n_samples // len(centers)
        for cx, cy in centers:
            pts = rng.normal([cx, cy], 0.4, (n_per, 2))
            pieces.append(pts)
        X = np.vstack(pieces)
    elif dataset == "pinwheel":
        n_arms = 5
        n_per = n_samples // n_arms
        pieces = []
        for k in range(n_arms):
            t = rng.uniform(0, 3, n_per)
            angle = 2 * np.pi * k / n_arms + t * 0.5
            r = t
            pts = np.stack([r * np.cos(angle), r * np.sin(angle)], axis=1)
            pts += rng.normal(0, 0.1, pts.shape)
            pieces.append(pts)
        X = np.vstack(pieces)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    split = int(0.8 * len(X))
    idx = rng.permutation(len(X))
    return X[idx[:split]], X[idx[split:]]


# ─────────────────────────────────────────────
# Anomaly Detection with EBM
# ─────────────────────────────────────────────

def anomaly_detection_demo(
    trainer: EBMTrainer,
    X_normal: np.ndarray,
    n_anomalies: int = 200,
    seed: int = 42,
) -> float:
    """
    Anomaly detection: score = E_θ(x), higher energy → more anomalous.
    Anomalies sampled from uniform distribution outside training support.

    Returns ROC-AUC.
    """
    rng = np.random.default_rng(seed)
    X_range = X_normal.max(0) - X_normal.min(0)
    center = X_normal.mean(0)

    # Out-of-distribution anomalies: uniform far from center
    X_anom = rng.uniform(
        center - 4 * X_range,
        center + 4 * X_range,
        size=(n_anomalies, X_normal.shape[1]),
    ).astype(np.float32)

    X_all = np.vstack([X_normal, X_anom])
    y_all = np.array([0] * len(X_normal) + [1] * n_anomalies)

    scores = trainer.energy(X_all)   # higher energy = anomaly
    auc = roc_auc_score(y_all, scores)
    print(f"Anomaly Detection ROC-AUC: {auc:.4f}")
    return auc


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_energy_landscape(
    trainer: EBMTrainer,
    X_train: np.ndarray,
    title: str = "EBM Energy Landscape",
    resolution: int = 200,
):
    """2D energy contour plot overlaid with training data and generated samples."""
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    energy_grid = trainer.energy(grid).reshape(resolution, resolution)

    # Clamp for visualization
    energy_grid = np.clip(energy_grid, np.percentile(energy_grid, 2), np.percentile(energy_grid, 98))

    generated = trainer.sample(500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13)

    # Energy landscape
    im = axes[0].contourf(xx, yy, energy_grid, levels=50, cmap="RdYlBu_r")
    axes[0].scatter(X_train[:, 0], X_train[:, 1], s=2, c="white", alpha=0.3, label="Train data")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("Energy E_θ(x)")
    axes[0].set_xlabel("x₁"); axes[0].set_ylabel("x₂")

    # Log-probability landscape
    log_p_grid = -energy_grid
    log_p_grid -= log_p_grid.max()   # shift for visualization
    im2 = axes[1].contourf(xx, yy, np.exp(log_p_grid), levels=50, cmap="Blues")
    axes[1].scatter(X_train[:, 0], X_train[:, 1], s=2, c="orange", alpha=0.3, label="Train data")
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_title("Unnormalized p_θ(x) ∝ exp(-E)")
    axes[1].set_xlabel("x₁")

    # Training data vs generated samples
    axes[2].scatter(X_train[:, 0], X_train[:, 1], s=5, alpha=0.4, label="Real data")
    axes[2].scatter(generated[:, 0], generated[:, 1], s=5, alpha=0.4, label="Generated", c="red")
    axes[2].set_title("Real vs Generated Samples")
    axes[2].set_xlabel("x₁")
    axes[2].legend(markerscale=2)

    plt.tight_layout()
    plt.savefig("ebm_energy_landscape.png", dpi=150)
    plt.show()
    print("Saved: ebm_energy_landscape.png")


def plot_score_field(
    trainer: NCSNTrainer,
    X_train: np.ndarray,
    sigma_idx: int = -1,
    resolution: int = 20,
):
    """Visualize score s_θ(x, σ) as a vector field."""
    sigma = trainer.sigmas[sigma_idx]

    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32, device=DEVICE
    )
    sigma_vec = sigma.expand(grid.shape[0])

    with torch.no_grad():
        scores = trainer.model(grid, sigma_vec).cpu().numpy()

    score_u = scores[:, 0].reshape(resolution, resolution)
    score_v = scores[:, 1].reshape(resolution, resolution)
    magnitude = np.sqrt(score_u ** 2 + score_v ** 2) + 1e-8
    score_u_norm = score_u / magnitude
    score_v_norm = score_v / magnitude

    generated = trainer.sample(500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Score Network  (σ = {sigma.item():.3f})", fontsize=12)

    axes[0].quiver(xx, yy, score_u_norm, score_v_norm, magnitude,
                   cmap="viridis", scale=30, alpha=0.7)
    axes[0].scatter(X_train[:, 0], X_train[:, 1], s=5, c="red", alpha=0.4)
    axes[0].set_title("Score field  s_θ(x, σ)")
    axes[0].set_xlabel("x₁"); axes[0].set_ylabel("x₂")

    axes[1].scatter(X_train[:, 0], X_train[:, 1], s=5, alpha=0.4, label="Real")
    axes[1].scatter(generated[:, 0], generated[:, 1], s=5, alpha=0.4, c="red", label="Generated")
    axes[1].set_title("Real vs Annealed Langevin Samples")
    axes[1].set_xlabel("x₁")
    axes[1].legend(markerscale=2)

    plt.tight_layout()
    plt.savefig("ncsn_score_field.png", dpi=150)
    plt.show()
    print("Saved: ncsn_score_field.png")


def plot_training_comparison(
    losses_cd: list[float],
    losses_nce: list[float],
    losses_dsm: list[float],
):
    """Compare training loss curves for CD, NCE, and DSM."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("EBM Training Objectives — Loss Curves", fontsize=12)

    for ax, losses, name, color in zip(
        axes,
        [losses_cd, losses_nce, losses_dsm],
        ["Contrastive Divergence (CD-k)", "Noise Contrastive Estimation", "Denoising Score Matching"],
        ["steelblue", "darkorange", "forestgreen"],
    ):
        ax.plot(losses, color=color)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ebm_training_comparison.png", dpi=150)
    plt.show()
    print("Saved: ebm_training_comparison.png")


def plot_langevin_chain(
    trainer: EBMTrainer,
    n_chains: int = 5,
    dataset_name: str = "moons",
):
    """
    Visualize Markov chain trajectories as Langevin converges.
    Shows how chains from random init migrate toward high-probability regions.
    """
    cfg = trainer.cfg
    x = torch.randn(n_chains, cfg["data_dim"], device=DEVICE)
    snapshots = [x.cpu().numpy().copy()]

    checkpoints = [50, 200, 500, 1000]
    step = 0
    for target in checkpoints:
        n = target - step
        x = langevin_dynamics(
            trainer.model, x, n_steps=n, step_size=cfg["mcmc_step_size"]
        )
        snapshots.append(x.cpu().numpy().copy())
        step = target

    X_train, _ = generate_data(dataset_name, n_samples=500)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    fig.suptitle("Langevin Chain Evolution", fontsize=12)
    titles = ["Init (t=0)"] + [f"t={c}" for c in checkpoints]

    for ax, snap, t in zip(axes, snapshots, titles):
        ax.scatter(X_train[:, 0], X_train[:, 1], s=5, alpha=0.2, c="gray", label="Data")
        ax.scatter(snap[:, 0], snap[:, 1], s=60, c="red", zorder=5, label="Chains")
        ax.set_title(t)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("ebm_langevin_chain.png", dpi=150)
    plt.show()
    print("Saved: ebm_langevin_chain.png")


# ─────────────────────────────────────────────
# Compare EBM Training Modes
# ─────────────────────────────────────────────

def compare_training_modes(
    dataset_name: str = "moons",
    n_epochs: int = 50,
) -> dict:
    """
    Train EBM with CD, NCE, and NCSN (DSM) on the same dataset.
    Compare via visual quality of generated samples.
    """
    X_train, X_test = generate_data(dataset_name, n_samples=CONFIG["n_samples"])
    cfg = {**CONFIG, "n_epochs": n_epochs}

    results = {}

    print("\n── Contrastive Divergence ──")
    cd_trainer = EBMTrainer(cfg)
    cd_trainer.train(X_train, mode="cd", verbose=True)
    results["cd"] = cd_trainer

    print("\n── Noise Contrastive Estimation ──")
    nce_trainer = EBMTrainer(cfg)
    nce_trainer.train(X_train, mode="nce", verbose=True)
    results["nce"] = nce_trainer

    print("\n── Denoising Score Matching (NCSN) ──")
    dsm_trainer = NCSNTrainer(cfg)
    dsm_trainer.train(X_train, verbose=True)
    results["dsm"] = dsm_trainer

    # Anomaly detection with CD-trained model
    anomaly_detection_demo(cd_trainer, X_test)

    # Loss comparison
    plot_training_comparison(
        cd_trainer.train_losses,
        nce_trainer.train_losses,
        dsm_trainer.train_losses,
    )

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}\n")

    # ── Dataset choice ──────────────────────────────────────────────────
    DATASET = "moons"   # # Placeholder: "moons" | "circles" | "checkerboard" | "gaussian_mixture" | "pinwheel"
    X_train, X_test = generate_data(DATASET, n_samples=CONFIG["n_samples"])
    print(f"Data: {DATASET} | train={len(X_train)}, test={len(X_test)}")

    # ── Train EBM (CD) ─────────────────────────────────────────────────
    print("\n=== Contrastive Divergence EBM ===")
    cd_trainer = EBMTrainer(CONFIG)
    cd_trainer.train(X_train, mode="cd")

    # Energy landscape visualization
    plot_energy_landscape(cd_trainer, X_train, title=f"CD-EBM: {DATASET}")

    # Langevin chain visualization
    plot_langevin_chain(cd_trainer, n_chains=8, dataset_name=DATASET)

    # Anomaly detection
    auc = anomaly_detection_demo(cd_trainer, X_test)

    # ── Train NCSN (DSM) ───────────────────────────────────────────────
    print("\n=== NCSN — Denoising Score Matching ===")
    dsm_trainer = NCSNTrainer(CONFIG)
    dsm_trainer.train(X_train)

    # Score field (finest noise level = last sigma)
    plot_score_field(dsm_trainer, X_train, sigma_idx=-1)

    # ── Multi-mode comparison ──────────────────────────────────────────
    print("\n=== Comparing Training Modes ===")
    compare_training_modes(DATASET, n_epochs=50)

    print(f"\nDone. Final anomaly AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
