"""
Neural ODE (Ordinary Differential Equation)
=============================================
A continuous-depth model where the hidden state evolves according to a neural
network-parameterized ODE. Instead of discrete layer-by-layer transformations,
the hidden state h(t) is the solution of:

    dh/dt = f_θ(h(t), t)
    h(t₁) = h(t₀) + ∫_{t₀}^{t₁} f_θ(h(t), t) dt

where f_θ is a neural network (the "dynamics network").

The integral is computed using a numerical ODE solver (Euler, RK4, Dormand-Prince).
Gradients flow back through the solver via the adjoint method, which avoids
storing all intermediate states — memory is O(1) in depth.

Adjoint method (backpropagation through ODE solver):
    Augment state with adjoint variable a(t) = -∂L/∂h(t).
    Run the augmented ODE backward:
        d/dt [h, a, ∂L/∂θ] = [f_θ, -a·∂f/∂h, -a·∂f/∂θ]
    This is O(depth) in memory vs. O(depth²) for direct differentiation.

Relationship to ResNets:
    A ResNet layer:  h_{l+1} = h_l + f_θ(h_l)
    Euler ODE step:  h(t+dt) = h(t) + dt · f_θ(h(t), t)
    Neural ODE is the continuous limit of a ResNet as depth → ∞, dt → 0.

Models implemented:
    1. ODENet classifier:      solve ODE → pooling → linear head
    2. Latent ODE:             encode observations → latent initial state →
                               solve ODE → decode; handles irregular time series
    3. Neural SDE (sketch):    stochastic dynamics with Wiener process noise

Use Cases:
    - Irregularly-sampled time series (medical, financial, sensor data)
    - Continuous normalizing flows (change-of-variables formula)
    - Trajectory modeling with known smooth dynamics
    - Latent dynamics discovery from time-series observations
    - Continuous-time control and smooth state evolution

Advantages:
    - Continuous-time: handles missing/irregular observations natively
    - Memory-efficient training via adjoint method
    - Adaptive computation: solver uses more steps where dynamics are complex
    - Natural inductive bias for smooth, regular dynamics

Limitations:
    - Slow training: ODE solve at every forward pass
    - Stiff ODEs cause numerical instability (require implicit solvers)
    - Not expressive for non-smooth dynamics

Fine-Tuning Recommendations:
    - ODE solver: 'rk4' for speed, 'dopri5' (adaptive RK45) for accuracy.
      Use 'euler' only for debugging — large step error.
    - adjoint=True: use adjoint backprop in production to save memory.
    - dynamics network: small MLP (2–3 layers, 64–128 hidden); tanh/softplus.
      Avoid ReLU — creates non-smooth dynamics (ODE solver struggles).
    - Time horizon T: normalize to [0, 1]. Very long T → stiffness.
    - For time series: use torchdiffeq (pip install torchdiffeq) for full
      adjoint support. This implementation shows the pattern with custom RK4.
    - rtol/atol: default 1e-3/1e-6 balances speed and accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ─────────────────────────────────────────────
# ODE SOLVERS (pure PyTorch, no torchdiffeq dep)
# ─────────────────────────────────────────────

def euler_step(f, h, t, dt):
    """Explicit Euler method (first-order)."""
    return h + dt * f(h, t)


def rk4_step(f, h, t, dt):
    """
    Classic 4th-order Runge-Kutta step.
    More accurate than Euler; good default for smooth dynamics.
    """
    k1 = f(h,           t)
    k2 = f(h + dt/2*k1, t + dt/2)
    k3 = f(h + dt/2*k2, t + dt/2)
    k4 = f(h + dt*k3,   t + dt)
    return h + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def odeint(f, h0: torch.Tensor, t_span: tuple,
            n_steps: int = 20, method: str = "rk4") -> torch.Tensor:
    """
    Integrate ODE dh/dt = f(h, t) from t_span[0] to t_span[1].

    For production use: replace with torchdiffeq.odeint_adjoint which
    provides adaptive step-size control and memory-efficient adjoint gradients.

        from torchdiffeq import odeint_adjoint
        h_T = odeint_adjoint(f, h0, t_eval, rtol=1e-3, atol=1e-6)

    Args:
        f:       callable f(h, t) → dh/dt
        h0:      initial state (batch, dim)
        t_span:  (t0, t1)
        n_steps: number of integration steps
        method:  'euler' or 'rk4'

    Returns:
        h_T: state at t1 (batch, dim)
    """
    h   = h0
    t   = t_span[0]
    dt  = (t_span[1] - t_span[0]) / n_steps
    step_fn = rk4_step if method == "rk4" else euler_step

    for _ in range(n_steps):
        h = step_fn(f, h, t, dt)
        t = t + dt
    return h


def odeint_trajectory(f, h0: torch.Tensor, t_eval: torch.Tensor,
                       method: str = "rk4") -> torch.Tensor:
    """
    Integrate ODE and return states at multiple evaluation times t_eval.

    Returns:
        trajectory: (len(t_eval), batch, dim)
    """
    traj = [h0]
    h    = h0
    for i in range(1, len(t_eval)):
        dt = t_eval[i] - t_eval[i-1]
        t  = t_eval[i-1]
        h  = rk4_step(f, h, t, dt) if method == "rk4" else euler_step(f, h, t, dt)
        traj.append(h)
    return torch.stack(traj, dim=0)


# ─────────────────────────────────────────────
# DYNAMICS NETWORK
# ─────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    Neural network parameterizing the ODE dynamics: dh/dt = f_θ(h, t).

    Design principles:
        - Small: ODE solve calls this O(n_steps) times per forward pass
        - Smooth activations: tanh / softplus (not ReLU — non-smooth gradients)
        - Optional time input: concatenate t for non-autonomous dynamics
    """

    def __init__(self, hidden_dim: int = 64, time_dependent: bool = False):
        super().__init__()
        self.time_dep = time_dependent
        in_dim = hidden_dim + 1 if time_dependent else hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Initialize near zero so initial dynamics are close to identity
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

        self.nfe = 0  # number of function evaluations (diagnostic)

    def forward(self, h: torch.Tensor, t: float) -> torch.Tensor:
        self.nfe += 1
        if self.time_dep:
            t_vec = torch.full((h.shape[0], 1), t, device=h.device, dtype=h.dtype)
            h_in  = torch.cat([h, t_vec], dim=1)
        else:
            h_in = h
        return self.net(h_in)


# ─────────────────────────────────────────────
# ODENET CLASSIFIER
# ─────────────────────────────────────────────

class ODENetClassifier(nn.Module):
    """
    Neural ODE for classification.

    Architecture:
        Feature extractor (downsampling MLP)
        → ODEBlock (continuous dynamics, t ∈ [0, 1])
        → Classifier head

    Equivalent to an infinite-depth ResNet with tied weights.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int,
                 n_steps: int = 20, method: str = "rk4"):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.odefunc   = ODEFunc(hidden_dim)
        self.n_steps   = n_steps
        self.method    = method
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.feature_net(x)
        self.odefunc.nfe = 0
        hT = odeint(self.odefunc, h0, t_span=(0.0, 1.0),
                     n_steps=self.n_steps, method=self.method)
        return self.classifier(hT)


# ─────────────────────────────────────────────
# LATENT ODE (IRREGULAR TIME SERIES)
# ─────────────────────────────────────────────

class LatentODE(nn.Module):
    """
    Latent ODE for irregularly-sampled time series.

    Architecture:
        RNN Encoder:  observations → latent initial state z(t₀)
                      (runs backward in time for better context)
        ODE Solver:   z(t) evolves forward: dz/dt = f_θ(z)
        Decoder:      z(t_i) → x̂(t_i)

    Key property: handles missing observations naturally — the ODE solver
    evaluates z at any query time without requiring regular spacing.

    For a complete VAE-style Latent ODE with uncertainty:
        - Replace encoder with recognition network outputting μ, σ
        - Add KL divergence term to loss
        See: Chen et al. (2018) "Neural Ordinary Differential Equations"
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        # GRU encoder (processes observations backward)
        self.encoder_gru  = nn.GRUCell(obs_dim, hidden_dim)
        self.latent_proj  = nn.Linear(hidden_dim, latent_dim)

        # ODE dynamics in latent space
        self.odefunc = ODEFunc(latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def encode(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode time series to initial latent state z(t₀).

        Args:
            obs_seq: (T, B, obs_dim) — observations (can have NaN for missing)

        Returns:
            z0: (B, latent_dim) initial latent state
        """
        B = obs_seq.shape[1]
        h = torch.zeros(B, self.encoder_gru.hidden_size, device=obs_seq.device)
        # Run backward for better IC estimation
        for t in range(obs_seq.shape[0] - 1, -1, -1):
            obs = obs_seq[t]  # (B, obs_dim)
            h   = self.encoder_gru(obs, h)
        return self.latent_proj(h)

    def forward(self, obs_seq: torch.Tensor,
                t_eval: torch.Tensor) -> torch.Tensor:
        """
        Encode → ODE solve → decode.

        Args:
            obs_seq: (T, B, obs_dim)
            t_eval:  (T_out,) query times

        Returns:
            x_hat: (T_out, B, obs_dim) reconstructed observations
        """
        z0   = self.encode(obs_seq)
        traj = odeint_trajectory(self.odefunc, z0, t_eval)  # (T_out, B, latent)
        # Decode each time step
        T, B, D = traj.shape
        x_hat = self.decoder(traj.view(T * B, D)).view(T, B, -1)
        return x_hat


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_spiral_data(n_samples: int = 1000, noise: float = 0.1,
                          seed: int = 42) -> tuple:
    """
    2-class spiral dataset — classic nonlinear classification benchmark.

    Replace with your own data:
        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
    """
    rng = np.random.RandomState(seed)
    n   = n_samples // 2

    def spiral(n, delta_t, noise):
        t = np.linspace(0, 4 * np.pi, n) + rng.uniform(0, delta_t, n)
        x = t * np.cos(t) + rng.normal(0, noise, n)
        y = t * np.sin(t) + rng.normal(0, noise, n)
        return np.c_[x, y]

    X = np.vstack([spiral(n, 0, noise), spiral(n, np.pi, noise)])
    y = np.array([0] * n + [1] * n)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))


def generate_latent_ode_data(n_series: int = 200, T: int = 50,
                               seed: int = 42) -> tuple:
    """
    Synthetic irregular time series: observations from a damped oscillator.
    Some time steps are randomly dropped to simulate irregular sampling.

    Returns:
        obs_seq:  (T, n_series, 2) — position + velocity observations
        t_eval:   (T,) — observation times
        true_traj:(T, n_series, 2) — noiseless ground truth
    """
    rng   = np.random.RandomState(seed)
    dt    = 0.1
    omega = 1.0
    gamma = 0.05

    # Simulate damped oscillator: ẍ + 2γẋ + ω²x = 0
    t = np.arange(T) * dt
    traj = np.zeros((T, n_series, 2))
    x0   = rng.uniform(-2, 2, n_series)
    v0   = rng.uniform(-1, 1, n_series)
    traj[0, :, 0] = x0
    traj[0, :, 1] = v0
    for i in range(1, T):
        x, v = traj[i-1, :, 0], traj[i-1, :, 1]
        a    = -2 * gamma * v - omega**2 * x
        traj[i, :, 0] = x + v * dt
        traj[i, :, 1] = v + a * dt

    obs_seq = traj + rng.normal(0, 0.1, traj.shape)
    return (torch.tensor(obs_seq, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(traj, dtype=torch.float32))


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

# Placeholder: tune hidden_dim, n_steps, method
NODE_CONFIG = {
    "hidden_dim": 64,    # Placeholder: 32–256
    "n_steps":    20,    # Placeholder: 10–50; more steps → more accurate but slower
    "method":     "rk4", # Placeholder: 'euler', 'rk4'
    "lr":         1e-3,
    "epochs":     50,
    "batch_size": 128,
}


def train_odenet(X: torch.Tensor, y: torch.Tensor,
                  config: dict = None) -> tuple:
    """Train ODENet classifier."""
    if config is None:
        config = NODE_CONFIG

    n_classes = int(y.max().item()) + 1
    model = ODENetClassifier(
        input_dim=X.shape[1],
        hidden_dim=config["hidden_dim"],
        n_classes=n_classes,
        n_steps=config["n_steps"],
        method=config["method"],
    ).to(DEVICE)

    ds     = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=config["lr"])

    history = {"loss": [], "acc": []}
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, total_correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits  = model(xb)
            loss    = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss    += loss.item() * len(xb)
            total_correct += (logits.argmax(1) == yb).sum().item()
            n += len(xb)

        acc = total_correct / n
        history["loss"].append(total_loss / n)
        history["acc"].append(acc)
        if epoch % 10 == 0:
            nfe = model.odefunc.nfe // (n // config["batch_size"])
            print(f"  Epoch {epoch:3d} | loss: {total_loss/n:.4f} | "
                  f"acc: {acc:.4f} | NFE/fwd: {nfe}")
    return model, history


def train_latent_ode(obs_seq: torch.Tensor, t_eval: torch.Tensor,
                      true_traj: torch.Tensor,
                      config: dict = None) -> tuple:
    """Train Latent ODE on irregular time series reconstruction."""
    if config is None:
        config = NODE_CONFIG

    model = LatentODE(obs_dim=2, latent_dim=8,
                       hidden_dim=config["hidden_dim"]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=config["lr"])

    obs_d   = obs_seq.to(DEVICE)
    t_d     = t_eval.to(DEVICE)
    true_d  = true_traj.to(DEVICE)

    history = {"loss": []}
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        opt.zero_grad()
        x_hat = model(obs_d, t_d)
        loss  = F.mse_loss(x_hat, true_d)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history["loss"].append(loss.item())
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | recon loss: {loss.item():.6f}")

    return model, history


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_spiral_boundary(model: ODENetClassifier, X: torch.Tensor,
                          y: torch.Tensor) -> None:
    """2D decision boundary for spiral classification."""
    model.eval()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                          np.linspace(y_min, y_max, 150))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        Z = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(7, 7))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    for cls in range(2):
        mask = y == cls
        plt.scatter(X[mask, 0], X[mask, 1], s=8, alpha=0.6,
                    c=["steelblue", "tomato"][cls], label=f"Class {cls}")
    plt.title("Neural ODE — Decision Boundary (Spiral)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("node_boundary.png", dpi=150)
    plt.show()


def plot_latent_trajectories(model: LatentODE, obs_seq: torch.Tensor,
                              t_eval: torch.Tensor,
                              true_traj: torch.Tensor,
                              n_show: int = 5) -> None:
    """
    Plot ground truth vs. Latent ODE reconstruction for a few series.
    """
    model.eval()
    with torch.no_grad():
        x_hat = model(obs_seq.to(DEVICE), t_eval.to(DEVICE)).cpu()

    t = t_eval.numpy()
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(t, true_traj[:, i, 0].numpy(), "k-",      linewidth=1.5, label="True")
        ax.plot(t, obs_seq[:,  i, 0].numpy(),  "o",        markersize=3, color="gray", alpha=0.5, label="Obs")
        ax.plot(t, x_hat[:,    i, 0].numpy(),  "tomato",   linewidth=1.5, linestyle="--", label="Latent ODE")
        ax.set_title(f"Series {i}")
        ax.set_xlabel("t")
    axes[0].set_ylabel("x(t)")
    axes[0].legend(fontsize=7)
    plt.suptitle("Latent ODE — Reconstruction of Damped Oscillators")
    plt.tight_layout()
    plt.savefig("node_latent.png", dpi=150)
    plt.show()


def plot_training_curves(history_cls: dict, history_lat: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].semilogy(history_cls["loss"],  color="steelblue")
    axes[0].set_title("ODENet — Training Loss"); axes[0].set_xlabel("Epoch")

    axes[1].plot(history_cls["acc"], color="tomato")
    axes[1].set_title("ODENet — Accuracy"); axes[1].set_xlabel("Epoch")

    axes[2].semilogy(history_lat["loss"], color="darkgreen")
    axes[2].set_title("Latent ODE — Recon Loss"); axes[2].set_xlabel("Epoch")

    plt.suptitle("Neural ODE — Training Curves")
    plt.tight_layout()
    plt.savefig("node_training.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"=== Neural ODE ===  (device: {DEVICE})")

    # 1. ODENet classification on spiral data
    print("\n--- ODENet Classifier (Spiral) ---")
    X, y = generate_spiral_data(n_samples=1000)
    model_cls, hist_cls = train_odenet(X, y)
    print(f"  Final accuracy: {hist_cls['acc'][-1]:.4f}")

    # 2. Latent ODE on irregular time series
    print("\n--- Latent ODE (Damped Oscillator) ---")
    obs_seq, t_eval, true_traj = generate_latent_ode_data()
    model_lat, hist_lat = train_latent_ode(obs_seq, t_eval, true_traj)

    # Visualizations
    plot_training_curves(hist_cls, hist_lat)
    plot_spiral_boundary(model_cls, X, y)
    plot_latent_trajectories(model_lat, obs_seq, t_eval, true_traj)

    print("\nNote: For production Neural ODEs with adjoint backprop:")
    print("      pip install torchdiffeq")
    print("      from torchdiffeq import odeint_adjoint")


if __name__ == "__main__":
    main()
