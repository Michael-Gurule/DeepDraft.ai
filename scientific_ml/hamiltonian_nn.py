"""
Hamiltonian Neural Network (HNN)
==================================
A physics-informed neural network that learns to respect energy conservation
by parameterizing the Hamiltonian H(q, p) rather than the dynamics directly.

Physical motivation:
    In Hamiltonian mechanics, a system's state is (q, p) — generalized
    coordinates and momenta. The equations of motion follow from H:

        dq/dt =  ∂H/∂p
        dp/dt = -∂H/∂q

    By learning H(q, p) rather than (dq/dt, dp/dt) directly, the network
    is structurally forced to conserve energy: the learned dynamics will
    keep H(q(t), p(t)) exactly constant along trajectories.

Network architecture:
    Input:  (q, p) — phase-space coordinates
    Output: H(q, p) — scalar energy (single neuron, no activation)

    Dynamics extracted via autograd:
        q̇_pred =  ∂H_θ/∂p   (autograd)
        ṗ_pred = -∂H_θ/∂q   (autograd)

Loss:
    L = MSE(q̇_pred, q̇_true) + MSE(ṗ_pred, ṗ_true)

    We supervise on known derivatives (from finite differences or direct
    measurements), not on trajectories. This avoids rollout error.

Comparison to vanilla Neural ODE:
    Neural ODE:    learns (q̇, ṗ) = f_θ(q, p) directly — may violate energy
    HNN:           learns H(q, p) → (q̇, ṗ) via symplectic structure — energy conserved
    Result:        HNN trajectories stay on correct energy surface indefinitely;
                   vanilla NODEs drift off the energy shell over long rollouts.

Symplectic integrators for long-horizon rollout:
    Standard RK4 does not exactly preserve the symplectic structure.
    For faithful long-horizon simulation, use the Störmer-Verlet (leapfrog)
    integrator which is exactly symplectic.

Lagrangian Neural Networks (LNN):
    Alternative: parameterize the Lagrangian L(q, q̇) and derive dynamics
    via the Euler-Lagrange equation. Equivalent power but different parameterization.
    Preferred when q̇ is observed more naturally than p.

Use Cases:
    - Molecular dynamics simulation (atomic systems conserve energy)
    - Orbital mechanics (N-body gravitational systems)
    - Robotic joint dynamics (mechanical energy conservation)
    - Plasma physics, fluid mechanics (Hamiltonian structure)
    - Vehicle and robot dynamics: enforce energy conservation in trajectory learning

Systems implemented:
    1. Simple pendulum:   H = (1/2)p² - cos(q)       (1 DOF)
    2. Spring system:     H = p²/2m + kq²/2            (1 DOF)
    3. Double pendulum:   H = T(q,p) + V(q)            (2 DOF, chaotic)

Fine-Tuning Recommendations:
    - Network: 3–4 hidden layers × 256–512 neurons; tanh activation.
      Large networks are needed to accurately capture energy surface geometry.
    - Input normalization: standardize (q, p) to mean 0, std 1. Hamiltonian
      networks are sensitive to input scale.
    - Data: collect dense (q, p, dq/dt, dp/dt) samples across phase space.
      Trajectories from a reference simulator work well.
    - dt for finite differences: small dt (1e-3 to 1e-4) reduces estimation error.
    - Leapfrog rollout: use for evaluation; RK4 for training gradient stability.
    - For Lagrangian NNs: pip install lagrangian-nns or implement manually
      using the Euler-Lagrange equation via autograd.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import autograd

from deepdraft.core import fast_or, save_figure

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# Higher-order autograd is more reliable on CPU
if DEVICE == "mps":
    DEVICE = "cpu"


# ─────────────────────────────────────────────
# HAMILTONIAN SYSTEMS (ground truth)
# ─────────────────────────────────────────────

def pendulum_hamiltonian(q: np.ndarray, p: np.ndarray,
                          m: float = 1.0, g: float = 9.8,
                          L: float = 1.0) -> np.ndarray:
    """True Hamiltonian for simple pendulum: H = p²/(2mL²) - mgL·cos(q)"""
    return p**2 / (2 * m * L**2) - m * g * L * np.cos(q)


def pendulum_dynamics(q: np.ndarray, p: np.ndarray,
                       m: float = 1.0, g: float = 9.8,
                       L: float = 1.0) -> tuple:
    """
    True equations of motion for simple pendulum.
        dq/dt =  ∂H/∂p = p / (mL²)
        dp/dt = -∂H/∂q = -mgL·sin(q)
    """
    dqdt = p / (m * L**2)
    dpdt = -m * g * L * np.sin(q)
    return dqdt, dpdt


def spring_dynamics(q: np.ndarray, p: np.ndarray,
                     m: float = 1.0, k: float = 4.0) -> tuple:
    """
    Harmonic oscillator: H = p²/2m + kq²/2
        dq/dt =  p/m
        dp/dt = -kq
    """
    return p / m, -k * q


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────

def generate_phase_space_data(system: str = "pendulum",
                               n_trajectories: int = 50,
                               n_steps: int = 100,
                               dt: float = 0.05,
                               seed: int = 42) -> dict:
    """
    Simulate phase-space trajectories and compute time derivatives.

    Derivatives computed via centered finite differences:
        dq/dt ≈ (q(t+dt) - q(t-dt)) / (2dt)

    Replace with real physical measurements where derivatives can be
    estimated from high-rate sensor data.

    Returns dict with:
        q, p:          phase-space coordinates (n_samples, dof)
        dqdt, dpdt:    time derivatives (n_samples, dof)
        energy:        total energy along each trajectory
    """
    rng = np.random.RandomState(seed)

    dynamics_fn = pendulum_dynamics if system == "pendulum" else spring_dynamics
    dof = 1

    all_q, all_p, all_dq, all_dp = [], [], [], []

    for _ in range(n_trajectories):
        # Random initial condition
        q = rng.uniform(-np.pi/2, np.pi/2, dof)
        p = rng.uniform(-1.5, 1.5, dof)

        qs, ps = [q.copy()], [p.copy()]

        # Simulate with RK4
        for _ in range(n_steps):
            dq, dp = dynamics_fn(q, p)
            k1q, k1p = dq, dp
            k2q, k2p = dynamics_fn(q + dt/2*k1q, p + dt/2*k1p)
            k3q, k3p = dynamics_fn(q + dt/2*k2q, p + dt/2*k2p)
            k4q, k4p = dynamics_fn(q + dt*k3q,   p + dt*k3p)
            q = q + dt/6 * (k1q + 2*k2q + 2*k3q + k4q)
            p = p + dt/6 * (k1p + 2*k2p + 2*k3p + k4p)
            qs.append(q.copy())
            ps.append(p.copy())

        qs, ps = np.array(qs), np.array(ps)

        # Centered finite differences for derivatives
        dqdt = (qs[2:] - qs[:-2]) / (2 * dt)
        dpdt = (ps[2:] - ps[:-2]) / (2 * dt)

        all_q.append(qs[1:-1])
        all_p.append(ps[1:-1])
        all_dq.append(dqdt)
        all_dp.append(dpdt)

    q_all  = np.vstack(all_q).astype(np.float32)
    p_all  = np.vstack(all_p).astype(np.float32)
    dq_all = np.vstack(all_dq).astype(np.float32)
    dp_all = np.vstack(all_dp).astype(np.float32)

    if system == "pendulum":
        energy = pendulum_hamiltonian(q_all.ravel(), p_all.ravel())
    else:
        energy = (p_all.ravel()**2 / 2 + 4 * q_all.ravel()**2 / 2)

    print(f"  Samples: {len(q_all)}  |  "
          f"Energy range: [{energy.min():.3f}, {energy.max():.3f}]")
    return {"q": q_all, "p": p_all, "dqdt": dq_all, "dpdt": dp_all,
            "energy": energy}


# ─────────────────────────────────────────────
# HAMILTONIAN NEURAL NETWORK
# ─────────────────────────────────────────────

class HNN(nn.Module):
    """
    Hamiltonian Neural Network.

    Learns a scalar Hamiltonian H_θ(q, p).
    Dynamics are derived analytically via autograd:
        q̇ =  ∂H_θ/∂p
        ṗ = -∂H_θ/∂q

    The symplectic structure (antisymmetric Jacobian) is exact by construction —
    not just approximately enforced through a loss term.
    """

    def __init__(self, dof: int = 1, hidden_dim: int = 200, n_layers: int = 3):
        super().__init__()
        input_dim = 2 * dof  # (q, p) concatenated

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]   # scalar output: H

        self.net = nn.Sequential(*layers)
        self.dof = dof

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Evaluate H(q, p). Shape: (N,) scalar per sample."""
        x = torch.cat([q, p], dim=1)
        return self.net(x)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> tuple:
        """
        Return predicted time derivatives via Hamilton's equations.

        Args:
            q: generalized coordinates (N, dof), requires_grad=True
            p: generalized momenta (N, dof), requires_grad=True

        Returns:
            dqdt: ∂H/∂p  (N, dof)
            dpdt: -∂H/∂q (N, dof)
        """
        H = self.hamiltonian(q, p)  # (N, 1)

        dHdp = autograd.grad(H.sum(), p, create_graph=True)[0]  # ∂H/∂p
        dHdq = autograd.grad(H.sum(), q, create_graph=True)[0]  # ∂H/∂q

        dqdt = dHdp       # canonical equation
        dpdt = -dHdq      # canonical equation (note sign)
        return dqdt, dpdt


class BaselineNODE(nn.Module):
    """
    Vanilla Neural ODE baseline (no Hamiltonian structure).
    Directly predicts (dq/dt, dp/dt) from (q, p).
    Used to compare energy conservation against HNN.
    """

    def __init__(self, dof: int = 1, hidden_dim: int = 200, n_layers: int = 3):
        super().__init__()
        input_dim = 2 * dof
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2 * dof))
        self.net = nn.Sequential(*layers)
        self.dof = dof

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> tuple:
        x   = torch.cat([q, p], dim=1)
        out = self.net(x)
        return out[:, :self.dof], out[:, self.dof:]


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

# Placeholder: tune hidden_dim and n_layers for your system complexity
HNN_CONFIG = {
    "hidden_dim": 200,    # Placeholder: 64–512; larger for complex Hamiltonians
    "n_layers":   3,      # Placeholder: 3–6
    "lr":         1e-3,
    "epochs":     fast_or(500, 20),
    "batch_size": 256,
}


def train_hnn(data: dict, config: dict = None,
               model_type: str = "hnn") -> tuple:
    """
    Train HNN or baseline NODE on phase-space derivative supervision.

    Args:
        data:       output of generate_phase_space_data()
        config:     HNN_CONFIG
        model_type: 'hnn' or 'baseline'
    """
    if config is None:
        config = HNN_CONFIG

    dof = data["q"].shape[1]
    if model_type == "hnn":
        model = HNN(dof=dof, hidden_dim=config["hidden_dim"],
                     n_layers=config["n_layers"]).to(DEVICE)
    else:
        model = BaselineNODE(dof=dof, hidden_dim=config["hidden_dim"],
                              n_layers=config["n_layers"]).to(DEVICE)

    # Prepare tensors
    q_t  = torch.tensor(data["q"],    device=DEVICE).requires_grad_(True)
    p_t  = torch.tensor(data["p"],    device=DEVICE).requires_grad_(True)
    dq_t = torch.tensor(data["dqdt"], device=DEVICE)
    dp_t = torch.tensor(data["dpdt"], device=DEVICE)

    n       = len(q_t)
    opt     = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50,
                                                           factor=0.5)
    history = {"loss": []}

    for epoch in range(1, config["epochs"] + 1):
        idx     = torch.randperm(n)[:config["batch_size"]]
        q_b  = q_t[idx];  p_b  = p_t[idx]
        dq_b = dq_t[idx]; dp_b = dp_t[idx]

        # Need requires_grad for HNN (autograd through H)
        q_b2 = q_b.detach().requires_grad_(True)
        p_b2 = p_b.detach().requires_grad_(True)

        dq_pred, dp_pred = model(q_b2, p_b2)
        loss = nn.MSELoss()(dq_pred, dq_b) + nn.MSELoss()(dp_pred, dp_b)

        opt.zero_grad(); loss.backward(); opt.step()
        sched.step(loss)
        history["loss"].append(loss.item())

        if epoch % 100 == 0:
            print(f"  {model_type.upper()} Epoch {epoch:4d} | loss: {loss.item():.6f}")

    return model, history


# ─────────────────────────────────────────────
# ROLLOUT & ENERGY CONSERVATION
# ─────────────────────────────────────────────

def leapfrog_rollout(model: HNN, q0: np.ndarray, p0: np.ndarray,
                      n_steps: int = 500, dt: float = 0.05) -> tuple:
    """
    Störmer-Verlet (leapfrog) symplectic integrator for long-horizon rollout.

    Leapfrog exactly preserves the symplectic structure of the phase space,
    ensuring bounded energy error over arbitrarily long times.

    Algorithm:
        p_{n+1/2} = p_n - (dt/2) ∂H/∂q(q_n, p_n)
        q_{n+1}   = q_n + dt ∂H/∂p(q_n, p_{n+1/2})
        p_{n+1}   = p_{n+1/2} - (dt/2) ∂H/∂q(q_{n+1}, p_{n+1/2})
    """
    model.eval()
    dof = len(q0)
    qs, ps = [q0.copy()], [p0.copy()]

    q = torch.tensor(q0[None], dtype=torch.float32, device=DEVICE)
    p = torch.tensor(p0[None], dtype=torch.float32, device=DEVICE)

    for _ in range(n_steps):
        q.requires_grad_(True); p.requires_grad_(True)
        H   = model.hamiltonian(q, p)
        # One grad call for both inputs: a second call would find the graph freed.
        dHdq, dHdp = autograd.grad(H.sum(), (q, p))

        p_half = (p - dt/2 * dHdq).detach()

        q_next = (q + dt * dHdp).detach()
        q_next.requires_grad_(True)
        p_half.requires_grad_(True)
        H2     = model.hamiltonian(q_next, p_half)
        dHdq2  = autograd.grad(H2.sum(), q_next)[0]

        p_next = (p_half - dt/2 * dHdq2).detach()

        q = q_next.detach()
        p = p_next

        qs.append(q.cpu().numpy()[0])
        ps.append(p.cpu().numpy()[0])

    return np.array(qs), np.array(ps)


def compute_energy_drift(model_hnn: HNN, model_base: BaselineNODE,
                          q0: np.ndarray, p0: np.ndarray,
                          system: str = "pendulum",
                          n_steps: int = 500, dt: float = 0.05) -> None:
    """
    Compare energy conservation between HNN (leapfrog) and baseline (RK4).
    Demonstrates HNN's structural advantage.
    """
    # HNN rollout (leapfrog)
    qs_hnn, ps_hnn = leapfrog_rollout(model_hnn, q0, p0, n_steps, dt)

    # Baseline rollout (RK4)
    model_base.eval()
    qs_base, ps_base = [q0.copy()], [p0.copy()]
    q_b = torch.tensor(q0[None], dtype=torch.float32, device=DEVICE)
    p_b = torch.tensor(p0[None], dtype=torch.float32, device=DEVICE)
    for _ in range(n_steps):
        with torch.no_grad():
            dq, dp = model_base(q_b, p_b)
        q_b = q_b + dt * dq
        p_b = p_b + dt * dp
        qs_base.append(q_b.cpu().numpy()[0])
        ps_base.append(p_b.cpu().numpy()[0])
    qs_base, ps_base = np.array(qs_base), np.array(ps_base)

    # True energy (pendulum)
    e_true = pendulum_hamiltonian(q0.ravel()[0], p0.ravel()[0])
    fn     = pendulum_hamiltonian if system == "pendulum" else \
             lambda q, p: p**2/2 + 4*q**2/2

    e_hnn  = fn(qs_hnn[:, 0], ps_hnn[:, 0])
    e_base = fn(qs_base[:, 0], ps_base[:, 0])

    t = np.arange(n_steps + 1) * dt
    plt.figure(figsize=(10, 4))
    plt.plot(t, np.abs(e_hnn  - e_true), "steelblue", linewidth=1.5, label="HNN (leapfrog)")
    plt.plot(t, np.abs(e_base - e_true), "tomato",    linewidth=1.5, label="Baseline NODE (RK4)")
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("|ΔE| (log scale)")
    plt.title("Hamiltonian NN — Energy Conservation Comparison")
    plt.legend()
    plt.tight_layout()
    save_figure("hnn_energy.png", dpi=150)

    print(f"  HNN   |ΔE| final: {abs(e_hnn[-1]  - e_true):.2e}")
    print(f"  Baseline |ΔE| final: {abs(e_base[-1] - e_true):.2e}")


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_phase_portrait(qs_true: np.ndarray, ps_true: np.ndarray,
                         qs_hnn: np.ndarray, ps_hnn: np.ndarray) -> None:
    """
    Phase portrait: orbit in (q, p) space.
    HNN should closely follow the true closed orbit (energy conservation).
    """
    plt.figure(figsize=(7, 7))
    plt.plot(qs_true[:, 0], ps_true[:, 0], "k-", linewidth=1.0,
             label="True (simulator)", alpha=0.7)
    plt.plot(qs_hnn[:, 0],  ps_hnn[:, 0],  "steelblue", linewidth=1.5,
             linestyle="--", label="HNN (leapfrog)")
    plt.scatter([qs_true[0, 0]], [ps_true[0, 0]], c="red", s=80, zorder=5,
                label="Initial condition")
    plt.xlabel("q (angle)")
    plt.ylabel("p (momentum)")
    plt.title("Hamiltonian NN — Phase Portrait")
    plt.legend()
    plt.tight_layout()
    save_figure("hnn_phase.png", dpi=150)


def plot_learned_hamiltonian(model: HNN, system: str = "pendulum") -> None:
    """
    2D contour plot of the learned H(q, p) surface.
    Contour lines = energy levels (conserved along trajectories).
    """
    model.eval()
    q_range = np.linspace(-np.pi, np.pi, 80)
    p_range = np.linspace(-3.0, 3.0, 80)
    Q, P    = np.meshgrid(q_range, p_range)

    q_t = torch.tensor(Q.ravel()[:, None], dtype=torch.float32, device=DEVICE)
    p_t = torch.tensor(P.ravel()[:, None], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        H_pred = model.hamiltonian(q_t, p_t).cpu().numpy().reshape(80, 80)

    # True Hamiltonian
    H_true = pendulum_hamiltonian(Q, P) if system == "pendulum" else P**2/2 + 4*Q**2/2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    levels = np.linspace(H_true.min(), H_true.max() * 0.8, 20)

    for ax, H, title in [(ax1, H_true, "True H(q,p)"),
                          (ax2, H_pred, "Learned H_θ(q,p)")]:
        cs = ax.contourf(Q, P, H, levels=levels, cmap="RdYlBu_r", alpha=0.8)
        ax.contour(Q, P, H, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
        plt.colorbar(cs, ax=ax)
        ax.set_xlabel("q"); ax.set_ylabel("p")
        ax.set_title(title)

    plt.suptitle("Hamiltonian Neural Network — Energy Surface")
    plt.tight_layout()
    save_figure("hnn_surface.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"=== Hamiltonian Neural Network ===  (device: {DEVICE})")

    system = "pendulum"  # Placeholder: 'pendulum' or 'spring'
    print(f"\nSystem: {system}")

    # Generate training data
    print("\nGenerating phase-space data...")
    data = generate_phase_space_data(system=system, n_trajectories=fast_or(60, 10),
                                      n_steps=80, dt=0.05)

    # Train HNN
    print("\nTraining HNN...")
    model_hnn, hist_hnn = train_hnn(data, model_type="hnn")

    # Train baseline for comparison
    print("\nTraining baseline NODE...")
    model_base, hist_base = train_hnn(data, model_type="baseline")

    # Rollout and energy comparison
    q0 = np.array([0.5])   # one degree of freedom: shape (dof,)
    p0 = np.array([0.3])

    print("\nLong-horizon rollout (500 steps):")
    qs_hnn, ps_hnn = leapfrog_rollout(model_hnn, q0, p0, n_steps=500, dt=0.05)

    # True trajectory for comparison
    qs_true, ps_true = [q0.copy()], [p0.copy()]
    q, p = q0.copy(), p0.copy()
    for _ in range(500):
        dq, dp = pendulum_dynamics(q, p)
        q = q + 0.05 * dq; p = p + 0.05 * dp
        qs_true.append(q.copy()); ps_true.append(p.copy())
    qs_true = np.array(qs_true); ps_true = np.array(ps_true)

    print("\nEnergy conservation comparison:")
    compute_energy_drift(model_hnn, model_base, q0, p0, system)

    # Visualizations
    plot_phase_portrait(qs_true, ps_true, qs_hnn, ps_hnn)
    plot_learned_hamiltonian(model_hnn, system)


if __name__ == "__main__":
    main()
