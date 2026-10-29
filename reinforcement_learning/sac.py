"""
Soft Actor-Critic (SAC)
=======================

SAC is an off-policy maximum-entropy RL algorithm for continuous action spaces.
It optimizes a trade-off between expected return and policy entropy:
  J(π) = Σ_t E_{(s,a)~ρ_π} [r(s,a) + α H(π(·|s))]

Key ideas:
  1. Maximum-entropy framework: agent maximizes reward AND entropy, encouraging
     exploration and preventing premature convergence to suboptimal deterministics.
  2. Twin Q-networks (Clipped Double-Q): two critics Q_θ1, Q_θ2; use min for
     target to reduce overestimation bias (same as TD3).
  3. Soft policy evaluation:
       V(s) = E_a[Q(s,a) - α log π(a|s)]
  4. Policy improvement: actor maximizes Q - α log π via reparameterization:
       a = tanh(μ(s) + σ(s) ⊙ ε),   ε ~ N(0, I)
       log π(a|s) corrected for tanh squashing
  5. Automatic entropy tuning: α adjusted so that H(π) ≈ target entropy -dim(A).

Architecture:
  Actor:   s → μ(s), log σ(s)    [Gaussian policy, squashed via tanh]
  Critic1: (s, a) → Q_θ1(s, a)  [off-policy TD target]
  Critic2: (s, a) → Q_θ2(s, a)  [twin for clipped double-Q]

Training loop (off-policy):
  1. Collect transitions (s, a, r, s', d) using current π; add to replay buffer
  2. Sample mini-batch from replay buffer
  3. Compute target:  y = r + γ(1-d)[min_i Q_θ'_i(s', ã') - α log π(ã'|s')]
     where ã' ~ π(·|s') (fresh sample)
  4. Update critics: minimize (Q_θi(s,a) - y)²
  5. Update actor: maximize E[Q(s,ã) - α log π(ã|s)]
  6. Soft-update target networks: θ' ← τθ + (1-τ)θ'
  7. Update α: minimize -α (log π + target_entropy)

Fine-tuning recommendations:
  - hidden_dim (#Placeholder: 256): larger networks for more complex environments
  - lr_actor / lr_critic (#Placeholder: 3e-4): standard Adam LR for SAC
  - gamma (#Placeholder: 0.99): discount factor
  - tau (#Placeholder: 0.005): soft target update coefficient (Polyak averaging)
  - alpha_init (#Placeholder: 0.2): initial entropy temperature; auto-tuned when auto_alpha=True
  - buffer_size (#Placeholder: 1_000_000): large replay buffer is key for off-policy stability
  - batch_size (#Placeholder: 256): larger batches stabilize training
  - learning_starts (#Placeholder: 10_000): steps of random exploration before learning

References:
  Haarnoja et al. (2018) — Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
  Haarnoja et al. (2019) — SAC Algorithms and Applications (auto-temperature tuning)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import gymnasium as gym

from deepdraft.core import fast_or, save_figure

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "env_name":         "Pendulum-v1",   # # Placeholder: any continuous-action Gym env
    "hidden_dim":       256,             # # Placeholder:
    "n_hidden":         2,               # # Placeholder: hidden layers in actor/critic
    "lr_actor":         3e-4,            # # Placeholder:
    "lr_critic":        3e-4,            # # Placeholder:
    "lr_alpha":         3e-4,            # # Placeholder:
    "gamma":            0.99,            # # Placeholder:
    "tau":              0.005,           # # Placeholder: soft target update
    "alpha_init":       0.2,             # # Placeholder: initial entropy temp
    "auto_alpha":       True,            # # Placeholder: auto entropy tuning
    "buffer_size":      fast_or(100_000, 5_000),         # # Placeholder:
    "batch_size":       fast_or(256, 64),             # # Placeholder:
    "learning_starts":  fast_or(5_000, 200),           # # Placeholder:
    "total_steps":      fast_or(50_000, 600),          # # Placeholder:
    "eval_every":       fast_or(5_000, 300),           # # Placeholder:
    "eval_episodes":    fast_or(5, 1),               # # Placeholder:
    "log_std_min":      -20,
    "log_std_max":      2,
    "seed":             42,
}

LOG_STD_MIN = CONFIG["log_std_min"]
LOG_STD_MAX = CONFIG["log_std_max"]


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr]      = obs
        self.acts[self.ptr]     = act
        self.rews[self.ptr]     = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, batch_size)
        return {
            "obs":      torch.tensor(self.obs[idx],      device=DEVICE),
            "acts":     torch.tensor(self.acts[idx],     device=DEVICE),
            "rews":     torch.tensor(self.rews[idx],     device=DEVICE),
            "next_obs": torch.tensor(self.next_obs[idx], device=DEVICE),
            "dones":    torch.tensor(self.dones[idx],    device=DEVICE),
        }

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dim: int, n_hidden: int, out_dim: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """
    Squashed Gaussian policy:
      μ(s), log σ(s) = MLP(s)
      ã = tanh(μ + σ ε),  ε ~ N(0,1)
      log π(ã|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))   ← squashing correction
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, n_hidden: int,
                 action_scale: float = 1.0):
        super().__init__()
        self.trunk = _mlp(obs_dim, hidden_dim, n_hidden - 1, hidden_dim)
        self.mean_layer    = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        self.action_scale  = action_scale

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob) with reparameterization."""
        h = F.relu(self.trunk(obs))
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterization
        eps = torch.randn_like(mean)
        x = mean + std * eps                            # pre-squash sample

        # Squash and compute log_prob
        action = torch.tanh(x) * self.action_scale
        log_prob = (
            -0.5 * ((x - mean) / std).pow(2)
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1)
        # Tanh squashing correction: -Σ log(1 - tanh²(x))
        log_prob -= torch.log(self.action_scale * (1 - torch.tanh(x).pow(2)) + 1e-6).sum(dim=-1)

        return action, log_prob

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        h = F.relu(self.trunk(obs_t))
        mean = self.mean_layer(h)
        if deterministic:
            return (torch.tanh(mean) * self.action_scale).squeeze(0).cpu().numpy()
        log_std = self.log_std_layer(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        x = mean + std * torch.randn_like(mean)
        return (torch.tanh(x) * self.action_scale).squeeze(0).cpu().numpy()


class Critic(nn.Module):
    """Q(s, a) → scalar value."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, hidden_dim, n_hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────
# SAC Agent
# ─────────────────────────────────────────────

class SAC:
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float, cfg: dict):
        h = cfg["hidden_dim"]
        n = cfg["n_hidden"]

        self.actor  = GaussianActor(obs_dim, act_dim, h, n, action_scale).to(DEVICE)
        self.critic1 = Critic(obs_dim, act_dim, h, n).to(DEVICE)
        self.critic2 = Critic(obs_dim, act_dim, h, n).to(DEVICE)
        self.critic1_tgt = Critic(obs_dim, act_dim, h, n).to(DEVICE)
        self.critic2_tgt = Critic(obs_dim, act_dim, h, n).to(DEVICE)
        self.critic1_tgt.load_state_dict(self.critic1.state_dict())
        self.critic2_tgt.load_state_dict(self.critic2.state_dict())

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=cfg["lr_actor"])
        self.opt_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg["lr_critic"],
        )

        # Automatic entropy temperature
        self.auto_alpha = cfg["auto_alpha"]
        self.target_entropy = -act_dim   # heuristic: -dim(A)
        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(cfg["alpha_init"]), requires_grad=True, device=DEVICE)
            self.opt_alpha = optim.Adam([self.log_alpha], lr=cfg["lr_alpha"])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg["alpha_init"]

        self.gamma = cfg["gamma"]
        self.tau   = cfg["tau"]

    def update(self, batch: dict) -> dict:
        obs, acts, rews, next_obs, dones = (
            batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["dones"]
        )

        with torch.no_grad():
            next_acts, next_log_pi = self.actor(next_obs)
            q1_tgt = self.critic1_tgt(next_obs, next_acts)
            q2_tgt = self.critic2_tgt(next_obs, next_acts)
            min_q_tgt = torch.min(q1_tgt, q2_tgt) - self.alpha * next_log_pi
            y = rews + self.gamma * (1 - dones) * min_q_tgt

        # Critic update
        q1 = self.critic1(obs, acts)
        q2 = self.critic2(obs, acts)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # Actor update
        new_acts, log_pi = self.actor(obs)
        q1_new = self.critic1(obs, new_acts)
        q2_new = self.critic2(obs, new_acts)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Alpha (temperature) update
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft target update
        for p, pt in zip(self.critic1.parameters(), self.critic1_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.critic2.parameters(), self.critic2_tgt.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "alpha_loss":  alpha_loss.item() if self.auto_alpha else 0.0,
        }


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def evaluate(agent: SAC, env: gym.Env, n_episodes: int) -> float:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            act = agent.actor.act(obs, deterministic=True)
            obs, rew, term, trunc, _ = env.step(act)
            ep_ret += rew
            done = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns))


def train(cfg: dict) -> tuple[SAC, dict]:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    env      = gym.make(cfg["env_name"])
    eval_env = gym.make(cfg["env_name"])
    env.reset(seed=cfg["seed"])

    obs_dim    = env.observation_space.shape[0]
    act_dim    = env.action_space.shape[0]
    act_scale  = float(env.action_space.high[0])

    buffer = ReplayBuffer(cfg["buffer_size"], obs_dim, act_dim)
    agent  = SAC(obs_dim, act_dim, act_scale, cfg)

    obs, _ = env.reset()
    ep_ret = 0.0
    ep_len = 0

    history = {"steps": [], "eval_returns": [], "actor_loss": [], "critic_loss": [], "alpha": []}

    for step in range(1, cfg["total_steps"] + 1):
        # Collect experience
        if step < cfg["learning_starts"]:
            act = env.action_space.sample()
        else:
            act = agent.actor.act(obs)

        next_obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc
        buffer.add(obs, act, rew, next_obs, float(term))
        obs = next_obs
        ep_ret += rew
        ep_len += 1

        if done:
            obs, _ = env.reset()
            ep_ret = 0.0
            ep_len = 0

        # Learn
        if step >= cfg["learning_starts"] and len(buffer) >= cfg["batch_size"]:
            batch = buffer.sample(cfg["batch_size"])
            info  = agent.update(batch)
            history["actor_loss"].append(info["actor_loss"])
            history["critic_loss"].append(info["critic_loss"])
            history["alpha"].append(info["alpha"])

        # Evaluate
        if step % cfg["eval_every"] == 0:
            ret = evaluate(agent, eval_env, cfg["eval_episodes"])
            history["steps"].append(step)
            history["eval_returns"].append(ret)
            print(f"  Step {step:6d} | Eval return: {ret:.1f} | α={agent.alpha:.4f}")

    env.close()
    eval_env.close()
    return agent, history


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_training(history: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("SAC Training", fontsize=12)

    axes[0].plot(history["steps"], history["eval_returns"], marker="o")
    axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Eval Return")
    axes[0].set_title("Evaluation Return"); axes[0].grid(True, alpha=0.3)

    smooth = lambda x, w=200: np.convolve(x, np.ones(w)/w, "valid")
    if len(history["actor_loss"]) > 200:
        axes[1].plot(smooth(history["actor_loss"]),  label="Actor",  color="steelblue")
        axes[1].plot(smooth(history["critic_loss"]), label="Critic", color="darkorange")
    else:
        axes[1].plot(history["actor_loss"],  label="Actor",  color="steelblue")
        axes[1].plot(history["critic_loss"], label="Critic", color="darkorange")
    axes[1].set_xlabel("Update step"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Actor / Critic Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["alpha"], color="forestgreen")
    axes[2].set_xlabel("Update step"); axes[2].set_ylabel("α (temperature)")
    axes[2].set_title("Entropy Temperature α"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("sac_training.png", dpi=150)
    print("Saved: sac_training.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}\n")
    print(f"=== SAC — {CONFIG['env_name']} ===")

    agent, history = train(CONFIG)
    plot_training(history)

    best = max(history["eval_returns"]) if history["eval_returns"] else 0
    print(f"\nBest eval return: {best:.1f}")


if __name__ == "__main__":
    main()
