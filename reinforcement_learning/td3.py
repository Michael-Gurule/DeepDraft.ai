"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
======================================================

TD3 extends DDPG with three key tricks that fix overestimation bias and
instability in the original actor-critic framework:

  1. Twin Critics (Clipped Double-Q):
       Use min(Q_θ1, Q_θ2) as the TD target to reduce overestimation.
       y = r + γ(1-d) min_i Q_θ'_i(s', π'(s') + ε̃)

  2. Delayed Policy Updates:
       Update actor and target networks every d steps (d=2 typical).
       Policy updates less frequently than value updates → more stable.

  3. Target Policy Smoothing:
       Add clipped Gaussian noise to target actions:
         ã = clip(π'(s') + clip(ε, -c, c), a_low, a_high)
         ε ~ N(0, σ_tilde)
       Prevents Q from exploiting narrow action-value peaks.

Comparison with SAC:
  - TD3 is deterministic; SAC is stochastic (entropy-based)
  - TD3 requires manual ε-greedy exploration noise; SAC explores via policy entropy
  - TD3 often more sample-efficient on easy envs; SAC more stable on hard ones
  - Both use off-policy replay buffers and twin critics

Architecture:
  Actor:   s → a = tanh(π_θ(s)) · action_scale    [deterministic]
  Critic1: (s, a) → Q_θ1(s, a)
  Critic2: (s, a) → Q_θ2(s, a)

Fine-tuning recommendations:
  - policy_noise (#Placeholder: 0.2): std of exploration noise during collection
  - noise_clip (#Placeholder: 0.5): clip target policy smoothing noise
  - policy_delay (#Placeholder: 2): actor/target update frequency
  - tau (#Placeholder: 0.005): Polyak coefficient for soft target updates
  - expl_noise (#Placeholder: 0.1): Gaussian noise added to deterministic actions
  - buffer_size (#Placeholder: 1_000_000): large replay for off-policy stability

References:
  Fujimoto et al. (2018) — Addressing Function Approximation Error in Actor-Critic
  Silver et al. (2014) — Deterministic Policy Gradient Algorithms (DDPG precursor)
  Lillicrap et al. (2016) — Continuous Control with Deep RL (DDPG)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CONFIG = {
    "env_name":       "Pendulum-v1",  # # Placeholder: any continuous-action Gym env
    "hidden_dim":     256,            # # Placeholder:
    "n_hidden":       2,              # # Placeholder:
    "lr_actor":       3e-4,           # # Placeholder:
    "lr_critic":      3e-4,           # # Placeholder:
    "gamma":          0.99,           # # Placeholder:
    "tau":            0.005,          # # Placeholder: Polyak coefficient
    "policy_noise":   0.2,            # # Placeholder: target policy smoothing σ
    "noise_clip":     0.5,            # # Placeholder: clip for target policy noise
    "expl_noise":     0.1,            # # Placeholder: exploration noise σ
    "policy_delay":   2,              # # Placeholder: actor update frequency
    "buffer_size":    100_000,        # # Placeholder:
    "batch_size":     256,            # # Placeholder:
    "learning_starts": 5_000,         # # Placeholder:
    "total_steps":    50_000,         # # Placeholder:
    "eval_every":     5_000,          # # Placeholder:
    "eval_episodes":  5,              # # Placeholder:
    "seed":           42,
}


# ─────────────────────────────────────────────
# Replay Buffer (shared pattern with SAC)
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts     = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews     = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.capacity = capacity

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr]      = obs
        self.acts[self.ptr]     = act
        self.rews[self.ptr]     = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, batch_size)
        return {k: torch.tensor(v[idx], device=DEVICE) for k, v in [
            ("obs", self.obs), ("acts", self.acts), ("rews", self.rews),
            ("next_obs", self.next_obs), ("dones", self.dones),
        ]}

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


class DeterministicActor(nn.Module):
    """Deterministic policy: s → a = tanh(MLP(s)) * action_scale."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, n_hidden: int,
                 action_scale: float = 1.0):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, n_hidden, act_dim)
        self.action_scale = action_scale

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.action_scale

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.0,
            act_low: float = -1.0, act_high: float = 1.0) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        act = self.forward(obs_t).squeeze(0).cpu().numpy()
        if noise_std > 0:
            act = act + np.random.normal(0, noise_std, act.shape)
            act = np.clip(act, act_low, act_high)
        return act


class Critic(nn.Module):
    """Q(s, a) → scalar."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, hidden_dim, n_hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────
# TD3 Agent
# ─────────────────────────────────────────────

class TD3:
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float,
                 act_low: float, act_high: float, cfg: dict):
        h = cfg["hidden_dim"]
        n = cfg["n_hidden"]
        self.action_scale = action_scale
        self.act_low  = act_low
        self.act_high = act_high

        self.actor     = DeterministicActor(obs_dim, act_dim, h, n, action_scale).to(DEVICE)
        self.actor_tgt = DeterministicActor(obs_dim, act_dim, h, n, action_scale).to(DEVICE)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic1     = Critic(obs_dim, act_dim, h, n).to(DEVICE)
        self.critic2     = Critic(obs_dim, act_dim, h, n).to(DEVICE)
        self.critic1_tgt = Critic(obs_dim, act_dim, h, n).to(DEVICE)
        self.critic2_tgt = Critic(obs_dim, act_dim, h, n).to(DEVICE)
        self.critic1_tgt.load_state_dict(self.critic1.state_dict())
        self.critic2_tgt.load_state_dict(self.critic2.state_dict())

        self.opt_actor  = optim.Adam(self.actor.parameters(), lr=cfg["lr_actor"])
        self.opt_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg["lr_critic"],
        )

        self.gamma        = cfg["gamma"]
        self.tau          = cfg["tau"]
        self.policy_noise = cfg["policy_noise"] * action_scale
        self.noise_clip   = cfg["noise_clip"]   * action_scale
        self.policy_delay = cfg["policy_delay"]
        self._update_count = 0

    def update(self, batch: dict) -> dict:
        self._update_count += 1
        obs, acts, rews, next_obs, dones = (
            batch["obs"], batch["acts"], batch["rews"], batch["next_obs"], batch["dones"]
        )

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(acts) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_acts = (self.actor_tgt(next_obs) + noise).clamp(
                self.act_low, self.act_high
            )
            # Clipped double-Q target
            q1_tgt = self.critic1_tgt(next_obs, next_acts)
            q2_tgt = self.critic2_tgt(next_obs, next_acts)
            y = rews + self.gamma * (1 - dones) * torch.min(q1_tgt, q2_tgt)

        # Critic update (every step)
        q1 = self.critic1(obs, acts)
        q2 = self.critic2(obs, acts)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        actor_loss = torch.tensor(0.0)
        if self._update_count % self.policy_delay == 0:
            # Delayed actor update
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            # Soft target updates
            for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.critic1.parameters(), self.critic1_tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.critic2.parameters(), self.critic2_tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def evaluate(agent: TD3, env: gym.Env, n_episodes: int) -> float:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            act = agent.actor.act(obs, noise_std=0.0)
            obs, rew, term, trunc, _ = env.step(act)
            ep_ret += rew
            done = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns))


def train(cfg: dict) -> tuple[TD3, dict]:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    env      = gym.make(cfg["env_name"])
    eval_env = gym.make(cfg["env_name"])
    env.reset(seed=cfg["seed"])

    obs_dim    = env.observation_space.shape[0]
    act_dim    = env.action_space.shape[0]
    act_scale  = float(env.action_space.high[0])
    act_low    = float(env.action_space.low[0])
    act_high   = float(env.action_space.high[0])

    buffer = ReplayBuffer(cfg["buffer_size"], obs_dim, act_dim)
    agent  = TD3(obs_dim, act_dim, act_scale, act_low, act_high, cfg)

    obs, _ = env.reset()
    history = {"steps": [], "eval_returns": [], "actor_loss": [], "critic_loss": []}

    for step in range(1, cfg["total_steps"] + 1):
        if step < cfg["learning_starts"]:
            act = env.action_space.sample()
        else:
            act = agent.actor.act(obs, noise_std=cfg["expl_noise"] * act_scale,
                                   act_low=act_low, act_high=act_high)

        next_obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc
        buffer.add(obs, act, rew, next_obs, float(term))
        obs = next_obs if not done else env.reset()[0]

        if step >= cfg["learning_starts"] and len(buffer) >= cfg["batch_size"]:
            info = agent.update(buffer.sample(cfg["batch_size"]))
            history["actor_loss"].append(info["actor_loss"])
            history["critic_loss"].append(info["critic_loss"])

        if step % cfg["eval_every"] == 0:
            ret = evaluate(agent, eval_env, cfg["eval_episodes"])
            history["steps"].append(step)
            history["eval_returns"].append(ret)
            print(f"  Step {step:6d} | Eval return: {ret:.1f}")

    env.close()
    eval_env.close()
    return agent, history


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_training(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("TD3 Training", fontsize=12)

    axes[0].plot(history["steps"], history["eval_returns"], marker="o", color="steelblue")
    axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Eval Return")
    axes[0].set_title("Evaluation Return"); axes[0].grid(True, alpha=0.3)

    smooth = lambda x, w=200: np.convolve(x, np.ones(w)/w, "valid")
    if len(history["critic_loss"]) > 200:
        axes[1].plot(smooth(history["critic_loss"]), label="Critic", color="darkorange")
        actor_nz = [v for v in history["actor_loss"] if v != 0.0]
        if actor_nz:
            axes[1].plot(smooth(actor_nz[:len(history["critic_loss"])]),
                         label="Actor", color="steelblue")
    axes[1].set_xlabel("Update step"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Actor / Critic Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("td3_training.png", dpi=150)
    plt.show()
    print("Saved: td3_training.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}\n")
    print(f"=== TD3 — {CONFIG['env_name']} ===")

    agent, history = train(CONFIG)
    plot_training(history)

    best = max(history["eval_returns"]) if history["eval_returns"] else 0
    print(f"\nBest eval return: {best:.1f}")


if __name__ == "__main__":
    main()
