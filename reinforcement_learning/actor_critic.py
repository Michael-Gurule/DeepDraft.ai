"""
Actor-Critic Methods: A2C / A3C
=================================
The Actor-Critic family bridges value-based (DQN) and policy-gradient (REINFORCE)
methods. The actor learns the policy π(a|s); the critic learns the value function
V(s) or Q(s,a) to reduce gradient variance via advantage estimation.

Advantage:
    A(s,a) = Q(s,a) - V(s)  ← how much better is action a vs. the average?

In practice, estimated as the TD residual (1-step or n-step):
    A_t ≈ r_t + γ V(s_{t+1}) - V(s_t)

This file implements:
    A2C  — Advantage Actor-Critic (synchronous, stable, simplest)
    A3C  — Asynchronous A3C (parallel workers, CPU-efficient)
           Implemented as synchronous parallel rollouts for simplicity.

Relationship to PPO:
    PPO is a constrained A2C — it adds the clipped surrogate objective
    to prevent destructively large policy updates. A2C is simpler but
    less sample-efficient and more sensitive to learning rate.

Relationship to MAPPO:
    MAPPO (Multi-Agent PPO) gives each agent a PPO actor and trains a
    centralized critic that sees the global state. Understanding A2C is a
    prerequisite for both PPO and MAPPO.

Use Cases:
    - Continuous and discrete control
    - Baseline for comparing PPO improvements
    - Environments where on-policy simplicity is preferred
    - Parallel simulation (A3C pattern with multiple environments)

Fine-Tuning Recommendations:
    - n_steps (rollout length): 5–20 for A2C, longer for smoother GAE.
    - learning_rate: 7e-4 (standard A2C starting point). More sensitive than PPO.
    - entropy_coef: 0.01. Increase to 0.05 if policy collapses early.
    - value_coef: 0.5. Balances actor vs. critic loss scale.
    - max_grad_norm: 0.5. Always use — prevents exploding gradients.
    - n_envs (parallel environments): 8–32. More → lower gradient variance,
      faster wall-clock time. Each env contributes independent transitions.
    - gamma: 0.99. Lower (0.95) for episodic tasks with short horizons.
    - RMSprop vs. Adam: A2C was originally designed with RMSprop.
      Adam often works better in practice.
    - Normalize advantages: zero mean, unit variance per rollout.
      Critical for A2C stability (more so than PPO).
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gymnasium as gym
from collections import deque

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# ─────────────────────────────────────────────
# SHARED ACTOR-CRITIC NETWORK
# ─────────────────────────────────────────────

class ActorCriticNetwork(nn.Module):
    """
    Shared-trunk Actor-Critic for A2C.

    The shared trunk learns a joint representation used by both
    actor and critic heads. This enforces representation sharing —
    state features useful for action selection are also useful for
    value estimation.

    Separate trunks (actor_trunk + critic_trunk) can be used when
    the two objectives interfere with each other, which can happen
    in complex environments.

    Args:
        obs_dim:     observation dimension
        action_dim:  action dimension (or number of discrete actions)
        hidden_sizes: shared trunk hidden layer widths
        action_type: 'continuous' (Normal) or 'discrete' (Categorical)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = None,  # Placeholder: [256, 256] for complex tasks
        action_type: str = "discrete",
    ):
        super().__init__()
        hidden_sizes  = hidden_sizes or [64, 64]
        self.action_type = action_type

        # Shared representation trunk
        layers, in_dim = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Actor head
        if action_type == "discrete":
            self.actor = nn.Linear(in_dim, action_dim)
        else:
            self.actor_mean = nn.Linear(in_dim, action_dim)
            self.log_std    = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(in_dim, 1)

        # Orthogonal initialization: improves gradient flow in deep networks
        self.apply(self._init_weights)
        # Scale output layers down for near-uniform initial policy
        if action_type == "discrete":
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor):
        """
        Returns (distribution, value).

        Args:
            obs: (batch, obs_dim)

        Returns:
            dist:  action distribution
            value: (batch, 1) state value
        """
        features = self.trunk(obs)
        value    = self.critic(features)

        if self.action_type == "discrete":
            dist = Categorical(logits=self.actor(features))
        else:
            mean = self.actor_mean(features)
            std  = self.log_std.exp().clamp(min=1e-6).expand_as(mean)
            dist = Normal(mean, std)

        return dist, value


# ─────────────────────────────────────────────
# VECTORIZED ENVIRONMENT WRAPPER
# ─────────────────────────────────────────────

class SyncVectorEnv:
    """
    Synchronous wrapper for running N identical environments in parallel.

    A2C collects one step from each environment simultaneously, giving
    n_envs independent transitions per step — reduces gradient variance
    without asynchronous complexity.

    This is the synchronous equivalent of A3C's async workers.

    Args:
        env_fns: list of callables that each return a new env instance
    """

    def __init__(self, env_fns: list):
        self.envs = [fn() for fn in env_fns]
        self.n    = len(self.envs)

    def reset(self):
        obs = [env.reset()[0] for env in self.envs]
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        results = [env.step(int(a) if not isinstance(a, np.ndarray) else a)
                   for env, a in zip(self.envs, actions)]
        obs, rewards, terms, truncs, infos = zip(*results)

        dones = np.array([t or u for t, u in zip(terms, truncs)], dtype=np.float32)
        obs_arr = np.array(obs, dtype=np.float32)

        # Auto-reset: replace obs for done environments
        for i, done in enumerate(dones):
            if done:
                obs_arr[i] = self.envs[i].reset()[0]

        return obs_arr, np.array(rewards, dtype=np.float32), dones

    def close(self):
        for env in self.envs:
            env.close()


# ─────────────────────────────────────────────
# A2C AGENT
# ─────────────────────────────────────────────

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent.

    Training loop:
        1. Collect n_steps from each of n_envs (= n_envs * n_steps transitions)
        2. Compute n-step returns and advantages
        3. Single gradient update on the entire rollout batch
        4. Repeat

    Key difference from PPO:
        No clipping — a single gradient update per rollout.
        Simpler and faster per update, but less stable and sample-efficient.

    Args:
        obs_dim:       observation dimension
        action_dim:    action dimension
        action_type:   'discrete' or 'continuous'
        n_envs:        number of parallel environments
        n_steps:       rollout steps per environment before update
        lr:            learning rate
        gamma:         discount factor
        gae_lambda:    GAE lambda (0 = TD, 1 = MC)
        value_coef:    weight for value loss
        entropy_coef:  weight for entropy bonus
        max_grad_norm: gradient clipping threshold
        use_gae:       use GAE (True) or n-step returns (False)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "discrete",
        hidden_sizes: list[int] = None,
        n_envs: int = 4,
        n_steps: int = 5,              # Placeholder: 5–20; longer → smoother advantages
        lr: float = 7e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,    # Placeholder: 0.0–0.05
        max_grad_norm: float = 0.5,
        use_gae: bool = True,
    ):
        self.n_envs       = n_envs
        self.n_steps      = n_steps
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm= max_grad_norm
        self.use_gae      = use_gae
        self.action_type  = action_type

        self.network   = ActorCriticNetwork(obs_dim, action_dim,
                                             hidden_sizes or [64, 64],
                                             action_type).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def collect_rollout(self, envs: SyncVectorEnv, obs: np.ndarray) -> tuple:
        """
        Collect n_steps from all environments.

        Returns stacked arrays of shape (n_steps, n_envs, ...).
        The last observation is used to bootstrap the value for incomplete episodes.
        """
        all_obs, all_actions, all_rewards, all_dones = [], [], [], []
        all_values, all_log_probs = [], []

        for _ in range(self.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                dist, value = self.network(obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                if self.action_type == "continuous":
                    log_prob = log_prob.sum(-1)

            action_np = action.cpu().numpy()
            next_obs, rewards, dones = envs.step(action_np)

            all_obs.append(obs)
            all_actions.append(action_np)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_values.append(value.squeeze(-1).cpu().numpy())
            all_log_probs.append(log_prob.cpu().numpy())

            obs = next_obs

        # Bootstrap value for the last state
        obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            _, last_value = self.network(obs_t)
        last_value = last_value.squeeze(-1).cpu().numpy()

        return (np.array(all_obs), np.array(all_actions), np.array(all_rewards),
                np.array(all_dones), np.array(all_values), np.array(all_log_probs),
                last_value, obs)

    def compute_advantages(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: np.ndarray,
    ) -> tuple:
        """
        Compute advantages using GAE-λ or n-step returns.

        GAE: smoother advantage estimates at the cost of computation.
        n-step: simpler, slightly higher variance.

        Args:
            rewards:    (n_steps, n_envs) rewards
            dones:      (n_steps, n_envs) done flags
            values:     (n_steps, n_envs) value estimates
            last_value: (n_envs,) bootstrap value

        Returns:
            advantages (n_steps*n_envs,), returns (n_steps*n_envs,)
        """
        if self.use_gae:
            advantages = np.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(self.n_steps)):
                next_val  = last_value if t == self.n_steps - 1 else values[t + 1]
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                gae   = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[t] = gae
            returns = advantages + values
        else:
            # n-step discounted returns (simpler, original A2C)
            returns = np.zeros_like(rewards)
            running = last_value
            for t in reversed(range(self.n_steps)):
                running  = rewards[t] + self.gamma * running * (1 - dones[t])
                returns[t] = running
            advantages = returns - values

        # Flatten (n_steps, n_envs) → (n_steps * n_envs,)
        advantages = advantages.flatten()
        returns    = returns.flatten()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(
        self,
        obs_arr: np.ndarray,
        actions_arr: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        log_probs_arr: np.ndarray,
    ) -> dict:
        """
        Single A2C gradient update.

        Unlike PPO, no minibatches and no repeated epochs — one forward+backward
        pass over the entire rollout.

        A2C loss:
            L = -E[A_t * log π(a_t|s_t)]  (actor: policy gradient)
              + c1 * E[(V(s_t) - R_t)²]   (critic: value function)
              - c2 * H[π]                  (entropy: exploration)
        """
        obs_t   = torch.tensor(obs_arr.reshape(-1, obs_arr.shape[-1]),
                               dtype=torch.float32).to(DEVICE)
        act_t   = torch.tensor(actions_arr.flatten(),
                               dtype=torch.float32 if self.action_type == "continuous"
                               else torch.int64).to(DEVICE)
        adv_t   = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        ret_t   = torch.tensor(returns,    dtype=torch.float32).to(DEVICE)

        dist, values = self.network(obs_t)
        log_probs    = dist.log_prob(act_t)
        if self.action_type == "continuous":
            log_probs = log_probs.sum(-1)

        # Policy (actor) loss — weighted log probability
        policy_loss = -(log_probs * adv_t).mean()

        # Value (critic) loss — MSE against discounted returns
        value_loss  = ((values.squeeze(-1) - ret_t) ** 2).mean()

        # Entropy bonus — encourages exploration
        entropy = dist.entropy()
        if self.action_type == "continuous":
            entropy = entropy.sum(-1)
        entropy = entropy.mean()

        total_loss = (policy_loss
                      + self.value_coef * value_loss
                      - self.entropy_coef * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "entropy":     entropy.item(),
        }


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

A2C_CONFIG = {
    "env_id":          "CartPole-v1",  # Placeholder: any Gymnasium env
    "n_envs":          8,              # Placeholder: 4–32 parallel envs
    "n_steps":         5,              # Placeholder: 5–20
    "hidden_sizes":    [64, 64],
    "lr":              7e-4,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "value_coef":      0.5,
    "entropy_coef":    0.01,
    "max_grad_norm":   0.5,
    "use_gae":         True,
    "total_timesteps": 200_000,
}


def train(config: dict = None) -> tuple:
    """
    A2C training loop with vectorized environments.

    Args:
        config: hyperparameter dict (uses A2C_CONFIG if None)

    Returns:
        agent, episode_returns
    """
    cfg    = config or A2C_CONFIG
    env_id = cfg["env_id"]

    # Create vectorized environments
    env_fns = [lambda: gym.make(env_id) for _ in range(cfg["n_envs"])]
    envs    = SyncVectorEnv(env_fns)

    # Infer dimensions from a single env
    probe = gym.make(env_id)
    obs_dim    = probe.observation_space.shape[0]
    action_dim = probe.action_space.n if hasattr(probe.action_space, "n") else probe.action_space.shape[0]
    action_type= "discrete" if hasattr(probe.action_space, "n") else "continuous"
    probe.close()

    agent = A2CAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_type=action_type,
        hidden_sizes=cfg["hidden_sizes"], n_envs=cfg["n_envs"],
        n_steps=cfg["n_steps"], lr=cfg["lr"], gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"], value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"], max_grad_norm=cfg["max_grad_norm"],
        use_gae=cfg["use_gae"],
    )

    obs             = envs.reset()
    episode_returns = []
    ep_buffers      = [0.0] * cfg["n_envs"]
    recent_returns  = deque(maxlen=20)
    total_steps     = 0
    update_count    = 0

    print(f"Training A2C on {env_id} | {cfg['n_envs']} envs × {cfg['n_steps']} steps | "
          f"obs={obs_dim} act={action_dim}")

    while total_steps < cfg["total_timesteps"]:
        (obs_arr, act_arr, rew_arr, done_arr,
         val_arr, lp_arr, last_val, obs) = agent.collect_rollout(envs, obs)

        # Track episode returns (approximate — envs auto-reset)
        for t in range(cfg["n_steps"]):
            for e in range(cfg["n_envs"]):
                ep_buffers[e] += rew_arr[t, e]
                if done_arr[t, e]:
                    episode_returns.append(ep_buffers[e])
                    recent_returns.append(ep_buffers[e])
                    ep_buffers[e] = 0.0

        advantages, returns = agent.compute_advantages(rew_arr, done_arr, val_arr, last_val)
        metrics = agent.update(obs_arr, act_arr, advantages, returns, lp_arr)

        total_steps  += cfg["n_envs"] * cfg["n_steps"]
        update_count += 1

        if update_count % 50 == 0:
            mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
            print(f"  Steps: {total_steps:7d} | MeanReturn: {mean_ret:7.2f} | "
                  f"PolicyLoss: {metrics['policy_loss']:7.4f} | "
                  f"Entropy: {metrics['entropy']:.4f}")

    envs.close()
    return agent, episode_returns


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training(episode_returns: list, window: int = 20) -> None:
    """Plot episode returns with smoothing."""
    returns  = np.array(episode_returns)
    if len(returns) < window:
        plt.figure(figsize=(10, 4))
        plt.plot(returns)
        plt.title("A2C — Episode Returns")
        plt.tight_layout()
        plt.savefig("a2c_training.png", dpi=150)
        plt.show()
        return
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 4))
    plt.plot(returns,  alpha=0.3, color="green", label="Episode Return")
    plt.plot(range(window - 1, len(returns)), smoothed, color="green",
             linewidth=2, label=f"Smoothed ({window}-ep)")
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("A2C — Training Returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig("a2c_training.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Advantage Actor-Critic (A2C) ===")
    agent, returns = train()
    plot_training(returns)


if __name__ == "__main__":
    main()
