"""
Proximal Policy Optimization (PPO)
====================================
The dominant on-policy policy gradient algorithm. Improves on vanilla policy
gradient (REINFORCE) by clipping the probability ratio between the new and old
policy, preventing destructively large updates while still making progress.

PPO trained OpenAI Five (Dota 2) and is the base of multi-agent extensions
such as MAPPO and of RLHF fine-tuning for language models.

Core idea — the clipped surrogate objective:
    L_CLIP(θ) = E[ min(r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t) ]

    Where:
        r_t(θ) = π_θ(a|s) / π_θ_old(a|s)   — probability ratio
        A_t    — advantage estimate (how much better was this action vs. average)
        ε      — clip range (typically 0.1–0.2); limits how far the policy can move

Full objective (three components):
    L(θ) = L_CLIP - c1 * L_VF + c2 * H[π_θ]

    Where:
        L_VF  = (V(s) - V_target)²          — value function loss (critic)
        H     = -E[log π(a|s)]               — policy entropy (encourages exploration)
        c1, c2 — coefficient hyperparameters

Advantage estimation — GAE-λ:
    A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)      — TD error
    λ controls bias-variance trade-off (λ=0 → TD, λ=1 → Monte Carlo)

Use Cases:
    - Continuous control (robotics, locomotion, flight)
    - Discrete action games (Atari, strategy games)
    - Multi-agent coordination (MAPPO extends this)
    - RLHF / fine-tuning LLMs with human feedback
    - Marketing budget allocation as sequential decision problem

Fine-Tuning Recommendations:
    - clip_range (ε): 0.1–0.2. Smaller → more conservative updates.
      Decrease if training is unstable; increase if learning is too slow.
    - learning_rate: 3e-4 is standard. Use LR annealing for long runs.
    - n_epochs: 10 update epochs per rollout. More → better data efficiency
      but risk overfitting the rollout. Typically 4–10.
    - gae_lambda: 0.95 (standard). 0.9 → lower variance, more bias.
    - gamma: 0.99 (standard). Lower for short-horizon tasks (0.9–0.95).
    - entropy_coef: 0.01. Increase (0.05–0.1) if policy collapses to
      deterministic too early. Decrease if entropy stays high.
    - value_coef: 0.5. Scales relative importance of critic loss.
    - max_grad_norm: 0.5. Essential — PPO training is unstable without it.
    - batch_size / n_minibatches: more minibatches → more gradient steps
      per rollout, better sample efficiency. Typical: 4–8 minibatches.
    - rollout_length: 2048 steps for continuous control; 128 for Atari.
      Longer → better GAE estimates, more memory.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from collections import deque
import gymnasium as gym

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# ─────────────────────────────────────────────
# ACTOR-CRITIC NETWORK
# ─────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic network for PPO.

    Shared trunk: both actor and critic learn a joint representation,
    which is more parameter-efficient and often trains more stably
    than fully separate networks for simple environments.

    For complex tasks (e.g., image-based), use separate networks or
    add convolutional feature extractors before the shared trunk.

    Architecture:
        obs → shared_trunk → actor_head → action distribution
                           → critic_head → state value V(s)

    Placeholder: swap 'continuous' for 'discrete' for environments like
    CartPole, LunarLander (discrete), or custom discrete action spaces.

    Args:
        obs_dim:     observation space dimension
        action_dim:  action space dimension
        hidden_sizes: list of hidden layer widths for shared trunk
        action_type: 'continuous' (Normal dist) or 'discrete' (Categorical)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = None,   # Placeholder: [64, 64] or [256, 256]
        action_type: str = "continuous",
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 64]
        self.action_type = action_type

        # ── Shared feature trunk ──────────────────────────────────
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            # Tanh is preferred over ReLU for RL policy networks:
            # bounded activations improve stability with policy gradient losses.
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # ── Actor head ────────────────────────────────────────────
        if action_type == "continuous":
            self.actor_mean = nn.Linear(in_dim, action_dim)
            # Log std as a learnable parameter (not input-dependent)
            # Placeholder: make log_std input-dependent for more expressive policies
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor_logits = nn.Linear(in_dim, action_dim)

        # ── Critic head ───────────────────────────────────────────
        self.critic = nn.Linear(in_dim, 1)

        # Initialize output layers with small weights for stable early training
        self._init_weights()

    def _init_weights(self):
        """Small output layer weights → near-uniform initial policy."""
        for layer in [self.critic]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        if self.action_type == "continuous":
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.zeros_(self.actor_mean.bias)
        else:
            nn.init.orthogonal_(self.actor_logits.weight, gain=0.01)
            nn.init.zeros_(self.actor_logits.bias)

    def forward(self, obs: torch.Tensor):
        """
        Compute action distribution and value estimate.

        Args:
            obs: (batch, obs_dim) observation tensor

        Returns:
            dist:  action distribution (Normal or Categorical)
            value: (batch, 1) state value estimate
        """
        features = self.trunk(obs)
        value     = self.critic(features)

        if self.action_type == "continuous":
            mean = self.actor_mean(features)
            std  = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
        else:
            logits = self.actor_logits(features)
            dist   = Categorical(logits=logits)

        return dist, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get only the value estimate (used during rollout collection)."""
        return self.critic(self.trunk(obs))


# ─────────────────────────────────────────────
# ROLLOUT BUFFER
# ─────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores a single rollout (fixed-length trajectory) for PPO updates.

    On-policy: buffer is completely cleared after each policy update.
    This is a key distinction from off-policy methods (DQN, SAC) which
    reuse experience via a replay buffer.

    Args:
        rollout_length: number of steps to collect before updating
        obs_dim:        observation dimension
        action_dim:     action dimension
        action_type:    'continuous' or 'discrete'
    """

    def __init__(
        self,
        rollout_length: int,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
    ):
        self.rollout_length = rollout_length
        self.action_type    = action_type
        self.reset()

    def reset(self):
        self.obs      = []
        self.actions  = []
        self.rewards  = []
        self.dones    = []
        self.values   = []
        self.log_probs = []
        self.ptr      = 0

    def store(self, obs, action, reward, done, value, log_prob):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.rollout_length

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute GAE-λ advantages and discounted returns.

        GAE (Generalized Advantage Estimation) balances:
            - Low variance (λ→0): uses bootstrapped TD target, one-step lookahead
            - Low bias  (λ→1):   uses full Monte Carlo return

        Returns are used as targets for the value function (critic).
        Advantages drive the policy gradient — normalized to zero mean,
        unit variance for training stability.

        Args:
            last_value:  V(s_{T+1}) bootstrap value for incomplete trajectories
            gamma:       discount factor
            gae_lambda:  GAE lambda parameter
        """
        rewards    = np.array(self.rewards)
        dones      = np.array(self.dones, dtype=np.float32)
        values     = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards)
        gae        = 0.0

        # Iterate backwards through the trajectory
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae   = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns    = advantages + np.array(self.values)

        # Normalize advantages: zero mean, unit variance
        # Reduces gradient variance; standard practice in PPO
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_tensors(self):
        """Convert stored numpy arrays to tensors for training."""
        obs       = torch.tensor(np.array(self.obs),       dtype=torch.float32)
        actions   = torch.tensor(np.array(self.actions),   dtype=torch.float32 if self.action_type == "continuous" else torch.int64)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32)
        returns   = torch.tensor(self.returns,              dtype=torch.float32)
        advantages= torch.tensor(self.advantages,           dtype=torch.float32)
        return obs, actions, log_probs, returns, advantages


# ─────────────────────────────────────────────
# PPO AGENT
# ─────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Implements the full PPO training loop:
        1. Collect rollout (interact with environment)
        2. Compute GAE advantages
        3. Update policy for n_epochs via clipped surrogate + value loss + entropy

    Args:
        obs_dim:        observation space size
        action_dim:     action space size
        action_type:    'continuous' or 'discrete'
        hidden_sizes:   actor-critic network hidden layer widths
        lr:             learning rate (same for actor and critic)
        gamma:          discount factor
        gae_lambda:     GAE lambda
        clip_range:     PPO clipping parameter ε
        n_epochs:       gradient update epochs per rollout
        n_minibatches:  number of minibatches per epoch
        value_coef:     weight for value function loss
        entropy_coef:   weight for entropy bonus
        max_grad_norm:  gradient clipping threshold
        rollout_length: steps per rollout collection
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        hidden_sizes: list[int] = None,  # Placeholder: [64,64] standard; [256,256] for complex tasks
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,         # Placeholder: 0.1 for fine-tuning; 0.2 standard
        n_epochs: int = 10,              # Placeholder: 4–10; more = better data efficiency
        n_minibatches: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,      # Placeholder: 0.0–0.05; higher = more exploration
        max_grad_norm: float = 0.5,
        rollout_length: int = 2048,      # Placeholder: 128 (Atari), 2048 (continuous control)
    ):
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_range    = clip_range
        self.n_epochs      = n_epochs
        self.n_minibatches = n_minibatches
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.action_type   = action_type

        self.network = ActorCritic(
            obs_dim, action_dim,
            hidden_sizes=hidden_sizes or [64, 64],
            action_type=action_type,
        ).to(DEVICE)

        # Single optimizer for both actor and critic (shared trunk)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        self.buffer = RolloutBuffer(rollout_length, obs_dim, action_dim, action_type)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray):
        """
        Sample action from current policy.

        Used during rollout collection. Returns action, log_prob, value
        needed to fill the rollout buffer.

        Args:
            obs: (obs_dim,) single observation

        Returns:
            action:   sampled action (numpy)
            log_prob: log probability of that action
            value:    state value estimate
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dist, value = self.network(obs_t)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        if self.action_type == "continuous":
            log_prob = log_prob.sum(-1)  # sum over action dimensions
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def update(self) -> dict:
        """
        Perform PPO update on the collected rollout.

        For each of n_epochs:
            - Shuffle the rollout data
            - Split into n_minibatches
            - For each minibatch:
                - Compute new log probs and values
                - Compute clipped surrogate loss
                - Compute value loss (MSE or clipped)
                - Compute entropy bonus
                - Backpropagate and clip gradients

        Returns:
            dict of training metrics (losses, clip fraction, approx KL)
        """
        last_obs = torch.tensor(self.buffer.obs[-1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        last_val = self.network.get_value(last_obs).item()
        self.buffer.compute_returns_and_advantages(last_val, self.gamma, self.gae_lambda)

        obs, actions, old_log_probs, returns, advantages = self.buffer.get_tensors()
        obs          = obs.to(DEVICE)
        actions      = actions.to(DEVICE)
        old_log_probs= old_log_probs.to(DEVICE)
        returns      = returns.to(DEVICE)
        advantages   = advantages.to(DEVICE)

        n            = len(obs)
        minibatch_sz = max(1, n // self.n_minibatches)

        metrics = {"policy_loss": [], "value_loss": [], "entropy": [],
                   "clip_frac": [], "approx_kl": []}

        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, minibatch_sz):
                mb_idx = idx[start:start + minibatch_sz]

                mb_obs    = obs[mb_idx]
                mb_act    = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_ret    = returns[mb_idx]
                mb_adv    = advantages[mb_idx]

                dist, values = self.network(mb_obs)
                new_log_probs = dist.log_prob(mb_act)
                if self.action_type == "continuous":
                    new_log_probs = new_log_probs.sum(-1)

                # ── Clipped surrogate (policy) loss ───────────────
                ratio      = (new_log_probs - mb_old_lp).exp()
                surr1      = ratio * mb_adv
                surr2      = ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value function loss ───────────────────────────
                # Clipped value loss prevents large updates to the critic
                # that would undermine the advantage estimates.
                values = values.squeeze(-1)
                value_loss = ((values - mb_ret) ** 2).mean()

                # ── Entropy bonus ─────────────────────────────────
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

                # Diagnostics
                clip_frac  = ((ratio - 1).abs() > self.clip_range).float().mean().item()
                approx_kl  = (mb_old_lp - new_log_probs).mean().item()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["clip_frac"].append(clip_frac)
                metrics["approx_kl"].append(approx_kl)

        self.buffer.reset()
        return {k: np.mean(v) for k, v in metrics.items()}


# ─────────────────────────────────────────────
# PPO WITH ADAPTIVE KL PENALTY (alternative to clipping)
# ─────────────────────────────────────────────

class PPOKLAgent(PPOAgent):
    """
    PPO with adaptive KL-penalty objective (Schulman et al., 2017 §4).

    Instead of clipping the probability ratio, this variant penalises
    deviations from the old policy via a KL-divergence term whose
    coefficient β is adapted automatically after every update:

        L_KL(θ) = E[r_t(θ) * A_t] - β * KL[π_old || π_θ]

    Adaptation rule (simple heuristic from the paper):
        if KL > 1.5 * target_kl:   β ← β * 2     (penalty too weak, policy moved too far)
        if KL < target_kl / 1.5:   β ← β / 2     (penalty too strong, policy barely moved)

    When to prefer KL penalty over clipping:
        - Clipping: simpler, more widely used, default choice.
        - KL penalty: cleaner objective; β gives interpretable "step size" signal.
          Useful when you want explicit KL control (e.g., RLHF fine-tuning of LLMs).

    The two approaches often converge to similar performance; the original
    PPO paper found clipping slightly better on most continuous-control tasks.

    Args:
        target_kl:   desired KL divergence per update (Placeholder: 0.01–0.05)
        beta_init:   initial KL coefficient (Placeholder: 1.0–3.0)
        beta_min:    minimum allowed β to prevent it collapsing to zero
        beta_max:    maximum allowed β
    """

    def __init__(
        self,
        *args,
        target_kl: float = 0.01,   # Placeholder: 0.01–0.05; tighter = more conservative
        beta_init: float = 1.0,    # Placeholder: start at 1.0; adapts automatically
        beta_min: float = 1e-4,
        beta_max: float = 1e4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_kl = target_kl
        self.beta      = beta_init
        self.beta_min  = beta_min
        self.beta_max  = beta_max

    def update(self) -> dict:
        """
        PPO-KL update: maximise E[r*A] subject to adaptive KL penalty.

        Overrides the clipped surrogate update in the parent PPOAgent.
        The KL coefficient β is updated once per rollout based on the
        mean KL observed across all minibatch updates.
        """
        last_obs = torch.tensor(self.buffer.obs[-1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        last_val = self.network.get_value(last_obs).item()
        self.buffer.compute_returns_and_advantages(last_val, self.gamma, self.gae_lambda)

        obs, actions, old_log_probs, returns, advantages = self.buffer.get_tensors()
        obs           = obs.to(DEVICE)
        actions       = actions.to(DEVICE)
        old_log_probs = old_log_probs.to(DEVICE)
        returns       = returns.to(DEVICE)
        advantages    = advantages.to(DEVICE)

        n            = len(obs)
        minibatch_sz = max(1, n // self.n_minibatches)

        metrics = {"policy_loss": [], "value_loss": [], "entropy": [],
                   "kl": [], "beta": []}

        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, minibatch_sz):
                mb_idx = idx[start:start + minibatch_sz]

                mb_obs    = obs[mb_idx]
                mb_act    = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_ret    = returns[mb_idx]
                mb_adv    = advantages[mb_idx]

                dist, values = self.network(mb_obs)
                new_log_probs = dist.log_prob(mb_act)
                if self.action_type == "continuous":
                    new_log_probs = new_log_probs.sum(-1)

                ratio = (new_log_probs - mb_old_lp).exp()

                # Surrogate objective (no clipping — KL handles the constraint)
                policy_loss = -(ratio * mb_adv).mean()

                # Approximate KL: E[log(π_old/π_new)] = E[old_lp - new_lp]
                approx_kl = (mb_old_lp - new_log_probs).mean()

                # KL penalty
                kl_loss = self.beta * approx_kl

                # Value loss
                values = values.squeeze(-1)
                value_loss = ((values - mb_ret) ** 2).mean()

                # Entropy bonus
                entropy = dist.entropy()
                if self.action_type == "continuous":
                    entropy = entropy.sum(-1)
                entropy = entropy.mean()

                total_loss = (policy_loss
                              + kl_loss
                              + self.value_coef * value_loss
                              - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["kl"].append(approx_kl.item())
                metrics["beta"].append(self.beta)

        # Adapt β based on mean KL across all updates this rollout
        mean_kl = np.mean(metrics["kl"])
        if mean_kl > 1.5 * self.target_kl:
            self.beta = min(self.beta * 2.0, self.beta_max)
        elif mean_kl < self.target_kl / 1.5:
            self.beta = max(self.beta / 2.0, self.beta_min)

        self.buffer.reset()
        return {k: np.mean(v) for k, v in metrics.items()}


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

PPO_CONFIG = {
    "env_id":         "Pendulum-v1",   # Placeholder: any Gymnasium env
    "hidden_sizes":   [64, 64],
    "lr":             3e-4,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "clip_range":     0.2,
    "n_epochs":       10,
    "n_minibatches":  4,
    "value_coef":     0.5,
    "entropy_coef":   0.0,             # Pendulum is continuous; no entropy needed
    "max_grad_norm":  0.5,
    "rollout_length": 2048,
    "total_timesteps": 200_000,
}


def train(config: dict = None) -> tuple:
    """
    Full PPO training loop.

    Interleaves rollout collection and policy updates until total_timesteps
    have been collected. Logs episodic returns for monitoring.

    Fine-tuning tip: monitor these diagnostics:
        approx_kl  > 0.02  → policy changing too fast; lower lr or clip_range
        clip_frac  > 0.2   → policy hitting clip boundary often; expected early
        entropy    → 0     → policy collapsed to deterministic; increase entropy_coef
        value_loss → ∞     → critic diverging; lower value_coef or lr

    Args:
        config: hyperparameter dict (uses PPO_CONFIG if None)

    Returns:
        agent, episode_returns list
    """
    cfg = config or PPO_CONFIG
    env = gym.make(cfg["env_id"])

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
    action_type = "continuous" if hasattr(env.action_space, "shape") else "discrete"

    agent = PPOAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_type=action_type,
        hidden_sizes=cfg["hidden_sizes"], lr=cfg["lr"], gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
        n_epochs=cfg["n_epochs"], n_minibatches=cfg["n_minibatches"],
        value_coef=cfg["value_coef"], entropy_coef=cfg["entropy_coef"],
        max_grad_norm=cfg["max_grad_norm"], rollout_length=cfg["rollout_length"],
    )

    obs, _ = env.reset()
    episode_return  = 0.0
    episode_returns = []
    recent_returns  = deque(maxlen=20)
    total_steps     = 0
    update_count    = 0

    print(f"Training PPO on {cfg['env_id']} | obs={obs_dim} act={action_dim} ({action_type})")

    while total_steps < cfg["total_timesteps"]:
        # ── Collect rollout ──────────────────────────────────────
        while not agent.buffer.is_full():
            action, log_prob, value = agent.get_action(obs)

            # Clip action to environment bounds (continuous envs)
            if action_type == "continuous":
                clipped = np.clip(action, env.action_space.low, env.action_space.high)
            else:
                clipped = int(action)

            next_obs, reward, terminated, truncated, _ = env.step(clipped)
            done = terminated or truncated

            agent.buffer.store(obs, action, reward, float(done), value, log_prob)
            obs            = next_obs
            episode_return += reward
            total_steps    += 1

            if done:
                episode_returns.append(episode_return)
                recent_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()

        # ── Update policy ────────────────────────────────────────
        metrics = agent.update()
        update_count += 1

        if update_count % 5 == 0:
            mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
            print(f"  Steps: {total_steps:7d} | "
                  f"MeanReturn: {mean_ret:8.2f} | "
                  f"PolicyLoss: {metrics['policy_loss']:7.4f} | "
                  f"ValueLoss: {metrics['value_loss']:7.4f} | "
                  f"ClipFrac: {metrics['clip_frac']:.3f} | "
                  f"ApproxKL: {metrics['approx_kl']:.4f}")

    env.close()
    return agent, episode_returns


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training(episode_returns: list, window: int = 20) -> None:
    """
    Plot raw and smoothed episodic returns over training.

    Key patterns to watch:
        Steady increase        → healthy learning
        Flat for many episodes → stuck in local optimum; adjust entropy_coef or lr
        Sudden drop then recover → policy update was too large; lower clip_range
        Noisy but trending up  → normal for PPO; smooth by increasing rollout_length
    """
    returns = np.array(episode_returns)
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(returns,  alpha=0.3, color="steelblue", label="Episode Return")
    plt.plot(range(window - 1, len(returns)), smoothed, color="steelblue",
             linewidth=2, label=f"Smoothed ({window}-ep)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO — Training Returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ppo_training.png", dpi=150)
    plt.show()


def evaluate(agent: PPOAgent, env_id: str, n_episodes: int = 10) -> float:
    """
    Evaluate a trained PPO agent deterministically (mean action).

    Args:
        agent:      trained PPOAgent
        env_id:     gymnasium environment ID
        n_episodes: number of evaluation episodes

    Returns:
        mean episode return
    """
    env = gym.make(env_id)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                dist, _ = agent.network(obs_t)
                action = dist.mean if agent.action_type == "continuous" else dist.probs.argmax(-1)
            action = action.squeeze(0).cpu().numpy()
            if agent.action_type == "continuous":
                action = np.clip(action, env.action_space.low, env.action_space.high)
            else:
                action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += reward
        returns.append(total)
    env.close()
    mean = np.mean(returns)
    print(f"Eval over {n_episodes} episodes: mean={mean:.2f} std={np.std(returns):.2f}")
    return mean


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def train_kl_variant(config: dict = None) -> tuple:
    """
    Train PPO with adaptive KL penalty instead of clipping.
    Drop-in replacement for train() — same interface, different update rule.
    """
    cfg = config or PPO_CONFIG
    env = gym.make(cfg["env_id"])

    obs_dim     = env.observation_space.shape[0]
    action_dim  = env.action_space.shape[0] if hasattr(env.action_space, "shape") else env.action_space.n
    action_type = "continuous" if hasattr(env.action_space, "shape") else "discrete"

    agent = PPOKLAgent(
        obs_dim=obs_dim, action_dim=action_dim, action_type=action_type,
        hidden_sizes=cfg["hidden_sizes"], lr=cfg["lr"], gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
        n_epochs=cfg["n_epochs"], n_minibatches=cfg["n_minibatches"],
        value_coef=cfg["value_coef"], entropy_coef=cfg["entropy_coef"],
        max_grad_norm=cfg["max_grad_norm"], rollout_length=cfg["rollout_length"],
        target_kl=0.01,    # Placeholder: tune alongside lr
        beta_init=1.0,     # Placeholder: adapts automatically; rarely needs tuning
    )

    obs, _ = env.reset()
    episode_return, episode_returns = 0.0, []
    recent_returns = deque(maxlen=20)
    total_steps, update_count = 0, 0

    print(f"Training PPO-KL on {cfg['env_id']}")
    while total_steps < cfg["total_timesteps"]:
        while not agent.buffer.is_full():
            action, log_prob, value = agent.get_action(obs)
            clipped = (np.clip(action, env.action_space.low, env.action_space.high)
                       if action_type == "continuous" else int(action))
            next_obs, reward, terminated, truncated, _ = env.step(clipped)
            done = terminated or truncated
            agent.buffer.store(obs, action, reward, float(done), value, log_prob)
            obs = next_obs
            episode_return += reward
            total_steps += 1
            if done:
                episode_returns.append(episode_return)
                recent_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()

        metrics = agent.update()
        update_count += 1
        if update_count % 5 == 0:
            mean_ret = np.mean(recent_returns) if recent_returns else float("nan")
            print(f"  Steps: {total_steps:7d} | MeanReturn: {mean_ret:8.2f} | "
                  f"KL: {metrics['kl']:.4f} | β: {agent.beta:.4f}")

    env.close()
    return agent, episode_returns


def main():
    print("=== Proximal Policy Optimization (PPO) ===")
    agent, returns = train()
    plot_training(returns)
    evaluate(agent, PPO_CONFIG["env_id"])

    # Uncomment to compare clipped PPO vs KL-penalty PPO:
    # print("\n=== PPO with Adaptive KL Penalty ===")
    # agent_kl, returns_kl = train_kl_variant()
    # plot_training(returns_kl)


if __name__ == "__main__":
    main()
