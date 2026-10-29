"""
Deep Q-Network (DQN) + Modern Variants
========================================
DQN extends Q-learning to continuous state spaces by approximating the
action-value function Q(s,a) with a deep neural network. Combines two
key ideas that stabilize training: experience replay and target networks.

The Bellman optimality equation (training target):
    Q(s,a) ← r + γ * max_{a'} Q_target(s', a')

    Where Q_target is a periodically-frozen copy of the main network.
    Without target stabilization, the network chases a moving target — diverges.

Variants implemented:
    DQN          — original (Mnih et al., 2015)
    Double DQN   — decouples action selection from value estimation (less overestimation)
    Dueling DQN  — separates value V(s) from advantage A(s,a) streams
    PER          — Prioritized Experience Replay (samples harder transitions more)

Double DQN:
    y = r + γ * Q_target(s', argmax_{a'} Q_online(s', a'))
    Uses online network to select action, target network to evaluate it.
    Reduces systematic overestimation of Q-values.

Dueling DQN:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    Separate streams for value and advantage, merged at output.
    Benefits tasks where value of state matters more than specific action.

Prioritized Experience Replay (PER):
    P(i) = |TD_error_i + ε|^α / Σ |TD_error_j + ε|^α
    Samples transitions with high TD error more frequently.
    Importance sampling weights correct the resulting bias.

Use Cases:
    - Discrete action spaces (Atari, board games, routing, scheduling)
    - Inventory management (discrete order quantities)
    - Ad bidding (discrete bid levels)
    - Marketing channel selection (discrete budget allocation)

Fine-Tuning Recommendations:
    - learning_rate: 1e-4 to 1e-3. DQN is sensitive — use lower lr for stability.
    - replay_buffer_size: 100k–1M. Larger → more diverse samples, more memory.
    - batch_size: 32 (Atari) to 256. Larger → more stable gradient estimates.
    - gamma: 0.99 standard. 0.95 for shorter-horizon tasks.
    - target_update_freq: 1000–10000 steps. More frequent → faster propagation
      but less stable. Soft updates (tau=0.005) are a smoother alternative.
    - epsilon schedule: start 1.0, end 0.01–0.05, decay over 10–30% of training.
      Too fast → insufficient exploration. Too slow → sample inefficiency.
    - Double DQN: always use — almost no cost, consistently better.
    - Dueling: use when state value matters more than specific action choice.
    - PER: use when some transitions are much more informative than others;
      adds complexity, tune alpha (0.6) and beta (0.4 → 1.0 annealed).
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import gymnasium as gym

from deepdraft.core import fast_or, save_figure, seed_everything
from deepdraft.evaluation import mean_confidence_interval

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# ─────────────────────────────────────────────
# NETWORK ARCHITECTURES
# ─────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Standard DQN Q-network.

    Maps (observation) → Q-values for all actions simultaneously.
    This is more efficient than evaluating each action separately.

    Args:
        obs_dim:     observation space dimension
        action_dim:  number of discrete actions
        hidden_sizes: hidden layer widths
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_sizes: list[int] = None):  # Placeholder: [128, 128] or [256, 256]
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128]
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN: separate value and advantage streams.

    Q(s,a) = V(s) + A(s,a) - (1/|A|) Σ A(s,a')

    The mean subtraction ensures identifiability: changing V by a constant
    while changing A by -constant would produce the same Q-values.
    This forces V to represent the true state value.

    Architecture:
        shared_trunk → value_stream  → V(s)     [scalar]
                     → advantage_stream → A(s,a) [vector]
        Q(s,a) = V(s) + A(s,a) - mean(A)

    Args:
        obs_dim:     observation dimension
        action_dim:  number of actions
        hidden_sizes: shared trunk hidden sizes
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_sizes: list[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128]
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes[:-1]:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        last_h = hidden_sizes[-1]
        # Value stream: estimates V(s) — how good is this state?
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, last_h), nn.ReLU(), nn.Linear(last_h, 1)
        )
        # Advantage stream: estimates A(s,a) — how much better is action a vs. average?
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, last_h), nn.ReLU(), nn.Linear(last_h, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features  = self.trunk(x)
        value     = self.value_stream(features)      # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, n_actions)
        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# ─────────────────────────────────────────────
# REPLAY BUFFERS
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Standard uniform experience replay buffer.

    Breaks temporal correlations in the training data by storing
    transitions and sampling random minibatches.

    Without replay: consecutive transitions are highly correlated →
    the network memorizes recent experience and forgets earlier ones.

    Args:
        capacity: maximum number of transitions to store
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),
            torch.tensor(actions,            dtype=torch.int64),
            torch.tensor(rewards,            dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones,              dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Samples transitions with probability proportional to |TD error|^alpha.
    Transitions with high TD error (surprising/informative) are replayed more.

    Bias correction: importance sampling weights w_i = (1/N * 1/P(i))^beta
    Beta is annealed from beta_start to 1.0 over training to reduce bias.

    Args:
        capacity:    maximum buffer size
        alpha:       prioritization exponent (0=uniform, 1=full prioritization)
        beta_start:  initial importance sampling exponent
        beta_frames: steps to anneal beta from beta_start to 1.0
        epsilon:     small constant added to priorities to ensure non-zero sampling
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,        # Placeholder: 0.4–0.7
        beta_start: float = 0.4,   # Placeholder: anneal to 1.0 over training
        beta_frames: int = 100_000,
        epsilon: float = 1e-5,
    ):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.epsilon     = epsilon
        self.frame       = 1

        self.buffer      = []
        self.priorities  = np.zeros(capacity, dtype=np.float32)
        self.pos         = 0

    @property
    def beta(self) -> float:
        """Linearly anneal beta from beta_start to 1.0."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, obs, action, reward, next_obs, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((obs, action, reward, next_obs, done))
        else:
            self.buffer[self.pos] = (obs, action, reward, next_obs, done)
        self.priorities[self.pos] = max_prio  # new transitions get max priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        N = len(self.buffer)
        probs = self.priorities[:N] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
        obs, actions, rewards, next_obs, dones = zip(*samples)

        # Importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.frame += 1

        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),
            torch.tensor(actions,            dtype=torch.int64),
            torch.tensor(rewards,            dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones,              dtype=torch.float32),
            torch.tensor(weights,            dtype=torch.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities with new TD errors after learning step."""
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = (abs(err) + self.epsilon) ** self.alpha

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# DQN AGENT
# ─────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent with Double DQN, optional Dueling architecture, and PER.

    Args:
        obs_dim:         observation dimension
        action_dim:      number of discrete actions
        hidden_sizes:    Q-network hidden layer widths
        lr:              learning rate
        gamma:           discount factor
        epsilon_start:   initial exploration rate
        epsilon_end:     final exploration rate
        epsilon_decay:   steps to decay epsilon from start to end
        batch_size:      training minibatch size
        target_update:   steps between hard target network updates
        buffer_size:     replay buffer capacity
        use_double:      enable Double DQN
        use_dueling:     enable Dueling architecture
        use_per:         enable Prioritized Experience Replay
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 50_000,
        batch_size: int = 64,
        target_update: int = 1000,     # Placeholder: 500–10000
        buffer_size: int = 100_000,
        use_double: bool = True,       # Placeholder: always True in practice
        use_dueling: bool = True,      # Placeholder: True for most tasks
        use_per: bool = False,         # Placeholder: True adds complexity but helps
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.use_double    = use_double
        self.use_per       = use_per
        self.steps         = 0

        NetClass = DuelingQNetwork if use_dueling else QNetwork
        self.online_net = NetClass(obs_dim, action_dim, hidden_sizes or [128, 128]).to(DEVICE)
        self.target_net = NetClass(obs_dim, action_dim, hidden_sizes or [128, 128]).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net never trains directly

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.buffer = (PrioritizedReplayBuffer(buffer_size) if use_per
                       else ReplayBuffer(buffer_size))

    def get_action(self, obs: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        Epsilon decays linearly from epsilon_start to epsilon_end over
        epsilon_decay steps: explores early, exploits later.

        Fine-tuning tip: if the agent never converges, epsilon may be
        decaying too fast — extend epsilon_decay.
        """
        self.epsilon = max(
            self.epsilon_end,
            1.0 - self.steps * (1.0 - self.epsilon_end) / self.epsilon_decay,
        )

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.online_net(obs_t).argmax(1).item()

    def update(self) -> float | None:
        """
        Sample a minibatch and perform one gradient step.

        Returns TD loss (or None if buffer is too small).
        """
        if len(self.buffer) < self.batch_size:
            return None

        if self.use_per:
            obs, actions, rewards, next_obs, dones, weights, indices = \
                self.buffer.sample(self.batch_size)
            weights = weights.to(DEVICE)
        else:
            obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size)
            indices = None

        obs, actions, rewards = obs.to(DEVICE), actions.to(DEVICE), rewards.to(DEVICE)
        next_obs, dones       = next_obs.to(DEVICE), dones.to(DEVICE)

        # Current Q values
        q_values = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            if self.use_double:
                # Double DQN: online selects action, target evaluates it
                next_actions = self.online_net(next_obs).argmax(1, keepdim=True)
                next_q       = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN: target selects and evaluates
                next_q = self.target_net(next_obs).max(1)[0]

            targets = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (q_values - targets).detach().cpu().numpy()

        # Weighted MSE loss (PER uses importance sampling weights)
        loss = (weights.to(DEVICE) * F.smooth_l1_loss(q_values, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for DQN stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update PER priorities
        if self.use_per and indices is not None:
            self.buffer.update_priorities(indices, td_errors)

        # Hard target network update
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

DQN_CONFIG = {
    "env_id":          "CartPole-v1",  # Placeholder: any discrete Gymnasium env
    "hidden_sizes":    [128, 128],
    "lr":              1e-3,
    "gamma":           0.99,
    "epsilon_start":   1.0,
    "epsilon_end":     0.01,
    "epsilon_decay":   10_000,
    "batch_size":      64,
    "target_update":   500,
    "buffer_size":     50_000,
    "use_double":      True,
    "use_dueling":     True,
    "use_per":         False,
    "total_timesteps": fast_or(100_000, 1_500),
    "learning_starts": fast_or(1000, 200),  # steps before first update
    "seeds":           fast_or([0, 1, 2, 3, 4], [0, 1]),  # each seed is a full training run
    "eval_episodes":   fast_or(10, 2),   # greedy episodes per trained agent
}


def train(config: dict = None, seed: int = 0) -> tuple:
    """
    DQN training loop with periodic evaluation.

    Args:
        config: hyperparameter dict (uses DQN_CONFIG if None)

    Returns:
        agent, episode_returns list
    """
    cfg = config or DQN_CONFIG
    seed_everything(seed)               # random, NumPy, and torch
    env = gym.make(cfg["env_id"])
    env.action_space.seed(seed)

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_sizes=cfg["hidden_sizes"], lr=cfg["lr"],
        gamma=cfg["gamma"], epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"], epsilon_decay=cfg["epsilon_decay"],
        batch_size=cfg["batch_size"], target_update=cfg["target_update"],
        buffer_size=cfg["buffer_size"], use_double=cfg["use_double"],
        use_dueling=cfg["use_dueling"], use_per=cfg["use_per"],
    )

    obs, _ = env.reset(seed=seed)       # seeds the environment's own RNG once
    episode_return  = 0.0
    episode_returns = []
    recent_returns  = deque(maxlen=20)
    total_steps     = 0
    losses          = []

    variant = ("Double+" if cfg["use_double"] else "") + \
              ("Dueling+" if cfg["use_dueling"] else "") + \
              ("PER " if cfg["use_per"] else "") + "DQN"
    print(f"Training {variant} on {cfg['env_id']} | obs={obs_dim} act={action_dim}")

    while total_steps < cfg["total_timesteps"]:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store `terminated`, not `done`, as the bootstrap mask. A terminated
        # state (the pole fell) has no future value, so the target is just r.
        # A truncated state (CartPole's 500-step time limit) is cut off by the
        # clock, not the dynamics: its target must still be r + γ max Q(s').
        # Masking it would teach the agent that its best states are worthless.
        agent.buffer.push(obs, action, reward, next_obs, float(terminated))
        obs            = next_obs
        episode_return += reward
        total_steps    += 1

        if total_steps >= cfg["learning_starts"]:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        if done:
            episode_returns.append(episode_return)
            recent_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

            if len(episode_returns) % 20 == 0:
                print(f"  Steps: {total_steps:7d} | Ep: {len(episode_returns):4d} | "
                      f"MeanReturn: {np.mean(recent_returns):7.2f} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Loss: {np.mean(losses[-100:]) if losses else 0:.4f}")

    env.close()
    return agent, episode_returns, losses


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training(episode_returns: list, losses: list, window: int = 20) -> None:
    """Plot episode returns and TD loss over training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    returns  = np.array(episode_returns)
    window = max(1, min(window, len(returns)))  # short runs (fast mode) have few episodes
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    ax1.plot(returns,  alpha=0.3, color="steelblue")
    ax1.plot(range(window - 1, len(returns)), smoothed, color="steelblue", linewidth=2)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Return")
    ax1.set_title("DQN — Episode Returns")

    if losses:
        ax2.plot(losses, alpha=0.4, color="coral")
        sm = np.convolve(losses, np.ones(200) / 200, mode="valid")
        ax2.plot(range(199, len(losses)), sm, color="coral", linewidth=2)
    ax2.set_xlabel("Update Step"); ax2.set_ylabel("TD Loss")
    ax2.set_title("DQN — Training Loss")

    plt.suptitle("DQN (Double + Dueling)")
    plt.tight_layout()
    save_figure("dqn_training.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def evaluate(agent: "DQNAgent", env_id: str, n_episodes: int, seed: int) -> float:
    """
    Mean return of the greedy policy (no exploration) on fresh episodes.

    Training returns mix in epsilon-greedy exploration, so they understate
    what the learned policy does. Evaluation episodes use seeds the training
    run never saw.
    """
    env = gym.make(env_id)
    returns = []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + 1_000 * seed + k)
        total, done = 0.0, False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = agent.online_net(obs_t).argmax(1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def main():
    print("=== Deep Q-Network (DQN) — Double + Dueling ===")
    cfg = DQN_CONFIG

    # One training run says little: RL results vary a lot between seeds.
    # Train once per seed and report the mean with a 95% confidence interval.
    scores = []
    for i, seed in enumerate(cfg["seeds"]):
        print(f"\n--- Seed {seed} ---")
        agent, returns, losses = train(cfg, seed=seed)
        scores.append(evaluate(agent, cfg["env_id"], cfg["eval_episodes"], seed))
        print(f"  Greedy evaluation return: {scores[-1]:.1f}")
        if i == 0:
            plot_training(returns, losses)

    mean, lo, hi = mean_confidence_interval(scores)
    print(f"\nGreedy return over {len(scores)} seeds: {mean:.1f}  "
          f"(95% CI [{lo:.1f}, {hi:.1f}]; per seed: {np.round(scores, 1).tolist()})")


if __name__ == "__main__":
    main()
