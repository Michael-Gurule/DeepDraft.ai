"""
QMIX — Monotonic Value Function Factorisation for MARL
========================================================
A cooperative multi-agent reinforcement learning (MARL) algorithm that learns
decentralized policies by training a centralized mixing network.

Core insight — Centralized Training, Decentralized Execution (CTDE):
    During training: share global state and all agents' Q-values.
    During execution: each agent acts on its own local observations only.
    This is practical because training is offline, execution must be real-time.

The QMIX factorization:
    Each agent i has an individual Q-value: Q_i(τ_i, a_i)
    A mixing network combines them: Q_total = f(Q_1, ..., Q_n; s)

    Monotonicity constraint:
        ∂Q_total / ∂Q_i ≥ 0  for all i
    Enforced by using non-negative weights in the mixing network (via absolute value).

    This enables: argmax_a Q_total = [argmax_a_i Q_i for each i]
    i.e., the joint greedy action = each agent acting greedily on its own Q.
    Coordination is learned implicitly through the mixing network.

    IGM (Individual-Global-Max) condition:
        argmax Q_total = [argmax Q_i]  ← QMIX satisfies this

Training (like DQN but multi-agent):
    1. Collect trajectories with ε-greedy per agent.
    2. Sample episodes from replay buffer.
    3. Compute per-agent Q-values via DRQN (GRU-based Q network).
    4. Mix with Q_mix; compute TD target; minimize Huber loss.
    5. Periodically copy online → target networks.

Use Cases:
    - Robot swarm coordination
    - Multi-player cooperative game AI (StarCraft II SMAC benchmark)
    - Traffic signal coordination (multi-intersection RL)
    - Autonomous vehicle platoons

Advantages:
    - Strictly more expressive than VDN (which just sums Q values)
    - Linear scaling with number of agents (no joint action space enumeration)
    - Decentralized execution — no communication needed at test time

Limitations:
    - Monotonicity is a strict assumption — suboptimal for tasks requiring
      non-monotonic agent interactions (use QPLEX or RODE for those)
    - CTDE requires access to global state during training
    - Hypernetwork architecture adds implementation complexity

Fine-Tuning Recommendations:
    1. mixing_embed_dim: 32 (default). 64–128 for complex tasks.
    2. agent_hidden_dim: 64 (GRU). 128–256 for complex observations.
    3. lr: 5e-4. Lower (1e-4) for stability; higher (1e-3) for speed.
    4. td_lambda: 0.6–0.8 for n-step returns (better than 1-step in MARL).
    5. epsilon_anneal_steps: anneal ε from 1.0 to 0.05 over 50K steps.
    6. batch_size: 32 episodes. Each episode = one full trajectory.
    7. For StarCraft / competitive: use MAPPO (policy gradient) instead —
       QMIX works best for fully cooperative tasks with dense rewards.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

class CooperativeNavEnv:
    """
    Simple cooperative navigation environment.

    n_agents must collectively cover n_landmarks.
    Each agent receives a reward based on how close any agent is to each landmark.
    Agents do not collide; the task is purely cooperative (shared reward).

    Observation (per agent):
        [own_x, own_y, dx_to_each_landmark..., dy_to_each_landmark...]
        Shape: (2 + 2 * n_landmarks,)

    Global state:
        Concatenation of all agents' observations.
        Shape: (n_agents * (2 + 2 * n_landmarks),)

    Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
    """

    N_ACTIONS = 5

    def __init__(
        self,
        n_agents: int = 3,
        n_landmarks: int = 3,      # Placeholder: match to n_agents for full coverage task
        grid_size: float = 2.0,
        max_steps: int = 50,
    ):
        self.n_agents    = n_agents
        self.n_landmarks = n_landmarks
        self.grid_size   = grid_size
        self.max_steps   = max_steps
        self.obs_dim     = 2 + 2 * n_landmarks
        self.state_dim   = n_agents * self.obs_dim

        self._step = 0
        self._agent_pos: np.ndarray | None = None
        self._landmark_pos: np.ndarray | None = None

    def reset(self) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Reset to random agent and landmark positions.

        Returns:
            obs:   list of per-agent observations (each shape: obs_dim)
            state: global state (shape: state_dim)
        """
        self._step = 0
        gs = self.grid_size
        self._agent_pos    = np.random.uniform(-gs, gs, (self.n_agents, 2))
        self._landmark_pos = np.random.uniform(-gs, gs, (self.n_landmarks, 2))
        return self._get_obs(), self._get_state()

    def _get_obs(self) -> list[np.ndarray]:
        """Per-agent observations: own position + relative landmark positions."""
        obs = []
        for i in range(self.n_agents):
            rel = (self._landmark_pos - self._agent_pos[i]).flatten()
            obs.append(np.concatenate([self._agent_pos[i], rel]).astype(np.float32))
        return obs

    def _get_state(self) -> np.ndarray:
        """Global state: concatenated agent observations."""
        return np.concatenate(self._get_obs()).astype(np.float32)

    def step(
        self,
        actions: list[int],
    ) -> tuple[list[np.ndarray], float, bool, np.ndarray]:
        """
        Apply one action per agent.

        Args:
            actions: list of action indices (one per agent)

        Returns:
            obs:    new per-agent observations
            reward: shared team reward (negative: min distance to each landmark)
            done:   True if max_steps reached
            state:  new global state
        """
        speed = 0.1
        deltas = {0: [0, 0], 1: [0, speed], 2: [0, -speed], 3: [-speed, 0], 4: [speed, 0]}

        for i, a in enumerate(actions):
            self._agent_pos[i] += deltas[a]
            self._agent_pos[i] = np.clip(self._agent_pos[i], -self.grid_size, self.grid_size)

        # Reward: for each landmark, reward = -min distance from any agent
        reward = 0.0
        for lm in self._landmark_pos:
            dists = np.linalg.norm(self._agent_pos - lm, axis=1)
            reward -= dists.min()

        self._step += 1
        done = self._step >= self.max_steps
        return self._get_obs(), reward, done, self._get_state()


# ─────────────────────────────────────────────
# AGENT NETWORK (DRQN — GRU-based Q)
# ─────────────────────────────────────────────

class AgentQNetwork(nn.Module):
    """
    Individual agent Q-network using a GRU to handle partial observability.

    Each agent maintains a hidden state h that summarizes its history
    of local observations — this is the "recurrent" part of DRQN.

    Architecture:
        obs → Linear → ReLU → GRU(hidden_dim) → Linear → Q(n_actions)

    Args:
        obs_dim:    observation dimension per agent
        n_actions:  number of discrete actions
        hidden_dim: GRU hidden state dimension
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,       # Placeholder: 128–256 for complex tasks
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1  = nn.Linear(obs_dim, hidden_dim)
        self.gru  = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim) agent observation at one timestep
            h:   (batch, hidden_dim) previous GRU hidden state

        Returns:
            q:     (batch, n_actions) Q-values
            h_new: (batch, hidden_dim) updated hidden state
        """
        x     = F.relu(self.fc1(obs))
        h_new = self.gru(x, h)
        q     = self.fc_q(h_new)
        return q, h_new

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(batch_size, self.hidden_dim, device=DEVICE)


# ─────────────────────────────────────────────
# MIXING NETWORK
# ─────────────────────────────────────────────

class QMixMixer(nn.Module):
    """
    QMIX monotonic mixing network.

    Takes per-agent Q-values and the global state, and produces Q_total.
    Monotonicity is enforced by ensuring all mixing weights are non-negative
    (via absolute value activation on the hypernetwork outputs).

    Hypernetworks: the mixing weights W_1, b_1, W_2, b_2 are generated
    by small networks conditioned on the global state. This gives the
    mixing weights state-dependent expressiveness while remaining monotonic.

    Args:
        n_agents:       number of agents
        state_dim:      global state dimension
        mixing_embed_dim: width of the mixing network hidden layer
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,  # Placeholder: 64–128 for complex tasks
    ):
        super().__init__()
        self.n_agents   = n_agents
        self.embed_dim  = mixing_embed_dim

        # Hypernetworks for W_1 and b_1 (first layer weights)
        self.hyper_W1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        # Hypernetworks for W_2 and b_2 (output layer weights)
        self.hyper_W2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(
        self,
        agent_qs: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix per-agent Q-values into Q_total.

        Args:
            agent_qs: (batch, n_agents) per-agent Q-values for chosen actions
            state:    (batch, state_dim) global state

        Returns:
            q_total: (batch, 1) joint Q-value
        """
        batch = agent_qs.size(0)

        # Layer 1: monotonic weights via abs()
        W1 = torch.abs(self.hyper_W1(state)).view(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), W1) + b1)  # (batch, 1, embed_dim)

        # Layer 2: monotonic output weights
        W2 = torch.abs(self.hyper_W2(state)).view(batch, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)

        q_total = torch.bmm(hidden, W2) + b2   # (batch, 1, 1)
        return q_total.view(batch, 1)


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────

class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for QMIX.

    Stores complete episodes (required for the GRU's hidden state reset).
    Each episode is a sequence of (obs, actions, reward, next_obs, done, state, next_state).

    Args:
        capacity: maximum number of episodes to store
    """

    def __init__(self, capacity: int = 500):   # Placeholder: 1000–5000 for longer training
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, episode: list[dict]) -> None:
        """Add a completed episode (list of step dicts) to the buffer."""
        self.buffer.append(episode)

    def sample(self, batch_size: int) -> list[list[dict]]:
        """Sample batch_size episodes uniformly at random."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────
# QMIX TRAINER
# ─────────────────────────────────────────────

class QMIXTrainer:
    """
    QMIX training loop.

    Maintains:
        - One shared AgentQNetwork (parameters shared across all agents)
        - One QMixMixer (centralized, only used during training)
        - Target networks for stable bootstrapping

    Args:
        env:              CooperativeNavEnv instance
        hidden_dim:       GRU hidden size per agent
        mixing_embed_dim: mixing network hidden size
        lr:               learning rate
        gamma:            discount factor
        epsilon_start:    initial exploration rate
        epsilon_end:      final exploration rate
        epsilon_decay:    episodes to anneal epsilon
        target_update:    episodes between target network updates
        batch_size:       number of episodes per training step
    """

    def __init__(
        self,
        env: CooperativeNavEnv,
        hidden_dim: int = 64,
        mixing_embed_dim: int = 32,
        lr: float = 5e-4,               # Placeholder: 1e-4 (stable) – 1e-3 (fast)
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,       # Placeholder: 1000–5000 for complex tasks
        target_update: int = 10,
        batch_size: int = 32,
    ):
        self.env        = env
        self.n_agents   = env.n_agents
        self.n_actions  = env.N_ACTIONS
        self.gamma      = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.epsilon        = epsilon_start
        self.epsilon_end    = epsilon_end
        self.epsilon_decay  = epsilon_decay

        # Shared agent network (same weights for all agents)
        self.agent_net    = AgentQNetwork(env.obs_dim, env.N_ACTIONS, hidden_dim).to(DEVICE)
        self.agent_target = AgentQNetwork(env.obs_dim, env.N_ACTIONS, hidden_dim).to(DEVICE)
        self.agent_target.load_state_dict(self.agent_net.state_dict())

        self.mixer        = QMixMixer(env.n_agents, env.state_dim, mixing_embed_dim).to(DEVICE)
        self.mixer_target = QMixMixer(env.n_agents, env.state_dim, mixing_embed_dim).to(DEVICE)
        self.mixer_target.load_state_dict(self.mixer.state_dict())

        params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.buffer = EpisodeReplayBuffer(capacity=1000)

    def _select_actions(
        self,
        obs_list: list[np.ndarray],
        hidden: list[torch.Tensor],
    ) -> tuple[list[int], list[torch.Tensor]]:
        """
        ε-greedy action selection for all agents simultaneously.

        Args:
            obs_list: list of per-agent observations
            hidden:   list of per-agent GRU hidden states

        Returns:
            actions:     list of chosen action indices
            new_hidden:  updated hidden states
        """
        actions    = []
        new_hidden = []
        for i, obs in enumerate(obs_list):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q, h_new = self.agent_net(obs_t, hidden[i])
            if random.random() < self.epsilon:
                actions.append(random.randint(0, self.n_actions - 1))
            else:
                actions.append(int(q.argmax(dim=1).item()))
            new_hidden.append(h_new)
        return actions, new_hidden

    def _collect_episode(self) -> tuple[list[dict], float]:
        """
        Run one episode with current policy; store all transitions.

        Returns:
            episode: list of step dicts
            ep_return: total undiscounted return
        """
        obs, state = self.env.reset()
        hidden = [self.agent_net.init_hidden(1) for _ in range(self.n_agents)]
        episode, ep_return = [], 0.0

        while True:
            actions, hidden = self._select_actions(obs, hidden)
            next_obs, reward, done, next_state = self.env.step(actions)
            ep_return += reward
            episode.append({
                "obs":       [o.copy() for o in obs],
                "actions":   actions,
                "reward":    reward,
                "next_obs":  [o.copy() for o in next_obs],
                "done":      done,
                "state":     state.copy(),
                "next_state": next_state.copy(),
            })
            obs, state = next_obs, next_state
            if done:
                break

        return episode, ep_return

    def _train_step(self) -> float:
        """
        Sample a batch of episodes and compute the QMIX TD loss.

        For each timestep in each episode:
            Q_i(τ_i, a_i) via agent_net
            Q_total = mix(Q_1, ..., Q_n; state)
            target  = r + γ * mix(max_a Q_target_1, ...; next_state)
            loss    = Huber(Q_total, target)

        Returns:
            mean batch loss
        """
        episodes = self.buffer.sample(self.batch_size)
        # Flatten all steps from all episodes
        all_obs, all_actions, all_rewards = [], [], []
        all_next_obs, all_dones, all_states, all_next_states = [], [], [], []

        for ep in episodes:
            for step in ep:
                all_obs.append(step["obs"])
                all_actions.append(step["actions"])
                all_rewards.append(step["reward"])
                all_next_obs.append(step["next_obs"])
                all_dones.append(float(step["done"]))
                all_states.append(step["state"])
                all_next_states.append(step["next_state"])

        T = len(all_rewards)
        if T == 0:
            return 0.0

        states      = torch.from_numpy(np.array(all_states)).float().to(DEVICE)
        next_states = torch.from_numpy(np.array(all_next_states)).float().to(DEVICE)
        rewards     = torch.tensor(all_rewards, dtype=torch.float32, device=DEVICE)
        dones       = torch.tensor(all_dones, dtype=torch.float32, device=DEVICE)

        # Per-agent Q-values for chosen actions (online network)
        agent_qs = []
        agent_qs_target = []
        h_on  = self.agent_net.init_hidden(T)
        h_tgt = self.agent_target.init_hidden(T)

        for i in range(self.n_agents):
            obs_i      = torch.from_numpy(np.array([s[i] for s in all_obs])).float().to(DEVICE)
            next_obs_i = torch.from_numpy(np.array([s[i] for s in all_next_obs])).float().to(DEVICE)
            act_i      = torch.tensor([a[i] for a in all_actions], dtype=torch.long, device=DEVICE)

            q_i, h_on  = self.agent_net(obs_i, h_on)
            q_chosen_i = q_i.gather(1, act_i.unsqueeze(1)).squeeze(1)
            agent_qs.append(q_chosen_i)

            with torch.no_grad():
                q_tgt_i, h_tgt = self.agent_target(next_obs_i, h_tgt)
                agent_qs_target.append(q_tgt_i.max(dim=1)[0])

        agent_qs_t      = torch.stack(agent_qs, dim=1)           # (T, n_agents)
        agent_qs_tgt_t  = torch.stack(agent_qs_target, dim=1)   # (T, n_agents)

        # Mix
        q_total        = self.mixer(agent_qs_t, states).squeeze(1)        # (T,)
        with torch.no_grad():
            q_total_tgt = self.mixer_target(agent_qs_tgt_t, next_states).squeeze(1)

        target = rewards + self.gamma * (1 - dones) * q_total_tgt

        loss = F.huber_loss(q_total, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()), max_norm=10.0
        )
        self.optimizer.step()
        return float(loss.item())

    def train(self, n_episodes: int = 500) -> dict:
        """
        Main training loop.

        Args:
            n_episodes: total episodes to train

        Returns:
            history dict with episode returns and losses
        """
        history = {"returns": [], "losses": []}

        for ep in range(1, n_episodes + 1):
            episode, ep_return = self._collect_episode()
            self.buffer.push(episode)
            history["returns"].append(ep_return)

            loss = 0.0
            if len(self.buffer) >= self.batch_size:
                loss = self._train_step()
            history["losses"].append(loss)

            # Anneal epsilon
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
            )

            # Update target networks
            if ep % self.target_update == 0:
                self.agent_target.load_state_dict(self.agent_net.state_dict())
                self.mixer_target.load_state_dict(self.mixer.state_dict())

            if ep % 50 == 0 or ep == 1:
                avg_return = np.mean(history["returns"][-50:])
                print(f"  Episode {ep:4d} | Avg Return: {avg_return:.2f} | "
                      f"Loss: {loss:.4f} | ε: {self.epsilon:.3f}")

        return history


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_history(history: dict, window: int = 50) -> None:
    """
    Plot smoothed episode returns and training loss.

    Returns should increase (become less negative) as agents learn coordination.
    A flat or decreasing return curve suggests the learning rate is too high
    or the mixing network capacity is insufficient.
    """
    returns = np.array(history["returns"])
    losses  = np.array([l for l in history["losses"] if l > 0])

    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(smooth(returns, window), label=f"Smoothed (w={window})", color="steelblue")
    axes[0].plot(returns, alpha=0.2, color="steelblue")
    axes[0].set_title("Episode Return"); axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward"); axes[0].legend()

    if len(losses):
        axes[1].plot(smooth(losses, min(window, len(losses))), color="darkorange")
        axes[1].set_title("TD Loss"); axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Huber Loss")

    plt.suptitle("QMIX — Training History")
    plt.tight_layout()
    plt.savefig("qmix_training.png", dpi=150)
    plt.show()


def plot_agent_trajectories(env: CooperativeNavEnv, trainer: QMIXTrainer) -> None:
    """
    Visualize agent and landmark positions during one evaluation episode.

    Plots the path each agent took and the landmark positions.
    Good coordination = each agent converges to a different landmark.
    """
    obs, state = env.reset()
    hidden = [trainer.agent_net.init_hidden(1) for _ in range(env.n_agents)]
    trajectories = [[] for _ in range(env.n_agents)]

    for _ in range(env.max_steps):
        for i in range(env.n_agents):
            trajectories[i].append(env._agent_pos[i].copy())
        actions, hidden = trainer._select_actions(obs, hidden)
        obs, _, done, state = env.step(actions)
        if done:
            break

    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)],
                alpha=0.7, linewidth=1.5, label=f"Agent {i+1}")
        ax.scatter(*traj[0], color=colors[i % len(colors)], marker="o", s=60)
        ax.scatter(*traj[-1], color=colors[i % len(colors)], marker="*", s=100)

    for j, lm in enumerate(env._landmark_pos):
        ax.scatter(*lm, color="black", marker="X", s=150,
                   label=f"Landmark {j+1}" if j == 0 else "")
        ax.annotate(f"L{j+1}", lm, textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    ax.set_xlim(-env.grid_size * 1.1, env.grid_size * 1.1)
    ax.set_ylim(-env.grid_size * 1.1, env.grid_size * 1.1)
    ax.set_title("QMIX — Agent Trajectories (★ = final position)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("qmix_trajectories.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "n_agents":       3,
    "n_landmarks":    3,
    "hidden_dim":     64,
    "mixing_embed":   32,
    "lr":             5e-4,
    "n_episodes":     500,
    "epsilon_decay":  300,
    "batch_size":     32,
    "target_update":  10,
}


def main():
    print("=== QMIX — Cooperative Multi-Agent RL ===")

    cfg = TRAINING_CONFIG

    # 1. Environment
    env = CooperativeNavEnv(
        n_agents=cfg["n_agents"],
        n_landmarks=cfg["n_landmarks"],
        max_steps=50,
    )
    print(f"Env: {cfg['n_agents']} agents, {cfg['n_landmarks']} landmarks")
    print(f"Obs dim: {env.obs_dim}, State dim: {env.state_dim}, Actions: {env.N_ACTIONS}")

    # 2. Trainer
    trainer = QMIXTrainer(
        env,
        hidden_dim=cfg["hidden_dim"],
        mixing_embed_dim=cfg["mixing_embed"],
        lr=cfg["lr"],
        epsilon_decay=cfg["epsilon_decay"],
        batch_size=cfg["batch_size"],
        target_update=cfg["target_update"],
    )

    total_params = (sum(p.numel() for p in trainer.agent_net.parameters()) +
                    sum(p.numel() for p in trainer.mixer.parameters()))
    print(f"Total parameters: {total_params:,}")

    # 3. Train
    history = trainer.train(n_episodes=cfg["n_episodes"])

    # 4. Visualize
    plot_training_history(history)
    plot_agent_trajectories(env, trainer)

    # 5. Evaluation
    print("\n--- Evaluation (10 greedy episodes) ---")
    trainer.epsilon = 0.0   # greedy
    eval_returns = []
    for _ in range(10):
        _, ep_ret = trainer._collect_episode()
        eval_returns.append(ep_ret)
    print(f"Mean return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")


if __name__ == "__main__":
    main()
