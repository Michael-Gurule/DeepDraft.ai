"""Checks for the RL fixes: time limits are not terminal states, recurrent
agents unroll through time, and a seed fixes the whole run."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location("rl_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimeLimitOnlyEnv(gym.Env):
    """Never terminates; every episode is truncated after `limit` steps."""

    def __init__(self, limit: int = 5, continuous: bool = False):
        self.limit = limit
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.action_space = (gym.spaces.Box(-1.0, 1.0, (1,), np.float32) if continuous
                             else gym.spaces.Discrete(2))
        self.t = 0

    def _obs(self):
        return np.full(3, self.t / self.limit, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 1.0, False, self.t >= self.limit, {}


def test_dqn_does_not_treat_time_limits_as_terminal(monkeypatch):
    dqn = load_module("reinforcement_learning/dqn.py")
    monkeypatch.setattr(dqn.gym, "make", lambda *a, **k: TimeLimitOnlyEnv(limit=5))
    cfg = {**dqn.DQN_CONFIG, "total_timesteps": 40, "learning_starts": 1_000,
           "hidden_sizes": [8], "buffer_size": 100}
    agent, returns, _ = dqn.train(cfg, seed=0)
    assert len(returns) == 8                                   # episodes did end...
    assert all(done == 0.0 for *_, done in agent.buffer.buffer)  # ...but none terminally


@pytest.mark.parametrize("terminated, truncated", [(False, True), (True, False)])
def test_ppo_bootstraps_truncated_steps_only(terminated, truncated):
    ppo = load_module("reinforcement_learning/ppo.py")
    agent = ppo.PPOAgent(obs_dim=3, action_dim=1, action_type="continuous",
                         hidden_sizes=[8], rollout_length=4)

    class OneStep(TimeLimitOnlyEnv):
        def step(self, action):
            return np.ones(3, dtype=np.float32), 1.0, terminated, truncated, {}

    env = OneStep(continuous=True)
    obs, _ = env.reset(seed=0)
    next_obs, reward, ended = ppo.collect_step(agent, env, obs, "continuous")
    assert reward == 1.0 and ended
    assert agent.buffer.dones[-1] == 1.0                       # GAE never crosses episodes
    expected = 1.0 + (agent.gamma * agent.value(next_obs) if truncated else 0.0)
    assert agent.buffer.rewards[-1] == pytest.approx(expected)


def test_ppo_bootstraps_rollout_from_the_state_after_the_last_step(monkeypatch):
    ppo = load_module("reinforcement_learning/ppo.py")
    agent = ppo.PPOAgent(obs_dim=3, action_dim=1, action_type="continuous",
                         hidden_sizes=[8], rollout_length=4, n_epochs=1, n_minibatches=1)
    env = TimeLimitOnlyEnv(limit=100, continuous=True)
    obs, _ = env.reset(seed=0)
    for _ in range(4):
        obs, _, _ = ppo.collect_step(agent, env, obs, "continuous")

    seen = {}
    original = agent.buffer.compute_returns_and_advantages
    def spy(last_value, *args, **kwargs):
        seen["last_value"] = last_value
        return original(last_value, *args, **kwargs)
    monkeypatch.setattr(agent.buffer, "compute_returns_and_advantages", spy)

    expected = agent.value(obs)
    agent.update(obs)
    assert seen["last_value"] == pytest.approx(expected)


def test_a2c_vector_env_reports_truncation_and_final_obs():
    a2c = load_module("reinforcement_learning/actor_critic.py")
    envs = a2c.SyncVectorEnv([lambda: TimeLimitOnlyEnv(limit=2)] * 2)
    envs.reset(seed=0)
    envs.step(np.array([0, 1]))
    obs, rewards, dones, terms, final_obs = envs.step(np.array([0, 1]))
    np.testing.assert_array_equal(dones, [1.0, 1.0])
    np.testing.assert_array_equal(terms, [0.0, 0.0])
    np.testing.assert_allclose(final_obs, 1.0)                 # state the episode ended on
    np.testing.assert_allclose(obs, 0.0)                       # auto-reset observation


def test_qmix_unrolls_each_agent_through_time():
    qmix = load_module("multi_agent_systems/qmix.py")
    from deepdraft.core import seed_everything

    seed_everything(0)
    trainer = qmix.QMIXTrainer(qmix.CooperativeNavEnv(), hidden_dim=16, batch_size=2)
    episode, _ = trainer._collect_episode()
    obs = torch.tensor(np.array([[step["obs"] for step in episode]]), dtype=torch.float32)

    q = trainer._unroll(trainer.agent_net, obs)
    for agent in range(trainer.n_agents):
        h = trainer.agent_net.init_hidden(1)
        for t in range(6):
            q_manual, h = trainer.agent_net(obs[:, t, agent], h)
        torch.testing.assert_close(q[0, 5, agent], q_manual[0])

    # The time limit is a truncation: no stored step is terminal
    assert not any(step["terminated"] for step in episode)
    trainer.buffer.push(episode)
    trainer.buffer.push(trainer._collect_episode()[0])
    assert np.isfinite(trainer._train_step())


def test_a_seed_fixes_the_whole_training_run():
    dqn = load_module("reinforcement_learning/dqn.py")
    cfg = {**dqn.DQN_CONFIG, "total_timesteps": 400, "learning_starts": 100,
           "hidden_sizes": [16], "buffer_size": 1_000}
    _, returns_a, losses_a = dqn.train(cfg, seed=3)
    _, returns_b, losses_b = dqn.train(cfg, seed=3)
    _, returns_c, _ = dqn.train(cfg, seed=4)
    assert returns_a == returns_b and losses_a == losses_b
    assert returns_a != returns_c
