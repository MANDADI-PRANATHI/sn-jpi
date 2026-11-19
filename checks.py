"""Reproduction exit checks for the simulator, trainer and baselines."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import gymnasium as gym
import numpy as np
import torch

from actor_critic import ActorCritic
from environments import ActionSwitch, make_env
from matrix import CONTROL, DOUBLE, MUJOCO
from rollout import collect_rollout
from train import run


class TestEnv(gym.Env):
    def __init__(self, termination=False):
        self.observation_space = gym.spaces.Box(-2, 2, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-.01, .01, (1,), dtype=np.float32)
        self.termination = termination

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return action.copy(), 10., self.termination and self.steps == 2, self.steps == 2, {}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    torch.set_num_threads(1)
    for task in MUJOCO + [DOUBLE] + CONTROL:
        env = make_env(task, False)
        obs, _ = env.reset(seed=32)
        assert env.observation_space.shape == obs.shape
        obs, reward, *_ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(reward)
        env.close()
        print('Reset and step passed', task, flush=True)
    for termination in (False, True):
        env = ActionSwitch(TestEnv(termination), True)
        env.reset()
        step = env.step(np.array([1.], dtype=np.float32))
        assert step[1] == 0 and step[0][0] > 0
        env.flipped = True
        step = env.step(np.array([1.], dtype=np.float32))
        assert step[0][0] < 0 and step[1] == .01 + int(not termination)
    env = ActionSwitch(TestEnv(), True)
    model = ActorCritic(1, 1)
    batch = collect_rollout(env, model, 3, .9, noise=np.random.default_rng(2))
    assert batch.transitions == 4 and batch.episode_lengths.tolist() == [2, 2]
    assert batch.episode_scores.tolist() == [0., 0.]
    torch.testing.assert_close(batch.value_targets, torch.tensor([.909, 1.01, .909, 1.01]))
    with torch.no_grad():
        policy, _ = model(batch.observations)
        torch.testing.assert_close(policy.log_prob(batch.latent_actions), batch.old_log_probs)
        assert not torch.allclose(policy.log_prob(batch.executed_actions), batch.old_log_probs)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for method in ('sn1', 'sn3', 'sn6', 'raw6', 'ppo', 'trpo', 'sac_keep', 'sac_reset'):
            case = run('initial', MUJOCO[0], method, 0, root, smoke=True)
            folder = root / 'raw' / case
            updates = read_rows(folder / 'updates.jsonl')
            if method.startswith(('sn', 'raw')):
                assert len(updates) == 2 * int(method[-1])
                assert all(np.isfinite(row['loss']) for row in updates)
            if method.startswith('sac'):
                assert updates[1]['replay_before'] == 64
                assert updates[1]['replay_after'] == (0 if method == 'sac_reset' else 64)
            print('Trainer smoke passed', method, flush=True)
        case = run('initial', MUJOCO[0], 'sn3', 0, root / 'repeat', smoke=True)
        first = root / 'raw' / case
        second = root / 'repeat' / 'raw' / case
        for file in ('updates.jsonl', 'evaluation.jsonl'):
            assert (first / file).read_bytes() == (second / file).read_bytes()
        env = make_env(MUJOCO[0], True)
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
        checkpoint = torch.load(first / 'checkpoint.pt', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        other = torch.load(second / 'checkpoint.pt', weights_only=True)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, other['model'][key], rtol=0, atol=0)
        env.close()
        print('Deterministic rerun and checkpoint round trip passed', flush=True)


if __name__ == '__main__':
    main()
