"""Check the result package against the published experiment matrix."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from actor_critic import ActorCritic
from environments import make_env
from matrix import jobs, schedule
from train import evaluate
from report import load_scores


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def validate_checkpoint(folder, protocol, task, method, expected):
    env = make_env(task, protocol in ('initial', 'stress'))
    env.flipped = True
    actor = None
    if method == 'trpo':
        from sb3_contrib import TRPO
        model = TRPO.load(folder / 'checkpoint.zip', device='cpu')
    elif method.startswith('sac'):
        from stable_baselines3 import SAC
        model = SAC.load(folder / 'checkpoint.zip', device='cpu')
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
        state = torch.load(folder / 'checkpoint.pt', weights_only=True)
        model.load_state_dict(state['model'])
        actor = model.numpy_actor()
    length, total = evaluate(env, model, actor, expected['eval_seed'])
    env.close()
    assert length == expected['length'] and abs(total - expected['episode_return']) < 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    root = Path('results')
    torch.set_num_threads(1)
    total, evaluated, validated = 0, 0, 0
    checkpoint_cases = set()
    for protocol, task, method, seed in jobs():
        folder = root / 'raw' / f'{protocol}__{task.replace("/", "-")}__{method}__{seed}'
        if args.allow_incomplete and not (folder / 'complete').exists():
            continue
        manifest = json.loads((folder / 'manifest.json').read_text())
        updates, switch, budget, episodes = schedule(protocol, task)
        assert (manifest['updates'], manifest['switch'], manifest['budget'], manifest['eval_episodes']) == (
                updates, switch, budget, episodes)
        assert (folder / 'complete').exists() and len(list(folder.glob('checkpoint.*'))) == 1
        evaluations = read_jsonl(folder / 'evaluation.jsonl')
        assert list(load_scores(root, protocol, task, method, seed))
        assert all(1 <= row['length'] <= 200 and np.isfinite(row['episode_return']) for row in evaluations)
        interactions = np.array([row['interactions'] for row in evaluations[::episodes]])
        spent = np.diff(interactions, prepend=0)
        local = method not in ('trpo', 'sac_keep', 'sac_reset')
        assert all(spent >= budget)
        assert all(spent < budget + 200) if local else all(spent == budget)
        metrics = read_jsonl(folder / 'updates.jsonl')
        first_rows = [metrics[0]] + [item for prior, item in zip(metrics, metrics[1:])
                                  if prior['update'] != item['update']]
        assert [row['update'] for row in first_rows] == list(range(1, updates+1))
        assert [row['interactions'] for row in first_rows] == list(interactions)
        if local:
            assert [sum(row['lengths']) for row in first_rows] == list(spent)
        if method.startswith(('raw', 'sn')):
            epochs = int(method[-1])
            assert [row['epoch'] for row in metrics] == list(range(1, epochs+1)) * updates
            for row in metrics:
                assert np.isfinite(row['loss']) and np.isfinite(row['grad_norm'])
                assert 0 < row['delta_inf'] <= row['delta2'] + 1e-12 <= 1 + 2e-12
                assert len(row['log_ratios']) == len(row['ratio_weights'])
                assert abs(sum(row['ratio_weights']) - 1) < 1e-10
        if method.startswith('sac'):
            row = first_rows[switch-1]
            assert row['replay_before'] == budget * (switch-1)
            assert row['replay_after'] == (0 if method == 'sac_reset' else row['replay_before'])
        case = (protocol, task, method)
        if case not in checkpoint_cases:
            validate_checkpoint(folder, protocol, task, method, evaluations[-1])
            checkpoint_cases.add(case)
        validated += 1
        total += int(interactions[-1])
        evaluated += len(evaluations)
    if not args.allow_incomplete:
        assert len(list((root / 'tables').glob('table*.csv'))) == 7
        assert len(list((root / 'figures').glob('figure*.png'))) == 4
    print(f'Validated {validated} runs, {total:,} training transitions and {evaluated:,} evaluation episodes.')
    print(f'Replayed the final evaluation episode from {len(checkpoint_cases)} saved checkpoints.')


if __name__ == '__main__':
    main()
