import json
import platform
import random
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from actor_critic import ActorCritic
from environments import make_env
from matrix import ASSUMPTIONS as C, schedule
from objectives import diagnostics, jpi
from rollout import collect_rollout


def local_update(model, optimizer, batch, method):
    logs = []
    is_ppo = method == 'ppo'
    epochs = C['ppo_epochs'] if is_ppo else int(method[-1])
    scores = batch.episode_scores.double()
    if is_ppo:
        with torch.no_grad():
            advantages = batch.value_targets - model(batch.observations)[1]
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    for epoch in range(1, epochs + 1):
        size = batch.transitions
        partitions = (torch.randperm(size).split(C['ppo_minibatch']) if is_ppo
                      else [slice(None)])
        for minibatch, rows in enumerate(partitions):
            policy, values = model(batch.observations[rows])
            change = policy.log_prob(batch.latent_actions[rows]) - batch.old_log_probs[rows]
            value_loss = .5 * (values - batch.value_targets[rows]).square().mean()
            entropy = policy.entropy().mean()
            kl = -change.mean()
            detail = {}
            if is_ppo:
                ratio = change.exp()
                adv = advantages[rows]
                actor = -torch.minimum(ratio * adv, ratio.clamp(1-C['ppo_clip'], 1+C['ppo_clip']) * adv).mean()
            else:
                traj = torch.zeros_like(scores).index_add(0, batch.episode_ids, change.double())
                actor = -jpi(traj, scores, normalized=method.startswith('sn'))
                detail = diagnostics(traj.detach(), scores, .8)
                detail.update(beta=.8, trajectory_kl=-traj.mean().item())
            loss = actor + C['value_coef'] * value_loss - C['entropy_coef'] * entropy
            if not is_ppo:
                loss = loss + C['kl_coef'] * kl
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), C['grad_cap'], error_if_nonfinite=True)
            optimizer.step()
            logs.append(dict(epoch=epoch, minibatch=minibatch, loss=loss.item(),
                             actor=actor.item(), value_loss=value_loss.item(),
                             entropy=entropy.item(), kl=kl.item(), grad_norm=grad.item(), **detail))
    return logs


def make_baseline(method, env, seed, budget):
    if method == 'trpo':
        from sb3_contrib import TRPO
        return TRPO('MlpPolicy', env, seed=seed, device='cpu', n_steps=budget,
                    batch_size=64, gamma=C['gamma'], learning_rate=C['lr'],
                    target_kl=C['trpo_target_kl'])
    from stable_baselines3 import SAC
    return SAC('MlpPolicy', env, seed=seed, device='cpu', gamma=C['gamma'],
               learning_rate=C['sac_lr'], learning_starts=C['sac_warmup'],
               buffer_size=C['sac_buffer'], batch_size=C['sac_batch_size'],
               train_freq=C['sac_train_freq'], gradient_steps=C['sac_gradient_steps'])


@torch.no_grad()
def evaluate(env, model, actor, seed):
    observation, _ = env.reset(seed=seed)
    length, total = 0, 0.
    while True:
        if actor is not None:
            action = actor(observation)
        else:
            action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        length += 1
        total += reward
        if terminated or truncated:
            return length, total


def run(protocol, task, method, seed, root='results', smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    updates, switch, budget, episodes = schedule(protocol, task)
    if smoke:
        updates, switch, budget, episodes = 2, 2, 64, 1
    run_id = f'{protocol}__{task.replace("/", "-")}__{method}__{seed}'
    folder = Path(root) / 'raw' / run_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete').exists():
        return run_id
    sparse = protocol in ('initial', 'stress')
    train = make_env(task, sparse)
    evaluation = make_env(task, sparse)
    train.reset(seed=seed)
    local = method not in ('trpo', 'sac_keep', 'sac_reset')
    if local:
        model = ActorCritic(train.observation_space.shape[0], train.action_space.shape[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=C['lr'])
    else:
        model = make_baseline(method, train, seed, budget)
    metadata = dict(run_id=run_id, protocol=protocol, task=task, method=method,
                    seed=seed, reward_mode='sparse' if sparse else 'dense',
                    updates=updates, switch=switch, budget=budget, eval_episodes=episodes,
                    assumptions=C, python=sys.version, platform=platform.platform(),
                    packages={p: version(p) for p in ('torch', 'numpy', 'gymnasium', 'mujoco',
                              'dm-control', 'stable-baselines3', 'sb3-contrib')})
    (folder / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    interactions = 0
    noise = np.random.default_rng(seed)
    with (folder / 'evaluation.jsonl').open('w') as eval_file, (folder / 'updates.jsonl').open('w') as log:
        for update in range(1, updates+1):
            train.flipped = evaluation.flipped = update >= switch
            if local:
                batch = collect_rollout(train, model, budget, C['gamma'], seed=seed + 1009 * update, noise=noise)
                interactions += batch.transitions
                rows = local_update(model, optimizer, batch, method)
                episode_data = dict(lengths=batch.episode_lengths.tolist(), returns=batch.episode_returns.tolist())
            else:
                replay_before = replay_after = None
                if method.startswith('sac'):
                    replay_before = model.replay_buffer.size()
                    if method == 'sac_reset' and update == switch:
                        model.replay_buffer.reset()
                    replay_after = model.replay_buffer.size()
                if update == switch:
                    model._last_obs = None
                before = model.num_timesteps
                model.learn(budget, reset_num_timesteps=False)
                if model.num_timesteps - before != budget:
                    raise RuntimeError('Unexpected baseline interaction count')
                rows = [dict(model.logger.name_to_value)]
                interactions += budget
                rows[0].update(replay_before=replay_before, replay_after=replay_after)
                episode_data = {}
            for row in rows:
                log.write(json.dumps(dict(update=update, interactions=interactions, **episode_data, **row)) + '\n')
                episode_data = {}
            actor = model.numpy_actor() if local else None
            for episode in range(episodes):
                eval_seed = 1_000_000 + 100_000*seed + 10*update + episode
                length, total = evaluate(evaluation, model, actor, eval_seed)
                eval_file.write(json.dumps(dict(run_id=run_id, task=task, method=method, seed=seed,
                                  reward_mode=metadata['reward_mode'], update=update,
                                  switched=train.flipped, episode=episode, eval_seed=eval_seed,
                                  length=length, episode_return=total, interactions=interactions)) + '\n')
            eval_file.flush()
            log.flush()
    if local:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), manifest=metadata),
                   folder / 'checkpoint.pt')
    else:
        model.save(folder / 'checkpoint.zip')
    train.close()
    evaluation.close()
    (folder / 'complete').touch()
    return run_id
