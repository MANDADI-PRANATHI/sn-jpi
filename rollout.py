"""Collect fresh complete episodes using the Gymnasium reset/step API."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Rollout:
    observations: torch.Tensor
    latent_actions: torch.Tensor
    executed_actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    episode_ids: torch.Tensor
    value_targets: torch.Tensor
    episode_returns: torch.Tensor
    episode_scores: torch.Tensor
    episode_lengths: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor

    @property
    def transitions(self):
        return self.rewards.numel()


@torch.no_grad()
def collect_rollout(env, model, min_transitions, gamma, *, seed=None, noise=None):
    """Collect at least min_transitions. Gamma is explicit for the critic."""
    if min_transitions < 1:
        raise ValueError("min_transitions must be positive")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")

    parameter = next(model.parameters())
    float_options = {"dtype": parameter.dtype, "device": parameter.device}
    obs_batch, latent_actions, old_log_probs = [], [], []
    executed_actions = []
    rewards, episode_ids, value_targets = [], [], []
    episode_returns, lengths, terminations, truncations = [], [], [], []

    actor = model.numpy_actor() if noise is not None else None
    if actor is not None:
        std = model.log_std.exp().numpy()

    while len(rewards) < min_transitions:
        observation, _ = env.reset(seed=seed if not lengths else None)
        episode_rewards = []
        terminated = truncated = False
        while not (terminated or truncated):
            if actor is None:
                obs = torch.as_tensor(observation, **float_options).clone()
                policy, _ = model(obs)
                sample = policy.sample()
                old_log_probs.append(policy.log_prob(sample))
                latent = sample.cpu().numpy()
            else:
                obs = np.asarray(observation, dtype=np.float32).copy()
                latent = actor(obs) + std * noise.standard_normal(std.shape, dtype=np.float32)
            action = np.clip(latent, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, info = env.step(
                action.astype(env.action_space.dtype, copy=False)
            )
            executed_actions.append(info.get("executed_action", action).copy())
            obs_batch.append(obs)
            latent_actions.append(latent)
            episode_rewards.append(float(reward))

        # Start at zero even when the episode ended at a time limit.
        targets = []
        future = 0.0
        for reward in reversed(episode_rewards):
            future = reward + gamma * future
            targets.append(future)
        value_targets.extend(reversed(targets))
        rewards.extend(episode_rewards)
        episode_ids.extend([len(lengths)] * len(episode_rewards))
        episode_returns.append(sum(episode_rewards))
        lengths.append(len(episode_rewards))
        terminations.append(terminated)
        truncations.append(truncated)

    returns = torch.tensor(episode_returns, **float_options)
    scores = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    device = parameter.device
    observations = (torch.stack(obs_batch) if actor is None
                    else torch.as_tensor(np.asarray(obs_batch), **float_options))
    latents = torch.as_tensor(np.asarray(latent_actions), **float_options)
    log_probs = (torch.stack(old_log_probs) if actor is None
                 else model(observations)[0].log_prob(latents))
    return Rollout(
        observations=observations,
        latent_actions=latents,
        executed_actions=torch.as_tensor(np.asarray(executed_actions), **float_options),
        old_log_probs=log_probs,
        rewards=torch.tensor(rewards, **float_options),
        episode_ids=torch.tensor(episode_ids, dtype=torch.long, device=device),
        value_targets=torch.tensor(value_targets, **float_options),
        episode_returns=returns,
        episode_scores=scores,
        episode_lengths=torch.tensor(lengths, dtype=torch.long, device=device),
        terminated=torch.tensor(terminations, dtype=torch.bool, device=device),
        truncated=torch.tensor(truncations, dtype=torch.bool, device=device),
    )
