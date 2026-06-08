"""Reward-prioritized PPO (PPO_V3).

This implementation adjusts minibatch sampling within PPO updates to prioritize
samples from high-return episodes. It computes per-episode returns from the
rollout buffer and samples training minibatches with probabilities proportional
to episode returns (shifted and normalized).

Note: This is a lightweight, in-repo approximation of episode-prioritized
sampling suitable for experiments. It uses the internal RolloutBuffer arrays
via swap_and_flatten and then samples indices according to episode priorities.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import explained_variance
from src.agents.ppo import MaskablePPO


class RewardPrioritizedPPO(MaskablePPO):
    def __init__(
        self,
        policy,
        env,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs=None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        prioritization_temperature: float = 1.0,
        min_priority_weight: float = 1e-3,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.prioritization_temperature = float(prioritization_temperature)
        self.min_priority_weight = float(min_priority_weight)

    def _compute_episode_priorities(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-sample episode ids and per-episode total returns.

        Returns:
            sample_episode_ids: shape (N,), episode id for each flattened sample
            episode_priorities: shape (num_episodes,), priority value for each episode
        """
        # Raw arrays shape: (buffer_size, n_envs)
        rewards = self.rollout_buffer.rewards
        episode_starts = self.rollout_buffer.episode_starts
        # Flatten as in RolloutBuffer.get (swap axes then reshape)
        flat_rewards = self.rollout_buffer.swap_and_flatten(rewards)
        flat_starts = self.rollout_buffer.swap_and_flatten(episode_starts)
        # Determine episode ids by cumulative sum of starts where start == 1
        # Treat any non-zero as start
        starts_bool = flat_starts.astype(bool)
        episode_ids = np.cumsum(starts_bool.astype(np.int32)) - 1
        if episode_ids.size == 0:
            return episode_ids, np.array([])
        num_eps = int(episode_ids.max()) + 1
        # Sum rewards per episode
        ep_returns = np.zeros(num_eps, dtype=np.float32)
        for ep in range(num_eps):
            ep_returns[ep] = flat_rewards[episode_ids == ep].sum()
        # Convert returns into positive priorities (shift) and apply temperature
        # Shift so min becomes zero
        shifted = ep_returns - ep_returns.min()
        shifted = np.maximum(shifted, 0.0)
        # Add small constant to avoid zero probability
        shifted += self.min_priority_weight
        # Softmax-like scaling with temperature, numerically stable
        if self.prioritization_temperature != 1.0:
            scaled_logits = shifted / self.prioritization_temperature
            scaled_logits = scaled_logits - np.max(scaled_logits)
            scaled = np.exp(scaled_logits)
        else:
            scaled = shifted
        scaled = np.nan_to_num(scaled, posinf=1.0, neginf=0.0)
        total = np.sum(scaled)
        if total <= 0 or not np.isfinite(total):
            priorities = np.ones_like(scaled, dtype=np.float32) / scaled.shape[0]
        else:
            priorities = scaled / total
        return episode_ids, priorities

    def train(self) -> None:
        """Override train to sample minibatches with episode-prioritized probabilities."""
        # Prepare standard training variables
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        else:
            clip_range_vf = None

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # Prepare flattened buffers once (like RolloutBuffer.get does)
        _tensor_names = [
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
        ]
        for tensor in _tensor_names:
            self.rollout_buffer.__dict__[tensor] = self.rollout_buffer.swap_and_flatten(
                self.rollout_buffer.__dict__[tensor]
            )
        total_samples = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs

        # Compute episode priorities and map sample -> episode
        sample_episode_ids, episode_priorities = self._compute_episode_priorities()
        if sample_episode_ids.size == 0 or episode_priorities.size == 0:
            # fallback to uniform sampling
            sample_probs = None
        else:
            # per-sample probability = episode_priorities[episode_id]
            sample_probs = episode_priorities[sample_episode_ids]
            sample_probs = sample_probs / (sample_probs.sum() + 1e-12)

        continue_training = True
        # train for n_epochs epochs using prioritized sampling
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # number of minibatches per epoch (cover whole buffer approximately)
            n_mb = max(1, total_samples // self.batch_size)
            used_indices = set()
            for _ in range(n_mb):
                if sample_probs is None:
                    batch_inds = np.random.choice(total_samples, size=self.batch_size, replace=False)
                else:
                    # draw without replacement with probabilities by repeated sampling until unique
                    # approach: sample bigger pool then take unique subset
                    if self.batch_size >= total_samples:
                        batch_inds = np.arange(total_samples)
                    else:
                        # draw with replacement more indices to then unique
                        pool = np.random.choice(total_samples, size=min(total_samples, self.batch_size * 3), replace=True, p=sample_probs)
                        batch_inds = np.unique(pool)
                        if batch_inds.size < self.batch_size:
                            # fill remaining uniformly
                            rem = self.batch_size - batch_inds.size
                            extra = np.random.choice(total_samples, size=rem, replace=False)
                            batch_inds = np.concatenate([batch_inds, extra])
                        else:
                            batch_inds = batch_inds[: self.batch_size]
                # ensure correct dtype
                batch_inds = batch_inds.astype(int)

                # Retrieve samples
                data = self.rollout_buffer._get_samples(batch_inds)
                observations, actions, values, old_log_probs, advantages, returns = data

                # Convert types for policy evaluation
                if isinstance(self.action_space, spaces.Discrete):
                    actions_eval = actions.long().flatten()
                else:
                    actions_eval = actions

                values_pred, log_prob, entropy = self.policy.evaluate_actions(observations, actions_eval)
                values_pred = values_pred.flatten()
                # Normalize advantage
                adv = advantages
                if self.normalize_advantage and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = th.exp(log_prob - old_log_probs)
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if clip_range_vf is None:
                    values_to_use = values_pred
                else:
                    values_to_use = returns  # simplified
                value_loss = th.nn.functional.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # approx kl
                with th.no_grad():
                    log_ratio = log_prob - old_log_probs
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # optimization
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses) if pg_losses else 0.0)
        self.logger.record("train/value_loss", np.mean(value_losses) if value_losses else 0.0)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions) if clip_fractions else 0.0)
        self.logger.record("train/loss", float(loss.item()) if 'loss' in locals() else 0.0)
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/explained_variance", explained_var)
        # priority stats
        if episode_priorities.size:
            self.logger.record("train/priority_min", float(np.min(episode_priorities)))
            self.logger.record("train/priority_max", float(np.max(episode_priorities)))
            self.logger.record("train/priority_mean", float(np.mean(episode_priorities)))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


__all__ = ["RewardPrioritizedPPO"]
