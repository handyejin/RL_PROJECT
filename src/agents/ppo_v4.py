"""KL-to-BC constrained PPO (PPO_V4).

이 구현은 BC(Behavioral Cloning) 정책에서 너무 멀어지지 않도록
KL divergence 제약을 적용하여 PPO를 훈련합니다.

특징:
- BC 정책(사전 학습된 모델)을 로드
- 훈련 중 현재 정책과 BC 정책 사이의 KL divergence를 계산
- Loss에 KL 페널티를 추가하여 BC에서의 거리 제한

사용:
    python scripts/train.py --algo ppo_v4 --pretrain path/to/bc_model.zip
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.save_util import load_from_zip_file
from src.agents.ppo import MaskablePPO


class KLtoBC_PPO(MaskablePPO):
    """KL-to-BC 제약이 적용된 PPO.
    
    BC 정책으로부터 일정 거리 이내에 있도록 KL divergence를 제약합니다.
    """
    
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
        bc_model_path: str | None = None,
        kl_coef: float = 1.0,
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
        self.bc_model_path = bc_model_path
        self.kl_coef = float(kl_coef)
        self.bc_policy = None
        
        # BC 모델 로드
        if bc_model_path:
            self._load_bc_policy(bc_model_path)

    def _load_bc_policy(self, bc_model_path: str) -> None:
        """BC 모델 정책 로드.

        PPO 형태 zip을 우선 시도. 실패 시 항상 경고 출력 (verbose 무관) —
        DQN 기반 BC zip은 evaluate_actions 호환이 없어 사용 불가.
        """
        try:
            from stable_baselines3 import PPO
            self.bc_policy = PPO.load(bc_model_path, device=self.device)
            # evaluate_actions 가용성 체크
            if not hasattr(self.bc_policy.policy, "evaluate_actions"):
                raise RuntimeError(
                    "Loaded BC policy has no evaluate_actions — must be a PPO-style policy."
                )
            print(f"[KLtoBC_PPO] Loaded BC policy from {bc_model_path}")
        except Exception as e:
            print(
                f"[KLtoBC_PPO] WARNING: Failed to load BC policy from {bc_model_path}: {e}\n"
                f"  → KL-to-BC penalty will be DISABLED (running as plain PPO). "
                f"BC zip must be a PPO-format model."
            )
            self.bc_policy = None

    def train(self) -> None:
        """Override train to include KL-to-BC constraint."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        else:
            clip_range_vf = None

        entropy_losses = []
        pg_losses, value_losses = [], []
        kl_losses = []
        clip_fractions = []

        # SB3 RolloutBuffer._get_samples는 swap_and_flatten된 평탄 배열을 가정.
        # ppo_v3과 동일하게 명시적으로 평탄화 수행 (n_envs > 1 지원).
        _tensor_names = [
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
        ]
        if not getattr(self.rollout_buffer, "generator_ready", False):
            for tensor in _tensor_names:
                self.rollout_buffer.__dict__[tensor] = self.rollout_buffer.swap_and_flatten(
                    self.rollout_buffer.__dict__[tensor]
                )
            self.rollout_buffer.generator_ready = True
        total_samples = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            n_mb = max(1, total_samples // self.batch_size)

            for _ in range(n_mb):
                # Random sampling of minibatch (uniform — KL-to-BC는 sampling이 아닌 loss term)
                batch_inds = np.random.choice(
                    total_samples, size=self.batch_size, replace=False
                )
                # Get samples from rollout buffer
                data = self.rollout_buffer._get_samples(batch_inds)
                observations, actions, values, old_log_probs, advantages, returns = data

                # Evaluate actions under current policy
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

                # PPO policy loss
                ratio = th.exp(log_prob - old_log_probs)
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                # Value loss
                value_loss = th.nn.functional.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # KL-to-BC constraint
                kl_loss_val = th.tensor(0.0, device=self.device)
                if self.bc_policy is not None and self.kl_coef > 0:
                    with th.no_grad():
                        # BC 정책의 log_prob 계산
                        bc_values, bc_log_prob, bc_entropy = self.bc_policy.policy.evaluate_actions(
                            observations, actions_eval
                        )
                    
                    # KL divergence: E[log(pi) - log(bc_pi)]
                    # 더 정확한 형태: KL(pi || bc_pi) = E[log(pi/bc_pi)]
                    kl_div = log_prob - bc_log_prob
                    kl_loss_val = th.mean(kl_div)
                    kl_losses.append(kl_loss_val.item())

                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.kl_coef * kl_loss_val

                # approx KL divergence (from current to old policy)
                with th.no_grad():
                    log_ratio = log_prob - old_log_probs
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
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
        self.logger.record("train/kl_to_bc_loss", np.mean(kl_losses) if kl_losses else 0.0)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions) if clip_fractions else 0.0)
        if 'loss' in locals():
            self.logger.record("train/loss", float(loss.item()))
        
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


__all__ = ["KLtoBC_PPO"]
