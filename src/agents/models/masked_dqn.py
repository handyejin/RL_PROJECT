"""DQN with invalid-action masking + optional Double DQN target.

`MaskableDQN`은 SB3 DQN을 확장해 epsilon-greedy 탐색·Q-argmax·predict에서
유효하지 않은 action을 제외한다. 마스크는 sb3-contrib 컨벤션을 따라
`env.action_masks()`에서 가져온다.

비고:
  - 환경이 `action_masks()`를 노출해야 한다 (RebalanceEnv는 이미 노출함).
  - 타깃 Q 계산에는 마스크를 적용하지 않는다 (next-state 마스크 미저장).
    본 환경에서 마스크는 "낭비 행동" 차단(다른 트럭의 목적지)이라 모든
    action이 기술적으론 유효하므로 학습에 영향이 크지 않다.
  - double_q=True면 Double DQN 타깃을 사용한다.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.noise import ActionNoise
from torch.nn import functional as F


class MaskableDQN(DQN):
    def __init__(self, *args, double_q: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_q = double_q

    # ------------------------------------------------------------------
    # 마스크 헬퍼
    # ------------------------------------------------------------------
    def _env_action_masks(self, n_envs: int) -> np.ndarray:
        """VecEnv의 각 sub-env에서 action_masks()를 받아 (n_envs, N) 배열로 반환."""
        masks_list = self.env.env_method("action_masks")
        return np.stack(masks_list).astype(bool)

    def _masked_argmax(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Q-값에 마스크 적용 후 argmax. obs: (n_envs, obs_dim), masks: (n_envs, N)."""
        with th.no_grad():
            obs_t = th.as_tensor(obs, device=self.device)
            q = self.q_net(obs_t).cpu().numpy()
        q = q.copy()
        q[~masks] = -np.inf
        return q.argmax(axis=1)

    @staticmethod
    def _random_masked(masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.array(
            [rng.choice(np.flatnonzero(m)) for m in masks], dtype=np.int64
        )

    # ------------------------------------------------------------------
    # 탐색·예측에 마스크 적용
    # ------------------------------------------------------------------
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._last_obs is not None
        masks = self._env_action_masks(n_envs)
        rng = np.random.default_rng()

        if self.num_timesteps < learning_starts:
            action = self._random_masked(masks, rng)
        elif rng.random() < self.exploration_rate:
            action = self._random_masked(masks, rng)
        else:
            action = self._masked_argmax(np.asarray(self._last_obs), masks)

        # Discrete action: buffer/action 동일
        return action, action

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        """마스크가 주어지면 적용, 아니면 vanilla DQN.predict."""
        if action_masks is None:
            return super().predict(observation, state, episode_start, deterministic)

        obs = np.asarray(observation)
        masks = np.atleast_2d(np.asarray(action_masks, dtype=bool))
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]

        if not deterministic and np.random.rand() < self.exploration_rate:
            rng = np.random.default_rng()
            action = self._random_masked(masks, rng)
        else:
            action = self._masked_argmax(obs, masks)

        return (int(action[0]) if single else action, state)

    # ------------------------------------------------------------------
    # Double DQN 타깃 (옵션)
    # ------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if not self.double_q:
            return super().train(gradient_steps, batch_size)

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Double DQN: online net으로 argmax 선택, target net으로 평가
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.gather(1, next_actions)
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


__all__ = ["MaskableDQN"]
