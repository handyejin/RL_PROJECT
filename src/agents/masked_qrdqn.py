"""QR-DQN with invalid-action masking.

`MaskableQRDQN`은 sb3-contrib QRDQN을 확장해 epsilon-greedy 탐색·argmax·predict에서
유효하지 않은 action을 제외한다. 마스크는 `env.action_masks()`에서 가져온다
(MaskableDQN과 동일한 컨벤션).

QRDQN은 Q값을 분위수(quantile) 분포로 추정하므로, argmax 전에
`quantile_net(obs)`의 분위수 평균(= 기대 Q)을 구한 뒤 마스크를 적용한다.

비고:
  - 환경이 `action_masks()`를 노출해야 한다 (RebalanceEnv는 이미 노출함).
  - 타깃 분위수 계산에는 마스크를 적용하지 않는다 (next-state 마스크 미저장).
    본 환경에서 마스크는 "낭비 행동"(다른 트럭 목적지·stay) 차단이라
    모든 action이 기술적으론 유효하므로 학습 영향이 크지 않다.
  - 학습 loss(quantile huber)는 QRDQN 그대로 사용 — 분포 학습이 핵심 안정화 장치.
"""

from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import QRDQN
from stable_baselines3.common.noise import ActionNoise


class MaskableQRDQN(QRDQN):
    # ------------------------------------------------------------------
    # 마스크 헬퍼
    # ------------------------------------------------------------------
    def _env_action_masks(self, n_envs: int) -> np.ndarray:
        """VecEnv의 각 sub-env에서 action_masks()를 받아 (n_envs, N) bool 배열로."""
        masks_list = self.env.env_method("action_masks")
        return np.stack(masks_list).astype(bool)

    def _masked_argmax(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """분위수 평균 Q에 마스크 적용 후 argmax. obs:(n_envs,obs_dim), masks:(n_envs,N)."""
        with th.no_grad():
            obs_t = th.as_tensor(obs, device=self.device)
            # quantile_net(obs): (batch, n_quantiles, n_actions) → 분위수 평균 = 기대 Q
            q = self.quantile_net(obs_t).mean(dim=1).cpu().numpy()
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
        """마스크가 주어지면 적용, 아니면 vanilla QRDQN.predict."""
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


__all__ = ["MaskableQRDQN"]
