"""PPO 알고리즘 래퍼.

stable-baselines3 PPO를 그대로 사용하며, 평가 시 optional action_masks를 받을 수 있도록
predict()를 확장합니다. 이 클래스는 주로 env.action_masks()를 활용한 deterministic 평가를
지원하기 위한 것입니다.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO


class MaskablePPO(PPO):
    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        """Predict action optionally respecting an action mask.

        If action_masks is provided, any invalid action selected by the underlying
        PPO predict() is replaced with the first valid action.
        """
        if action_masks is None:
            return super().predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
            )

        action, state = super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        action = np.asarray(action)
        masks = np.asarray(action_masks, dtype=bool)

        single = action.ndim == 0
        if single:
            action = action[None]
        if masks.ndim == 1:
            masks = masks[None]

        if action.shape[0] != masks.shape[0]:
            masks = np.broadcast_to(masks, action.shape)

        for idx, act in enumerate(action):
            if not masks[idx, int(act)]:
                valid = np.flatnonzero(masks[idx])
                if valid.size > 0:
                    action[idx] = int(valid[0])

        if single:
            return int(action[0]), state
        return action, state


__all__ = ["MaskablePPO"]