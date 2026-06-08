"""Top-K action mask wrapper.

원래 RebalanceEnv는 트럭 현재 위치 등을 제외한 valid 정류소 ~145개를 모두 후보로
제공. Top-K wrapper는 휴리스틱(most_imbalanced)이 쓰는 score로 valid 후보 중
상위 K개만 True로 좁힘 → 정책이 K개 안에서 선택.

휴리스틱 score는 트럭 적재 상태에 따라:
  - load == 0: bikes - target (잉여 큰 정류소 우선 — 적재)
  - load == capacity: target - bikes (부족 큰 정류소 우선 — 하차)
  - 그 외: |bikes - target| (불균형 큰 정류소)

평가 시 MaskedEvalCallback이 wrapper.action_masks()를 호출해 predict에 넘김 →
deterministic argmax가 K개 안에서 선택. 학습 rollout에서 PPO가 무효 action을
샘플링하더라도 env.step()이 그대로 받아 처리(기존 동작 유지).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class TopKMaskWrapper(gym.Wrapper):
    """현재 트럭 상태에 따른 휴리스틱 score 기준 상위 K개 정류소만 mask로 노출."""

    def __init__(self, env, k: int = 12):
        super().__init__(env)
        self.k = int(k)

    def action_masks(self) -> np.ndarray:
        base_mask = self.env.action_masks()
        valid_idx = np.where(base_mask)[0]
        if valid_idx.size <= self.k:
            return base_mask

        inner = self.env
        truck = inner.trucks[inner.current_truck]
        bikes = inner.bikes.astype(np.float32)
        target = inner.data.capacity.astype(np.float32) * inner.target_fill_ratio
        cap = inner.truck_capacity

        if truck.load == 0:
            score = bikes - target
        elif truck.load >= cap:
            score = target - bikes
        else:
            score = np.abs(bikes - target)

        valid_scores = score[valid_idx]
        # 큰 값(=most imbalanced) 상위 K개 인덱스 (valid_idx 내 위치)
        topk_local = np.argpartition(valid_scores, -self.k)[-self.k:]
        new_mask = np.zeros_like(base_mask)
        new_mask[valid_idx[topk_local]] = True
        return new_mask


__all__ = ["TopKMaskWrapper"]
