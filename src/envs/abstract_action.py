"""추상 action wrapper — 146지선다(정류소)를 소수의 "의도(intent)"로 축소.

RL은 "어디로?" 대신 "어떤 전략?"만 학습한다. 각 의도는 현재 상태를 보고
결정적 규칙으로 구체 정류소 인덱스로 변환된다. 규칙은 MostImbalancedPolicy가
쓰는 primitive를 그대로 재사용한다.

ACTIONS (기본 5개):
  0 stay            현재 위치 유지 (이동 안 함)
  1 most_surplus    (bikes - target) 최대 정류소 → 자전거 수거
  2 most_deficit    (target - bikes) 최대 정류소 → 자전거 배달
  3 nearest_urgent  가장 가까운 위급(빈/꽉찬) 정류소
  4 most_imbalanced |bikes - target| 최대 정류소

모든 후보에서 자기 위치 + 다른 트럭 목적지를 제외해 중복 이동을 막는다
(휴리스틱과 동일한 조정 로직).

masking: env_method("action_masks")로 마스킹 에이전트와 호환. 크기는 n_actions.
대상 정류소가 없는 의도는 마스킹되고, stay는 항상 허용.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 의도 라벨 (로깅/replay용)
# index 5 predictive: 미래 H스텝 후 예상 불균형 큰 곳 선점 (반응형보다 선행)
ACTION_NAMES = ["stay", "most_surplus", "most_deficit", "nearest_urgent",
                "most_imbalanced", "predictive"]


class AbstractActionWrapper(gym.ActionWrapper):
    def __init__(self, env, pred_horizon: int = 3,
                 forecast_rent=None, forecast_ret=None):
        super().__init__(env)
        self.n_actions = len(ACTION_NAMES)
        self.action_space = spaces.Discrete(self.n_actions)
        # predictive 의도용 — forecast_* 주어지면 그걸로, 아니면 oracle(env.data 실제 미래)
        self.pred_horizon = int(pred_horizon)
        self.forecast_rent = forecast_rent
        self.forecast_ret = forecast_ret

    # ------------------------------------------------------------------
    # 상태 헬퍼
    # ------------------------------------------------------------------
    def _base(self):
        """RebalanceEnv 본체 (중첩 wrapper 관통)."""
        e = self.env
        while not hasattr(e, "trucks"):
            e = e.env
        return e

    def _excluded(self, env):
        """자기 위치 + 다른 트럭 목적지 인덱스 집합."""
        truck = env.trucks[env.current_truck]
        ex = {int(truck.location)}
        for i, other in enumerate(env.trucks):
            if i == env.current_truck:
                continue
            if not other.is_idle:
                ex.add(int(other.destination))
        return ex

    def _target_station(self, abstract_action: int):
        """의도 → 구체 정류소 인덱스. 유효 후보 없으면 None."""
        env = self._base()
        truck = env.trucks[env.current_truck]
        if abstract_action == 0:  # stay
            return int(truck.location)

        bikes = env.bikes.astype(np.float32)
        target = env.data.capacity.astype(np.float32) * env.target_fill_ratio

        if abstract_action == 1:      # most_surplus
            scores = bikes - target
        elif abstract_action == 2:    # most_deficit
            scores = target - bikes
        elif abstract_action == 4:    # most_imbalanced
            scores = np.abs(bikes - target)
        elif abstract_action == 3:    # nearest_urgent
            cap = np.maximum(env.data.capacity.astype(np.float32), 1)
            ratio = bikes / cap
            urgent = (ratio <= env.urgent_low_ratio) | (ratio >= env.urgent_high_ratio)
            dist = env.data.distance_matrix[int(truck.location)].astype(np.float32)
            scores = np.full(env.N, -np.inf, dtype=np.float32)
            scores[urgent] = -dist[urgent]  # 가까울수록 높은 점수
        elif abstract_action == 5:    # predictive — 미래 H스텝 후 예상 불균형
            H = self.pred_horizon
            T = env.data.rentals.shape[0]
            t_end = min(env.t + H, T)
            if self.forecast_rent is not None:        # forecast(과거평균) 사용
                fr = self.forecast_rent[env.t:t_end].sum(axis=0).astype(np.float32)
                fe = self.forecast_ret[env.t:t_end].sum(axis=0).astype(np.float32)
            else:                                     # oracle(실제 미래) 사용
                fr = env.data.rentals[env.t:t_end].sum(axis=0).astype(np.float32)
                fe = env.data.returns[env.t:t_end].sum(axis=0).astype(np.float32)
            predicted = bikes + (fe - fr)             # 개입 없을 때 도달할 상태
            if truck.load == 0:
                scores = predicted - target           # 곧 넘칠 곳 → 수거
            elif truck.load >= env.truck_capacity:
                scores = target - predicted           # 곧 소진될 곳 → 배달
            else:
                scores = np.abs(predicted - target)
        else:
            return None

        scores = scores.astype(np.float32).copy()
        for idx in self._excluded(env):
            scores[idx] = -np.inf

        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            return None
        return best

    # ------------------------------------------------------------------
    # ActionWrapper 인터페이스
    # ------------------------------------------------------------------
    def action(self, abstract_action):
        station = self._target_station(int(abstract_action))
        if station is None:  # 유효 후보 없으면 stay
            return int(self._base().trucks[self._base().current_truck].location)
        return station

    def action_masks(self) -> np.ndarray:
        """각 의도가 유효한지 (대상 정류소 존재). stay는 항상 True."""
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[0] = True  # stay
        for a in range(1, self.n_actions):
            mask[a] = self._target_station(a) is not None
        if not mask.any():
            mask[0] = True
        return mask


__all__ = ["AbstractActionWrapper", "ACTION_NAMES"]
