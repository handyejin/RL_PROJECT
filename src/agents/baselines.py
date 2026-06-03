"""베이스라인 정책 (Phase 3).

RL과 비교할 비학습 정책. 환경 내부 상태(env.bikes, env.trucks 등)를 직접 보고 결정.

- NoopPolicy: 트럭이 자기 위치에 머무름 (재배치 없음)
- MostImbalancedPolicy: 트럭 적재량에 따라 가장 균형 어긋난 정류소로 이동
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.envs.rebalance_env import RebalanceEnv


class BasePolicy(ABC):
    name: str = "base"

    @abstractmethod
    def act(self, env: RebalanceEnv) -> int:
        """현재 결정 트럭(env.current_truck)이 갈 정류소 idx 반환."""
        ...


class NoopPolicy(BasePolicy):
    """현재 위치에 머무름."""

    name = "noop"

    def act(self, env: RebalanceEnv) -> int:
        return env.trucks[env.current_truck].location


class MostImbalancedPolicy(BasePolicy):
    """탐욕적 균형 정책.

    - 트럭 비어있음(load==0): 가장 잉여(bikes - target)가 큰 정류소로 → 적재
    - 트럭 가득(load==capacity): 가장 부족(target - bikes)이 큰 정류소로 → 하차
    - 부분 적재: 절대 불균형(|bikes - target|) 가장 큰 정류소로

    다른 트럭의 목적지는 제외하여 중복 이동 방지.
    """

    name = "most_imbalanced"

    def act(self, env: RebalanceEnv) -> int:
        truck = env.trucks[env.current_truck]
        bikes = env.bikes.astype(np.float32)
        target = env.data.capacity.astype(np.float32) * env.target_fill_ratio

        if truck.load == 0:
            scores = bikes - target  # 잉여 큰 곳일수록 높음
        elif truck.load >= env.truck_capacity:
            scores = target - bikes  # 부족 큰 곳일수록 높음
        else:
            scores = np.abs(bikes - target).astype(np.float32)

        # 자기 위치 + 다른 트럭 목적지 제외
        scores = scores.copy()
        scores[truck.location] = -np.inf
        for i, other in enumerate(env.trucks):
            if i == env.current_truck:
                continue
            if not other.is_idle:
                scores[other.destination] = -np.inf

        best = int(np.argmax(scores))
        # 모든 후보가 -inf면 (이론상 거의 없음) 자기 위치 머무름
        if not np.isfinite(scores[best]):
            return truck.location
        return best


POLICY_REGISTRY: dict[str, type[BasePolicy]] = {
    NoopPolicy.name: NoopPolicy,
    MostImbalancedPolicy.name: MostImbalancedPolicy,
}


def get_policy(name: str) -> BasePolicy:
    if name not in POLICY_REGISTRY:
        raise ValueError(f"unknown policy '{name}'. choices: {list(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name]()
