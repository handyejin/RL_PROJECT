"""RebalanceEnv 위에 Poisson 확률 수요를 얹는 서브클래스 (기존 env 무수정).

기존 `RebalanceEnv`는 기록된 실제 수요를 그대로 결정적으로 replay 한다
(`self.data.rentals[self.t]`). 결정적 replay에서는 그날 일어난 일이 고정이라
"미래를 아는" forecast 정보의 가치가 작다 → 반응형 휴리스틱과 격차가 잘 안 벌어진다.

이 서브클래스는 reset 시점에 rentals/returns 를 **Poisson(기록값)** 으로 다시
샘플링해 매 에피소드 다른 수요 실현(realization)을 만든다. forecast(예측 = 기댓값,
`agent_demand_forecast`)는 그대로 두고 **실현값만 확률화**하므로:

    - agent(state에 forecast 보유)는 "기대 분포"를 알고 사전 대비 가능
    - 휴리스틱(현재 재고만 봄)은 실현된 무작위성에 항상 사후 대응

→ 확률성 아래에서 forecast 기반 선제 라우팅의 이점이 드러난다(옛 SmallProblem 조건 재현).

원본 env/data_loader 는 일절 수정하지 않고, _tick 이 읽는 `self.data` 를
Poisson 복사본으로 바꾸는 방식만 사용한다.
"""

from __future__ import annotations

import copy

import numpy as np

from src.envs.rebalance_env import RebalanceEnv


class StochasticRebalanceEnv(RebalanceEnv):
    """reset마다 수요를 Poisson(기록값)으로 재샘플하는 RebalanceEnv."""

    def __init__(self, *args, demand_noise: str = "poisson", demand_rate_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if demand_noise not in {"none", "poisson"}:
            raise ValueError(f"unknown demand_noise: {demand_noise}")
        self.demand_noise = demand_noise
        self.demand_rate_scale = float(demand_rate_scale)

    def reset(self, seed=None, options=None):
        # 원본 reset이 episode를 고르고 self.data/bikes/_rng 등을 세팅한다.
        obs, info = super().reset(seed=seed, options=options)
        if self.demand_noise != "poisson":
            return obs, info

        # super().reset이 고른 self.data 는 공유 객체이므로 mutate 금지.
        # 얕은 복사로 동적 속성(agent_demand_forecast 등)을 보존한 뒤
        # rentals/returns 만 Poisson 재샘플 배열로 교체한다.
        rate_rent = np.clip(self.data.rentals.astype(np.float64) * self.demand_rate_scale, 0.0, None)
        rate_ret = np.clip(self.data.returns.astype(np.float64) * self.demand_rate_scale, 0.0, None)
        sampled = copy.copy(self.data)
        sampled.rentals = self._rng.poisson(rate_rent).astype(self.data.rentals.dtype)
        sampled.returns = self._rng.poisson(rate_ret).astype(self.data.returns.dtype)
        self.data = sampled
        # initial_bikes / capacity / _last_potential 은 수요와 무관 → 재계산 불필요.
        return obs, info
