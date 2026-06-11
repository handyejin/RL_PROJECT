"""agent-local potential-based reward shaping wrapper.

원칙:
    팀 공통 RebalanceEnv의 reward 식은 수정하지 않는다.
    이 wrapper는 `src/agents/ours` 실험에서만 선택적으로 감싸서 사용한다.

PBRS:
    r'_t = r_t + scale * (gamma * Phi(s') - Phi(s))

여기서 Phi(s)는 정류소 재고가 target_fill_ratio에 가까울수록 0에 가까워지는
잠재함수다. reward 자체를 새로 설계하는 것이 아니라, sparse/delayed reward의
학습 신호를 조금 더 빠르게 주는 보조 신호로만 사용한다.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class AgentPotentialRewardWrapper(gym.Wrapper):
    """원본 reward에 agent-local PBRS 항을 더한다."""

    VALID_MODES = {"projected_imbalance"}

    def __init__(self, env, mode: str = "projected_imbalance", scale: float = 0.0, gamma: float = 0.99):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown agent shaping mode: {mode}")
        super().__init__(env)
        self.mode = mode
        self.scale = float(scale)
        self.gamma = float(gamma)
        self._last_phi = 0.0
        self.cum_agent_shaping = 0.0

    def __getattr__(self, name):
        """wrapper에 없는 속성은 원본 env로 위임한다."""
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """episode 시작 시점의 potential을 저장한다."""
        obs, info = self.env.reset(*args, **kwargs)
        self._last_phi = self._potential()
        self.cum_agent_shaping = 0.0
        return obs, info

    def step(self, action):
        """원본 step reward에 PBRS 항을 더한다."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_now = self._potential()
        shaped = self.gamma * phi_now - self._last_phi
        bonus = self.scale * shaped
        self._last_phi = phi_now
        self.cum_agent_shaping += float(bonus)

        info = dict(info)
        info["cum_agent_shaping"] = round(self.cum_agent_shaping, 3)
        return obs, float(reward) + float(bonus), terminated, truncated, info

    def _potential(self) -> float:
        """Phi(s): target 재고와의 평균 차이를 음수로 둔다."""
        bikes = self.env.bikes.astype(np.float32)
        if self.mode == "projected_imbalance":
            bikes = self._projected_bikes_from_forecast(bikes)
        capacity = np.maximum(self.env.data.capacity.astype(np.float32), 1.0)
        target = capacity * self.env.target_fill_ratio
        imbalance = np.abs(bikes - target) / capacity
        return float(-np.mean(imbalance))

    def _projected_bikes_from_forecast(self, bikes: np.ndarray) -> np.ndarray:
        """forecast가 있으면 1시간 뒤 예상 재고로 potential을 계산한다."""
        forecast = getattr(self.env.data, "agent_demand_forecast", None)
        if forecast is None or len(forecast) == 0:
            return bikes
        idx = min(int(self.env.t), len(forecast) - 1)
        net = forecast[idx, :, 2].astype(np.float32)
        capacity = self.env.data.capacity.astype(np.float32)
        return np.clip(bikes + net, 0.0, capacity)


def maybe_wrap_agent_reward_shaping(env, args):
    """CLI 옵션에 따라 agent-local PBRS wrapper를 적용한다."""
    scale = float(getattr(args, "agent_shaping_scale", 0.0) or 0.0)
    if scale == 0.0:
        return env
    return AgentPotentialRewardWrapper(
        env,
        mode=getattr(args, "agent_shaping_mode", "projected_imbalance"),
        scale=scale,
        gamma=getattr(args, "agent_shaping_gamma", 0.99),
    )
