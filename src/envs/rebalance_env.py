"""따릉이 재배치 Gymnasium 환경.

Parameter sharing single-agent 형태:
- 1 RL step = 1 트럭의 1 결정 (다음 갈 정류소 선택)
- 환경이 turn을 관리 — agent는 항상 "지금 결정해야 할 1대 트럭"의 obs를 받음
- 모든 트럭이 이동 중이면 환경 시계를 진행시키며 demand replay와 도착 처리

State:
  [bike_ratio (N), truck_loc_idx_norm (n_trucks), truck_load_ratio (n_trucks),
   truck_remaining_steps_norm (n_trucks), current_truck_onehot (n_trucks),
   sin_t, cos_t, sin_day, cos_day]

Action: Discrete(N) — 다음 갈 정류소 idx (자기 위치 선택 시 머무름)

Reward (per RL step, 마지막 결정 이후 누적):
  stockout * w_stockout + full * w_full + travel_km * w_km + travel_step * w_step
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.data_loader import EpisodeData

logger = logging.getLogger(__name__)


@dataclass
class TruckState:
    location: int          # 현재(idle) 또는 출발 정류소 idx
    destination: int       # 목적지 정류소 idx (idle이면 location과 동일)
    load: int              # 적재 자전거 수
    remaining_steps: int   # 도착까지 남은 step (0이면 idle)

    @property
    def is_idle(self) -> bool:
        return self.remaining_steps == 0


class RebalanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_data: EpisodeData | list[EpisodeData],
        n_trucks: int = 3,
        truck_capacity: int = 20,
        target_fill_ratio: float = 0.5,
        w_stockout: float = -1.0,
        w_full: float = -0.8,
        w_travel_km: float = -0.01,
        w_travel_step: float = -0.005,
        max_travel_steps: int = 10,
        use_action_mask: bool = True,
        urgent_low_ratio: float = 0.0,    # bikes/capacity ≤ 이 비율이면 빈 위급
        urgent_high_ratio: float = 1.0,   # bikes/capacity ≥ 이 비율이면 가득 위급
        urgent_bonus: float = 0.0,        # 위급 정류소 도착 시 보너스 reward (shaping)
        strict_urgent_mask: bool = False, # True: 위급 정류소만 action 가능 (자기 위치 포함)
        explore_bonus_scale: float = 0.0, # 방문 빈도 기반 탐색 보너스 (count 적은 정류소 +)
        shaping_scale: float = 0.0,       # Potential-based shaping 스케일. 0=꺼짐.
        shaping_gamma: float = 0.99,      # shaping에 쓰는 γ (DQN gamma와 일치 권장)
        w_work_per_bike: float = 0.0,     # 적재/하차 1대당 양수 reward
        w_idle_visit: float = 0.0,        # 도착했는데 0대 옮긴 경우 페널티 (양수로 설정 → 음수로 적용)
        future_demand_horizon: int = 0,   # 0이면 비활성, >0이면 향후 N step 정류소별 net demand를 obs에 포함
        seed: int | None = None,
    ):
        super().__init__()
        if isinstance(episode_data, EpisodeData):
            self._episodes = [episode_data]
        else:
            if len(episode_data) == 0:
                raise ValueError("episode_data list cannot be empty")
            self._episodes = list(episode_data)
        # 모든 episode는 같은 정류소 셋 / 같은 N을 가져야 함 (마포구 고정)
        self.data = self._episodes[0]
        self.N = self.data.n_stations
        self.T = self.data.n_steps
        self.n_trucks = n_trucks
        self.truck_capacity = truck_capacity
        self.target_fill_ratio = target_fill_ratio
        self.w_stockout = w_stockout
        self.w_full = w_full
        self.w_travel_km = w_travel_km
        self.w_travel_step = w_travel_step
        self.max_travel_steps = max_travel_steps
        self.use_action_mask = use_action_mask
        self.urgent_low_ratio = urgent_low_ratio
        self.urgent_high_ratio = urgent_high_ratio
        self.urgent_bonus = urgent_bonus
        self.strict_urgent_mask = strict_urgent_mask
        self.explore_bonus_scale = explore_bonus_scale
        self.shaping_scale = shaping_scale
        self.shaping_gamma = shaping_gamma
        self.w_work_per_bike = w_work_per_bike
        self.w_idle_visit = w_idle_visit
        self.cum_urgent_bonus: float = 0.0
        self.cum_explore_bonus: float = 0.0
        self.cum_shaping: float = 0.0
        self.cum_work: float = 0.0
        self._last_potential: float = 0.0
        # visit_count는 reset에서 초기화

        self.action_space = spaces.Discrete(self.N)
        # 미래 demand horizon: 0이면 비활성, >0이면 향후 N step의 정류소별 net flow를 obs에 추가
        self.future_demand_horizon = future_demand_horizon
        obs_dim = (
            self.N                    # 정류소 자전거 비율
            + self.n_trucks            # 트럭 위치 idx 정규화
            + self.n_trucks            # 트럭 적재 비율
            + self.n_trucks            # 트럭 남은 이동 step 정규화
            + self.n_trucks            # current truck one-hot
            + 4                        # sin/cos hour & day-of-episode
            + 5                        # 캘린더: sin/cos(dow), is_weekend, is_holiday, is_holiday_eve
            + 4                        # 날씨: temp, precip, wind, humidity (정규화)
            + (self.N if self.future_demand_horizon > 0 else 0)  # 미래 demand (옵션)
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        # 상태 변수는 reset()에서 초기화
        self.bikes: np.ndarray = np.zeros(self.N, dtype=np.int32)
        self.trucks: list[TruckState] = []
        self.t: int = 0
        self.current_truck: int = 0
        self.cum_stockout: int = 0
        self.cum_full: int = 0
        self.cum_travel_km: float = 0.0
        self.cum_travel_steps: int = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # episode 선택: options에 episode_idx가 있으면 명시 선택 (deterministic 평가용),
        # 아니면 random 회전 (학습 시 다양한 날짜 노출)
        if options and "episode_idx" in options:
            self.data = self._episodes[options["episode_idx"]]
        elif len(self._episodes) > 1:
            idx = int(self._rng.integers(len(self._episodes)))
            self.data = self._episodes[idx]

        self.bikes = self.data.initial_bikes.copy()
        truck_starts = self._rng.choice(self.N, size=self.n_trucks, replace=True)
        self.trucks = [
            TruckState(location=int(s), destination=int(s), load=0, remaining_steps=0)
            for s in truck_starts
        ]
        self.t = 0
        self.cum_stockout = 0
        self.cum_full = 0
        self.cum_travel_km = 0.0
        self.cum_travel_steps = 0
        self.cum_urgent_bonus = 0.0
        self.cum_explore_bonus = 0.0
        self.cum_shaping = 0.0
        self.cum_work = 0.0
        self.visit_count = np.zeros(self.N, dtype=np.int32)
        self._last_potential = self._potential()

        # episode 시작 시점에는 모든 트럭이 idle → 0번 트럭부터 결정
        self.current_truck = 0
        return self._get_obs(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        truck = self.trucks[self.current_truck]
        assert truck.is_idle, "current truck must be idle when step() is called"

        # 1) 현재 트럭에 action 적용 (출발)
        dest = int(action)
        reward = 0.0
        if dest == truck.location:
            # 자기 위치 머무름 — 이동 비용 없음. 다음 결정까지 1 step 후로 미룸.
            truck.remaining_steps = 1
            truck.destination = dest
        else:
            steps = int(self.data.travel_steps[truck.location, dest])
            steps = max(steps, 1)
            km = float(self.data.distance_matrix[truck.location, dest])
            truck.destination = dest
            truck.remaining_steps = steps
            # 이동 거리 비용 즉시 청구 (step 비용은 진행 중 누적)
            reward += self.w_travel_km * km
            self.cum_travel_km += km

        # 2) 다음 idle 트럭이 생길 때까지 시간 진행
        reward_advance, done = self._advance_until_next_decision()
        reward += reward_advance

        # 3) Potential-based reward shaping (Ng et al. 1999, policy invariance 보장)
        #    shaped = γ·Φ(s') − Φ(s), Φ(s) = -Σ|bikes - target|
        if self.shaping_scale != 0.0:
            phi_now = self._potential()
            shaped = self.shaping_gamma * phi_now - self._last_potential
            reward += self.shaping_scale * shaped
            self.cum_shaping += self.shaping_scale * shaped
            self._last_potential = phi_now

        truncated = False
        obs = self._get_obs()
        info = self._info()
        return obs, float(reward), done, truncated, info

    def _potential(self) -> float:
        """Φ(s) = -Σ|bikes_i - target_i|. 균형 잡힐수록 0에 가까움 (최대값=0).

        Potential-based reward shaping의 잠재함수.
        모든 정류소가 capacity*target_fill_ratio에 가까울수록 좋은 상태.
        """
        target = self.data.capacity.astype(np.float32) * self.target_fill_ratio
        return float(-np.sum(np.abs(self.bikes.astype(np.float32) - target)))

    # ------------------------------------------------------------------
    # 내부 시뮬레이션
    # ------------------------------------------------------------------
    def _advance_until_next_decision(self) -> tuple[float, bool]:
        """다음 결정 시점까지 환경 시계를 진행. (누적 reward, done) 반환.

        SMDP 트리거: idle 트럭이 있어도 위급 정류소가 없으면 시계만 흐름 (DQN 호출 skip).
        urgent_low_ratio=0, urgent_high_ratio=1이면 모든 정류소가 항상 위급 → 기존 동작.
        """
        reward = 0.0
        while True:
            next_idle = self._pick_next_idle(exclude=self.current_truck)
            # 트리거 조건: idle 트럭 존재 AND (위급 정류소 존재 OR 모든 트럭 idle)
            if next_idle is not None and self._needs_decision():
                self.current_truck = next_idle
                return reward, False

            # 트리거 미충족 → 시간 1 step 진행
            if self.t >= self.T:
                return reward, True
            reward += self._tick()
            if self.t >= self.T:
                return reward, True

    def _needs_decision(self) -> bool:
        """위급 정류소(빈 위험·가득 위험)가 하나라도 있는가."""
        cap = self.data.capacity
        ratio = self.bikes / np.maximum(cap, 1)
        low_thr = self.urgent_low_ratio
        high_thr = self.urgent_high_ratio
        # 0/1 트리비얼 임계치는 사실상 "항상 위급"으로 fast-path
        if low_thr <= 0.0 and high_thr >= 1.0:
            return True
        return bool(((ratio <= low_thr) | (ratio >= high_thr)).any())

    def _pick_next_idle(self, exclude: int) -> int | None:
        for i, tr in enumerate(self.trucks):
            if i == exclude:
                continue
            if tr.is_idle:
                return i
        # 방금 출발한 트럭 자신도 다시 idle이면(머무름) 자기 차례
        if self.trucks[exclude].is_idle:
            return exclude
        return None

    def _tick(self) -> float:
        """환경 1 step 진행: demand replay + 트럭 이동/도착. reward 반환."""
        reward = 0.0

        # demand 처리
        rentals = self.data.rentals[self.t]
        returns = self.data.returns[self.t]

        served_rent = np.minimum(rentals, self.bikes)
        stockout = (rentals - served_rent).sum()
        self.bikes -= served_rent

        available_space = self.data.capacity - self.bikes
        served_return = np.minimum(returns, available_space)
        full = (returns - served_return).sum()
        self.bikes += served_return

        self.cum_stockout += int(stockout)
        self.cum_full += int(full)
        reward += self.w_stockout * stockout + self.w_full * full

        # 트럭 이동 진행 + 도착 처리
        for tr in self.trucks:
            if tr.is_idle:
                continue
            reward += self.w_travel_step  # 이동 중 step 비용
            self.cum_travel_steps += 1
            tr.remaining_steps -= 1
            if tr.remaining_steps == 0:
                tr.location = tr.destination
                bonus = self._apply_rebalance(tr)  # urgent + explore 보너스 (내부에서 누적)
                reward += bonus

        self.t += 1
        return reward

    def _apply_rebalance(self, truck: TruckState) -> float:
        """도착 정류소에서 적재/하차 휴리스틱 실행.

        @return urgent_bonus + explore_bonus (둘 다 0이면 0.0).
        """
        s = truck.location
        cap = int(self.data.capacity[s])
        target = int(cap * self.target_fill_ratio)
        current = int(self.bikes[s])

        # 위급 정류소 도착 보너스 (도착 시점 비율 기준)
        urgent = 0.0
        if self.urgent_bonus != 0.0 and cap > 0:
            ratio = current / cap
            if ratio <= self.urgent_low_ratio or ratio >= self.urgent_high_ratio:
                urgent = self.urgent_bonus
        self.cum_urgent_bonus += urgent

        # 탐색 보너스 — 방문 횟수가 적은 정류소에 가중치 (1/sqrt(n))
        self.visit_count[s] += 1
        explore = 0.0
        if self.explore_bonus_scale != 0.0:
            explore = self.explore_bonus_scale / float(np.sqrt(self.visit_count[s]))
            self.cum_explore_bonus += explore

        qty_moved = 0  # 실제 옮긴 자전거 수 (적재/하차 통합)
        if current > target:
            qty_moved = min(current - target, self.truck_capacity - truck.load)
            self.bikes[s] -= qty_moved
            truck.load += qty_moved
        elif current < target:
            qty_moved = min(target - current, truck.load, self.data.capacity[s] - current)
            self.bikes[s] += qty_moved
            truck.load -= qty_moved

        # work reward: 옮긴 양에 비례한 양수, 허탕(qty=0) 시 페널티
        work = self.w_work_per_bike * qty_moved
        if qty_moved == 0:
            work -= self.w_idle_visit
        self.cum_work += work

        return urgent + explore + work

    # ------------------------------------------------------------------
    # Observation / Info
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        bike_ratio = self.bikes.astype(np.float32) / self.data.capacity.astype(np.float32)
        loc_norm = np.array(
            [tr.location / max(self.N - 1, 1) for tr in self.trucks], dtype=np.float32
        )
        load_ratio = np.array(
            [tr.load / self.truck_capacity for tr in self.trucks], dtype=np.float32
        )
        rem_norm = np.array(
            [min(tr.remaining_steps, self.max_travel_steps) / self.max_travel_steps
             for tr in self.trucks], dtype=np.float32
        )
        cur_onehot = np.zeros(self.n_trucks, dtype=np.float32)
        cur_onehot[self.current_truck] = 1.0

        # 시간 인코딩 — episode 내 진행도 + 하루 중 시각
        episode_frac = self.t / max(self.T, 1)
        hour_frac = (self.t * 10 / 60) % 24 / 24  # 10분 step → 시각
        time_enc = np.array(
            [np.sin(2 * np.pi * hour_frac), np.cos(2 * np.pi * hour_frac),
             np.sin(2 * np.pi * episode_frac), np.cos(2 * np.pi * episode_frac)],
            dtype=np.float32,
        )

        # 캘린더 인코딩 — episode 날짜 기준 (24h 내 변화 없음)
        dow = getattr(self.data, "dayofweek", 0)
        cal_enc = np.array(
            [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
             1.0 if getattr(self.data, "is_weekend", False) else 0.0,
             1.0 if getattr(self.data, "is_holiday", False) else 0.0,
             1.0 if getattr(self.data, "is_holiday_eve", False) else 0.0],
            dtype=np.float32,
        )

        # 날씨 인코딩 — 현재 step의 날씨 (temp, precip, wind, humidity 정규화)
        w_arr = getattr(self.data, "weather", None)
        if w_arr is not None and len(w_arr) > 0:
            idx = min(self.t, len(w_arr) - 1)
            w = w_arr[idx]
            weather_enc = np.array(
                [(w[0] + 20.0) / 60.0,           # temp_c: -20~40 → 0~1
                 min(w[1] / 30.0, 1.0),          # precip_mm: 0~30 → 0~1 (clip)
                 min(w[2] / 10.0, 1.0),          # wind_ms: 0~10 → 0~1 (clip)
                 w[3] / 100.0],                  # humidity_pct: 0~100 → 0~1
                dtype=np.float32,
            )
        else:
            weather_enc = np.zeros(4, dtype=np.float32)

        parts = [bike_ratio, loc_norm, load_ratio, rem_norm, cur_onehot,
                 time_enc, cal_enc, weather_enc]

        # 미래 demand 인코딩 — horizon>0일 때만 포함 (옵션)
        if self.future_demand_horizon > 0:
            H = self.future_demand_horizon
            t_end = min(self.t + H, self.T)
            if t_end > self.t:
                future_rent = self.data.rentals[self.t:t_end].sum(axis=0)
                future_ret = self.data.returns[self.t:t_end].sum(axis=0)
                future_net = (future_ret - future_rent).astype(np.float32)
                cap_f = self.data.capacity.astype(np.float32)
                future_net_norm = future_net / np.maximum(cap_f, 1.0)
                future_enc = np.clip(future_net_norm, -1.0, 1.0)
            else:
                future_enc = np.zeros(self.N, dtype=np.float32)
            parts.append(future_enc)

        return np.concatenate(parts)

    def action_masks(self) -> np.ndarray:
        """이동 가능한 정류소 마스크. sb3-contrib 컨벤션.

        - 자기 현재 위치(stay)는 기본 차단 — trivial stay 솔루션 회피.
        - 다른 트럭이 in-flight로 향하고 있는 목적지는 차단 (중복 작업 회피).
        - strict_urgent_mask=True면 위급 정류소만 허용.
        - 안전장치: 갈 곳이 정말 없을 때만 자기 위치 fallback.
        - use_action_mask=False면 항상 all-ones.
        """
        mask = np.ones(self.N, dtype=bool)
        if not self.use_action_mask:
            return mask
        for i, tr in enumerate(self.trucks):
            if i == self.current_truck or tr.is_idle:
                continue
            mask[tr.destination] = False

        # 자기 위치 stay 차단 — trivial 솔루션(한 곳에 모여 영원히 머무름) 직접 회피
        self_loc = self.trucks[self.current_truck].location
        mask[self_loc] = False

        # strict 모드: 위급 정류소만 가능
        if self.strict_urgent_mask:
            cap = self.data.capacity
            ratio = self.bikes / np.maximum(cap, 1)
            urgent = (ratio <= self.urgent_low_ratio) | (ratio >= self.urgent_high_ratio)
            mask &= urgent

        # 안전장치: 모두 막혔으면 자기 위치는 강제 허용 (fallback)
        if not mask.any():
            mask[self_loc] = True
        return mask

    # 구버전 호환 alias
    get_action_mask = action_masks

    def _info(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "current_truck": self.current_truck,
            "cum_stockout": self.cum_stockout,
            "cum_full": self.cum_full,
            "cum_travel_km": round(self.cum_travel_km, 3),
            "cum_travel_steps": self.cum_travel_steps,
            "cum_urgent_bonus": round(self.cum_urgent_bonus, 3),
            "cum_explore_bonus": round(self.cum_explore_bonus, 3),
            "cum_shaping": round(self.cum_shaping, 3),
            "cum_work": round(self.cum_work, 3),
        }
