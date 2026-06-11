"""미래 수요 관측값을 추가하는 agent 전용 wrapper.

공통 환경(`src/envs/rebalance_env.py`)은 수정하지 않고, agent가 받는
observation 뒤에 미래 수요 feature만 덧붙인다.

State 확장:
    s'_t = concat(s_t, future_demand_t)

미래 수요 feature:
    - oracle_net:
        향후 H개 step 동안 정류소별 (returns - rentals)을 capacity로 나눈 값
    - oracle_inout:
        향후 H개 step 동안 정류소별 rentals, returns를 각각 capacity로 나눈 값
    - history_net:
        학습 episode들에서 같은 요일/시간대의 평균 (returns - rentals)을 만든 뒤,
        현재 시점부터 H개 step의 평균 net demand를 사용한다.
    - history_projected_travel:
        history_net으로 예상 재고 편차, 현재 트럭에서 각 정류소까지의 이동 시간,
        heuristic residual score를 함께 제공한다.
    - forecast_net / forecast_inout / forecast_projected_travel:
        외부 예측 parquet의 1시간 예상 대여/반납을 사용한다.

주의:
    oracle 모드는 실제 미래 값을 보는 upper-bound 실험이다. 실제 배포용
    예측 모델이 아니라, "미래 수요 정보가 있으면 성능이 좋아지는가"를
    확인하기 위한 고도화 실험으로 사용한다.
    history_net은 평가 episode의 미래 값을 직접 보지 않는 현실형 feature다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class DemandForecastProvider:
    """정류소별 1시간 대여/반납 예측 parquet을 episode 격자로 변환한다.

    입력 parquet 형식:
        t, station_id, pred_rentals_1h, pred_returns_1h, pred_net_1h

    pred_net_1h는 pred_returns_1h - pred_rentals_1h 이다.
    이 값은 실제 재고가 아니라, 현재 재고에 더해볼 수 있는 미래 수요 압력이다.
    """

    COLUMNS = ["pred_rentals_1h", "pred_returns_1h", "pred_net_1h"]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"forecast file not found: {self.path}")
        frame = pd.read_parquet(self.path)
        required = {"t", "station_id", *self.COLUMNS}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"forecast file missing columns: {sorted(missing)}")
        frame = frame[["t", "station_id", *self.COLUMNS]].copy()
        frame["t"] = pd.to_datetime(frame["t"])
        frame = frame.drop_duplicates(["t", "station_id"], keep="last")
        self.frame = frame.set_index(["t", "station_id"]).sort_index()

    def grid_for_episode(self, episode) -> tuple[np.ndarray, int, int]:
        """episode의 timestamp/station 순서에 맞춘 (T, N, 3) forecast grid를 만든다."""
        index = pd.MultiIndex.from_product(
            [episode.timestamps, episode.station_ids],
            names=["t", "station_id"],
        )
        sub = self.frame.reindex(index)
        missing_rows = int(sub[self.COLUMNS[0]].isna().sum())
        total_rows = int(len(sub))
        values = sub[self.COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
        grid = values.reshape(len(episode.timestamps), len(episode.station_ids), len(self.COLUMNS))
        return grid.astype(np.float32), total_rows - missing_rows, total_rows


def build_history_net_profile(episodes: list) -> dict[str, np.ndarray]:
    """과거 episode들로 요일/시간대별 평균 net demand profile을 만든다.

    반환값:
        by_dow: (7, T, N) 요일별 평균 returns-rentals
        overall: (T, N) 전체 날짜 평균 returns-rentals

    같은 요일 데이터가 없으면 overall profile로 fallback한다.
    """
    if not episodes:
        raise ValueError("history profile needs at least one episode")

    T = episodes[0].n_steps
    N = episodes[0].n_stations
    sums = np.zeros((7, T, N), dtype=np.float32)
    counts = np.zeros(7, dtype=np.float32)
    overall_sum = np.zeros((T, N), dtype=np.float32)
    overall_count = 0.0

    for ep in episodes:
        if ep.n_steps != T or ep.n_stations != N:
            raise ValueError("history profile episodes must share T and N")
        net = (ep.returns - ep.rentals).astype(np.float32)
        dow = int(getattr(ep, "dayofweek", 0))
        sums[dow] += net
        counts[dow] += 1.0
        overall_sum += net
        overall_count += 1.0

    by_dow = np.zeros_like(sums)
    for dow in range(7):
        if counts[dow] > 0:
            by_dow[dow] = sums[dow] / counts[dow]
    overall = overall_sum / max(overall_count, 1.0)
    return {"by_dow": by_dow, "overall": overall, "counts": counts}


class FutureDemandObservationWrapper(gym.Wrapper):
    """RebalanceEnv의 observation을 미래 수요 feature로 확장한다."""

    VALID_MODES = {
        "none",
        "oracle_net",
        "oracle_inout",
        "history_net",
        "history_projected_travel",
        "forecast_net",
        "forecast_inout",
        "forecast_projected_travel",
    }

    def __init__(self, env, mode: str = "none", horizon: int = 6, history_profile: dict | None = None):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown future demand mode: {mode}")
        if horizon < 1:
            raise ValueError("future demand horizon must be >= 1")
        super().__init__(env)
        self.mode = mode
        self.horizon = int(horizon)
        self.history_profile = history_profile

        base_space = env.observation_space
        # *_net은 정류소별 net demand 1개씩 추가하고,
        # oracle_inout은 rentals/returns를 각각 추가하므로 2N개가 늘어난다.
        # *_projected_travel은 net, projected deviation, travel, residual score를 추가한다.
        if mode == "none":
            extra_dim = 0
        elif mode in {"history_projected_travel", "forecast_projected_travel"}:
            extra_dim = env.N * 4
        elif mode.endswith("_net"):
            extra_dim = env.N
        else:
            extra_dim = env.N * 2

        low = np.concatenate(
            [
                np.asarray(base_space.low, dtype=np.float32),
                np.full(extra_dim, -1.0, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.asarray(base_space.high, dtype=np.float32),
                np.full(extra_dim, 1.0, dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.action_space

    def __getattr__(self, name):
        """wrapper에 없는 속성은 원본 env로 위임한다."""
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """원본 env reset 후 observation만 확장한다."""
        obs, info = self.env.reset(*args, **kwargs)
        return self._augment(obs), info

    def step(self, action):
        """원본 env step 후 next observation만 확장한다."""
        obs, reward, done, truncated, info = self.env.step(action)
        return self._augment(obs), reward, done, truncated, info

    def action_masks(self):
        """action mask는 원본 env의 규칙을 그대로 사용한다."""
        return self.env.action_masks()

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return obs
        return np.concatenate([obs.astype(np.float32, copy=False), self._future_features()])

    def _future_features(self) -> np.ndarray:
        capacity = np.maximum(self.env.data.capacity.astype(np.float32), 1.0)
        if self.mode == "history_projected_travel":
            return self._projected_travel_features(capacity, self._history_window())
        if self.mode == "forecast_projected_travel":
            _, _, net = self._forecast_values()
            return self._projected_travel_features(capacity, net)

        # net demand = returns - rentals
        # 음수면 대여가 더 많아 곧 자전거 부족 가능성이 크고,
        # 양수면 반납이 더 많아 거치대 부족 가능성이 크다.
        if self.mode.endswith("_net"):
            if self.mode == "history_net":
                net = self._history_window()
            elif self.mode == "forecast_net":
                _, _, net = self._forecast_values()
            else:
                rentals, returns = self._oracle_window()
                net = returns - rentals
            net = net / capacity
            return np.clip(net, -1.0, 1.0).astype(np.float32)

        # in/out 모드는 대여량과 반납량을 분리해서 agent에게 제공한다.
        if self.mode == "forecast_inout":
            rentals, returns, _ = self._forecast_values()
        else:
            rentals, returns = self._oracle_window()
        rentals_norm = np.clip(rentals / capacity, 0.0, 1.0)
        returns_norm = np.clip(returns / capacity, 0.0, 1.0)
        return np.concatenate([rentals_norm, returns_norm]).astype(np.float32)

    def _oracle_window(self) -> tuple[np.ndarray, np.ndarray]:
        """현재 시점부터 horizon까지의 실제 rentals/returns 합을 반환한다."""
        t_end = min(self.env.t + self.horizon, self.env.T)
        if t_end <= self.env.t:
            zeros = np.zeros(self.env.N, dtype=np.float32)
            return zeros, zeros
        rentals = self.env.data.rentals[self.env.t:t_end].sum(axis=0).astype(np.float32)
        returns = self.env.data.returns[self.env.t:t_end].sum(axis=0).astype(np.float32)
        return rentals, returns

    def _history_window(self) -> np.ndarray:
        """과거 profile에서 현재 요일/시간대의 평균 net demand를 가져온다."""
        if not self.history_profile:
            return np.zeros(self.env.N, dtype=np.float32)

        by_dow = self.history_profile["by_dow"]
        overall = self.history_profile["overall"]
        counts = self.history_profile["counts"]
        dow = int(getattr(self.env.data, "dayofweek", 0))
        t_end = min(self.env.t + self.horizon, self.env.T)
        if t_end <= self.env.t:
            return np.zeros(self.env.N, dtype=np.float32)

        # 같은 요일 데이터가 있으면 요일별 평균, 없으면 전체 평균을 사용한다.
        source = by_dow[dow] if counts[dow] > 0 else overall
        return source[self.env.t:t_end].sum(axis=0).astype(np.float32)

    def _forecast_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """외부 예측 parquet에서 현재 step의 1시간 예상 대여/반납을 가져온다."""
        forecast = getattr(self.env.data, "agent_demand_forecast", None)
        if forecast is None or len(forecast) == 0:
            zeros = np.zeros(self.env.N, dtype=np.float32)
            return zeros, zeros, zeros
        idx = min(int(self.env.t), len(forecast) - 1)
        current = forecast[idx].astype(np.float32)
        rentals = current[:, 0]
        returns = current[:, 1]
        net = current[:, 2]
        return rentals, returns, net

    def _projected_travel_features(self, capacity: np.ndarray, net: np.ndarray) -> np.ndarray:
        """예상 불균형, 이동 시간, residual policy용 score를 함께 만든다."""
        bikes = self.env.bikes.astype(np.float32)
        target = capacity * self.env.target_fill_ratio
        projected_bikes = np.clip(bikes + net, 0.0, capacity)

        # net_norm: 앞으로 1시간 동안 평균적으로 자전거가 늘/줄 방향.
        history_net_norm = np.clip(net / capacity, -1.0, 1.0)

        # projected_deviation: 예상 재고가 목표보다 얼마나 높은/낮은지.
        projected_deviation = np.clip((projected_bikes - target) / capacity, -1.0, 1.0)

        truck = self.env.trucks[self.env.current_truck]
        travel = self.env.data.travel_steps[truck.location].astype(np.float32)
        max_travel = max(float(getattr(self.env, "max_travel_steps", 10)), 1.0)
        travel_norm = np.clip(travel / max_travel, 0.0, 1.0)

        # residual_score는 baseline heuristic의 station scoring을 정규화한 값이다.
        # policy는 이 점수 위에 learned residual을 더해 baseline에서 조금씩 벗어난다.
        if truck.load == 0:
            score = projected_bikes - target
        elif truck.load >= self.env.truck_capacity:
            score = target - projected_bikes
        else:
            score = np.abs(projected_bikes - target)
        score = score.astype(np.float32)
        score = score - np.nanmean(score)
        scale = float(np.nanstd(score) + 1e-6)
        residual_score = np.clip(score / scale, -3.0, 3.0) / 3.0

        return np.concatenate(
            [
                history_net_norm.astype(np.float32),
                projected_deviation.astype(np.float32),
                travel_norm.astype(np.float32),
                residual_score.astype(np.float32),
            ]
        )


def maybe_wrap_future_demand(env, args):
    """CLI 옵션에 따라 env를 미래 수요 wrapper로 감싼다."""
    mode = getattr(args, "future_mode", "none")
    if mode == "none":
        return env
    return FutureDemandObservationWrapper(
        env,
        mode=mode,
        horizon=getattr(args, "future_horizon", 6),
        history_profile=getattr(args, "history_profile", None),
    )
