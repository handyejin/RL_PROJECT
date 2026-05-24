"""RebalanceEnv 데이터 로더.

processed parquet에서 1 episode 분량의 정적 데이터를 묶어 환경에 공급한다.

- 정류소 마스터 → 자치구 필터, 거리/이동 step 행렬
- demand_10min → (T, N) shape의 rentals/returns 격자 (episode 슬라이싱)
- 초기 자전거 분포 추정 (data_based / uniform)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.geo import (
    compute_distance_matrix,
    distance_to_travel_steps,
)


@dataclass
class EpisodeData:
    station_ids: list[str]          # 길이 N
    station_coords: np.ndarray      # (N, 2) lat/lon
    distance_matrix: np.ndarray     # (N, N) km
    travel_steps: np.ndarray        # (N, N) int, 같은 정류소=0
    capacity: np.ndarray            # (N,) int — 일괄 20
    initial_bikes: np.ndarray       # (N,) int
    rentals: np.ndarray             # (T, N) int — 매 step 발생 대여 요청
    returns: np.ndarray             # (T, N) int — 매 step 발생 반납 요청
    timestamps: pd.DatetimeIndex    # 길이 T
    # 캘린더 feature — episode 시작 날짜 기준 (24h 내 변화 없으므로 scalar)
    dayofweek: int = 0              # 0=월 ~ 6=일
    is_weekend: bool = False        # 토/일
    is_holiday: bool = False        # 공휴일
    is_holiday_eve: bool = False    # 휴일 전날(평일 중 내일이 주말/공휴일)
    # 날씨 — (T, 4): temp_c, precip_mm, wind_ms, humidity_pct
    weather: np.ndarray | None = None

    @property
    def n_stations(self) -> int:
        return len(self.station_ids)

    @property
    def n_steps(self) -> int:
        return self.rentals.shape[0]


def _filter_stations(stations: pd.DataFrame, district: str) -> pd.DataFrame:
    mask = stations["gu"] == district
    if not mask.any():
        raise ValueError(f"district '{district}' not found in stations.gu")
    return stations[mask].sort_values("station_id").reset_index(drop=True)


def _build_demand_grid(
    demand: pd.DataFrame,
    station_ids: list[str],
    timestamps: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """sparse (t, station_id, rentals, returns) → dense (T, N) 두 행렬."""
    id_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    t_to_idx = {t: i for i, t in enumerate(timestamps)}

    sub = demand[
        demand["station_id"].isin(id_to_idx)
        & demand["t"].isin(t_to_idx)
    ]
    T, N = len(timestamps), len(station_ids)
    rentals = np.zeros((T, N), dtype=np.int32)
    returns = np.zeros((T, N), dtype=np.int32)

    if len(sub) > 0:
        ti = sub["t"].map(t_to_idx).to_numpy()
        si = sub["station_id"].map(id_to_idx).to_numpy()
        rentals[ti, si] = sub["rentals"].to_numpy()
        returns[ti, si] = sub["returns"].to_numpy()
    return rentals, returns


def _estimate_initial_bikes(
    capacity: np.ndarray,
    rentals: np.ndarray,
    returns: np.ndarray,
    mode: str,
    fill_ratio: float,
) -> np.ndarray:
    """초기 자전거 분포 추정.

    data_based: 첫 step의 (returns - rentals) net flow가 음수면 빈 자리,
                양수면 가득 찬 자리를 의미하므로 그걸 반영해 fill_ratio 주변에서 조정.
                MVP는 단순화: fill_ratio + 첫 6 step의 net flow / capacity 보정.
    uniform:    모든 정류소 capacity * fill_ratio.
    """
    N = len(capacity)
    if mode == "uniform":
        return (capacity * fill_ratio).astype(np.int32)

    # data_based — 첫 1시간(6 step)의 net flow로 시작 분포 추정
    horizon = min(6, rentals.shape[0])
    net = returns[:horizon].sum(axis=0) - rentals[:horizon].sum(axis=0)
    base = capacity * fill_ratio - net  # rental 많을수록 시작 자전거가 많이 있어야
    bikes = np.clip(base, 0, capacity).astype(np.int32)
    return bikes


def load_episode(
    processed_dir: Path | str,
    district: str = "마포구",
    episode_start: pd.Timestamp | str = "2025-01-01 00:00",
    episode_duration_min: int = 1440,
    step_duration_min: int = 10,
    capacity_per_station: int = 20,
    speed_kmh: float = 25.0,
    initial_distribution: str = "data_based",
    initial_fill_ratio: float = 0.5,
) -> EpisodeData:
    processed_dir = Path(processed_dir)
    episode_start = pd.Timestamp(episode_start)

    stations_all = pd.read_parquet(processed_dir / "stations.parquet")
    stations = _filter_stations(stations_all, district)

    station_ids = stations["station_id"].tolist()
    coords = stations[["lat", "lon"]].to_numpy(dtype=np.float64)
    dist = compute_distance_matrix(coords)
    travel = distance_to_travel_steps(dist, speed_kmh, step_duration_min)

    n_steps = episode_duration_min // step_duration_min
    timestamps = pd.date_range(
        start=episode_start,
        periods=n_steps,
        freq=f"{step_duration_min}min",
    )

    demand = pd.read_parquet(processed_dir / "demand_10min.parquet")
    rentals, returns = _build_demand_grid(demand, station_ids, timestamps)

    # 날씨 + 캘린더 슬라이스 (전처리 parquet에서 가져옴)
    weather_arr, cal_flags = _build_weather_and_calendar(processed_dir, timestamps)

    capacity = np.full(len(station_ids), capacity_per_station, dtype=np.int32)
    initial_bikes = _estimate_initial_bikes(
        capacity, rentals, returns, initial_distribution, initial_fill_ratio
    )

    # 캘린더 — episode 시작 시점 기준 (요일은 datetime에서, 나머지는 전처리 parquet에서)
    dow = int(episode_start.dayofweek)
    is_weekend = bool(cal_flags["is_weekend"])
    is_holiday = bool(cal_flags["is_holiday"])
    is_holiday_eve = bool(cal_flags["is_holiday_eve"])

    return EpisodeData(
        station_ids=station_ids,
        station_coords=coords,
        distance_matrix=dist,
        travel_steps=travel,
        capacity=capacity,
        initial_bikes=initial_bikes,
        rentals=rentals,
        returns=returns,
        timestamps=timestamps,
        dayofweek=dow,
        is_weekend=is_weekend,
        is_holiday=is_holiday,
        is_holiday_eve=is_holiday_eve,
        weather=weather_arr,
    )


# 날씨 정규화 범위 (config/default.yaml과 일치)
WEATHER_COLS = ("temp_c", "precip_mm", "wind_ms", "humidity_pct")
WEATHER_RANGE = {
    "temp_c": (-20.0, 40.0),       # → 0~1
    "precip_mm": (0.0, 30.0),      # → 0~1 (시간당 30mm는 폭우)
    "wind_ms": (0.0, 10.0),        # → 0~1 (10m/s는 강풍)
    "humidity_pct": (0.0, 100.0),  # → 0~1
}


CALENDAR_COLS = ("is_weekend", "is_holiday", "is_holiday_eve")


def _build_weather_and_calendar(
    processed_dir: Path,
    timestamps: pd.DatetimeIndex,
) -> tuple[np.ndarray, dict]:
    """weather_10min.parquet에서 episode 기간의 날씨 시계열 + 시작 시점 캘린더 추출.

    @return (weather_arr (T,4), calendar_flags dict)
    - 날씨: temp_c, precip_mm, wind_ms, humidity_pct (정규화 안 됨)
    - 캘린더: is_weekend, is_holiday, is_holiday_eve (episode 시작 시각 기준 1개 값)
    """
    weather_zero = np.zeros((len(timestamps), len(WEATHER_COLS)), dtype=np.float32)
    cal_zero = {c: False for c in CALENDAR_COLS}

    path = processed_dir / "weather_10min.parquet"
    if not path.exists():
        return weather_zero, cal_zero

    weather = pd.read_parquet(path).set_index("t")

    # 캘린더 — episode 시작 시각의 행 (없으면 가장 가까운 과거)
    cal_flags = dict(cal_zero)
    if all(c in weather.columns for c in CALENDAR_COLS):
        try:
            row = weather[list(CALENDAR_COLS)].reindex([timestamps[0]]).ffill()
            if row.isna().all().any():
                # 없으면 asof로 가장 가까운 과거
                idx = weather.index.asof(timestamps[0])
                if pd.notna(idx):
                    row = weather.loc[[idx], list(CALENDAR_COLS)]
            cal_flags = {c: bool(row[c].iloc[0]) for c in CALENDAR_COLS}
        except (KeyError, IndexError):
            pass

    # 날씨 시계열
    if not all(c in weather.columns for c in WEATHER_COLS):
        return weather_zero, cal_flags

    sliced = weather[list(WEATHER_COLS)].reindex(timestamps).ffill().bfill().fillna(0.0)
    return sliced.to_numpy(dtype=np.float32), cal_flags
