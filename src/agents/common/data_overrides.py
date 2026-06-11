"""agent 전용 데이터 보강 helper.

팀원 공통 env/data_loader 파일은 수정하지 않고, 학습 agent가 받은
EpisodeData 객체에만 추가 데이터를 붙인다.

현재 지원:
    - capacity_path: 정류소별 실제 거치대 수(capacity)를 episode에 반영
    - forecast_path: 정류소별 1시간 대여/반납 예측값을 episode에 반영

이 파일의 처리는 실험용 데이터 보강이며, 원본 parquet 파일 자체를 덮어쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.common.future_demand import DemandForecastProvider


def _has_path(value: str | None) -> bool:
    """빈 문자열/None은 옵션이 꺼진 것으로 본다."""
    return value is not None and str(value).strip() != ""


def _read_table(path: str | Path) -> pd.DataFrame:
    """확장자에 맞춰 parquet/csv 테이블을 읽는다."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def apply_capacity_override(
    episodes: list,
    capacity_path: str | None,
    initial_fill_ratio: float = 0.5,
) -> dict[str, float] | None:
    """정류소별 capacity를 episode에 반영하고 초기 재고도 다시 추정한다.

    기존 env는 stations.parquet에 capacity가 없으면 모든 정류소를 20대로 본다.
    이 helper는 외부 stations.parquet/csv의 capacity를 읽어 agent 실험에서만 사용한다.
    """
    if not _has_path(capacity_path):
        return None
    table = _read_table(capacity_path)
    required = {"station_id", "capacity"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"capacity table missing columns: {sorted(missing)}")

    capacity_map = (
        table[["station_id", "capacity"]]
        .dropna(subset=["station_id", "capacity"])
        .assign(capacity=lambda df: pd.to_numeric(df["capacity"], errors="coerce"))
        .dropna(subset=["capacity"])
        .drop_duplicates("station_id", keep="last")
        .set_index("station_id")["capacity"]
        .to_dict()
    )

    matched = 0
    total = 0
    capacity_sum = 0.0
    for ep in episodes:
        old_capacity = ep.capacity.astype(np.int32)
        new_capacity = np.array(
            [int(capacity_map.get(station_id, old_capacity[i])) for i, station_id in enumerate(ep.station_ids)],
            dtype=np.int32,
        )
        new_capacity = np.maximum(new_capacity, 1)
        matched += int(sum(station_id in capacity_map for station_id in ep.station_ids))
        total += len(ep.station_ids)
        capacity_sum += float(new_capacity.sum())

        # data_loader의 data_based 초기화와 같은 생각으로 첫 1시간 net flow를 반영한다.
        horizon = min(6, ep.rentals.shape[0])
        net = ep.returns[:horizon].sum(axis=0) - ep.rentals[:horizon].sum(axis=0)
        base = new_capacity.astype(np.float32) * float(initial_fill_ratio) - net
        ep.capacity = new_capacity
        ep.initial_bikes = np.clip(base, 0, new_capacity).astype(np.int32)

    return {
        "capacity_matched": float(matched),
        "capacity_total": float(total),
        "capacity_mean": capacity_sum / max(total, 1),
    }


def attach_forecast_override(episodes: list, forecast_path: str | None) -> dict[str, float] | None:
    """외부 demand_forecast parquet을 episode별 (T, N, 3) 배열로 붙인다."""
    if not _has_path(forecast_path):
        return None
    provider = DemandForecastProvider(forecast_path)
    matched = 0
    total = 0
    for ep in episodes:
        grid, ep_matched, ep_total = provider.grid_for_episode(ep)
        ep.agent_demand_forecast = grid
        matched += ep_matched
        total += ep_total
    return {
        "forecast_matched": float(matched),
        "forecast_total": float(total),
    }
