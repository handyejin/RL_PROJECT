"""Train station-level 1-hour demand forecasts for RL state features.

The model uses only calendar/time, station id, historical profile from train
dates, and past observed demand. It then writes:

- data/processed/demand_forecast_h1.joblib
- data/processed/demand_forecast_1h.parquet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.demand_forecast import (  # noqa: E402
    FEATURE_COLS,
    DemandForecastArtifact,
    attach_profiles,
    build_dense_demand,
    build_feature_table,
    make_profile_tables,
    predict_forecast_grid,
    save_model,
)
def filter_stations(stations: pd.DataFrame, district: str) -> pd.DataFrame:
    """자치구 기준으로 정류소를 필터링한다."""

    mask = stations["gu"] == district
    if not mask.any():
        raise ValueError(f"district '{district}' not found in stations.gu")
    return stations[mask].sort_values("station_id").reset_index(drop=True)


def filter_active_stations(stations: pd.DataFrame, demand_rows: pd.DataFrame) -> pd.DataFrame:
    """수요 기록이 있는 정류소만 남긴다.

    원본 환경 코드를 수정하지 않기 위해 수요예측 스크립트 안에서만 사용하는
    보조 필터다. 마포구 전체 정류소 중 2025년 대여/반납 기록이 관측된
    정류소를 대상으로 1시간 수요예측 feature를 만든다.
    """

    active_ids = set(demand_rows["station_id"].dropna().unique())
    return stations[stations["station_id"].isin(active_ids)].reset_index(drop=True)


def date_range(start: str, end: str) -> list[str]:
    import datetime

    current = datetime.date.fromisoformat(start)
    end_date = datetime.date.fromisoformat(end)
    dates: list[str] = []
    while current <= end_date:
        dates.append(current.isoformat())
        current += datetime.timedelta(days=1)
    return dates


def split_dates(seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    dates = date_range("2025-01-01", "2025-12-31")
    rng.shuffle(dates)
    split_idx = int(len(dates) * 0.8)
    return dates[:split_idx], sorted(dates[split_idx:])


def split_dates_for_rl_holdout(
    seed: int,
    n_train_dates: int,
    n_eval_dates: int,
    date_manifest: str | None,
    train_sampling: str,
) -> tuple[list[str], list[str]]:
    """RL 평가 split과 같은 방식의 holdout 날짜를 만든다.

    현재 팀원 원본에는 difficulty manifest split helper가 없으므로, 이
    스크립트는 seed 기반 80/20 split을 자체 제공한다. 고급 manifest 기반
    split이 필요하면 별도 manifest loader를 추가해야 한다.
    """

    if date_manifest is not None:
        raise ValueError("--date-manifest is not supported by this standalone script.")
    if train_sampling != "balanced":
        raise ValueError("--train-sampling must be 'balanced' without --date-manifest.")
    train_pool, eval_pool = split_dates(seed)
    return train_pool[:n_train_dates], eval_pool[:n_eval_dates]


def sample_rows(
    x: pd.DataFrame,
    y_rent: np.ndarray,
    y_ret: np.ndarray,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(x) <= max_rows:
        return x, y_rent, y_ret
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x.iloc[idx].reset_index(drop=True), y_rent[idx], y_ret[idx]


def save_forecast_parquet(
    path: Path,
    timestamps: pd.DatetimeIndex,
    station_ids: list[str],
    forecast_grid: np.ndarray,
) -> None:
    T, N, _ = forecast_grid.shape
    out = pd.DataFrame(
        {
            "t": np.repeat(timestamps.to_numpy(), N),
            "station_id": np.tile(np.array(station_ids, dtype=object), T),
            "pred_rentals_1h": forecast_grid[:, :, 0].reshape(-1),
            "pred_returns_1h": forecast_grid[:, :, 1].reshape(-1),
            "pred_net_1h": forecast_grid[:, :, 2].reshape(-1),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn demand forecast for RL features.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-steps", type=int, default=6, help="6 x 10min = next 1h")
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--max-eval-rows", type=int, default=200_000)
    parser.add_argument("--max-iter", type=int, default=140)
    parser.add_argument("--model-out", default="data/processed/demand_forecast_h1_rlholdout_seed42.joblib")
    parser.add_argument("--forecast-out", default="data/processed/demand_forecast_1h_rlholdout_seed42.parquet")
    parser.add_argument("--metrics-out", default="logs/demand_forecast_h1_rlholdout_seed42_metrics.json")
    parser.add_argument(
        "--holdout-dates",
        nargs="*",
        default=None,
        help="이 날짜들은 forecast 모델 학습에서 제외하고 eval로 사용한다. 예: 2025-01-04 2025-02-20",
    )
    parser.add_argument(
        "--holdout-from-rl-split",
        action="store_true",
        help="RL 실험 split과 같은 eval 날짜를 forecast eval/holdout으로 사용한다.",
    )
    parser.add_argument("--date-manifest", default=None)
    parser.add_argument("--n-train-dates", type=int, default=60)
    parser.add_argument("--n-eval-dates", type=int, default=10)
    parser.add_argument("--train-sampling", choices=["balanced", "hard", "natural"], default="balanced")
    args = parser.parse_args()

    t0 = time.time()
    processed_dir = Path(args.processed_dir)
    stations_all = pd.read_parquet(processed_dir / "stations.parquet")
    stations = filter_stations(stations_all, args.district)
    demand_rows = pd.read_parquet(processed_dir / "demand_10min.parquet")
    stations = filter_active_stations(stations, demand_rows)
    station_ids = stations["station_id"].tolist()

    print(f"=== demand forecast | district={args.district} | stations={len(station_ids)} ===")
    print("[1/5] loading dense demand grid...")
    timestamps, rentals, returns = build_dense_demand(processed_dir, station_ids)
    print(f"  grid: T={len(timestamps):,}, N={len(station_ids):,}")

    print("[2/5] building supervised feature table...")
    features, y_rent, y_ret = build_feature_table(
        timestamps, rentals, returns, station_ids, horizon_steps=args.horizon_steps
    )
    if args.holdout_from_rl_split:
        _, eval_dates = split_dates_for_rl_holdout(
            args.seed,
            args.n_train_dates,
            args.n_eval_dates,
            args.date_manifest,
            args.train_sampling,
        )
        all_dates = date_range("2025-01-01", "2025-12-31")
        train_dates = [d for d in all_dates if d not in set(eval_dates)]
    elif args.holdout_dates:
        eval_dates = sorted(set(args.holdout_dates))
        all_dates = date_range("2025-01-01", "2025-12-31")
        train_dates = [d for d in all_dates if d not in set(eval_dates)]
    else:
        train_dates, eval_dates = split_dates(args.seed)
    row_dates = features["t"].dt.date.astype(str)
    train_mask = row_dates.isin(train_dates).to_numpy()
    eval_mask = row_dates.isin(eval_dates).to_numpy()

    train_base = features.loc[train_mask].reset_index(drop=True)
    eval_base = features.loc[eval_mask].reset_index(drop=True)
    y_train_rent = y_rent[train_mask]
    y_train_ret = y_ret[train_mask]
    y_eval_rent = y_rent[eval_mask]
    y_eval_ret = y_ret[eval_mask]

    profile_by_time, profile_by_station = make_profile_tables(
        train_base, y_train_rent, y_train_ret
    )
    train_x = attach_profiles(train_base, profile_by_time, profile_by_station)
    eval_x = attach_profiles(eval_base, profile_by_time, profile_by_station)

    train_x, y_train_rent, y_train_ret = sample_rows(
        train_x, y_train_rent, y_train_ret, args.max_train_rows, args.seed
    )
    eval_x, y_eval_rent, y_eval_ret = sample_rows(
        eval_x, y_eval_rent, y_eval_ret, args.max_eval_rows, args.seed + 1
    )
    print(f"  train rows: {len(train_x):,}, eval rows: {len(eval_x):,}")

    print("[3/5] fitting HistGradientBoostingRegressor models...")
    model_kwargs = dict(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=args.seed,
    )
    x_train_np = train_x[FEATURE_COLS].to_numpy(dtype=np.float32)
    model_rentals = HistGradientBoostingRegressor(**model_kwargs)
    model_returns = HistGradientBoostingRegressor(**model_kwargs)
    model_rentals.fit(x_train_np, y_train_rent)
    model_returns.fit(x_train_np, y_train_ret)

    artifact = DemandForecastArtifact(
        station_ids=station_ids,
        model_rentals=model_rentals,
        model_returns=model_returns,
        profile_by_station_dow_hour=profile_by_time,
        profile_by_station=profile_by_station,
        horizon_steps=args.horizon_steps,
        step_duration_min=10,
        train_dates=sorted(train_dates),
        feature_cols=FEATURE_COLS,
    )
    save_model(artifact, args.model_out)

    print("[4/5] evaluating holdout dates...")
    pred_rent, pred_ret = artifact.predict(eval_x)
    baseline_rent = eval_x["profile_rent_1h"].to_numpy(dtype=np.float32)
    baseline_ret = eval_x["profile_ret_1h"].to_numpy(dtype=np.float32)
    metrics = {
        "district": args.district,
        "n_stations": len(station_ids),
        "horizon_min": args.horizon_steps * 10,
        "train_rows": int(len(train_x)),
        "eval_rows": int(len(eval_x)),
        "model_rent_mae": float(mean_absolute_error(y_eval_rent, pred_rent)),
        "model_return_mae": float(mean_absolute_error(y_eval_ret, pred_ret)),
        "model_rent_rmse": float(root_mean_squared_error(y_eval_rent, pred_rent)),
        "model_return_rmse": float(root_mean_squared_error(y_eval_ret, pred_ret)),
        "profile_rent_mae": float(mean_absolute_error(y_eval_rent, baseline_rent)),
        "profile_return_mae": float(mean_absolute_error(y_eval_ret, baseline_ret)),
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "  MAE rent/return: "
        f"{metrics['model_rent_mae']:.3f}/{metrics['model_return_mae']:.3f} "
        "(profile "
        f"{metrics['profile_rent_mae']:.3f}/{metrics['profile_return_mae']:.3f})"
    )

    print("[5/5] writing full forecast parquet...")
    forecast_grid = predict_forecast_grid(artifact, timestamps, rentals, returns)
    save_forecast_parquet(Path(args.forecast_out), timestamps, station_ids, forecast_grid)
    print(f"  model: {args.model_out}")
    print(f"  forecast: {args.forecast_out}")
    print(f"  metrics: {args.metrics_out}")
    print(f"done ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
