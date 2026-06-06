"""Station-level short-horizon demand forecasting.

  features at time t -> rentals/returns during [t, t + horizon)

The saved artifact can be loaded by the environment data loader to append
forecast features to each episode observation without leaking future eval data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class DemandForecastArtifact:
    station_ids: list[str]
    model_rentals: object
    model_returns: object
    profile_by_station_dow_hour: pd.DataFrame
    profile_by_station: pd.DataFrame
    horizon_steps: int
    step_duration_min: int
    train_dates: list[str]
    feature_cols: list[str]

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features = frame[self.feature_cols].to_numpy(dtype=np.float32)
        rentals = np.maximum(self.model_rentals.predict(features), 0.0)
        returns = np.maximum(self.model_returns.predict(features), 0.0)
        return rentals.astype(np.float32), returns.astype(np.float32)


def save_model(model: DemandForecastArtifact, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str) -> DemandForecastArtifact:
    return joblib.load(path)


def build_dense_demand(
    processed_dir: Path | str,
    station_ids: list[str],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Load sparse demand parquet into dense (T, N) rentals/returns arrays."""
    processed_dir = Path(processed_dir)
    demand = pd.read_parquet(processed_dir / "demand_10min.parquet")
    demand = demand[demand["station_id"].isin(station_ids)].copy()
    timestamps = pd.date_range(
        demand["t"].min(),
        demand["t"].max(),
        freq="10min",
    )
    t_to_idx = {t: i for i, t in enumerate(timestamps)}
    id_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    rentals = np.zeros((len(timestamps), len(station_ids)), dtype=np.float32)
    returns = np.zeros_like(rentals)
    ti = demand["t"].map(t_to_idx).to_numpy()
    si = demand["station_id"].map(id_to_idx).to_numpy()
    rentals[ti, si] = demand["rentals"].to_numpy(dtype=np.float32)
    returns[ti, si] = demand["returns"].to_numpy(dtype=np.float32)
    return timestamps, rentals, returns


def _rolling_past_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Past sum over [t-window, t), excluding current t."""
    csum = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float32), np.cumsum(arr, axis=0)])
    out = np.zeros_like(arr)
    for t in range(len(arr)):
        start = max(0, t - window)
        out[t] = csum[t] - csum[start]
    return out


def _future_sum(arr: np.ndarray, horizon: int) -> np.ndarray:
    """Future sum over [t, t+horizon)."""
    csum = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float32), np.cumsum(arr, axis=0)])
    out = np.zeros_like(arr)
    T = len(arr)
    for t in range(T):
        end = min(T, t + horizon)
        out[t] = csum[end] - csum[t]
    return out


def build_feature_table(
    timestamps: pd.DatetimeIndex,
    rentals: np.ndarray,
    returns: np.ndarray,
    station_ids: list[str],
    horizon_steps: int = 6,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create station-time supervised examples for demand forecasting."""
    N = len(station_ids)
    T = len(timestamps)
    past_rent_1h = _rolling_past_sum(rentals, 6)
    past_ret_1h = _rolling_past_sum(returns, 6)
    past_rent_3h = _rolling_past_sum(rentals, 18)
    past_ret_3h = _rolling_past_sum(returns, 18)
    y_rent = _future_sum(rentals, horizon_steps)
    y_ret = _future_sum(returns, horizon_steps)

    # Historical profile using all provided rows. This is safe only if the caller
    # passes train-period data when fitting the model.
    idx = pd.MultiIndex.from_product([timestamps, station_ids], names=["t", "station_id"])
    flat = pd.DataFrame(index=idx).reset_index()
    flat["station_idx"] = np.tile(np.arange(N), T)
    flat["dow"] = flat["t"].dt.dayofweek.astype(np.int16)
    flat["hour"] = flat["t"].dt.hour.astype(np.int16)
    flat["slot"] = (flat["t"].dt.hour * 6 + flat["t"].dt.minute // 10).astype(np.int16)
    flat["month"] = flat["t"].dt.month.astype(np.int16)
    flat["is_weekend"] = (flat["dow"] >= 5).astype(np.float32)

    flat["past_rent_1h"] = past_rent_1h.reshape(-1)
    flat["past_ret_1h"] = past_ret_1h.reshape(-1)
    flat["past_rent_3h"] = past_rent_3h.reshape(-1)
    flat["past_ret_3h"] = past_ret_3h.reshape(-1)
    target_rent = y_rent.reshape(-1)
    target_ret = y_ret.reshape(-1)
    return flat, target_rent.astype(np.float32), target_ret.astype(np.float32)


def add_train_profiles(df: pd.DataFrame, y_rent: np.ndarray, y_ret: np.ndarray) -> pd.DataFrame:
    """Add station/dow/hour historical averages computed on the given rows."""
    out = df.copy()
    tmp = out[["station_id", "dow", "hour"]].copy()
    tmp["target_rent"] = y_rent
    tmp["target_ret"] = y_ret
    prof = (
        tmp.groupby(["station_id", "dow", "hour"], observed=True)[["target_rent", "target_ret"]]
        .mean()
        .rename(columns={"target_rent": "profile_rent_1h", "target_ret": "profile_ret_1h"})
        .reset_index()
    )
    station_prof = (
        tmp.groupby(["station_id"], observed=True)[["target_rent", "target_ret"]]
        .mean()
        .rename(columns={"target_rent": "station_avg_rent_1h", "target_ret": "station_avg_ret_1h"})
        .reset_index()
    )
    out = out.merge(prof, on=["station_id", "dow", "hour"], how="left")
    out = out.merge(station_prof, on=["station_id"], how="left")
    out[["profile_rent_1h", "profile_ret_1h", "station_avg_rent_1h", "station_avg_ret_1h"]] = (
        out[["profile_rent_1h", "profile_ret_1h", "station_avg_rent_1h", "station_avg_ret_1h"]]
        .fillna(0.0)
        .astype(np.float32)
    )
    return out


def make_profile_tables(
    train_df: pd.DataFrame,
    y_rent: np.ndarray,
    y_ret: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = train_df[["station_id", "dow", "hour"]].copy()
    tmp["target_rent"] = y_rent
    tmp["target_ret"] = y_ret
    profile_by_time = (
        tmp.groupby(["station_id", "dow", "hour"], observed=True)[["target_rent", "target_ret"]]
        .mean()
        .rename(columns={"target_rent": "profile_rent_1h", "target_ret": "profile_ret_1h"})
        .reset_index()
    )
    profile_by_station = (
        tmp.groupby(["station_id"], observed=True)[["target_rent", "target_ret"]]
        .mean()
        .rename(columns={"target_rent": "station_avg_rent_1h", "target_ret": "station_avg_ret_1h"})
        .reset_index()
    )
    return profile_by_time, profile_by_station


def attach_profiles(
    df: pd.DataFrame,
    profile_by_time: pd.DataFrame,
    profile_by_station: pd.DataFrame,
) -> pd.DataFrame:
    out = df.merge(profile_by_time, on=["station_id", "dow", "hour"], how="left")
    out = out.merge(profile_by_station, on=["station_id"], how="left")
    cols = ["profile_rent_1h", "profile_ret_1h", "station_avg_rent_1h", "station_avg_ret_1h"]
    out[cols] = out[cols].fillna(0.0).astype(np.float32)
    return out


FEATURE_COLS = [
    "station_idx",
    "dow",
    "hour",
    "slot",
    "month",
    "is_weekend",
    "past_rent_1h",
    "past_ret_1h",
    "past_rent_3h",
    "past_ret_3h",
    "profile_rent_1h",
    "profile_ret_1h",
    "station_avg_rent_1h",
    "station_avg_ret_1h",
]


def predict_forecast_grid(
    artifact: DemandForecastArtifact,
    timestamps: pd.DatetimeIndex,
    rentals: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Predict (T, N, 3) rentals/returns/net_pressure for the given timeline."""
    features, _, _ = build_feature_table(
        timestamps,
        rentals,
        returns,
        artifact.station_ids,
        horizon_steps=artifact.horizon_steps,
    )
    features = attach_profiles(
        features,
        artifact.profile_by_station_dow_hour,
        artifact.profile_by_station,
    )
    pred_rent, pred_ret = artifact.predict(features)
    T, N = len(timestamps), len(artifact.station_ids)
    pred_rent = pred_rent.reshape(T, N)
    pred_ret = pred_ret.reshape(T, N)
    pred_net = pred_ret - pred_rent
    return np.stack([pred_rent, pred_ret, pred_net], axis=-1).astype(np.float32)
