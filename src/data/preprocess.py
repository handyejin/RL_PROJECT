"""따릉이 데이터 전처리.

원본 CSV (CP949) → parquet 변환.
- 정류소 마스터: 위경도 0인 폐쇄 정류소 제거, 자치구 파생
- 대여 이력: 월별 파일 결합, 시간 파싱, 비정상 trip 필터
- 수요 테이블: (시각, 정류소) × (rentals, returns) — 시뮬레이터 replay 입력
- 날씨 (ASOS): QC플래그 드롭, 결측 보간, step 단위 resample
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data"
OUT_DIR = RAW_DIR / "processed"
ENCODING = "cp949"

STATION_COLS = {
    "대여소_ID": "station_id",
    "주소1": "address1",
    "주소2": "address2",
    "위도": "lat",
    "경도": "lon",
}

TRIP_COLS = {
    "자전거번호": "bike_id",
    "대여일시": "start_time",
    "반납일시": "end_time",
    "대여 대여소번호": "start_code",
    "대여 대여소명": "start_name",
    "대여거치대": "start_rack",
    "반납대여소번호": "end_code",
    "반납대여소명": "end_name",
    "반납거치대": "end_rack",
    "대여대여소ID": "start_station_id",
    "반납대여소ID": "end_station_id",
    "이용시간(분)": "duration_min",
    "이용거리(M)": "distance_m",
    "생년": "birth_year",
    "성별": "gender",
    "이용자종류": "user_type",
    "자전거구분": "bike_type",
}

WEATHER_COLS = {
    "지점": "stn_id",
    "지점명": "stn_name",
    "일시": "ts",
    "기온(°C)": "temp_c",
    "강수량(mm)": "precip_mm",
    "풍속(m/s)": "wind_ms",
    "습도(%)": "humidity_pct",
    "일조(hr)": "sunshine_hr",
    "일사(MJ/m2)": "solar_mj",
}


# ──────────────────────────────────────────────────────────────
# 정류소 마스터
# ──────────────────────────────────────────────────────────────
def load_stations(path: Path | None = None) -> pd.DataFrame:
    path = path or (RAW_DIR / "stations_master.csv")
    df = pd.read_csv(path, encoding=ENCODING).rename(columns=STATION_COLS)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    before = len(df)
    df = df[(df["lat"].abs() > 1e-6) & (df["lon"].abs() > 1e-6)].copy()
    logger.info("stations: dropped %d closed/zero-coord rows", before - len(df))

    df["gu"] = df["address1"].str.extract(r"서울특별시\s+(\S+구)", expand=False)
    return df.drop_duplicates("station_id").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 대여 이력
# ──────────────────────────────────────────────────────────────
def load_trips(
    stations: pd.DataFrame | None = None,
    keep_gu: list[str] | None = None,
    min_min: float = 1.0,
    max_min: float = 24 * 60,
    glob_pattern: str = "trips_*.csv",
) -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / glob_pattern)))
    if not files:
        raise FileNotFoundError(f"no trip files matched: {glob_pattern}")
    logger.info("trips: loading %d files", len(files))

    df = pd.concat(
        [pd.read_csv(f, encoding=ENCODING, low_memory=False) for f in files],
        ignore_index=True,
    ).rename(columns=TRIP_COLS)

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["start_time", "end_time", "start_station_id", "end_station_id"])
    df = df[df["duration_min"].between(min_min, max_min) & (df["end_time"] > df["start_time"])]
    logger.info("trips: dropped %d abnormal rows (kept %d)", before - len(df), len(df))

    if keep_gu:
        if stations is None:
            raise ValueError("keep_gu requires `stations` DataFrame")
        ids = set(stations.loc[stations["gu"].isin(keep_gu), "station_id"])
        df = df[df["start_station_id"].isin(ids) & df["end_station_id"].isin(ids)].copy()
        logger.info("trips: filtered to %s → %d rows", keep_gu, len(df))

    return df.sort_values("start_time").reset_index(drop=True)


def trips_to_demand(trips: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """(시각, 정류소) 별 출발/도착 카운트. README §2.2: 1 step = 10분."""
    starts = (
        trips.assign(t=trips["start_time"].dt.floor(freq))
        .groupby(["t", "start_station_id"])
        .size()
        .rename("rentals")
        .reset_index()
        .rename(columns={"start_station_id": "station_id"})
    )
    ends = (
        trips.assign(t=trips["end_time"].dt.floor(freq))
        .groupby(["t", "end_station_id"])
        .size()
        .rename("returns")
        .reset_index()
        .rename(columns={"end_station_id": "station_id"})
    )
    return (
        starts.merge(ends, on=["t", "station_id"], how="outer")
        .fillna(0)
        .astype({"rentals": "int32", "returns": "int32"})
        .sort_values(["t", "station_id"])
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────
# 날씨 (ASOS 시간자료)
# ──────────────────────────────────────────────────────────────
def load_weather(glob_pattern: str = "weather_asos_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / glob_pattern)))
    if not files:
        raise FileNotFoundError(f"no weather files matched: {glob_pattern}")

    df = pd.concat([pd.read_csv(f, encoding=ENCODING) for f in files], ignore_index=True)
    df = df.loc[:, ~df.columns.str.contains("QC플래그")].rename(columns=WEATHER_COLS)

    if "ts" not in df.columns:
        raise ValueError(f"weather: '일시' column missing. got {df.columns.tolist()}")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    missing = [c for c in ("temp_c", "precip_mm", "wind_ms", "humidity_pct") if c not in df.columns]
    if missing:
        logger.warning("weather: value columns missing %s — 기상자료개방포털 재다운로드 필요", missing)

    if "precip_mm" in df.columns:
        df["precip_mm"] = df["precip_mm"].fillna(0.0)
    for c in ("temp_c", "wind_ms", "humidity_pct"):
        if c in df.columns:
            df[c] = df[c].interpolate(limit=3).ffill().bfill()

    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def resample_weather(weather: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    drop_cols = [c for c in ("stn_id", "stn_name") if c in weather.columns]
    w = weather.set_index("ts").drop(columns=drop_cols)
    return w.resample(freq).ffill().reset_index().rename(columns={"ts": "t"})


# ──────────────────────────────────────────────────────────────
# 통합 실행
# ──────────────────────────────────────────────────────────────
def run(
    keep_gu: list[str] | None = None,
    step_freq: str = "10min",
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = load_stations()
    trips = load_trips(stations=stations, keep_gu=keep_gu)
    demand = trips_to_demand(trips, freq=step_freq)
    weather = resample_weather(load_weather(), freq=step_freq)

    paths = {
        "stations": out_dir / "stations.parquet",
        "trips": out_dir / "trips.parquet",
        "demand": out_dir / f"demand_{step_freq}.parquet",
        "weather": out_dir / f"weather_{step_freq}.parquet",
    }
    stations.to_parquet(paths["stations"], index=False)
    trips.to_parquet(paths["trips"], index=False)
    demand.to_parquet(paths["demand"], index=False)
    weather.to_parquet(paths["weather"], index=False)
    for name, p in paths.items():
        logger.info("wrote %s → %s", name, p)
    return paths
