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

import holidays
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data"
OUT_DIR = RAW_DIR / "processed"
ENCODING = "cp949"  # 서울시·기상청 공공데이터 원본 인코딩 (UTF-8 아님)

# 한글 컬럼명 → 영문 스네이크 케이스 매핑. 이후 파이프라인은 영문 키만 다룸.
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
    "현지기압(hPa)": "pressure_hpa",
    "일조(hr)": "sunshine_hr",
    "일사(MJ/m2)": "solar_mj",
    "적설(cm)": "snow_cm",
    "시정(10m)": "visibility_10m",
}


# ──────────────────────────────────────────────────────────────
# 정류소 마스터
# ──────────────────────────────────────────────────────────────
def load_stations(path: Path | None = None) -> pd.DataFrame:
    path = path or (RAW_DIR / "stations_master.csv")
    df = pd.read_csv(path, encoding=ENCODING).rename(columns=STATION_COLS)

    # 위/경도 숫자 변환
    # coerce: 변환 실패 시 NaN (예: 빈 문자열 → NaN)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # (0,0) 좌표는 폐쇄된 정류소이므로 제거. 실제로 서울에 위도 0 또는 경도 0인 정류소는 없으므로, 1e-6 정도 작은 값도 제거.
    before = len(df)
    df = df[(df["lat"].abs() > 1e-6) & (df["lon"].abs() > 1e-6)].copy()
    logger.info("stations: dropped %d closed/zero-coord rows", before - len(df))

    # 서울특별시가 들어가는 주소에서 구 추출 (예: "서울특별시 마포구 월드컵북로 400" → "마포구"). 실패하면 NaN.
    df["gu"] = df["address1"].str.extract(r"서울특별시\s+(\S+구)", expand=False)

    # station_id 기준 중복 제거
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
    """월별 trip CSV를 파일 단위로 읽어 필터링 후 concat.

    - 전체 concat 후 필터링하면 메모리 폭증 → 파일별로 필터링 먼저 적용.
    - keep_gu 지정 시 stations에서 해당 자치구 station_id 집합을 만들어 필터.
    - min_min/max_min: 비정상 trip(테스트 주행, 반납 실패 등) 컷오프.
    """
    files = sorted(glob.glob(str(RAW_DIR / glob_pattern)))
    if not files:
        raise FileNotFoundError(f"no trip files matched: {glob_pattern}")
    logger.info("trips: loading %d files", len(files))

    # 자치구 필터: 매 파일마다 stations를 훑지 않도록 station_id 집합을 미리 구성
    if keep_gu:
        if stations is None:
            raise ValueError("keep_gu requires `stations` DataFrame")
        keep_ids = set(stations.loc[stations["gu"].isin(keep_gu), "station_id"])
    else:
        keep_ids = None

    parts: list[pd.DataFrame] = []
    total_dropped = 0
    for f in files:
        df = pd.read_csv(f, encoding=ENCODING, low_memory=False).rename(columns=TRIP_COLS)

        # 시간/숫자 컬럼 파싱. coerce → 파싱 실패값은 NaN (아래 dropna에서 제거)
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
        df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
        df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")

        # 비정상 trip 필터:
        #  - 필수 키(시간/정류소 ID)가 NaN인 행 제거
        #  - duration이 [min_min, max_min] 범위 밖 (너무 짧거나 24h 초과) 제거
        #  - 종료 ≤ 시작인 비논리 trip 제거
        before = len(df)
        df = df.dropna(subset=["start_time", "end_time", "start_station_id", "end_station_id"])
        df = df[df["duration_min"].between(min_min, max_min) & (df["end_time"] > df["start_time"])]
        total_dropped += before - len(df)

        # 자치구 필터: 출발·도착 모두 keep_gu 내 정류소인 trip만 유지
        if keep_ids is not None:
            df = df[df["start_station_id"].isin(keep_ids) & df["end_station_id"].isin(keep_ids)]

        parts.append(df.copy())
        logger.info("trips: %s → %d rows kept", Path(f).name, len(df))

    df = pd.concat(parts, ignore_index=True)
    logger.info("trips: total %d rows (dropped %d abnormal across all files)", len(df), total_dropped)
    # 시뮬레이터 replay는 시간순으로 trip을 소비하므로 정렬해서 반환
    return df.sort_values("start_time").reset_index(drop=True)


def trips_to_demand(trips: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """(시각, 정류소) 별 출발/도착 카운트. README §2.2: 1 step = 10분.

    출발(rentals)과 도착(returns)이 서로 다른 정류소·시각에 발생하므로
    각각 따로 집계 후 outer merge → 한쪽만 발생한 칸은 0으로 채움.
    """
    # 출발 카운트: start_time을 freq 단위로 floor → (시각, 출발 정류소)별 trip 수
    starts = (
        trips.assign(t=trips["start_time"].dt.floor(freq))
        .groupby(["t", "start_station_id"])
        .size()
        .rename("rentals")
        .reset_index()
        .rename(columns={"start_station_id": "station_id"})
    )
    # 도착 카운트: end_time 기준으로 동일 처리
    ends = (
        trips.assign(t=trips["end_time"].dt.floor(freq))
        .groupby(["t", "end_station_id"])
        .size()
        .rename("returns")
        .reset_index()
        .rename(columns={"end_station_id": "station_id"})
    )
    # outer merge: 출발만 있던 칸·도착만 있던 칸 모두 보존, 빈칸은 0 → int32로 다운캐스트(메모리 절감)
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
def load_weather(glob_pattern: str = "OBS_ASOS_*.csv") -> pd.DataFrame:
    """ASOS 시간자료 CSV들을 읽어 정제된 시간별 날씨 DF 반환.

    - QC플래그 컬럼은 분석에 안 쓰므로 드롭.
    - 누적치(강수/적설/일조)와 순간값(기온/풍속 등)을 다르게 보간:
      누적치 NaN = "관측 안 됨/0" → 0으로 채움.
      순간값 NaN = "센서 결측" → 시간 보간(과도 보간 방지를 위해 limit=3).
    """
    files = sorted(glob.glob(str(RAW_DIR / glob_pattern)))
    if not files:
        raise FileNotFoundError(f"no weather files matched: {glob_pattern}")
    logger.info("weather: loading %d files", len(files))

    df = pd.concat([pd.read_csv(f, encoding=ENCODING) for f in files], ignore_index=True)
    # QC플래그(품질 관리 코드) 컬럼 일괄 제거 후 영문 컬럼명으로 정규화
    df = df.loc[:, ~df.columns.str.contains("QC플래그")].rename(columns=WEATHER_COLS)

    # 시간 컬럼이 없으면 이후 모든 처리가 의미 없음 → 조기 실패
    if "ts" not in df.columns:
        raise ValueError(f"weather: '일시' column missing. got {df.columns.tolist()}")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # 핵심 관측치 누락 → 다운로드가 불완전했을 가능성 (경고만, 진행은 함)
    missing = [c for c in ("temp_c", "precip_mm", "wind_ms", "humidity_pct") if c not in df.columns]
    if missing:
        logger.warning("weather: value columns missing %s — 기상자료개방포털 재다운로드 필요", missing)

    # 누적/측정 없으면 0으로 해석되는 항목: 결측 = 무강수/무적설/무일조
    for c in ("precip_mm", "snow_cm", "sunshine_hr"):
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    # 연속 물리량: 짧은 결측만 선형 보간(limit=3 = 최대 3시간), 양끝은 ffill/bfill로 마감
    for c in ("temp_c", "wind_ms", "humidity_pct", "pressure_hpa", "visibility_10m"):
        if c in df.columns:
            df[c] = df[c].interpolate(limit=3).ffill().bfill()

    # ts 파싱 실패행 제거 후 시간순 정렬
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def resample_weather(weather: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """시간 단위 날씨를 step 단위(기본 10분)로 다운샘플(forward fill).

    수요 테이블(`trips_to_demand`)과 동일한 시간 격자에 맞춰 merge 가능하게 함.
    관측소 식별자(stn_*)는 단일 지점 사용 가정이라 드롭. 컬럼명 ts→t로 통일.
    """
    drop_cols = [c for c in ("stn_id", "stn_name") if c in weather.columns]
    w = weather.set_index("ts").drop(columns=drop_cols)
    # ffill: 1시간 관측값을 그 시간대 6개 10분 칸에 동일 적용 (계단형)
    return w.resample(freq).ffill().reset_index().rename(columns={"ts": "t"})


# ──────────────────────────────────────────────────────────────
# 캘린더 피처 (공휴일·주말·휴일 전날)
# ──────────────────────────────────────────────────────────────
def add_calendar_features(df: pd.DataFrame, time_col: str = "t") -> pd.DataFrame:
    """주말·공휴일 플래그 추가. is_holiday_eve는 비주말 평일 중 다음날이 휴일(공휴일/주말)인 경우.
    """
    if time_col not in df.columns:
        raise ValueError(f"add_calendar_features: '{time_col}' column missing")

    # 시간 부분 제거(자정 기준). 같은 날짜의 여러 step이 동일한 공휴일 판정을 공유하게 함.
    dates = pd.to_datetime(df[time_col]).dt.normalize()
    # 데이터에 등장하는 연도만 공휴일 테이블 로드 (전 연도 로드 불필요)
    years = sorted({d.year for d in dates.dropna().unique()})
    kr = holidays.SouthKorea(years=years)

    is_weekend = dates.dt.dayofweek >= 5  # 토(5)·일(6)
    is_holiday = dates.dt.date.map(lambda d: d in kr).astype(bool)

    # 휴일 전날(eve): "내일이 쉬는 날인 평일" → 금요일 또는 공휴일 전날 평일
    # 주말·공휴일 자체는 제외해야 "이브" 의미가 살아남 (토요일은 일요일 전날이지만 이미 주말)
    next_day = dates + pd.Timedelta(days=1)
    next_weekend = next_day.dt.dayofweek >= 5
    next_holiday = next_day.dt.date.map(lambda d: d in kr).astype(bool)
    is_eve = (~is_weekend & ~is_holiday) & (next_weekend | next_holiday)

    # 원본 보존을 위해 copy. to_numpy()로 인덱스 정렬 이슈 방지.
    out = df.copy()
    out["is_weekend"] = is_weekend.to_numpy()
    out["is_holiday"] = is_holiday.to_numpy()
    out["is_holiday_eve"] = is_eve.to_numpy()
    return out


# ──────────────────────────────────────────────────────────────
# 통합 실행
# ──────────────────────────────────────────────────────────────
def run(
    keep_gu: list[str] | None = None,
    step_freq: str = "10min",
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """전체 전처리 파이프라인 실행 → parquet 4종 저장 후 경로 dict 반환.

    parquet 채택 이유: CSV 대비 10~30배 작고 컬럼 타입 유지 + 부분 컬럼 로딩 가능.
    keep_gu로 자치구 서브셋만 처리하면 학습 초기 반복 실험에서 I/O·메모리 부담↓.
    """
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 의존성 순서: stations → trips(keep_gu 필터에 필요) → demand → weather
    stations = load_stations()
    trips = load_trips(stations=stations, keep_gu=keep_gu)
    demand = add_calendar_features(trips_to_demand(trips, freq=step_freq))
    weather = add_calendar_features(resample_weather(load_weather(), freq=step_freq))

    # demand/weather는 step 단위가 파일명에 반영 → 여러 해상도를 동시에 보관 가능
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
