# 데이터 전처리

원본 따릉이/기상청 CSV → 학습용 parquet 4종 변환 과정.

---

## 1. 한 줄 요약

> **"CP949 인코딩 + sparse + 불규칙 raw CSV"를 "UTF-8 + dense + 정형 parquet"으로 정리. 학습 시점에 반복 읽기 빠르게."**

한 번 돌려두면 학습이 그 parquet만 읽으면 됨.

---

## 2. 입력 (data/)

| 파일 | 출처 | 인코딩 | 크기 |
|---|---|---|---|
| `stations_master.csv` | 서울시 공공데이터 | CP949 | 정류소 마스터 |
| `trips_YYYYMM.csv` (12개) | 서울시 공공자전거 대여이력 | CP949 | 월별, 500K rows/월 |
| `OBS_ASOS_*.csv` | 기상청 ASOS 시간자료 | CP949 | 1년치 시간 단위 |

**왜 CP949?** 한국 공공기관 CSV의 기본 인코딩. UTF-8로 열면 한글 깨짐.

---

## 3. 처리 흐름 — 5단계

```
[원본 CSV]                       [출력 parquet]
─────────────                    ──────────────
stations_master.csv  ─┐
                      │  ① load_stations()
                      ├──────────────────────→  stations.parquet
                      │  (위경도 0 제거 + 자치구 파생)
                      │
trips_*.csv (12개)   ─┤
                      │  ② load_trips()
                      ├──────────────────────→  trips.parquet
                      │  (비정상 trip 필터링)
                      │
                      │  ③ trips_to_demand()
                      └──────────────────────→  demand_10min.parquet
                         (시각×정류소 카운트)
                                                
OBS_ASOS_*.csv       ─┐  ④ load_weather()
                      ├──────────────────────→ (raw 시간 단위)
                      │  (QC플래그 제거 + 결측 처리)
                      │
                      │  ⑤ resample_weather() + add_calendar_features()
                      └──────────────────────→  weather_10min.parquet
                         (10분 다운샘플 + 캘린더 피처)
```

---

## 4. 단계별 상세

### 4.1 정류소 마스터 (`load_stations`)

**입력 → 출력 변환**:
```
원본 컬럼 (한글)        →   출력 컬럼 (영문)
─────────────────────       ────────────────────────
대여소_ID                   station_id
주소1                       address1
주소2                       address2
위도                        lat
경도                        lon
                            gu (파생)
```

**주요 처리 4단계**:

1. **숫자 변환** — `pd.to_numeric(lat, errors="coerce")`
   - 빈 문자열·`-` 같은 비숫자값을 자동으로 NaN으로 처리
   - 이후 단계에서 NaN 행은 자연스럽게 걸러짐

2. **(0, 0) 좌표 제거** — `lat.abs() > 1e-6 & lon.abs() > 1e-6`
   - 폐쇄 정류소는 좌표가 `(0, 0)`으로 등록됨 → 학습 데이터에서 제외
   - 부동소수 오차 방어를 위해 `> 0` 대신 `> 1e-6` 사용

3. **자치구 추출** — `address1`에 정규식 적용
   - 패턴: `서울특별시 + 공백 + (○○구)`
   - 예: "서울특별시 마포구 월드컵북로 400" → 추출 결과: "마포구"
   - 매칭 실패 시 NaN (비서울 정류소)

4. **중복 제거** — `drop_duplicates("station_id")`
   - 같은 station_id가 여러 행 있으면 첫 번째만 유지

**결과**:
- 서울 전체 정류소 **3,341개**
- 자치구별 분포 top 5:
  ```
  송파구  245
  강서구  232
  영등포구 202
  강남구  201
  서초구  171
  ```
- 28개는 비서울 (gu = NaN) → 학습 시 keep_gu 필터로 자동 제외

### 4.2 대여 이력 (`load_trips`)

**메모리 절감 설계 — 파일별 스트리밍**:
```python
for file in monthly_files:        # 1년치 12개 파일
    df = read_csv(file, encoding="cp949")
    df = clean_and_filter(df)     # 비정상 trip 제거
    df = filter_by_gu(df)         # 자치구 필터
    parts.append(df)              # 정제된 것만 메모리 보관
df = pd.concat(parts)
```
→ 전체 concat 후 필터링하면 메모리 폭증. 파일별 정제 후 합침.

**비정상 trip 필터 4종**:

1. **시간 파싱 실패** — `start_time` 또는 `end_time`이 NaN
   → 시간 정보 없으면 시뮬레이션 불가

2. **너무 짧은 trip** — `duration_min` 결측 또는 1분 미만
   → 테스트 주행 / 사용자 실수 / 오류 데이터

3. **너무 긴 trip** — `duration_min` 24h 초과
   → 반납 안 한 분실 자전거 / 시스템 오류

4. **시간 역전** — `end_time` ≤ `start_time`
   → 데이터 입력 오류 (도착이 출발보다 빠를 수 없음)

**결과**: 1년치 **5,071,283 trips** (필터 후)
- 평균 이동 시간: 20.7분 (median 10분)
- 평균 이동 거리: 1,966m (median 1,189m)

### 4.3 수요 테이블 (`trips_to_demand`) ⭐ — 학습의 핵심

**아이디어**: 한 trip은 두 카운트에 기여:
```
trip: A 09:15 출발 → B 09:30 도착
     │                  │
     └─ A의 rentals (09:10 칸) +1
                        └─ B의 returns (09:30 칸) +1
```

**구현**:
```python
starts = trips.groupby([floor(start_time, "10min"), start_station]).size()
ends   = trips.groupby([floor(end_time, "10min"), end_station]).size()
demand = starts.merge(ends, how="outer").fillna(0)
```

**outer merge가 왜 필요?**
- 어떤 정류소·시각엔 rentals만 발생 (도착 X)
- 다른 곳엔 returns만
- inner merge하면 그 칸들이 사라짐 → 정보 손실

**결과**: 1년치 **5,786,019 rows** sparse 테이블
- 컬럼: `t, station_id, rentals, returns, is_weekend, is_holiday, is_holiday_eve`
- rentals/returns 합계 = 5,071,283 (trips 수와 정확히 일치) ✅
- max rentals per (t, station): 36건 (출퇴근시간 인기 정류소)

### 4.4 날씨 (`load_weather`)

**입력**: ASOS 시간자료 CSV (`기상자료개방포털`에서 다운로드).

**처리 흐름**:
```
원본 (한글 컬럼):                정제 후 (영문):
  지점, 일시, 기온(°C), ...   →   stn_id, ts, temp_c, ...
  QC플래그 (X.X)              →   (드롭)
```

**결측 처리 — 누적치 vs 순간값 다르게**:

**누적치 (`precip_mm`, `snow_cm`, `sunshine_hr`)** — 결측 시 **0으로 채움**
- 의미: "그 시간에 비/눈/일조가 0이었다" 추정 합리적

**순간값 (`temp_c`, `wind_ms`, `humidity_pct`, `pressure_hpa`, `visibility_10m`)** — 시간 보간
- 처리: `df[c].interpolate(limit=3).ffill().bfill()`
- 짧은 결측(≤3시간)만 선형 보간, 양끝은 ffill/bfill로 마감
- 너무 긴 결측은 보간 안 함 (센서 장기 고장 시 추정 신뢰 X)

### 4.5 10분 리샘플 (`resample_weather`) + 캘린더 (`add_calendar_features`)

**다운샘플**:
```
원본: 1시간 단위 (예: 09:00, 10:00, 11:00, ...)
출력: 10분 단위 (예: 09:00, 09:10, 09:20, ..., 10:00, ...)
```
**ffill** 방식: 09:00 관측값을 09:00~09:50 6개 칸에 동일 적용 (계단형).
**왜?** demand_10min과 같은 시간 격자에 맞춰서 환경에서 매 step 인덱싱 가능하게.

**캘린더 피처 추가**:
```python
is_weekend     = dayofweek >= 5                    # 토/일
is_holiday     = ts.date() in holidays.SouthKorea  # 공휴일
is_holiday_eve = (평일 not holiday) AND (내일이 휴일 OR 주말)
```

**예시**:
| 날짜 | dow | weekend | holiday | eve |
|---|---|---|---|---|
| 1/15 (수) | 2 | 0 | 0 | 0 |
| 1/17 (금) | 4 | 0 | 0 | **1** ← 다음날 토 |
| 1/29 (수) | 2 | 0 | **1** | 0 ← 설날 |
| 5/5 (월) | 0 | 0 | **1** | 0 ← 어린이날 |
| 10/3 (금) | 4 | 0 | **1** | 0 ← 개천절 |

---

## 5. 출력 — parquet 4종

| 파일 | 크기 | 행 수 | 학습 사용 |
|---|---|---|---|
| `stations.parquet` | 171 KB | 3,341 | ✅ 정류소 마스터 (load_episode 시 자치구 필터) |
| `trips.parquet` | 149 MB | 5,071,283 | ❌ (demand 재생성용, 학습엔 미사용) |
| `demand_10min.parquet` ⭐ | 12.6 MB | 5,786,019 | ✅ **시뮬레이션 핵심** (매 step rentals/returns) |
| `weather_10min.parquet` | 650 KB | 52,411 | ✅ obs의 날씨·캘린더 |

### 5.1 demand_10min.parquet 스키마

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `t` | datetime | 10분 단위 timestamp |
| `station_id` | string | 정류소 ID |
| `rentals` | int32 | 그 정류소에서 그 10분 구간에 출발한 trip 수 |
| `returns` | int32 | 그 정류소에 그 10분 구간에 도착한 trip 수 |
| `is_weekend` | bool | 그날 주말 여부 |
| `is_holiday` | bool | 그날 공휴일 여부 |
| `is_holiday_eve` | bool | 그날 휴일 전날 여부 |

### 5.2 weather_10min.parquet 스키마

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `t` | datetime | 10분 단위 (1년치) |
| `temp_c` | float | 기온 |
| `precip_mm` | float | 시간당 강수량 |
| `wind_ms` | float | 풍속 |
| `humidity_pct` | int | 습도 |
| `pressure_hpa` | float | 기압 |
| `sunshine_hr` | float | 일조 |
| `snow_cm` | float | 적설 |
| `visibility_10m` | int | 시정 |
| `is_weekend` / `is_holiday` / `is_holiday_eve` | bool | 캘린더 |

---

## 6. 데이터 특성 (1년치)

### 6.1 정류소
- 서울 전체 **3,341개**
- 자치구 25개에 분포 (송파 245 > 강서 232 > 영등포 202 > 강남 201 > 서초 171 > ...)
- 마포구 학습 권역: **146개**

### 6.2 Trip
- 1년 총 **5,071,283건**
- 평균 이동: 20.7분 / 1,966m
- 분포: 짧은 이용 많음 (median 10분 / 1.2km)

### 6.3 수요 패턴 (demand_10min)
- 10분 한 칸 평균: 0.88 trip
- 최대 한 칸 36 trip (출퇴근 시간 인기 정류소)
- 1 episode (24h × 146개) 총 평균: **2~5천 trip**

### 6.4 날씨 (1년)
- 기온 범위: **-12.1℃ ~ +37.6℃** (한겨울 한파 ~ 한여름 폭염 모두 포함)
- 강수 최대: 35.2mm/h (장마 폭우)
- 캘린더: 주말 ~104일, 공휴일 ~17일 (대체공휴일 포함)

---

## 7. 실행 방법

### 7.1 한 번에 (CLI)

```bash
# 마포구만, 10분 step
python scripts/run_preprocess.py

# 여러 자치구
python scripts/run_preprocess.py --gu 마포구 영등포구 강남구

# 서울 전체
python scripts/run_preprocess.py --gu all

# 다른 step (예: 5분)
python scripts/run_preprocess.py --step 5min
```

### 7.2 처리 시간 (참고)

| 단계 | 시간 (마포구) | 메모리 |
|---|---|---|
| stations | ~1초 | <100MB |
| trips (12 파일) | ~30~60초 | ~2GB peak |
| demand 집계 | ~5초 | ~500MB |
| weather | ~3초 | <100MB |
| 총 | **~1~2분** | |

---

## 8. 학습 파이프라인에서 어떻게 쓰이나

### 8.1 전체 데이터 흐름

```
[data/processed/*.parquet]
        │
        │ ① load_episode (날짜별 24h 슬라이스, 시작 시 1회)
        ▼
[EpisodeData × 60] (train pool) + [EpisodeData × 7] (eval set)
        │
        │ ② RebalanceEnv에 list로 주입
        ▼
[train_env] [eval_env]
        │
        │ ③ DQN.learn() — reset/step 반복
        ▼
[학습된 모델 (.zip)]
```

### 8.2 load_episode — parquet → 메모리 numpy 변환

```python
# data_loader.py:load_episode()
stations = pd.read_parquet("stations.parquet")
stations = stations[stations["gu"] == "마포구"]     # 146개만

demand   = pd.read_parquet("demand_10min.parquet")
rentals, returns = _build_demand_grid(demand, ...)  # sparse → (144, 146) dense

weather  = pd.read_parquet("weather_10min.parquet")
weather_slice, cal_flags = _build_weather_and_calendar(...)

return EpisodeData(
    station_ids, station_coords, distance_matrix, travel_steps,
    rentals, returns, weather_slice,
    dayofweek, is_weekend, is_holiday, is_holiday_eve,
)
```

→ **한 episode당 약 425KB**. 60일 + 7일 = 67개 × 425KB ≈ **메모리 약 30MB**.

### 8.3 train.py 4단계

#### [1/4] 60일 + 7일 episodes 로드 (학습 시작 시 1회)

```python
# scripts/train.py
train_dates = TRAIN_DATES[: args.n_train_dates]   # 60일 random sample
train_episodes = [
    load_episode("data/processed", district="마포구", episode_start=f"{d} 00:00")
    for d in train_dates
]
eval_episodes = [load_episode(...) for d in EVAL_DATES]   # 7일 (학습 분리)
```

#### [2/4] 휴리스틱 baseline 측정

```python
heuristic_reward = evaluate_heuristic(eval_episodes, ...)
# eval set 7일에 most_imbalanced 정책으로 미리 돌려 평균 reward 계산
# → 학습 중 비교 기준선으로 사용
```

#### [3/4] 환경 구성 + DQN 학습

```python
train_env = build_env(train_episodes, ...)   # episode 60개 리스트로 주입
eval_env  = build_env(eval_episodes, ...)    # episode 7개

model = MaskableDQN("MlpPolicy", train_env, ...)
model.learn(total_timesteps=2_000_000, callback=eval_callback)
```

#### [4/4] 모델 저장
```python
model.save(log_root / "masked_dqn_final.zip")
np.save(log_root / "history.npy", history)
```

### 8.4 환경 내부 — 60개 episode 무작위 회전

`RebalanceEnv`는 episode 리스트를 받아 **reset마다 무작위 선택**:

```python
# rebalance_env.py
def reset(self, ...):
    if len(self._episodes) > 1:
        idx = int(self._rng.integers(len(self._episodes)))
        self.data = self._episodes[idx]      # ★ 60개 중 무작위 1개
    self.bikes = self.data.initial_bikes.copy()
    ...
```

→ `env.reset()` 호출마다 60일 중 무작위 하루 → 다양한 환경에서 학습.

### 8.5 매 시뮬레이션 step에서 parquet 활용

#### `_tick`에서 demand 인덱싱 (매 10분 시계 진행)

```python
def _tick(self):
    rentals = self.data.rentals[self.t]      # (146,) — O(1) 인덱싱
    returns = self.data.returns[self.t]      # (146,)
    # demand 처리
    served = np.minimum(rentals, self.bikes)
    stockout = (rentals - served).sum()      # → reward 페널티
    ...
```

#### `_get_obs`에서 캘린더 + 날씨 사용 (DQN에게 obs 전달)

```python
def _get_obs(self):
    # 정류소 상태
    bike_ratio = self.bikes / self.data.capacity

    # 시각
    time_enc = [sin(hour), cos(hour), ...]

    # ★ 캘린더 — EpisodeData에서 (parquet에서 미리 가져옴)
    cal_enc = [sin(2π·dow/7), cos(2π·dow/7),
               is_weekend, is_holiday, is_holiday_eve]

    # ★ 날씨 — 매 step 시계열 인덱싱
    w = self.data.weather[self.t]            # (4,)
    weather_enc = [(w[0]+20)/60, min(w[1]/30, 1), ...]

    return concat([bike_ratio, 트럭정보, time_enc, cal_enc, weather_enc])
    # → 171 dim obs (DQN 입력)
```

### 8.6 학습 1 step에서 일어나는 일 — 시퀀스

```
DQN.learn() 안의 한 RL step:

1. env.step(action) 호출
   └─ RebalanceEnv.step()
      ├─ action 적용 (트럭 출발)
      ├─ _tick() 여러 번:
      │   ├─ self.data.rentals[t] ← 인덱싱
      │   ├─ self.data.returns[t] ← 인덱싱
      │   └─ stockout/full/reward 계산
      └─ _get_obs() ← 171 dim 벡터 생성
          ├─ self.data.weather[t] ← 인덱싱
          ├─ self.data.dayofweek ← 캘린더
          └─ 트럭 상태 + 시간 + 캘린더 + 날씨

2. transition (obs, action, reward, next_obs) → replay buffer

3. train_freq=4마다 학습 1회:
   batch = buffer.sample(64)
   target = r + γ · max Q_target(s')
   loss = MSE(Q(s,a), target)
   gradient step
```

### 8.7 정리 — 효율의 핵심

| 단계 | 시점 | parquet 읽기 | 메모리 |
|---|---|---|---|
| **load_episode** | 학습 시작 1회 | ✅ stations/demand/weather 전체 | peak 약 1GB |
| **EpisodeData 보관** | 학습 전 기간 | ❌ | 약 30MB 유지 |
| **env.reset()** (매 episode) | 학습 중 | ❌ | 포인터 교체만 |
| **env.step()** (매 시뮬 step) | 학습 중 | ❌ | numpy 인덱싱만 |

**핵심 설계**: parquet은 학습 전에 numpy로 한 번만 풀어두고, 학습 중에는 그 array만 인덱싱 → I/O 병목 없이 빠른 학습.

### 8.8 코드 위치 요약

| 동작 | 파일·함수 |
|---|---|
| parquet → EpisodeData | `src/envs/data_loader.py` : `load_episode` |
| episode 풀 구성 | `scripts/train.py` : `[1/4] loading episodes` 부분 |
| 환경에 list 주입 | `scripts/train.py` : `build_env` |
| reset마다 무작위 회전 | `src/envs/rebalance_env.py` : `reset` |
| 매 step parquet 활용 | `src/envs/rebalance_env.py` : `_tick`, `_get_obs` |
| 학습 루프 | SB3 `DQN.learn()` (라이브러리 내부) |

---

## 9. 트러블슈팅

### 9.1 CP949 디코딩 에러
```
UnicodeDecodeError: 'utf-8' codec can't decode byte ...
```
→ `preprocess.py`의 `ENCODING = "cp949"` 확인. UTF-8로 변환된 CSV를 받았다면 이 값을 `"utf-8"`로 바꿔야 함.

### 9.2 weather 컬럼 누락
원본 ASOS CSV에 QC플래그만 있고 값 컬럼 누락된 경우 → 기상자료개방포털에서 "지상관측자료 → 종관기상관측" 다시 다운로드.

### 9.3 정류소 ID 매칭 안 됨
trip의 `대여대여소ID`와 stations의 `대여소_ID`가 같은 형식인지 확인. 일부 데이터는 prefix 다를 수 있음.

### 9.4 메모리 부족
trips 파일이 매우 클 경우 `pd.read_csv(file, chunksize=...)`로 chunk별 처리로 변경. 현재는 파일별 스트리밍으로 충분.

---

## 10. 관련 코드 / 문서

- **`src/data/preprocess.py`** — 전처리 핵심 모듈
- **`scripts/run_preprocess.py`** — CLI 진입점
- **`src/envs/data_loader.py`** — 전처리 결과 → 환경 입력 (EpisodeData) 변환
- **`docs/source_guide.html`** — 함수별 상세 가이드 (인터랙티브)
- **`docs/experiments.md`** — 학습 실험 로그
- **`README.md`** — 프로젝트 전체 흐름
