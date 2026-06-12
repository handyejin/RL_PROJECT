# 따릉이 재배치 강화학습 (Bike Rebalancing RL)

서울시 공공자전거 **따릉이**의 자치구 단위 재배치 문제를 강화학습으로 푸는 프로젝트.
N대의 트럭이 정류소를 돌며 자전거를 적재·하차하여 **대여 실패(stockout)** 와
**반납 실패(full)** 의 24시간 누적치를 최소화하는 정책을 학습한다.
REINFORCE / A2C / DQN / PPO 를 같은 환경·같은 평가 holdout 위에서 비교한다.

---

## 1. 프로젝트 개요

| 항목 | 값 |
|---|---|
| 환경 | `gymnasium` 커스텀 환경 — `src/envs/rebalance_env.py` |
| 시간 해상도 | 1 step = 10분, 1 episode = 24시간 (144 step) |
| 트럭 제어 | Parameter sharing single-agent (트럭 한 대씩 결정) |
| State | 정류소 점유율, 트럭 위치/적재/이동 잔여, 시간·캘린더·날씨 인코딩, 미래 수요 예측 |
| Action | Top-K 후보 정류소 중 한 곳 |
| Reward | -stockout, -full, -이동거리, -이동시간 + 선택적 셰이핑 |
| 데이터 | 2025년 1~12월 따릉이 대여이력 + 정류소 마스터 + 기상 + 공휴일 |
| 평가 | chronological 80/20 split — 마지막 73일 holdout |

알고리즘 4종은 모두 동일한 `RebalanceEnv` 위에서 동작하며, `most_imbalanced`
휴리스틱을 공통 baseline으로 비교한다.

---

## 2. 디렉토리 구조

```
rl_project/
├── config/
│   ├── default.yaml                  # 팀 공통 기본 설정
│   └── ours/                         # 보고서 기준 ours 설정 (실행기 진입점)
│       ├── reinforce_topk12.yaml
│       ├── a2c_topk12.yaml
│       ├── a2c_topk12_vae.yaml
│       ├── dqn_topk12.yaml
│       ├── dqn_topk3.yaml
│       └── ppo_topk12.yaml
│
├── src/
│   ├── envs/
│   │   ├── rebalance_env.py          # Gymnasium 커스텀 환경
│   │   ├── data_loader.py            # parquet → EpisodeData 변환
│   │   └── topk_mask_wrapper.py
│   ├── agents/
│   │   ├── run_interactive.py        # 대화형/CLI 실행기 (모든 알고리즘 통합)
│   │   ├── run_from_config.py        # YAML config 실행기
│   │   ├── algorithms/
│   │   │   ├── reinforce/core.py     # REINFORCE + value baseline
│   │   │   ├── a2c/core.py           # 1-step TD Actor-Critic
│   │   │   ├── dqn/core.py           # MaskableDQN (Double/Dueling)
│   │   │   ├── dqn_small/core.py     # 25구 축소 환경 DQN
│   │   │   └── ppo/core.py           # MaskablePPO
│   │   ├── models/                   # 커스텀 SB3 policy/Q 네트워크
│   │   └── common/                   # 공통 유틸 (state 보강, BC, candidate-K 등)
│   ├── data/                         # 전처리 / 수요예측 / VAE latent 학습
│   └── utils/
│
├── scripts/                          # 전처리, 평가, 보고서 빌드 스크립트
├── data/                             # 원본 CSV, 전처리 parquet, forecast/VAE 파생물
├── logs/                             # 학습 로그 및 가중치 (gitignored)
├── docs/
│   ├── design_notes.md               # 환경/state/reward 설계 노트
│   ├── ours_run_guide.md             # ours 알고리즘 재현 가이드
│   └── ...                           # 실험 보고서, replay 등
└── requirements.txt
```

---

## 3. 설치

Python 3.10+ 권장.

```bash
git clone https://github.com/handyejin/RL_PROJECT.git
cd rl_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

주요 의존성: `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`,
`numpy`, `pandas`, `pyarrow`, `pyyaml`, `holidays`, `tqdm`.

---

## 4. 데이터 준비

### 4.1 원본 CSV 배치

`data/` 아래에 다음 파일들을 둔다.

- `data/trips_2025_01.csv` ~ `data/trips_2025_12.csv` (따릉이 대여이력)
- `data/stations_master.csv` (정류소 마스터)
- `data/OBS_ASOS_TIM_*.csv` (서울 기상)

### 4.2 전처리

```bash
PYTHONPATH=. python scripts/run_preprocess.py \
  --gu all \
  --out data/processed_seoul_all
```

완료 후 `data/processed_seoul_all/{stations,trips,demand_10min,weather_10min}.parquet` 생성.

### 4.3 수요예측 모델 학습 (구별 1시간 예측)

```bash
PYTHONPATH=. python scripts/train_demand_forecast.py \
  --processed-dir data/processed_seoul_all \
  --district 강남구 \
  --model-out data/forecast_by_gu/demand_forecast_1h_강남구.joblib
```

25개 구 모두 만들 때는 위 명령을 구별로 반복 실행.

---

## 5. 실행 방법

### 5.1 대화형 실행 — `run_interactive`

알고리즘·자치구·Top-K·VAE 옵션을 메뉴로 고른다.

```bash
PYTHONPATH=. python -m src.agents.run_interactive
```

CLI 인자로 한 번에 지정해도 된다.

```bash
# DQN, 강남구, Top-K=12
PYTHONPATH=. python -m src.agents.run_interactive \
  --algorithm dqn --district 강남구 --candidate-top-k 12

# 25개 구 전체 순차 실행
PYTHONPATH=. python -m src.agents.run_interactive \
  --algorithm a2c --district ALL --candidate-top-k 12
```

지원 알고리즘: `reinforce`, `a2c`, `dqn`, `ppo`.

### 5.2 YAML config로 실행 — `run_from_config`

보고서 기준 설정은 모두 `config/ours/*.yaml` 에 명시되어 있다.

```bash
# 보고서 기준 A2C
PYTHONPATH=. python -m src.agents.run_from_config \
  --config config/ours/a2c_topk12.yaml

# 보고서 기준 DQN (Top-K=12, seed=42, dqn_small 축소환경)
PYTHONPATH=. python -m src.agents.run_from_config \
  --config config/ours/dqn_topk12.yaml

# 25구 전체 — district override
PYTHONPATH=. python -m src.agents.run_from_config \
  --config config/ours/dqn_topk12.yaml --district ALL
```

---

## 6. 학습된 모델 (Pre-trained weights)

학습 가중치는 용량 문제로 git에 포함되지 않는다. 필요한 알고리즘만 받아 `logs/` 아래에 푼다.

| 알고리즘 | 다운로드                                                                                |
|---|-------------------------------------------------------------------------------------|
| REINFORCE | https://drive.google.com/file/d/17OVftX6w1r-akMI-NwfC8soWZbzYFQzA/view?usp=sharing  |
| A2C       | https://drive.google.com/file/d/1C7wd3HwU3Ff6OAAHdI8kjANVMLIjMaTF/view?usp=sharing  |
| DQN       | https://drive.google.com/file/d/1noCav9oCU9sqGBB6onpNRScGJgNdjXbx/view?usp=drive_link |
| PPO       | https://drive.google.com/file/d/1UV_WRSJU1JuLn3ArVknMIXKyw1Uk5-ba/view?usp=sharing           |

```bash
unzip <algo>.zip -d logs/
```

압축 해제 후 `logs/` 아래 학습 시 사용한 디렉토리 구조가 그대로 들어가므로 `scripts/export_replay.py` / `run_from_config.py` 가 바로 모델을 찾아 쓴다.

---

## 7. 학습 결과 시각화 (Replay Viewer)

학습된 모델로 1 episode를 굴려 트럭 이동·정류소 점유율·누적 reward 변화를 지도 위에서 재생하는 정적 HTML 뷰어가 `docs/ours_replay_viewer.html` 에 들어 있다.

### 7.1 뷰어 서버 띄우기

뷰어는 같은 디렉토리에 있는 replay JSON을 fetch로 읽기 때문에 정적 HTTP 서버가 필요하다. 프로젝트 루트에서:

```bash
python -m http.server 8765 --directory docs
```

브라우저에서 [http://localhost:8765/ours_replay_viewer.html](http://localhost:8765/ours_replay_viewer.html) 열기. 우측 상단 셀렉터에서 보고 싶은 replay를 고른다.

### 7.2 셀렉터에 자동 등장하는 파일명 규칙

뷰어는 `docs/` 디렉토리에서 다음 패턴의 JSON을 자동으로 스캔한다.

```
<구>_<알고리즘>_<YYYY-MM-DD>.json
```

- `<알고리즘>` 은 `REINFORCE` / `A2C` / `PPO` / `DQN` 중 하나 (대소문자 고정)
- 예: `강남구_DQN_2025-10-20.json`, `노원구_A2C_2025-10-20.json`

이 규칙에서 벗어난 파일은 셀렉터에 뜨지 않는다 (URL로 직접 호출은 가능).

### 7.3 새 replay JSON 생성

평가 holdout의 임의 날짜에 대해 학습된 모델 추론을 1 episode 굴려 JSON으로 export 한다.

**일반 환경(전체 정류소)에서 학습한 DQN/MaskableDQN:**

```bash
PYTHONPATH=. python scripts/export_replay.py \
  --algo masked_dqn \
  --model logs/<디렉토리>/best_model.zip \
  --district 강남구 --date 2025-10-20 \
  --out docs/강남구_DQN_2025-10-20.json
```

**`dqn_small` 축소 환경(top-N 정류소 + 소수 트럭)에서 학습한 모델** 은 학습 시점과 동일하게 정류소 subset·wrapper 체인을 재구성해야 하므로 별도 스크립트를 쓴다.

```bash
PYTHONPATH=. python scripts/export_replay_dqn_small.py \
  --model logs/dqn_seed42_k12_dqn_small_강남구/best_model.zip \
  --district 강남구 --date 2025-10-20 \
  --out docs/강남구_DQN_2025-10-20.json
```

> 학습 시 `--max-stations`, `--n-trucks`, `--candidate-*`, `--future-*` 설정과 인자를 동일하게 맞춰야 observation 차원이 일치한다. 차이가 있으면 스크립트가 명확히 에러를 띄운다.

생성된 JSON을 `docs/` 안에 두면 다음 새로고침 시 셀렉터에 자동 등장한다.

---

## 8. 참고 문서

- `docs/design_notes.md` — 환경·state·reward 설계 결정 노트
- `docs/ours_run_guide.md` — ours 알고리즘별 재현 명령 모음
- `docs/ours_replay_viewer.html` — 한 episode 시뮬레이션 시각화

---

## 9. 라이선스 및 데이터 출처

- 원본 데이터: 서울특별시 공공자전거 대여 이력 (서울 열린데이터 광장)
- 기상 데이터: 기상청 ASOS 종관기상관측
