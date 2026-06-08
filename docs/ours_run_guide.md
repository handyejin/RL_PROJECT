# 우리 실험 코드 실행 가이드

이 문서는 `src/agents/ours/` 아래에 추가한 실험 코드를 팀원이 재현하기 위한 실행 순서이다.

핵심 목표는 다음 알고리즘을 수정 state에서 실행하는 것이다.

| 알고리즘 | 설명 |
|---|---|
| REINFORCE | Reward-to-Go와 Value Network baseline을 사용하는 policy gradient |
| A2C | Actor-Critic 구조, `r + gamma V(s') - V(s)` advantage 사용 |
| DQN | action mask를 적용한 Double DQN + Dueling Q 안정화 옵션 |
| PPO | action mask를 적용한 MaskablePPO |

## 1. 실험에서 사용하는 데이터

실험은 원본 따릉이 CSV를 바로 사용하지 않고, 먼저 parquet 형태로 전처리한 뒤 학습한다.

| 데이터 | 기본 경로 | 설명 |
|---|---|---|
| 원본 대여/반납 CSV | `data/trips_2025_*.csv` | 2025년 월별 따릉이 이용 기록 |
| 정류소 마스터 | `data/stations_master.csv` | 정류소 위치, 구 정보, 거치대 수 등 |
| 전처리 결과 | `data/processed_seoul_all/` | 서울 전체 episode 생성을 위한 parquet |
| 구별 1시간 수요예측 | `data/forecast_by_gu/` | 각 구별 `rent/return/net` 예측 feature |
| 정류소 capacity | `data/processed/station_capacity.csv` | 정류소별 최대 거치 가능 수 |

## 2. 전처리 실행

서울 전체 데이터를 새로 만들 때는 다음 명령을 실행한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python scripts/run_preprocess.py \
  --gu all \
  --out data/processed_seoul_all
```

완료 후 아래 파일들이 있어야 한다.

```text
data/processed_seoul_all/stations.parquet
data/processed_seoul_all/trips.parquet
data/processed_seoul_all/demand_10min.parquet
data/processed_seoul_all/weather_10min.parquet
```

## 3. 수요예측 파일 생성

각 구별 학습에는 1시간 수요예측 parquet이 필요하다.

예를 들어 영등포구만 만들려면:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python scripts/train_demand_forecast.py \
  --processed-dir data/processed_seoul_all \
  --district 영등포구 \
  --max-train-rows 500000 \
  --max-eval-rows 200000 \
  --max-iter 140 \
  --model-out data/forecast_by_gu/demand_forecast_1h_영등포구.joblib \
  --forecast-out data/forecast_by_gu/demand_forecast_1h_영등포구.parquet \
  --metrics-out data/forecast_by_gu/demand_forecast_1h_영등포구_metrics.json
```

25개 구 전체 forecast와 A2C 결과를 한 번에 만들려면 아래 runner를 사용할 수 있다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_gu_a2c_full \
  --episodes 500 \
  --eval-every 50 \
  --n-train-dates 200 \
  --run-tag gu_a2c_topk_no_bc_2026-06-06 \
  --device cpu
```

## 4. 쉽게 실행하는 방법

터미널 선택형 wrapper를 추가했다.

```bash
PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_interactive
```

실행하면 다음을 고른다.

```text
1. REINFORCE
2. A2C
3. DQN (Double DQN)
4. PPO

1. ALL
2. 영등포구
3. 마포구
4. 관악구
5. 직접 입력
```

명령형으로 바로 실행할 수도 있다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_interactive \
  --algorithm a2c \
  --district 영등포구 \
  --episodes 500 \
  --eval-every 50 \
  --progress
```

### 4.1 YAML 파일로 실행하는 방법

Top-K처럼 실험마다 바꾸는 값은 YAML 파일로 관리할 수 있다.

팀원 공통 `config/default.yaml`은 수정하지 않고, 우리 실험 설정은 `config/ours/` 아래에 둔다.

| 파일 | 용도 |
|---|---|
| `config/ours/dqn_topk3.yaml` | DQN action 후보를 3개로 줄인 실험 |
| `config/ours/dqn_topk12.yaml` | 기존 DQN Top-K 12 비교 실험 |
| `config/ours/reinforce_topk12.yaml` | REINFORCE 보고서 기준 실험 |
| `config/ours/a2c_topk12.yaml` | A2C 보고서 기준 실험 |
| `config/ours/ppo_topk12.yaml` | PPO 보고서 기준 실험 |

예를 들어 DQN Top-K 3 실험은 다음처럼 실행한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml
```

YAML 안의 `district`를 바꾸면 다른 구를 실행할 수 있다.

```yaml
algorithm: dqn
district: 강남구

candidate_action:
  top_k: 3
```

CLI에서 임시로 override할 수도 있다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml \
  --district 영등포구 \
  --candidate-top-k 4
```

25개 구 전체를 같은 설정으로 돌리고 싶으면 `district: ALL`로 바꾸거나 CLI에서 지정한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml \
  --district ALL
```

`--dry-run`을 붙이면 실제 학습은 하지 않고 실행될 명령만 확인한다.

```bash
PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml \
  --dry-run
```

Top-K는 다음 의미를 가진다.

| 값 | 의미 |
|---:|---|
| `top_k: 12` | 후보를 넓게 둠. A2C/PPO/REINFORCE 기본 비교에 사용 |
| `top_k: 3` | 후보를 강하게 줄임. DQN처럼 큰 action space에 약한 알고리즘 점검용 |

즉, Top-K는 단순 출력 옵션이 아니라 **action space 크기를 바꾸는 구조 하이퍼파라미터**다.

REINFORCE는 다음처럼 실행한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_interactive \
  --algorithm reinforce \
  --district 영등포구 \
  --episodes 500 \
  --eval-every 50 \
  --progress
```

DQN은 timestep 기준으로 실행한다. wrapper에서는 다른 알고리즘의 학습 방식은 바꾸지 않고, DQN에만 다음 안정화 옵션을 적용한다.

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--double-q` | on | target overestimation 완화 |
| `--dueling-q` | on | `Q(s,a)=V(s)+A(s,a)-mean(A)` 구조 사용 |
| `--dqn-reward-scale` | `0.01` | 학습 TD target scale만 축소, 평가 reward는 원본 유지 |
| `--dqn-exploration-initial-eps` | `0.3` | Top-K 후보 구조에 맞춰 초기 랜덤 탐색 완화 |
| `--dqn-exploration-final-eps` | `0.02` | 후반 랜덤 탐색 완화 |

rollback과 early stopping은 사용하지 않는다. 학습은 끝까지 진행하고, 대표 성능은 저장된 Best checkpoint로 평가한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_interactive \
  --algorithm dqn \
  --district 영등포구 \
  --total-timesteps 170000 \
  --eval-every-timesteps 20000 \
  --progress
```

PPO도 timestep 기준으로 실행한다. 기본 설정은 Top-K action 후보에서 정책이 너무 크게 흔들리지 않도록 보수적인 update 값을 사용한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_interactive \
  --algorithm ppo \
  --district 영등포구 \
  --total-timesteps 170000 \
  --eval-every-timesteps 20000 \
  --progress
```

PPO 기본 하이퍼파라미터는 다음과 같다.

| 옵션 | 값 | 의미 |
|---|---:|---|
| `learning_rate` | `1e-4` | policy update 크기 완화 |
| `ent_coef` | `0.003` | 후반 탐색 강도 감소 |
| `target_kl` | `0.03` | policy가 너무 멀리 바뀌면 update 중단 |
| `clip_range` | `0.1` | PPO clipping 범위 축소 |
| `n_epochs` | `5` | 같은 rollout 반복 학습 감소 |
| `n_steps` | `256` | 더 짧은 rollout 단위 |

25개 구 전체를 순차 실행하려면:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_interactive \
  --algorithm a2c \
  --district ALL \
  --episodes 500 \
  --eval-every 50 \
  --progress
```

## 5. 진행률 확인

`--progress` 옵션을 사용하면 `tqdm`으로 진행률을 볼 수 있다.

REINFORCE/A2C는 episode 기준, DQN/PPO는 timestep 기준으로 표시된다.

표시되는 값은 다음과 같다.

| 표시값 | 의미 |
|---|---|
| `eval` | 현재 checkpoint의 7일 평균 평가 reward |
| `base` | 같은 구의 MostImbalanced baseline reward |
| `delta` | `eval - base`, 0보다 크면 baseline보다 좋음 |
| `best` | 지금까지 가장 좋은 baseline 대비 delta |

학습 마지막에 출력되는 날짜별 평가표는 모든 알고리즘에서 **Best checkpoint** 기준으로 통일했다.

```text
best reward: 학습 중 평가 reward가 가장 좋았던 checkpoint
final reward: 마지막 학습 step/episode의 checkpoint
```

따라서 메인 성능 비교는 `*_best` 표를 사용하고, `final reward`는 학습 후반 안정성을 해석할 때 함께 본다. 이 방식은 REINFORCE/A2C/PPO/DQN 모두 동일하다.

예:

```text
A2C 영등포구: 50/500 [eval=-2390.2, base=-2440.1, delta=+49.9, best=+49.9]
DQN 영등포구: 20000/170000 [eval=-2393.2, base=-2440.1, delta=+46.9, best=+46.9]
```

## 6. Episode Cache

구별 episode 로딩은 기본적으로 `data/episode_cache/`에 cache된다.

첫 실행은 parquet에서 episode를 만들기 때문에 기존과 비슷하게 걸리지만, 같은 구/날짜/전처리 경로로 다시 실행하면 cache를 바로 읽는다.

```text
DQN 금천구 load train cache: hit=200, miss=0, dir=data/episode_cache
DQN 금천구 load eval cache: hit=7, miss=0, dir=data/episode_cache
```

cache는 순수 episode만 저장한다. capacity와 forecast는 기존처럼 cache 로딩 후 다시 적용되므로 수요예측 파일을 바꿔도 episode cache를 재사용할 수 있다.

cache를 끄고 싶으면 다음 옵션을 붙인다.

```bash
--no-episode-cache
```

## 7. Baseline 해석

Reward는 구마다 scale이 다르다.

따라서 raw reward만 비교하지 않고, 같은 구의 `MostImbalanced` baseline과 비교한다.

```text
Delta = Model Reward - MostImbalanced Baseline Reward
```

| Delta | 의미 |
|---:|---|
| `> 0` | 모델이 baseline보다 좋음 |
| `= 0` | baseline과 비슷함 |
| `< 0` | 모델이 baseline보다 나쁨 |

## 8. 결과 파일

학습 결과는 `logs/` 아래에 저장된다.

```text
logs/a2c_{tag}/history.npy
logs/a2c_{tag}/best/best_model.pt
logs/a2c_{tag}/actor_critic_final.pt

logs/reinforce_{tag}/history.npy
logs/reinforce_{tag}/best/best_model.pt
logs/reinforce_{tag}/reinforce_final.pt

logs/dqn_{tag}/history.npy
logs/dqn_{tag}/best_model.zip
logs/dqn_{tag}/final_model.zip

logs/ppo_{tag}/history.npy
logs/ppo_{tag}/best_model.zip
logs/ppo_{tag}/final_model.zip
```

`logs/`와 모델 파일은 git에 올리지 않는다.

25개 구 batch runner 결과 요약은 `docs/` 아래에 저장된다.

```text
docs/gu_a2c_topk_no_bc_2026-06-06_summary.csv
docs/gu_a2c_topk_no_bc_2026-06-06_summary.md
```

## 9. 빠른 테스트

PR 전 간단히 실행만 확인하려면 episode를 작게 줄인다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_interactive \
  --algorithm a2c \
  --district 영등포구 \
  --episodes 2 \
  --eval-every 1 \
  --progress
```

## 10. 주의사항

- `data/processed_seoul_all/`와 `data/forecast_by_gu/`는 용량이 커서 git에 올리지 않는다.
- 수요예측 파일이 없으면 `run_interactive.py`가 어떤 파일이 없는지 알려준다.
- 평가 reward는 7일 평가셋 기준이다.
- `Best Ep`는 `eval-every` 간격으로 평가한 checkpoint 중 최고 시점이다.
