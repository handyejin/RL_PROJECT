# 팀원용 DQN/PPO 재현 및 확장 지침서

작성일: 2026-06-06

## 1. 목적

이 문서는 팀원이 원본 GitHub 프로젝트와 본인의 실험 데이터를 기준으로, 내가 사용한 방식의 DQN/PPO 실험을 재현하거나 확장할 수 있도록 만든 실무 지침서다.

팀원들이 AI 코딩 도구를 사용한다는 전제를 두고, 아래 내용을 그대로 AI에게 전달해도 구현 방향을 이해할 수 있게 작성하였다.

핵심 목표는 다음과 같다.

```text
원본 환경과 기존 DQN 파일은 유지하면서,
agent 쪽에 수요예측 feature와 후보 action wrapper를 추가해
DQN/PPO도 REINFORCE/A2C와 같은 기준으로 비교한다.
```

## 2. 반드시 지킬 원칙

| 원칙 | 내용 |
|---|---|
| 원본 환경 보존 | `src/envs/*`는 직접 수정하지 않는다. |
| 기존 DQN 보존 | 팀원이 만든 `src/agents/masked_dqn.py`는 직접 수정하지 않는다. |
| 공통 학습 스크립트 보존 | `scripts/train.py`, `config/*`는 수정하지 않는다. |
| 추가 코드는 분리 | 새 코드는 `src/agents/ours/` 아래에 둔다. |
| 평가 reward 통일 | 학습 중 보조 wrapper를 쓰더라도 평가는 원본 reward로 한다. |
| 결과 해석 분리 | BC 직후 성능과 RL fine-tuning 이후 성능을 반드시 분리해서 본다. |

## 3. 팀원이 이해해야 할 핵심 아이디어

### 3.1 State 보강

기본 state는 현재 재고 중심이다. 재배치 문제에서는 "현재 부족한 곳"보다 "곧 부족해질 곳"을 미리 방문하는 것이 중요할 수 있다.

따라서 agent-local wrapper에서 다음 feature를 observation 뒤에 붙인다.

```text
pred_net_1h = pred_returns_1h - pred_rentals_1h
projected_bikes = current_bikes + pred_net_1h
projected_deviation = (projected_bikes - target_bikes) / capacity
```

필요한 입력 파일은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `station_capacity.csv` | 정류소별 capacity를 읽기 위한 추가 파일 |
| `demand_forecast_1h_*.parquet` | 정류소별 1시간 대여/반납 예측 파일 |

팀원 데이터에 예측 파일이 없으면 먼저 예측 파일을 만들거나, `future-mode none`으로 원본 state 기준 DQN/PPO부터 실행한다.

### 3.2 Action 후보 줄이기

기본 action은 146개 정류소 중 하나를 직접 선택한다. 이 방식은 action space가 커서 DQN/PPO가 불안정해질 수 있다.

개선 방식은 다음과 같다.

```text
1. 현재 state와 1시간 수요예측을 사용해 정류소별 후보 점수를 계산한다.
2. 점수가 높은 상위 K개 정류소를 고른다. 본 실험에서는 K=12를 사용했다.
3. agent action은 실제 정류소 번호가 아니라 후보 rank 0~11이 된다.
4. wrapper가 후보 rank를 실제 정류소 action으로 변환해 원본 env.step()에 전달한다.
```

후보 점수는 다음 개념이다.

```text
candidate_score =
    forecast_imbalance
  - travel_penalty
  - zone_penalty
```

### 3.3 BC와 rollback

BC는 `MostImbalanced` 휴리스틱 행동을 먼저 따라하게 하는 예습 학습이다.

하지만 BC 후 RL fine-tuning이 성능을 떨어뜨릴 수 있으므로, 다음 기준을 사용한다.

| 경우 | 해석 |
|---|---|
| BC 없이 baseline을 넘음 | RL 자체의 개선으로 볼 수 있음 |
| BC 직후보다 RL 후 좋아짐 | BC 이후 RL이 추가 개선한 것 |
| BC 직후가 최고이고 RL 후 나빠짐 | RL 개선이 아니라 BC policy 유지 |

`rollback-to-best-on-eval`은 평가가 나빠질 때 best checkpoint로 되돌리는 장치다.

## 4. 복사/추가해야 하는 파일

원본 프로젝트에 아래 파일들을 추가한다. 기존 파일을 덮어쓰기보다 `src/agents/ours/` 폴더를 새로 추가하는 방식이 안전하다.

### 4.1 공통 파일

| 파일 | 역할 |
|---|---|
| `src/agents/ours/__init__.py` | package marker |
| `src/agents/ours/common/__init__.py` | common package marker |
| `src/agents/ours/common/data_overrides.py` | capacity/forecast 데이터를 episode에 붙임 |
| `src/agents/ours/common/future_demand.py` | observation에 forecast feature 추가 |
| `src/agents/ours/common/candidate_actions.py` | 후보 12개 action wrapper |
| `src/agents/ours/common/bc_utils.py` | BC teacher action 수집 |
| `src/agents/ours/common/reward_shaping.py` | 선택적 PBRS wrapper |
| `src/agents/ours/common/dqn_core.py` | DQN 학습/evaluation core |
| `src/agents/ours/common/ppo_core.py` | PPO 학습/evaluation core |

### 4.2 DQN/PPO 실행 파일

최소 재현에는 아래 4개 파일이면 된다.

| 파일 | 목적 |
|---|---|
| `src/agents/ours/experiments/dqn_topk_forecast_plus_stable_no_bc.py` | BC 없는 DQN 안정화 실험 |
| `src/agents/ours/experiments/dqn_topk_forecast_plus_stable_bc.py` | BC 포함 DQN 안정화 실험 |
| `src/agents/ours/experiments/ppo_topk_forecast_plus_conservative_no_bc.py` | BC 없는 PPO 보수적 업데이트 실험 |
| `src/agents/ours/experiments/ppo_topk_forecast_plus_conservative_bc.py` | BC 포함 PPO 보수적 업데이트 실험 |

선택 실험으로는 다음 파일을 추가할 수 있다.

| 파일 | 목적 |
|---|---|
| `src/agents/ours/experiments/dqn_topk_forecast_plus_pbrs_no_bc.py` | DQN PBRS no-BC 실험 |
| `src/agents/ours/experiments/ppo_topk_forecast_plus_pbrs_no_bc.py` | PPO PBRS no-BC 실험 |

현재 결과 기준으로 PBRS no-BC는 권장하지 않는다. 성능이 기존 안정화 설정보다 낮았다.

## 5. 의존성

원본 프로젝트 환경에 다음 패키지가 필요하다.

```bash
pip install stable-baselines3 sb3-contrib torch numpy pandas pyarrow gymnasium
```

이미 `.venv`가 있으면 다음처럼 확인한다.

```bash
.venv/bin/python - <<'PY'
import torch
import stable_baselines3
import sb3_contrib
import pandas
import pyarrow
print("dependencies ok")
PY
```

## 6. 데이터 준비

### 6.1 필수 전처리 산출물

기본 env 실행에는 원본 전처리 산출물이 필요하다.

```text
data/processed/station_capacity.csv
data/processed/trips.parquet
data/processed/demand_10min.parquet
data/processed/weather_10min.parquet
```

### 6.2 수정 state 실험용 추가 파일

내 실험 방식의 DQN/PPO를 실행하려면 다음 두 파일이 필요하다.

```text
station_capacity.csv                     # 정류소별 capacity 추가 테이블
demand_forecast_1h_rlholdout_seed42.parquet # 1시간 수요예측
```

현재 내 실행 파일은 기본 경로가 다음처럼 되어 있다.

```text
--capacity-path data/processed/station_capacity.csv
--forecast-path data/processed/demand_forecast_1h_rlholdout_seed42.parquet
```

팀원 환경에서는 파일 위치가 다를 수 있으므로 실행 시 필요하면 경로를 본인 환경에 맞게 바꾼다.

예:

```bash
--capacity-path data/processed/station_capacity.csv
--forecast-path data/processed/demand_forecast_1h_rlholdout_seed42.parquet
```

예측 파일이 없으면 아래 스크립트로 다시 생성한다.

```bash
PYTHONPATH=. .venv/bin/python scripts/train_demand_forecast.py \
  --processed-dir data/processed \
  --district 마포구 \
  --seed 42 \
  --horizon-steps 6 \
  --holdout-from-rl-split \
  --n-train-dates 200 \
  --n-eval-dates 7 \
  --forecast-out data/processed/demand_forecast_1h_rlholdout_seed42.parquet \
  --model-out data/processed/demand_forecast_h1_rlholdout_seed42.joblib \
  --metrics-out logs/demand_forecast_h1_rlholdout_seed42_metrics.json
```

필수 컬럼:
- t 또는 timestamp
- station_id
- pred_rentals_1h
- pred_returns_1h
- pred_net_1h = pred_returns_1h - pred_rentals_1h

주의:
- 평가 날짜를 forecast 평균 계산에 포함할지 여부를 명시해줘.
- 실제 미래 6 step을 직접 합산한 oracle forecast를 쓰는 경우,
  보고서에는 upper-bound 또는 oracle 실험이라고 명확히 표시해줘.

## 7. 실행 명령

아래 명령은 프로젝트 루트에서 실행한다.

### 7.1 DQN no-BC 안정화

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python \
  -m src.agents.ours.experiments.dqn_topk_forecast_plus_stable_no_bc \
  --total-timesteps 170000 \
  --eval-every 20000 \
  --n-train-dates 200 \
  --capacity-path data/processed/station_capacity.csv \
  --forecast-path data/processed/demand_forecast_1h_rlholdout_seed42.parquet \
  --tag teammate_dqn_no_bc \
  --device cpu
```

### 7.2 DQN BC 안정화

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python \
  -m src.agents.ours.experiments.dqn_topk_forecast_plus_stable_bc \
  --total-timesteps 170000 \
  --eval-every 20000 \
  --n-train-dates 200 \
  --capacity-path data/processed/station_capacity.csv \
  --forecast-path data/processed/demand_forecast_1h_rlholdout_seed42.parquet \
  --tag teammate_dqn_bc \
  --device cpu
```

### 7.3 PPO no-BC 보수적 설정

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python \
  -m src.agents.ours.experiments.ppo_topk_forecast_plus_conservative_no_bc \
  --total-timesteps 170000 \
  --eval-every 20000 \
  --n-train-dates 200 \
  --capacity-path data/processed/station_capacity.csv \
  --forecast-path data/processed/demand_forecast_1h_rlholdout_seed42.parquet \
  --tag teammate_ppo_no_bc \
  --device cpu
```

### 7.4 PPO BC 보수적 설정

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python \
  -m src.agents.ours.experiments.ppo_topk_forecast_plus_conservative_bc \
  --total-timesteps 170000 \
  --eval-every 20000 \
  --n-train-dates 200 \
  --capacity-path data/processed/station_capacity.csv \
  --forecast-path data/processed/demand_forecast_1h_rlholdout_seed42.parquet \
  --tag teammate_ppo_bc \
  --device cpu
```

## 8. 핵심 하이퍼파라미터

### 8.1 DQN no-BC 안정화

| 항목 | 값 |
|---|---:|
| Double DQN | True |
| learning rate | 0.00005 |
| n-step | 3 |
| train_freq | 1 |
| gradient_steps | 1 |
| target_update_interval | 500 |
| exploration_initial_eps | 0.30 |
| exploration_fraction | 0.60 |
| exploration_final_eps | 0.02 |
| candidate_top_k | 12 |
| candidate_mode | forecast_imbalance |
| rollback_to_best_on_eval | True |

### 8.2 DQN BC 안정화

| 항목 | 값 |
|---|---:|
| BC epochs | 10 |
| BC policy | masked_heuristic |
| learning rate | 0.00003 |
| exploration_initial_eps | 0.05 |
| exploration_final_eps | 0.005 |
| finetune_patience | 2 |
| rollback_to_best_on_eval | True |

### 8.3 PPO no-BC 보수적 설정

| 항목 | 값 |
|---|---:|
| learning rate | 0.00005 |
| clip_range | 0.05 |
| ent_coef | 0.003 |
| n_epochs | 3 |
| target_kl | 0.01 |
| n_steps | 512 |
| batch_size | 256 |
| candidate_top_k | 12 |
| rollback_to_best_on_eval | True |

### 8.4 PPO BC 보수적 설정

| 항목 | 값 |
|---|---:|
| BC epochs | 10 |
| BC policy | masked_heuristic |
| learning rate | 0.00003 |
| clip_range | 0.05 |
| ent_coef | 0.0 |
| n_epochs | 3 |
| target_kl | 0.01 |
| finetune_patience | 4 |
| rollback_to_best_on_eval | True |

## 9. 결과 파일 위치

실행 후 결과는 `logs/` 아래에 저장된다.

예:

```text
logs/dqn_teammate_dqn_no_bc/history.npy
logs/dqn_teammate_dqn_no_bc/best_model.zip
logs/dqn_teammate_dqn_no_bc/final_model.zip

logs/ppo_teammate_ppo_bc/history.npy
logs/ppo_teammate_ppo_bc/best_model.zip
logs/ppo_teammate_ppo_bc/final_model.zip
```

`history.npy`에는 평가 시점별 reward가 들어 있다.

간단히 확인하는 스크립트:

```bash
.venv/bin/python - <<'PY'
import numpy as np
from pathlib import Path

for p in sorted(Path("logs").glob("*/history.npy")):
    if "teammate" not in str(p):
        continue
    rows = []
    for x in np.load(p, allow_pickle=True):
        d = dict(x) if not isinstance(x, dict) else x
        if "eval_reward" in d:
            rows.append((d.get("timesteps"), float(d["eval_reward"]), d.get("stage", "")))
    if not rows:
        continue
    best = max(rows, key=lambda r: r[1])
    final = rows[-1]
    print(p.parent.name, "best=", best, "final=", final)
PY
```

## 10. 결과 해석 기준

기본 비교값은 같은 환경의 `MostImbalanced` baseline이다.

```text
Delta = model_reward - baseline_reward
Delta > 0 이면 baseline보다 좋다.
```

BC 실험은 다음처럼 해석한다.

| 관찰 | 해석 |
|---|---|
| BC 직후보다 RL 후 Best가 증가 | RL fine-tuning 개선 있음 |
| BC 직후가 Best이고 RL 후 악화 | RL 개선 아님. BC policy 유지 |
| Final이 Best와 같음 | rollback으로 best checkpoint를 final로 복구한 것일 수 있음 |
| PBRS no-BC가 낮음 | reward shaping scale 또는 방향이 맞지 않는 것 |

## 11. 우리 실험에서 나온 기준 결과

팀원이 같은 데이터를 쓰지 않으면 reward scale은 달라질 수 있다. 반드시 본인 baseline 대비 Delta로 비교한다.

| 실험 | Best Reward | Baseline 대비 |
|---|---:|---:|
| DQN no-BC 안정화 | -435.9 | +12.4 |
| DQN BC 안정화 | -417.7 | +30.6 |
| PPO no-BC 보수적 | -445.2 | +3.1 |
| PPO BC 보수적 | -404.0 | +44.3 |

주의:

```text
DQN BC는 baseline은 넘었지만 BC 이후 RL 개선은 없었다.
PPO BC는 BC 직후 -417.3에서 RL 후 -404.0으로 개선되었다.
```

## 12. AI 코딩 도구에 줄 수 있는 구현 지시문

팀원이 AI 코딩 도구에 바로 전달할 수 있는 형태의 지시문이다.

```text
원본 환경 파일(src/envs/*), 기존 DQN 파일(src/agents/masked_dqn.py),
scripts/train.py, config/*는 수정하지 마라.

src/agents/ours/ 아래에 agent-local wrapper와 DQN/PPO 실행 파일을 추가해라.

필수 구현:
1. data_overrides.py
   - episode data에 정류소별 capacity와 1시간 forecast grid를 attach한다.
2. future_demand.py
   - observation 뒤에 forecast_projected_travel feature를 추가한다.
3. candidate_actions.py
   - 원본 Discrete(N) station action을 Discrete(12) candidate-rank action으로 감싼다.
   - candidate는 forecast_imbalance, travel penalty, static3 zone penalty로 고른다.
4. dqn_core.py
   - 기존 MaskableDQN을 import해서 사용한다.
   - Double DQN 기본값을 True로 둔다.
   - n_steps, exploration_initial_eps, rollback_to_best_on_eval 옵션을 지원한다.
5. ppo_core.py
   - sb3_contrib MaskablePPO를 사용한다.
   - action_masks를 predict와 learn에 적용한다.
   - conservative PPO 설정: lr, clip_range, target_kl, n_epochs를 CLI로 조정 가능하게 한다.
6. BC
   - MostImbalanced teacher action을 수집해 policy를 사전학습한다.
   - BC 직후 eval reward를 history에 stage='bc'로 저장한다.
7. 평가
   - 학습 중 reward shaping이 있더라도 평가에서는 원본 reward 기준으로 7일 평균 reward를 출력한다.
   - 날짜별 reward, 평균 reward, baseline 대비 Delta를 출력한다.
8. 안전장치
   - eval reward가 best보다 낮으면 best checkpoint로 rollback할 수 있게 한다.
   - BC 이후 개선이 없으면 patience로 early stop한다.

실행 파일:
- dqn_topk_forecast_plus_stable_no_bc.py
- dqn_topk_forecast_plus_stable_bc.py
- ppo_topk_forecast_plus_conservative_no_bc.py
- ppo_topk_forecast_plus_conservative_bc.py

마지막으로 py_compile, smoke run, 170k full run을 순서대로 실행하고,
MostImbalanced baseline 대비 Delta와 BC 이후 RL 개선 여부를 표로 정리해라.
```

## 13. 검증 체크리스트

실험 전에 다음을 확인한다.

| 체크 | 명령/기준 |
|---|---|
| 공통 코드 미수정 | `git diff -- src/envs src/agents/masked_dqn.py scripts/train.py config --stat`가 비어 있어야 함 |
| 문법 검사 | `PYTHONPATH=. .venv/bin/python -m py_compile src/agents/ours/common/dqn_core.py src/agents/ours/common/ppo_core.py` |
| DQN smoke | `--total-timesteps 1000 --eval-every 1000 --n-train-dates 10` |
| PPO smoke | `--total-timesteps 1024 --eval-every 1024 --n-train-dates 10` |
| 결과 파일 | `logs/<agent_tag>/history.npy` 생성 |
| 평가 기준 | 7일 평균 reward와 baseline 대비 Delta 출력 |

## 14. 최종 정리

팀원이 내 방식으로 DQN/PPO를 하려면 핵심은 알고리즘 자체보다 다음 세 가지다.

```text
1. state에 1시간 수요예측과 capacity 정보를 넣는다.
2. action을 전체 정류소 선택에서 후보 12개 선택으로 줄인다.
3. DQN/PPO는 update가 강하므로 보수적 하이퍼파라미터와 rollback을 사용한다.
```

이 기준으로 실행하면 DQN/PPO도 REINFORCE/A2C와 같은 실험 구조에서 비교할 수 있다.
