# 팀원용 DQN/PPO 재현 지침서

작성일: 2026-06-06

## 1. 목적

이 문서는 DQN/PPO를 우리 수정 state와 Top-K 후보 action 구조에서 재현하기 위한 최소 실행 지침이다.

기본 원칙은 다음과 같다.

```text
원본 env, 기존 DQN 파일, 공통 train.py는 수정하지 않는다.
실험 코드는 src/agents/ours/ 아래에서 실행한다.
실험 설정은 config/ours/*.yaml에서 관리한다.
```

## 2. 핵심 아이디어

### 2.1 State 보강

정류소별 capacity와 1시간 수요예측을 observation에 추가한다.

```text
pred_net_1h = pred_returns_1h - pred_rentals_1h
projected_bikes = current_bikes + pred_net_1h
```

필요 파일:

```text
data/processed/station_capacity.csv
data/processed_seoul_all/
data/forecast_by_gu/
```

### 2.2 Top-K 후보 action

기존 action은 전체 정류소 중 하나를 직접 선택한다.
Top-K 구조에서는 현재 상태에서 유망한 K개 정류소만 후보로 만들고, agent는 후보 rank를 선택한다.

```text
기존: action = 실제 정류소 index
변경: action = 후보 rank 0..K-1
```

DQN은 action 후보가 많을수록 Q값 학습이 불안정해질 수 있으므로, 먼저 `top_k: 3`을 권장한다.

## 3. 실행 파일 구조

| 파일 | 역할 |
|---|---|
| `src/agents/ours/run_from_config.py` | YAML 설정을 읽어 학습 실행 |
| `src/agents/ours/run_interactive.py` | 터미널 선택형 실행기 |
| `src/agents/ours/algorithms/dqn/core.py` | DQN 학습/evaluation core |
| `src/agents/ours/algorithms/ppo/core.py` | PPO 학습/evaluation core |
| `src/agents/ours/common/candidate_actions.py` | Top-K 후보 action wrapper |
| `config/ours/dqn_topk3.yaml` | DQN Top-K 3 권장 설정 |
| `config/ours/dqn_topk12.yaml` | DQN Top-K 12 비교 설정 |
| `config/ours/ppo_topk12.yaml` | PPO 보고서 기준 설정 |

## 4. 실행 방법

먼저 dry-run으로 명령이 정상 생성되는지 확인한다.

```bash
PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml \
  --dry-run
```

DQN Top-K 3 실행:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml
```

다른 구 실행:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml \
  --district 영등포구
```

Top-K만 바꿔서 실행:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml \
  --candidate-top-k 4
```

25개 구 전체 실행:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml \
  --district ALL
```

PPO 실행:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/ppo_topk12.yaml
```

## 5. YAML에서 바꾸는 값

`config/ours/dqn_topk3.yaml`의 핵심은 아래 부분이다.

```yaml
algorithm: dqn
district: 강남구

candidate_action:
  top_k: 3
  mode: forecast_imbalance
  travel_coef: 0.20
  zone_mode: static3
  zone_penalty: 1.0
  feature_mode: basic
```

`top_k`는 단순 출력 옵션이 아니라 action space 크기를 바꾸는 구조 하이퍼파라미터다.

## 6. 평가 기준

평가는 고정 7개 날짜의 평균 reward로 한다.

```text
Delta = Model Reward - MostImbalanced Baseline Reward
```

| Delta | 의미 |
|---:|---|
| `> 0` | 모델이 baseline보다 좋음 |
| `= 0` | baseline과 비슷함 |
| `< 0` | 모델이 baseline보다 나쁨 |

대표 성능은 학습 중 가장 좋았던 Best checkpoint로 보고, Final checkpoint는 학습 후반 안정성을 볼 때 함께 확인한다.

## 7. 결과 저장 위치

```text
logs/dqn_{tag}/history.npy
logs/dqn_{tag}/best_model.zip
logs/dqn_{tag}/final_model.zip

logs/ppo_{tag}/history.npy
logs/ppo_{tag}/best_model.zip
logs/ppo_{tag}/final_model.zip
```

`logs/`와 모델 파일은 Git에 올리지 않는다.
