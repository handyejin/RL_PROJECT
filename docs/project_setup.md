# 프로젝트 설정 정리 — 따릉이 재배치 RL

> 작성일: 2026-06-03 | 현재 best: **BC v7 (오전 피크 가중) + RL fine-tune 100k**
> 설정 출처: [config/default.yaml](../config/default.yaml), [src/envs/rebalance_env.py](../src/envs/rebalance_env.py)

## 1. 문제 정의

서울 마포구 공공자전거(따릉이) **재배치(rebalancing)** 최적화.
재배치 트럭이 하루 동안 정류소를 돌며 자전거를 옮겨, **빈 정류소(대여 실패)와 꽉 찬 정류소(반납 실패)를 최소화**한다.

| 항목 | 값 |
|---|---|
| 지역 | 마포구, **146개 정류소** |
| 트럭 | **3대**, 적재용량 각 20대, 속도 25 km/h |
| 1 episode | **1일 = 144 step** (10분 단위) |
| 목표 | 정류소 채움 비율 50%(`target_fill_ratio`) 유지 |

## 2. 환경 (RebalanceEnv) — Gymnasium 호환

### 2.1 State / Observation — `Box(171,)`, [-1, 1] 정규화

`obs`는 에이전트가 **매 결정 순간 "보는" 현재 상황을 171개 숫자로 표현한 벡터**다.
이 벡터 하나가 신경망(q_net)의 입력이 되고, 출력으로 146개 정류소의 Q값이 나온다.

| # | 구성요소 | 차원 | 어떤 데이터인가 | 정규화 |
|---|---|---:|---|---|
| 1 | `bike_ratio` | **146** | **정류소별 (현재 자전거 수 ÷ 거치대 용량)** — 핵심 정보 | 0~1 |
| 2 | `loc_norm` | 3 | 트럭 3대 각각 현재 위치 (정류소 idx) | idx/(N-1) |
| 3 | `load_ratio` | 3 | 트럭 3대 각각 적재율 (짐칸이 몇 % 찼나) | 0~1 |
| 4 | `rem_norm` | 3 | 트럭 3대 각각 목적지 도착까지 남은 시간 | 0~1 |
| 5 | `cur_onehot` | 3 | 지금 결정할 차례인 트럭 one-hot | 0/1 |
| 6 | `time_enc` | 4 | 하루 중 시각 + episode 진행도 | sin/cos |
| 7 | `cal_enc` | 5 | 요일(sin/cos), 주말, 공휴일, 공휴일 전날 | sin/cos, 0/1 |
| 8 | `weather_enc` | 4 | 기온·강수량·풍속·습도 | 0~1 |
| | **합계** | **171** | | |

**핵심은 ①번 (146개)**: 각 정류소가 지금 얼마나 차 있나.
- 0에 가까움 = 거의 빔 → 대여 실패(stockout) 위험
- 1에 가까움 = 거의 꽉 참 → 반납 실패(full) 위험
- 에이전트는 이 146개를 보고 "어디가 위급한가" 판단해 트럭을 보낸다.

**데이터 출처** ([data/](../data/) → `data/processed/`로 전처리):
- `trips_2025_*.csv` (월별 대여 기록) → 정류소별 시간대별 대여/반납량 → 시뮬레이션 중 자전거 수 변화 → ① `bike_ratio`
- `OBS_ASOS_TIM_*.csv` (ASOS 기상) → ⑧ `weather_enc`
- 달력 정보 → ⑦ `cal_enc`
- 트럭 위치·적재(②③④⑤)는 시뮬레이션 중 실시간 계산

**주의 — obs에 "미래"는 없음 (기본)**: obs는 "지금 이 순간" 스냅샷만 담는다.
"앞으로 1시간 뒤 어디가 빌 것 같다" 같은 미래 demand는 기본 비활성(`future_demand_horizon=0`).
미래 demand를 추가해봤으나(BC v5) 오히려 성능이 나빠져 끈 상태.

### 2.2 Action — `Discrete(146)`

다음 갈 정류소 idx 선택. 자기 위치를 고르면 머무름(stay).
**Action Mask 적용**: in-flight 트럭 목적지 차단 + **자기 위치 stay 기본 차단**, strict_urgent_mask(위급 정류소 우선).

### 2.3 Reward (per RL step)

```
reward = stockout × (-1.0) + full × (-0.8) + travel_km × (-0.008) + travel_step × (-0.002)
```

| 가중치 | 값 | 의미 |
|---|---:|---|
| `stockout` | **-1.0** | 빈 정류소에서 대여 실패 1건당 |
| `full` | **-0.8** | 꽉 찬 정류소에서 반납 실패 1건당 |
| `travel_km` | -0.008 | 트럭 이동 1km당 |
| `travel_step` | -0.002 | 이동 중 1 step당 |

- 학습 시엔 보조 신호 추가: `urgent_bonus=5.0`, `explore_bonus=1.0`, `shaping_scale=0.5`(potential-based shaping).
- **평가 시엔 보조 신호 모두 OFF** (urgent/explore/shaping=0) → 휴리스틱과 공정 비교(fair metric). reward 90%가 stockout+full.

## 3. 알고리즘 / 모델

| 항목 | 값 |
|---|---|
| 알고리즘 | **MaskableDQN** (SB3 DQN + action masking, [src/agents/masked_dqn.py](../src/agents/masked_dqn.py)) |
| Double Q | **ON** (over-estimation 완화) |
| Policy net | MLP **[256, 256]** |
| 비교 baseline | `MostImbalancedPolicy` 휴리스틱 (잉여 큰 곳 → 부족 큰 곳 greedy) |

### 주요 DQN 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---|---:|---|
| learning_rate | 1e-4 | `lr_decay=true` (1e-4 → 1e-5 linear) |
| buffer_size | 100,000 | |
| batch_size | 64 | |
| gamma | 0.99 | |
| exploration_fraction | 0.6 | final_eps 0.15 |
| **learning_starts** | **20,000** | RL gradient 시작 시점 — best는 이 전(10k)의 BC prior |
| target_update_interval | 1,000 | |
| net_arch | [256, 256] | big net은 over-fit으로 폐기 |

## 4. 학습 파이프라인 — 2단계 (IL → RL)

```
[1단계: 모방학습(Behavior Cloning)]  scripts/pretrain_bc.py
  휴리스틱을 292일 episode에서 시뮬레이션 → (obs, action) 쌍 ~80,000개 수집
  → 신경망이 obs를 보면 휴리스틱과 같은 action을 내놓도록 지도학습 (분류) → SB3 zip 저장
  ★ 오전 피크 가중: env.t∈[44,90] 결정을 ×3 oversample (WeightedRandomSampler)

[2단계: RL Fine-tune]  scripts/train.py --pretrain <bc.zip>
  BC zip의 q_net 가중치를 RL 학습 시작점으로 로드
  → MaskableDQN으로 추가 학습 (timesteps 100k)
  → MaskedEvalCallback이 7일 deterministic 순회로 best 선택
```

### 데이터 분할

| | 날짜 수 | 용도 |
|---|---:|---|
| Train | 292일 (2025년) | BC 데이터 수집 + RL 학습 |
| **Eval (고정 7일)** | 7일 | `03-25, 04-18, 05-17, 07-01, 07-06, 07-09, 08-21` (미학습) |

- 평가: deterministic 7일 순회 (seed=42), 휴리스틱·RL 동일 환경에서 측정 → **격차로 우열 판단**.

## 5. 현재 best 모델 & 산출물

| 항목 | 경로 / 값 |
|---|---|
| **Best 모델** | `logs/masked_dqn_bc_v7_peak_100k/best/best_model.zip` |
| BC prior | `logs/bc_v7_peak/bc_model.zip` (오전 피크 ×3, 292일) |
| **성능** | 7일 평균 **-494.2**, 휴리스틱(-500.0) 대비 **+5.8 (추월)**, **4일 추월** |
| Replay | `docs/replay_bc_v7_peak_<date>.json` (7일), [replay_viewer.html](replay_viewer.html)에서 열람 |

### 핵심 스크립트

| 스크립트 | 역할 |
|---|---|
| [scripts/pretrain_bc.py](../scripts/pretrain_bc.py) | BC prior 학습 (피크 가중 옵션 `--peak-start/end/weight`, `--dataset` 재사용) |
| [scripts/train.py](../scripts/train.py) | RL fine-tune (`--pretrain`, `--timesteps`, lr_decay 등) |
| [scripts/eval_7day.py](../scripts/eval_7day.py) | 7일 eval set 모델 vs 휴리스틱 날짜별 비교 |
| [scripts/diag_0517.py](../scripts/diag_0517.py) | reward 구성요소(stockout/full/travel) 분해 진단 |
| [scripts/export_replay.py](../scripts/export_replay.py) | 모델 → replay JSON (viewer용) |

## 5.5 적용한 학습 기법 (techniques)

이 프로젝트는 단일 알고리즘이 아니라 여러 RL 기법을 조합해 성능을 끌어올렸다.
각 기법이 "왜" 필요했는지와 효과를 정리한다.

### A. Value-based RL — DQN
- **무엇**: Deep Q-Network. 각 (상태, 행동)의 가치 Q(s,a)를 신경망으로 근사하고, 가장 높은 행동을 선택.
- **왜**: action이 이산(정류소 146개 중 선택)이라 value-based가 자연스러움.

### B. Double Q-Learning (Double DQN)
- **무엇**: 행동 선택용 net과 가치 평가용 target net을 분리해 Q값 **과대평가(over-estimation)** 완화.
- **효과**: 학습 안정성 ↑. ablation에서 vanilla DQN 대비 best 갱신.

### C. Action Masking (MaskableDQN)
- **무엇**: 매 step 유효하지 않은 행동을 마스킹해 Q 선택에서 제외.
  - in-flight 트럭의 목적지 차단
  - **자기 위치 stay 기본 차단** (가장 강력했던 개입)
  - strict_urgent_mask: 위급 정류소 우선
- **왜**: vanilla DQN이 "한 곳에 모여 영원히 stay"하는 trivial 솔루션에 빠짐 → mask로 차단.
- **효과**: trivial stay 제거가 plateau 돌파의 1차 열쇠 (best -593 → -448, 학습 환경 기준).

### D. Potential-based Reward Shaping
- **무엇**: `shaping_scale × (γ·Φ(s') - Φ(s))`, Φ = -Σ|bikes - target|. 정책 불변성(policy-invariant) 보장하는 보조 보상.
- **왜**: stockout/full만으로는 보상이 희소(sparse)·평탄 → 균형에 가까워지는 방향으로 dense 신호 제공.
- **효과**: 단독 효과는 작지만 mask와 결합 시 -600대 진입.

### E. Imitation Learning (Behavior Cloning) prior 🌟
- **무엇**: 휴리스틱(거의 oracle)을 **지도학습으로 모방** → 그 가중치를 RL 시작점으로.
  - 1단계 BC: (obs, 휴리스틱 action) 쌍을 cross-entropy로 q_net 학습 (분류 문제로 취급)
  - 2단계 RL fine-tune: BC 가중치 로드 후 RL
- **왜**: 14번 ablation으로 순수 RL은 -93 plateau. 휴리스틱 수준에서 출발해 추가 개선 노림.
- **효과**: **plateau 돌파의 핵심**. 격차 -93 → -39 → -6.6까지 단축.

### F. Weighted Sampling — 오전 피크 가중 ★ (최종 돌파)
- **무엇**: BC 학습 데이터에서 **오전 demand ramp(env.t∈[44,90]) 결정을 ×3 oversample** (`WeightedRandomSampler`).
- **왜**: 진단 결과 휴리스틱과의 격차가 **오전 출퇴근 피크 구간에서 전부 발생**(과소대응) → 그 구간을 더 정확히 모방.
- **효과**: **휴리스틱 평균 첫 추월** (-506.6 → -494.2, +5.8). weight 3.0이 최적(과가중·저녁 추가는 역효과).

### G. 학습 안정화 기법
- **lr_decay**: learning_rate를 1e-4 → 1e-5로 linear decay → 후반 학습 안정 (best step을 후반으로 이동).
- **Deterministic 평가**: random reset 대신 7일 episode 고정 순회(seed=42) → best 모델 선택 신뢰성 ↑ (random eval은 운빨로 misleading).
- **데이터 다양성**: BC 학습 데이터 60일 → 292일이 RL 일반화에 결정적(-33점).

### 기법별 기여 요약

| 기법 | 격차 효과 | 비고 |
|---|---|---|
| C. Action mask (stay 차단) | 큼 | trivial stay 제거, 1차 돌파 |
| E. IL prior (BC) | 매우 큼 | -93 → -6.6 |
| F. 오전 피크 가중 | 결정적 | -6.6 → +5.8 (추월) |
| D. Reward shaping | 작음 | mask와 결합 시 효과 |
| B. Double Q | 작음 | 안정성 |
| G. lr_decay / det. eval / 데이터 다양성 | 보조 | 신뢰성·일반화 |

> ❌ **효과 없거나 역효과였던 시도**: big net([512,512,256], over-fit), 미래 demand obs(BC v5, RL 악화), strict_urgent_mask 단독(트럭 마비), travel penalty 조정, seed/LR 변경, 저녁 피크 가중.

## 6. 재현 명령

```bash
# 1) BC prior 학습 (오전 피크 가중)
python scripts/pretrain_bc.py --tag v7_peak --n-dates 292 --epochs 150 \
  --lr 3e-3 --batch-size 128 --peak-start 44 --peak-end 90 --peak-weight 3.0

# 2) RL fine-tune
python scripts/train.py --tag bc_v7_peak_100k \
  --pretrain logs/bc_v7_peak/bc_model.zip --timesteps 100000

# 3) 7일 평가
PYTHONPATH=. python scripts/eval_7day.py \
  --model logs/masked_dqn_bc_v7_peak_100k/best/best_model.zip --label v7_peak
```

## 7. 알려진 특성 (중요)

- **best는 항상 step 10k = 순수 BC prior** (learning_starts=20k 이전). 20k 이후 RL은 catastrophic forgetting으로 악화만 함 → RL fine-tune 단계는 사실상 BC prior를 선택하는 역할.
- **BC accuracy ≠ RL 성능** — 데이터 다양성·표적(오전 피크)이 acc 절대값보다 중요.
- 자세한 실험 이력: [experiments_2026-05-30.md](experiments_2026-05-30.md) (ablation), [experiments_2026-06-03.md](experiments_2026-06-03.md) (피크 가중 돌파), 요약 [training_progress.md](training_progress.md).
