# 2026-05-30 실험 로그 — DQN/DDQN ablation, plateau 분석

따릉이 재배치 RL — 마포구 146 정류소, 트럭 3대, 1일(144 step) episode.

## 0. 환경 & 평가 셋업

- **Train**: 2025년 1년 random shuffle 후 앞 60일 (1~12월 mixed 계절·요일)
- **Eval (7일)**: `2025-03-25, 04-18, 05-17, 07-01, 07-06, 07-09, 08-21` (학습 안 함)
- **휴리스틱 baseline**: `most_imbalanced` greedy (잉여 큰 곳 → 부족 큰 곳)
- **평가 방식**: 학습 도중엔 `MaskedEvalCallback`, 학습 후엔 `export_replay.py` (deterministic seed=42)

## 1. 코드/문서 보정 (실험 전)

| 변경 | 위치 | 이유 |
|---|---|---|
| 정류소별 capacity 배열 사용 | `data_loader.py` | stations.parquet에 컬럼이 이미 있는데 문서·코드는 "일괄 20" 가정 |
| `truck.capacity` config 배선 | `train.py` | config 값이 train.py에 안 들어가고 있던 버그 |
| `station_capacities` JSON 필드 | `export_replay.py` | 첫 정류소 capacity만 저장하던 버그 |
| HTML viewer fallback | `replay_viewer.html` | 옛 JSON 포맷도 로드 가능하도록 |
| README/docs 갱신 | `README.md` | "일괄 20"·"action mask placeholder" stale 정보 정정 |

## 2. 학습 ablation — 시간 순

### 2.1 Vanilla DQN 500k

- 명령: `python scripts/train.py --algo dqn --timesteps 500000 --tag replay_500k`
- 결과: **best -657.6 (step 180k)**, 휴리스틱 -500.0, 격차 -157.6 ❌
- 관찰: 3대 트럭이 정류소 17번에 모여 영원히 stay (replay 분석으로 확인). cum_km 5.8만, trivial 솔루션.
- 원인: vanilla DQN은 action mask 무시 → in-flight 목적지 차단 못 함 → 우연히 한 곳에 모인 후 stay가 Q값 최고로 학습됨.

### 2.2 MaskableDQN 500k

- 변경: `--algo masked_dqn` (action mask 적용, in-flight 목적지 차단)
- 결과: **best -604.2 (step 190k)**, 격차 -104.2 ❌
- 관찰: cum_km 5.8 → 227.2로 폭증, 트럭이 활발히 이동. mask가 trivial stay는 깸.
- 그러나 휴리스틱은 못 넘음 — 움직이긴 하지만 어디로 갈지 학습 부족.

### 2.3 Shaped DQN 500k (`shaping_scale=0.5`)

- 변경: `env.shaping_scale: 0.5` 추가, `--algo dqn` 그대로
- 결과: **best -653.2 (step 180k)**, 격차 -153.2 ❌
- 관찰: potential-based shaping `γ·Φ(s')-Φ(s)`, Φ=-Σ|bikes-target|.
- vanilla DQN 단독(-657.6)과 거의 동일 → shaping 단독으로는 trivial stay 못 깸. mask가 더 결정적.

### 2.4 Masked + Shaped 500k

- 변경: `--algo masked_dqn` + `shaping_scale=0.5`
- 결과: **best -598.2 (step 190k)**, 격차 -98.2 ❌
- 관찰: 두 보조 결합. 첫 -600대 진입. cum_km 168.9.

### 2.5 Masked + Shaped + Exploration boost 500k

- 변경: `urgent_bonus 2.0→5.0`, `explore_bonus 0.3→1.0`, `learning_starts=20000` (사용자 직접 변경)
- 결과: **best -597.5 (step 190k)**, 격차 -97.5 ❌
- 관찰: 평균 cum_km 168.9 → 28.3 (다시 보수적). urgent_bonus 키운 게 "위급 도착 후 stay하며 bonus 수확"이라는 reward hacking 유도.

### 2.6 Masked + strict_urgent_mask 500k

- 변경: `strict_urgent_mask: true` (위급 정류소 + stay만 허용)
- 결과: **best -596.2 (step 190k)**, 격차 -96.2 ❌
- 관찰: cum_km 평균 6.7로 감소. 위급 없을 때 안전장치(자기 위치 fallback)로 stay 강제 → 트럭 마비.

### 2.7 DDQN (Double Q) 500k

- 변경: `double_q: true`
- 결과: **best -593.6 (step 190k)**, 격차 -93.6 ❌
- 관찰: Q-value over-estimation 완화. 학습 안정성 ↑, best 1.3점 갱신.
- **2.1~2.7은 plateau** — best 5개가 -596~-657 좁은 구간에 갇힘. 노브 변경만으로 한계.

### 2.8 DDQN + stay 차단 500k 🎯 **plateau 깸**

- 변경: `action_masks()`에서 자기 위치 stay를 **기본 차단** (안전장치는 유지)
- 결과 (학습 환경 random eval): **best -448.77** ✅ 휴리스틱 초과!
- 그러나 deterministic 7일 평가 시: **평균 -655.1** ❌ 휴리스틱 -500.0 대비 -155 부족
- 핵심 발견: **MaskedEvalCallback의 random reset 평가가 misleading**. random하게 7 episode 뽑은 평균은 진짜 정책 성능과 괴리.
- 그래도 cum_km 평균 798로 폭증 → 행동 다양성 확보, trivial stay 완전히 차단.

> ⚠️ **stay 차단은 `rebalance_env.py:action_masks()`에 hard-coded됨**. 이후 2.9~2.13 (그리고 현재 진행 중인 이후 학습) **모두 stay 차단 유지**. 노브가 아니라 환경 자체의 영구 변경이므로 끄려면 코드 수정 필요.

### 2.9 DDQN + stay 차단 1M (timesteps 2배)

- 변경: `timesteps: 500000 → 1000000`
- 결과: **best -457.04 (step 860k)** (random eval), det. 평균 -645.2
- 관찰: timestep 2배에 deterministic 평균 +9.9 (작은 개선). best step 860k로 후반 이동.

### 2.10 DDQN + stay 차단 + bignet `[512,512,256]` 1M

- 변경: `net_arch: [512, 512, 256]` (4배 큰 net)
- 결과: best -461.5 (random eval), det. 평균 **-685.6 (악화 -40)**
- 관찰: 학습 환경에선 더 좋아 보였지만 deterministic 평가는 악화. **Over-fit** — 큰 net이 random eval의 noise를 외움.

### 2.11 DDQN + Deterministic eval 1M 🎯 **평가 방식 개선**

- 변경: `MaskedEvalCallback`을 random reset 대신 **7일 episode를 (idx 0~6) 순서 + seed=42**로 순회. `RebalanceEnv.reset(options={"episode_idx": i})` 추가.
- 결과: **best -602.6 (step 110k, deterministic)**
- 직전 random eval 모델 deterministic 측정 -645.2 → **+42.6 개선**
- 핵심: 진짜 best 모델이 선택됨. best가 학습 초반(110k)에 나오고 이후 발산.

### 2.12 DDQN + lr_decay + target_fill 0.8 1M

- 변경: `target_fill_ratio: 0.5 → 0.8`, `lr_decay: true` (1.0→0.1 linear clamp)
- 결과: **best -694.9 (det. 평균)**, 휴리스틱(0.8 env) -610.8, 격차 -84.1
- 관찰:
  - target=0.8 환경 자체가 어려워짐 — 휴리스틱조차 -500 → -610
  - 그러나 **휴리스틱-RL 격차는 좁아짐** (-103 → -84)
  - **lr_decay 효과 확인**: best step 110k → **880k**로 이동, 후반 학습 지속

### 2.13 DDQN + lr_decay + target_fill 0.5 1M 🏆 **이때까지 best**

- 변경: target_fill 0.5 회귀, lr_decay 유지
- 결과: **best -593.0 (det. 평균, step 90k)**, 격차 -93.0 ❌
- 관찰: 격차 -103 → -93로 +9.7 개선. 2025-07-06 (가장 쉬운 날)에서 휴리스틱 -242.6 vs DDQN -256.3, **차이 단 13.7점**.

### 2.14 Work reward 추가 (`w_work_per_bike=0.05`, `w_idle_visit=0.1`)

- 변경:
  - 적재/하차 1대당 +0.05 양수 보상
  - 적정 정류소 도착해서 0대 옮기면 -0.1 페널티 (허탕 방문)
  - 학습+eval 둘 다 적용 (환경 reward 함수 자체 변경)
- 결과:
  - 휴리스틱 (새 환경): **-437.0** (이전 -500 대비 +63점 — work reward가 휴리스틱에 더 유리)
  - DDQN best: **-536.8 (det. 평균)**, 격차 -99.8
- 관찰:
  - 정책 우열은 거의 안 바뀜 — reward 단위만 변화
  - 격차 -93 → -100 미세 악화. 휴리스틱이 잉여/부족 큰 곳만 가니까 매번 많이 옮김 → work bonus 잘 받음
  - best step 90k → 270k 이동 — 학습 신호 풍부해져 더 안정

### 휴리스틱 reward 측정 메커니즘 (참고)

- 휴리스틱(`MostImbalancedPolicy`)은 RL 모델이 아니라 **코드 기반 결정 규칙**
- `evaluate_heuristic`이 **학습/eval과 동일한 RebalanceEnv를 생성**해서 정책을 episode 끝까지 돌림
- 환경이 reward를 부여하는 방식은 RL과 100% 동일 (stockout, full, travel, work, …)
- 즉 **환경 reward 함수가 바뀌면 휴리스틱·RL 모두 reward 값이 바뀜**. 우열은 격차로 판단해야 공정.

---

## 2.B IL (Imitation Learning) prior 도입 — plateau 돌파

전제: 2.1~2.14의 14번 ablation으로 격차 -93에서 plateau. 휴리스틱이 거의 oracle 수준이므로 **prior로 활용** 시도.

### 2.15 IL 파이프라인 설계

**개념**: "휴리스틱을 모방한 정책으로 시작 → RL이 추가 개선" 2단계 학습

```
[1단계: Behavior Cloning]
휴리스틱을 60일 episode에서 시뮬레이션 → (obs, action) 쌍 수집 → SB3 DQN의 q_net을 cross-entropy로 학습

[2단계: RL Fine-tune]
BC로 학습된 q_net 가중치를 RL 학습 시작점으로 사용 → 평소처럼 RL 학습
```

신규 코드:
- `scripts/pretrain_bc.py`: data 수집 + BC supervised 학습 + SB3 zip 저장
- `scripts/train.py`: `--pretrain` 옵션 — BC zip의 q_net 가중치를 RL 학습 시작 전 로드 (q_net + q_net_target 둘 다)

### 2.16 BC 학습 hyperparameter 튜닝 (v1~v3)

휴리스틱 60일 시뮬레이션 → 약 16,500개 (obs, action) 쌍 수집.

| 버전 | epoch | lr | batch | schedule | best acc | 비고 |
|---|---:|---:|---:|---|---:|---|
| v1 | 20 | 1e-3 | 256 | constant | 6.5% | 학습 부족 |
| **v2** | 100 | 3e-3 | 128 | constant | **22.4%** | hyperparam tuning 천장 |
| v3 | 300 | 5e-3 | 128 | cosine | 18.0% | lr 키워서 후퇴 |

관찰:
- v2 → v3: lr 5e-3 + cosine schedule이 v2보다 학습 망가짐
- v2의 22%가 hyperparameter tuning의 천장 — **휴리스틱이 obs로부터 완벽 예측 어려움** (146개 중 top-1 선택)
- random baseline 0.7% 대비 32배 — prior로는 의미 있는 수준

### 2.17 BC v2 + RL fine-tune 1M 🎯 **plateau 돌파**

명령: `python scripts/train.py --tag bc_finetune_1M --pretrain logs/bc_v2/bc_model.zip`

결과:
- 학습 환경 best: **-539.65 (step 10,000!)** — 전체 1M의 1% 지점
- Deterministic 7일 평균: **-539.6**, 격차 -93 → **-39.6** (휴리스틱 거리 **57% 단축**)
- **2025-07-06에서 최초 휴리스틱 추월** ✅ (휴 -242.6 vs DQN -233.5, +9.2점)

학습 곡선의 특이점:
- step 0~10k: BC prior 활용해서 빠르게 best 도달
- step 100k~1M: -650~-700 진동 — **catastrophic forgetting** (RL이 BC prior 망가뜨림)
- 1M 학습의 99%가 사실상 의미 없음

### 2.18 fine-tune timesteps 단축 (1M → 100k)

가설: best가 step 10k에 나오니 1M까지 학습 불필요.

결과: best 모델 **정확히 동일** (-539.65, step 5k), deterministic 7일 평균도 **모든 날짜에서 0점 차이**. 학습 시간 **11.4분 → 1.3분 (10배 단축)**.

→ BC prior + 짧은 RL 학습이 BC prior + 긴 RL 학습과 동일. 시간 효율 검증 완료.

### 2.19 환경 obs 확장 — 미래 demand feature 추가

가설: BC acc 22% 천장은 obs 정보 부족. 휴리스틱이 보는 정류소 imbalance와 미래 demand 사이 상관관계를 obs에 추가하면 학습 ↑.

**변경 (`rebalance_env.py:_get_obs()`)**:
- 향후 1시간(6 step)의 정류소별 net demand = `returns[t:t+6].sum() - rentals[t:t+6].sum()`
- capacity로 정규화 + [-1, 1] clip
- `obs_dim 171 → 317` (+146)

`future_demand_horizon = 6` 속성 추가.

### 2.20 BC v5 (새 obs) — acc 2배 도약

| 버전 | obs_dim | best acc |
|---|---:|---:|
| v2 (옛 obs) | 171 | 22.4% |
| **v5 (미래 demand 포함)** | **317** | **49.8%** |

미래 demand 정보가 BC 학습에 결정적 도움 — 휴리스틱 결정의 **절반을 정확히 예측**. acc 2배 도약.

### 2.21 BC v5 + RL fine-tune 100k 🔍 **반직관적 결과**

명령: `python scripts/train.py --tag bc_v5_finetune_100k --pretrain logs/bc_v5/bc_model.zip --timesteps 100000`

결과:
- 학습 환경 best: **-557.58 (step 5,000)** — BC v2 best(-539.65)보다 17점 **악화**
- Deterministic 7일 평균: **-557.6**, 격차 **-57.6** (BC v2 -39.6에서 악화)
- 6/7일에서 BC v2보다 안 좋음 (2025-07-01 1일만 +28점 개선)
- 가장 큰 악화: 2025-07-09 (-81점)
- **2025-07-06 추월 사라짐** (BC v2 +9.2 → BC v5 -16.7로 역전)

| 날짜 | 휴리스틱 | BC v2 | **BC v5** | Δ(v5-휴) | vs v2 |
|---|---:|---:|---:|---:|---:|
| 03-25 | -574.2 | -595.5 | -600.4 | -26.1 | -4.8 |
| 04-18 | -682.3 | -727.4 | -739.1 | -56.8 | -11.7 |
| 05-17 | -280.3 | -348.1 | -362.4 | -82.1 | -14.3 |
| 07-01 | -377.2 | -420.3 | **-392.0** | -14.8 | +28.3 |
| 07-06 | -242.6 | **-233.5** | -259.3 | -16.7 | -25.8 |
| 07-09 | -592.8 | -623.2 | -704.3 | -111.6 | -81.2 |
| 08-21 | -750.7 | -829.5 | -845.5 | -94.8 | -16.0 |
| **평균** | **-500.0** | **-539.6** | **-557.6** | **-57.6** | **-17.9** |

**놀라움**: BC acc 22% → 50% (2배 향상), 그러나 RL fine-tune 결과는 **악화**.

**가설**:
- **BC acc는 RL 효율의 직접 지표가 아님** — 모방 정확도와 RL 정책 품질이 따로 움직임
- **새 obs (미래 demand 146 dim)가 RL 학습에 노이즈**일 가능성 — feature 추가가 항상 도움 안 됨
- **BC v5가 휴리스틱을 너무 정확히 모방** → RL이 휴리스틱 너머로 개선할 여지 줄어듦. BC v2는 22%만 모방해서 RL이 채워줄 공간이 많았을 수 있음
- **Catastrophic forgetting이 v5에서 더 심각** — sharp한 prior가 작은 RL 변경에도 큰 손상

결론: **BC v2가 여전히 best**. 미래 demand obs 추가는 BC accuracy엔 도움, 최종 RL 정책엔 손해.

---

### 2.22 BC v6 — 학습 데이터 60일 → 292일 🏆🔥 **거의 추월**

가설 (2.21에서 미래 demand feature가 역효과 → 다른 변경 시도): BC 학습 데이터 다양성 부족이 일반화 한계일 수 있음. 전체 train pool 292일 사용.

**옵션화**: env의 미래 demand obs를 hard-coded → config 옵션 (`future_demand_horizon: int = 0`, 기본 비활성). v2 셋업 (옛 obs 171)으로 회귀.

#### BC v6 학습 (292일 data)

| 버전 | n_dates | epoch | data 쌍 | best acc |
|---|---:|---:|---:|---:|
| v2 (이전 best) | 60 | 100 | 16,486 | **22.4%** |
| v6 (epoch 50) | 292 | 50 | 80,269 | 18.6% (underfit) |
| **v6b** (epoch 150) | 292 | 150 | 80,269 | **20.2%** |

**반직관**: 데이터 5배 늘렸는데 BC acc는 **떨어짐** (22.4% → 20.2%). 가설: 데이터 다양성↑ → label noise(같은 obs에 다른 트럭 차례 → 다른 action) 더 보임 → train acc memorize 어려움.

#### BC v6b + RL fine-tune 100k 결과

| 날짜 | 휴리스틱 | BC v2 | **BC v6 (292d)** | vs 휴 | vs v2 |
|---|---:|---:|---:|---:|---:|
| **03-25** | -574.2 | -595.5 | **-569.0** | **+5.3** ✅ | +26.6 |
| **04-18** | -682.3 | -727.4 | **-672.8** | **+9.5** ✅ | +54.5 |
| 05-17 | -280.3 | -348.1 | -341.6 | -61.3 | +6.6 |
| 07-01 | -377.2 | -420.3 | -384.6 | -7.4 | +35.8 |
| 07-06 | -242.6 | -233.5 | -255.8 | -13.1 | -22.3 |
| 07-09 | -592.8 | -623.2 | -598.1 | -5.4 | +25.0 |
| **08-21** | -750.7 | -829.5 | **-724.2** | **+26.5** ✅ | **+105.3** |
| **평균** | **-500.0** | -539.6 | **-506.6** | **-6.6** | **+33.1** |

#### 성과
- **3일 추월** (03-25, 04-18, **08-21**) — BC v2의 1일에서 3배 증가
- **평균 격차 -39.6 → -6.6** (휴리스틱 거리 83% 단축)
- **demand peak (08-21) 추월** ⭐: 휴리스틱 -750.7 vs RL **-724.2** = +26.5점. RL이 가장 약했던 어려운 날에서 휴리스틱 추월 — RL의 진짜 가치 입증
- 7/7일 중 6일이 BC v2 대비 개선

#### 핵심 발견: **BC accuracy ≠ RL 일반화**

| | BC train acc | RL fine-tune 결과 |
|---|---:|---:|
| BC v2 (60일) | 22.4% | -539.6 |
| BC v5 (60일 + 미래 demand) | 49.8% | -557.6 (악화) |
| **BC v6 (292일)** | **20.2%** | **-506.6** (개선) |

→ **train 데이터 다양성이 BC accuracy보다 RL 일반화에 훨씬 중요**. acc는 memorize 정도 지표, 다양성↑이 일반화↑로 이어짐.

---

## 2.B 요약 — IL prior의 효과와 핵심 발견

- ✅ **plateau 완전 돌파**: 14번 ablation의 -93에서 **-6.6**까지 좁힘
- ✅ **3일 휴리스틱 추월** (03-25, 04-18, **08-21**)
- ✅ **demand peak 추월** — RL의 진짜 가치 입증
- ✅ **데이터 다양성이 결정적** — 60일 → 292일이 -33점 도약
- ❌ **BC accuracy는 misleading 지표** — v2(22%)보다 v5(50%)가 RL 결과 안 좋고, v6(20%)이 가장 좋음
- ❌ **미래 demand obs는 역효과** (BC v5) — feature 추가가 항상 도움 안 됨
- ❌ **catastrophic forgetting 여전** — 100k면 충분, 그 이상은 손해
- 🎯 **완전 추월까지 -6.6점**

## 3. 종합 비교 (Deterministic 기준)

| # | 학습 | best (det.) | 격차 |
|---|---|---:|---:|
| — | 휴리스틱 | -500.0 | 0 |
| 2.1 | Vanilla DQN 500k | -657.6 (random) | -157.6 |
| 2.2 | MaskableDQN 500k | -604.2 (random) | -104.2 |
| 2.4 | Masked + Shaped 500k | -598.2 (random) | -98.2 |
| 2.7 | DDQN 500k | -593.6 (random) | -93.6 |
| 2.8 | DDQN + stay 차단 500k | -655.1 (det.) | -155.1 |
| 2.9 | DDQN + stay 차단 1M | -645.2 (det.) | -145.2 |
| 2.10 | DDQN + bignet 1M | -685.6 (det.) | -185.6 |
| 2.11 | DDQN + det. eval 1M | -602.6 (det.) | -102.6 |
| 2.12 | DDQN + lr_decay + fill 0.8 1M | -694.9 (det.) | -84.1 (0.8 env) |
| 2.13 | DDQN + lr_decay + fill 0.5 1M | -593.0 (det.) | -93.0 |
| 2.14 | DDQN + work reward 1M (다른 env) | -536.8 (det.) | -99.8 (work env) |
| 2.17 | BC v2 + RL fine-tune 1M (=100k) | -539.6 (det.) | -39.6 |
| 2.21 | BC v5 (미래 demand) + RL 100k | -557.6 (det.) | -57.6 |
| **2.22** | **BC v6 (292일 data) + RL 100k** 🏆🔥 | **-506.6 (det.)** | **-6.6** (3일 추월) |

> 2.1~2.7은 random eval로 학습한 모델의 deterministic 측정 안 했음. 단순 비교는 불공정.
> 2.14는 환경 reward 함수가 다름(work reward 추가) → 다른 학습의 절대값과 직접 비교 불가, 격차로만 판단.
> 2.21은 obs_dim이 다름(317 vs 171) — 같은 환경 reward라 격차는 직접 비교 가능.
> 2.22는 v2와 같은 obs(171), 같은 reward — 데이터 다양성만 변경.

## 4. 핵심 발견

### 4.1 trivial stay 솔루션의 위력
- DQN의 7개 변형 모두 "자기 위치 stay"의 Q값을 최고로 학습 → 트럭이 한 곳에 모여 영원히 stay
- 이동 비용은 음수, stay 비용은 0 → 단기 reward 우위
- `action_masks()`에서 stay를 막는 **한 줄 변경**이 가장 강력한 개입이었음 (best -593 → -448 학습 환경 기준)

### 4.2 평가 방식이 결과를 결정
- `MaskedEvalCallback`의 random reset 평가 → best가 운빨로 선택됨
- 학습 시 best `-457` ↔ deterministic 측정 `-655` (200점 괴리)
- **deterministic 평가가 진짜 정책 성능**. 학습 중 평가도 같은 방식으로 통일하니 best 선택 정확해짐.

### 4.3 plateau는 실재함, 그러나 점진적 개선 가능
- mask + shaping 결합 이후 -596~-598에 5개 학습 갇힘
- 추월의 핵심은 (a) stay 차단, (b) deterministic 평가, (c) lr_decay
- 누적 개선: -657.6 → -593.0 (**64점**, 휴리스틱과 격차도 -157 → -93)

### 4.4 big net은 답이 아님
- `[512,512,256]`로 키웠더니 학습 환경 best는 비슷한데 deterministic 평가 -40점 악화
- Q-net 표현력보다 **일반화 능력**이 더 중요. small net `[256,256]`이 더 잘 일반화.

### 4.5 lr_decay 효과는 작지만 일관됨
- 평균 +9.7점 개선
- best step이 학습 후반으로 이동 (110k → 880k) — 후반 학습 지속 효과는 확인됨
- 다만 best 절대값 갱신은 작음

### 4.6 환경 자체의 어려움 — target_fill_ratio
- target=0.5 vs 0.8 환경에서 휴리스틱 자체가 -500 vs -610 (110점 차이)
- 어려운 환경(0.8)에서 RL의 상대적 가치 ↑ (격차 -84 vs -93)
- 즉 **best 격차 좁히기**가 목표면 0.8, **best 절대값**이면 0.5

### 4.7 demand peak에 약함
- 2025-07-06 (4k 대여) — 휴리스틱과 차이 단 -13.7점 (거의 추월)
- 2025-08-21 (5k 대여, 휴일 직전) — 차이 -126.7점 (큰 격차)
- demand 많은 날에 정책이 못 따라감 → action space 한계 또는 학습 부족

## 5. 코드 변경 사항 (전체)

### 환경 (`src/envs/`)
- `rebalance_env.py`:
  - `action_masks()`: 자기 위치 stay **hard-coded 차단** (안전장치만 유지). 2.8 이후 모든 학습에 영구 적용.
  - `reset()`: `options={"episode_idx": i}` 받아 deterministic episode 선택
  - `_get_obs()`: 2.19 — 미래 1시간 정류소별 net demand 추가 (obs_dim 171 → 317). 2.22에서 옵션화 (`future_demand_horizon: int = 0` 기본 비활성).
- `data_loader.py`: 정류소별 capacity 배열 사용 (이미 동작, 문서만 갱신)

### 학습 (`scripts/train.py`)
- `MaskedEvalCallback._on_step()`: 7일 deterministic 순회
- `--learning-starts`, `--truck-capacity`, `--target-fill-ratio`, `--lr-decay` 인자 추가
- `policy_kwargs.net_arch` config에서 읽도록 배선
- `lr_decay=true`면 `lr = base × max(0.1, progress)` callable로 감쌈
- `--pretrain` 옵션 추가 (2.15) — BC zip의 q_net 가중치 로드해 RL 시작

### IL pretrain (`scripts/pretrain_bc.py`) — 신규 (2.15)
- 휴리스틱을 train episode에서 시뮬레이션 → (obs, action) 쌍 수집
- SB3 MaskableDQN의 q_net을 cross-entropy로 학습 (분류 문제)
- best acc 가중치 자동 보존
- SB3 zip으로 저장 → `train.py --pretrain`이 로드
- 옵션: `--epochs`, `--lr`, `--lr-schedule`(constant/cosine), `--batch-size`, `--n-dates`(데이터 수집할 날짜 수)
- 2.22 — `--n-dates 292`로 데이터 5배 늘려서 RL 일반화 결정적 개선

### 평가 (`scripts/export_replay.py`)
- `--strict-mask`, `--w-travel-km`, `--w-travel-step`, `--target-fill-ratio` 인자 추가 (학습 환경 일치용)
- `station_capacities` 배열 저장

### Viewer (`docs/replay_viewer.html`)
- 정류소 hover tooltip (`남은수량 N / 수용가능 M` + 위급 태그)
- 메타에 `현재 위급 (빈/가득)`, `평균 fill` 통계 추가
- 옛 JSON 포맷 fallback (`station_capacity` 스칼라)
- 셀렉터: 13개 학습 결과 optgroup으로 그룹화

### Config (`config/default.yaml`)
- `truck.target_fill_ratio: 0.5`
- `env.shaping_scale: 0.5`, `strict_urgent_mask: true`
- `env.urgent_bonus: 5.0`, `explore_bonus: 1.0`
- `dqn.learning_starts: 20000`, `lr_decay: true`, `net_arch: [256, 256]`
- `training.timesteps: 1000000`

## 6. 남은 후보 (다음 세션)

1. **2M timesteps + lr_decay** — 후반 학습 지속 효과 더 보기
2. **buffer_size 100k → 300k** — replay buffer 다양성
3. **batch_size 64 → 128** — gradient 안정성
4. **Reward 가중치 재조정** (예: w_full -0.8 → -1.0)
5. **단순 우월 알고리즘 시도** (PPO, A2C) — DQN의 후반 발산 회피
6. **Multi-agent (IQL)** — 트럭당 독립 policy

## 7. 진척 요약

| 단계 | det. reward | 격차 |
|---|---:|---:|
| 시작점: Vanilla DQN | (random eval -657.6) | -157.6 |
| 14번 ablation 후 (DDQN+lr_decay) | -593.0 | -93.0 |
| IL prior 도입 (BC v2 + RL fine-tune) | -539.6 | -39.6 |
| **BC v6 (292일 학습 데이터)** 🏆🔥 | **-506.6** | **-6.6** |

- **누적 개선: 151점, 격차 96% 단축**
- **3일 휴리스틱 추월** (03-25, 04-18, **08-21 demand peak**)
- **demand peak 추월 (08-21 +26.5점)** — RL이 가장 약했던 어려운 날에서 휴리스틱 능가
- 완전 추월까지 단 **-6.6점**
- **결정적 인사이트**: BC train acc는 misleading 지표. 데이터 다양성이 본질

## 8. 격차를 더 좁히는 방법 — 가설과 제안

### 8.1 왜 -93에서 못 내려가는가 (분석)

- **휴리스틱이 거의 oracle 수준**: `MostImbalancedPolicy`는 매 결정 시 "잉여/부족 큰 곳"으로 가는 그리디 최적. 환경의 `_apply_rebalance`도 도착 시 자동으로 거의 최적 분량 옮김. RL이 추가로 학습할 여지가 작음.
- **Reward landscape가 평탄**: stockout/full이 reward의 90% 차지. 정책의 미세 변화가 reward에 거의 안 보임.
- **Demand peak 대응 약함**: 2025-08-21처럼 demand 많은 날 -127점 격차. RL이 미래 demand 예측 활용 못 함.
- **DQN의 capability 천장**: 14번 ablation으로 노브 다 돌렸음. 동일 알고리즘으로 -93 이하는 어려움.

### 8.2 다음 시도 후보 (우선순위순)

#### A. Imitation Learning prior 🌟
- 휴리스틱 행동(action 시퀀스)을 데이터로 수집 → 지도학습으로 policy 모방
- 모방 후 RL로 fine-tune → **휴리스틱 수준에서 시작**해서 추가 개선만
- 현재 RL은 -650 ~ -593 진동, IL prior 후 fine-tune은 -500 부근에서 시작 → 추월 가능성 ↑
- 구현: 1.5시간, 학습은 IL + RL fine-tune ~20분
- **가장 큰 도약 기대**

#### B. State에 미래 demand 예측 feature 추가
- 현재 obs에 없음: 향후 1시간(6 step)의 정류소별 net demand
- 추가 시 RL이 "곧 빌 정류소 미리 가" 같은 휴리스틱이 못 하는 선제 행동 가능
- 구현: `data_loader.py`/`rebalance_env.py`에 미래 demand window 추가, obs_dim 확장
- 비용: 1시간, 효과 가능성 큼 (휴리스틱이 못 하는 것)

#### C. Reward 가중치 재조정
- 현재 stockout(-1) vs full(-0.8) 가중치. 사용자 우선순위 따라:
  - 빌리기 우선 → stockout을 -1.5나 -2.0
  - 반납 우선 → full을 -1.0
- 구현: config 한 줄, 휴리스틱 baseline 자동 재측정
- 비용: 5분, 효과 작음 (정책 우선순위 살짝 변화)

#### D. Action space 재설계
- 현재: `Discrete(146)` — 146개 정류소 indices
- 대안: `Discrete(K)` — Top-K 위급 정류소만 (K=10 정도)
- 매 step마다 mask가 K개만 valid → 학습 효율 ↑
- 구현: env 수정 + obs 인코딩 변경. 2시간
- 비용: 중간, 효과 가능성 큼 (action space 14배 축소)

#### E. Curriculum learning
- 학습 초반엔 쉬운 episode(demand 적은 날), 후반엔 어려운 episode
- 또는 초반엔 trucks 5대(쉬움) → 점점 3대(어려움)
- 구현: train.py 수정 1시간
- 효과: demand peak 대응 향상 가능성

### 8.3 추천 순서

1. **A (Imitation Learning)** — 가장 직접적, 격차 -93 → 0 또는 추월 가능성
2. **B (미래 demand feature)** — 휴리스틱이 못 하는 영역, RL 가치 명확
3. **D (action space K개)** — DQN 학습 효율, 시간 비용 중간
4. **C (reward 가중치)** — 빠른 실험, 작은 효과
5. **E (curriculum)** — demand peak 약점 직접 공격

### 8.4 실제 진행 결과 (사후)

- A 진행 (2.15~2.18) → 격차 -93 → -40 (BC v2, **첫 1일 추월**)
- B 진행 (2.19~2.21) → **역효과** (BC accuracy ↑이지만 RL -57.6 악화)
- **데이터 다양성 강화** (2.22, A 후속) → 격차 -40 → **-6.6** (3일 추월, demand peak 포함)
- 가장 효과 큰 단일 변경: **train 데이터 5배** (60일 → 292일)
- 가장 효과 작거나 음인 변경: BC accuracy 강화 (B), big net (2.10), strict_urgent_mask (2.6)

### 8.5 남은 도전 (격차 -6.6)

- 어려운 날(05-17 -61, 07-06 -13)에서 여전히 RL이 약함
- 평균 추월까지 -6.6점 — 다음 시도 후보:
  - KL penalty / EWC로 RL이 BC prior 더 보존
  - BC를 더 큰 net으로 (단 over-fit 주의)
  - 5,000 step보다 더 짧게 fine-tune (3k, 1k)
  - Eval set에서 어려운 날(05-17)만 추가 학습 — domain-specific tuning
