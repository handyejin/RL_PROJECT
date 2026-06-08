# DQN 소규모 — 휴리스틱 추월 레시피 정리

> **한 줄 요약:** "작게 줄인 환경"에서 **forecast를 입력으로 받은 순수 DQN**이 예측형
> 휴리스틱(SLA)을 **리워드 기준 약 2배로 추월**한다. 146정류소·전체 환경에선 못 넘던
> 문제를, 학습 가능한 크기(10~20정류소·트럭 1대)로 줄이자 RL이 베이스라인을 압도했다.
>
> 코드: [scripts/dqn_small.py](../scripts/dqn_small.py) (환경·평가는 [scripts/rtdp_small.py](../scripts/rtdp_small.py) 재사용)
> 원본 결과 기록: [docs/dqn_small_results.md](dqn_small_results.md) · 브랜치 `feat/rl-small-scale-overtake`
> 재현 검증: 2026-06-07, N=15에서 DQN −71.70 / SLA −124.34 / **Δ +52.64** (원본 +52.6과 일치)

---

## 1. 환경 (Environment)

추월을 만든 환경은 전체 `RebalanceEnv`가 아니라, 같은 동역학·비용을 쓰되 **학습 가능한
크기로 축소한 `SmallProblem`** ([scripts/rtdp_small.py](../scripts/rtdp_small.py))이다.
RTDP·휴리스틱·DQN이 **완전히 동일한 잣대**로 비교되도록 한 환경을 공유한다.

| 항목 | 설정 |
|---|---|
| 자치구 | **마포구** (`data/processed`, `config/default.yaml` 기본값) |
| 정류소 | **출퇴근 불균형 압력 top-N** (기본 N=10, 추월 최대는 N=15) + **depot 1개(무한 버퍼)** |
| 정류소 선택 기준 | `press = \|반납−대여\|`를 아침피크(07–10시)·저녁피크(17–21시) 구간 합산 → 상위 N |
| 트럭 | **1대**, 용량 30 |
| 시간축 | **전일 144 step** (10분/step, 출근·퇴근 피크 모두 포함) |
| 수요 | **Poisson(60일 forecast 평균)** — 매 에피소드 다른 확률 실현 |
| 초기 재고 | 용량 × target_ratio(0.5) 반올림 |
| depot | 정류소 좌표 무게중심에 배치, 트럭이 자전거를 무한히 적재/하역 |

**비용(=리워드) 정의** — 전체 환경 `RebalanceEnv`와 동일하게 맞춰 절대 비교는 못 해도
상대 비교는 공정:

```
reward = stockout × (−1.0)        # 대여 미충족 (자전거 없어서 못 빌림)
       + full     × (−0.8)        # 반납 미충족 (거치대 꽉 차서 못 반납)
       + 이동km   × (−0.008)
       + 이동step × (−0.002)
```
높을수록 좋음. 미충족수요(stockout+full)가 핵심 지표.

> **왜 축소가 핵심인가:** 전체(146·하루·3트럭)에선 상태·신용할당 폭발로 model-free DQN이
> 학습 자체를 못 했다. "퍼진 정류소(이동 2~4 step) + 확률 수요"는 **반응형이 항상 늦는**
> 구조라, 미리 배치할 줄 아는 예측형이 빛나고, 그걸 DQN이 학습으로 더 잘 해낼 여지가 생긴다.

---

## 2. 비교 휴리스틱 (Baselines)

세 가지 휴리스틱과 비교했다. 모두 같은 `SmallProblem`·같은 Poisson 실현에서 평가.

| 휴리스틱 | 성격 | 동작 |
|---|---|---|
| **do-nothing** | 무행동 | 재배치 안 함, depot 대기 |
| **STR** | **반응형·최소 재배치** | 현재 재고가 밴드 밖이면 **밴드 가장자리까지만** 고치고, 밴드 밖 정류소 중 **가장 가까운 곳**으로 이동. 현재 상태만 봄 |
| **SLA** | **예측형 lookahead** ⭐ | 목표(50%)로 재배치하되, **향후 6step 예측 순수요**를 더한 `\|inv+미래수요−target\|`가 **가장 큰 정류소**로 이동. forecast를 탐욕적으로 사용 |

**SLA가 핵심 베이스라인**이다 — "예측을 쓰는 휴리스틱"으로, 우리 DQN이 같은 forecast
정보로 이걸 넘느냐가 프로젝트의 질문이다. (SLA = 매 step 예측 불균형 1위를 그리디하게
처리하는 강한 예측형 베이스라인.)

---

## 3. DQN 기법 (Techniques)

`scripts/dqn_small.py` — stable-baselines3 vanilla `DQN`(MlpPolicy)을 from scratch 학습.

### 3.1 관측 (Observation) — **forecast를 state에 넣는 게 핵심**
`obs_dim = 3K+3` (K=정류소 수):

| 구성 | 차원 | 설명 |
|---|---|---|
| 재고율 | K | `inv / capacity` |
| 적재율 | 1 | `truck_load / truck_cap` |
| 위치 one-hot | K+1 | 트럭 현재 위치(정류소 또는 depot) |
| 시각 | 1 | 정규화된 step |
| **forecast 미래 순수요율** | K | **향후 6step 예측 (반납−대여) / capacity** ← 예측형과 동등한 정보 |

### 3.2 행동 (Action)
`(K+1) × 3` — **목적지(K개 정류소 + depot) × 재배치량 레벨 {비움 / 중간 / 채움}**.
목적지뿐 아니라 **얼마나 채울지(레벨)**까지 고른다. (ablation 결과 이 레버는 필수는 아님 — §5)

### 3.3 리워드 셰이핑 (선택)
Potential-based shaping (Ng 1999, **정책 불변**): `Φ(s) = −예측 불균형`
→ 조밀한 신용할당. shaping_scale=1.0 기본. (ablation: 없어도 추월 — §5)

### 3.4 하이퍼파라미터
| | 값 |
|---|---|
| 알고리즘 | stable-baselines3 `DQN` (MlpPolicy, net_arch [256, 256]) |
| total_timesteps | **400,000** (학습 약 4분) |
| learning_rate | 5e-4 |
| gamma | 0.99 |
| buffer_size | 200,000 |
| learning_starts | 5,000 |
| batch_size | 128 |
| train_freq | 4 |
| target_update_interval | 2,000 |
| exploration_fraction | 0.3, final_eps 0.05 |

> **주의:** MaskableDQN·candidate Top-K·BC pretrain은 **쓰지 않았다**. 핵심 레시피는
> "**작은 환경 + forecast 받은 순수 DQN(from scratch)**"이며, 보조 장치는 불필요(§5).

---

## 4. 결과 (Results)

리워드 정의 = 원본 RebalanceEnv(stockout −1.0 / full −0.8 / km −0.008 / step −0.002), **높을수록 좋음**.

### N=15 (추월 폭 최대), 2026-06-07 재현값

| 정책 | 리워드(확률30) | 미충족(확률) | 리워드(실제7) |
|---|---:|---:|---:|
| do-nothing | −191.37 | 218.5 | −249.97 |
| STR (반응형) | −128.71 | 143.7 | −186.03 |
| SLA (예측형) | −124.34 | 138.2 | −160.72 |
| **DQN (학습)** | **−71.70** ⭐ | **77.2** | **−124.51** |

→ **DQN vs SLA +52.64, vs STR +57.01.** 미충족수요 기준 SLA의 138.2 → 77.2로 **거의 절반**.

### N=10 (헤드라인)
DQN −32.6 vs SLA −67.5 → **+34.8** (미충족 73.6 → 36.9, 절반).

### 정류소 수 스윕 — "배울 수 있는 크기"의 경계 (고정 400k)
| N | DQN | SLA(예측형) | DQN−SLA | 결과 |
|---|---:|---:|---:|---|
| 10 | −32.6 | −67.5 | **+34.8** | DQN 압도 ✅ |
| 15 | −71.7 | −124.3 | **+52.6** | DQN 압도 ✅ (최대) |
| 20 | −149.1 | −173.4 | **+24.3** | DQN 우위 ✅ |
| 30 | −270.8 | −238.5 | −32.3 | SLA 역전 ❌ |
| 50 | −440.1 | −383.6 | −56.5 | SLA 역전 ❌ |

단, **N=30도 학습량 2배(800k)면 +4.5로 추월 회복** → 경계는 "정류소 수의 벽"이 아니라
**"문제 크기 ↔ 학습량"의 균형점**. 커질수록 필요한 학습량이 가파르게 증가한다.

---

## 5. 귀인 (Ablation) — 추월은 견고하고, 보조장치 덕이 아니다

| 조건 | DQN(확률) | 해석 |
|---|---:|---|
| 기본 (seed 1) | 36.90 | — |
| seed 2 | 36.80 | ✅ 재현됨 (운빨 아님) |
| shaping = 0 | 30.20 | shaping 없어도 추월 |
| no-amount (목적지만, 50% 고정 = SLA와 동일 행동력) | 36.07 | 재배치량 레버 원인 아님 |
| **no-forecast** (현재 상태만) | **69.53** | **forecast가 승리폭의 핵심** |

※ 위 표는 N=10·미충족수요(↓ 좋음) 기준(원본 기록).

### 귀인 분해
```
SLA 예측형 (forecast 탐욕 사용)         73.6
DQN forecast 없음 (라우팅 학습만)       69.5  ← 라우팅 학습만으론 SLA와 거의 동률
DQN forecast 있음 (라우팅 + forecast)   36.9  ← 둘 결합 시 압도
```
→ **큰 추월 = ① forecast 입력(승리폭 대부분) + ② 순수 라우팅 학습(단독 동률, 결합 시 시너지).**
shaping·재배치량 레버는 불필요.

---

## 6. 핵심 레시피 요약

```
작은 환경 (마포구 출퇴근 압력 top-10~20 정류소 + 트럭 1대, 전일 144 step)
  + Poisson 확률 수요
  + forecast 미래 순수요를 obs에 입력            ← 승리폭의 핵심
  + from-scratch 순수 DQN (MlpPolicy [256,256], 400k)
  + 예측형 휴리스틱(SLA)과 동일 환경·동일 실현으로 공정 비교
  = 예측형을 리워드 기준 약 2배로 추월
```

**왜 전체 환경(146·3트럭)에선 안 됐나:** 알고리즘 결함이 아니라 **문제 크기 대비 학습
난이도** — 상태·신용할당 폭발로 model-free DQN이 학습 불가. 크기를 줄이면(또는 학습량을
대폭 늘리면) RL이 예측형 휴리스틱의 탐욕적 빈틈을 발견해 넘어선다.

---

## 부록. 재현 방법

추월 코드는 `feat/rl-small-scale-overtake` 브랜치에 있다(master tip엔 없음).

```bash
# 헤드라인 N=10
python scripts/dqn_small.py --timesteps 400000 --shaping 1.0

# 추월 폭 최대 N=15
python scripts/dqn_small.py --n-stations 15 --timesteps 400000 --shaping 1.0

# 어블레이션
python scripts/dqn_small.py --no-forecast   # forecast 제거 → 추월폭 급감
python scripts/dqn_small.py --shaping 0      # shaping 제거 → 추월 유지
python scripts/dqn_small.py --no-amount      # 재배치량 레버 제거 → 추월 유지
```
