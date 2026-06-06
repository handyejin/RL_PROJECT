# 서영현 논문 기반 재배치 전략 재현 — 작업 정리 (2026-06-05~06)

대상 논문
- 서영현 (2020), 서울대 박사 "실시간 동적 계획법 및 강화학습 기반의 공공자전거 시스템의 동적 재배치 전략" (지도 고승영)
- 영문 저널판: **Seo et al. (2022), _Journal of Advanced Transportation_, "Rebalancing Docked Bicycle Sharing System with Approximate Dynamic Programming and Reinforcement Learning"** (open access)

우리 프로젝트: 마포구 146정류소·3트럭, 하루(144×10분) 결정론 평가(7일). 휴리스틱 천장 = 반응형 -500.02 / 예측형 forecast -459.65 / oracle -382.79.

---

## 0. 한눈 요약

| 시도 | 내용 | 결과 |
|---|---|---|
| ① 예측오차 보정/타게팅 | forecast 잔차로 실시간 보정 + 오차 큰 정류소만 탐색 | ❌ 우리 환경선 악화 (모든 변형 < -459.65) |
| 논문 정독·비교 | 전체 논문으로 환경·방법 대조, §14 해석 정정 | 휴리스틱 추월 엔진은 **RTDP**(소규모 확률적 DP)지 model-free RL 아님 |
| RTDP 소규모 재현 (갈래 A) | 6정류소·1트럭·2~3h, 퍼진 정류소+적재 lever+분석 백업 | ❌ RTDP ≈ STR(반응형), **예측형 SLA(5.63)엔 못 미침**. 추월 실패 확정 |

핵심 교훈: **논문의 "휴리스틱 추월"은 model-free RL이 아니라 RTDP(테이블형 확률적 동적계획)의 성과**이며, 그것도 ① 이동거리 있는 인스턴스 ② 적재량 예측 lever ③ 제약된 baseline이 결합된 조건에서 나왔다. 우리 대규모(146·하루·결정론)·강한 반응형 환경에는 그대로 오지 않는다.

---

## 1. 논문 환경·방법 (전체 정독)

### 환경 (그들 vs 우리)
| | 논문 | 우리 |
|---|---|---|
| 지역 | 여의도 2.9km² | 마포구 |
| 정류소 | RTDP **5~7개** / A2C 31개 | **146개** |
| 트럭 | 1대 (적재 15) | 3대 |
| 기간 | **2시간**(07–09 또는 18–20 피크) | **하루 전체**(144 step) |
| 수요 | 확률 Poisson(RF예측이 평균), Skellam 전이 | 결정론 7일 eval |
| step | 10분 | 10분 |
| 상태 | (시각, 차량위치, **정류소별 "안전밴드 안/밖" 이진 fill-rate index**) | raw 재고/비율 |
| 행동 | 다음 정류소 + 적재량(**안전재고 규칙으로 자동**) | 트럭 목적지 146(적재 자동) |
| 보상 | 미충족수요(stockout+full) 최소화 | 동일 |

### 방법 — 두 개의 엔진
- **RTDP** (Barto 1995, Real-Time Dynamic Programming): 테이블형 비동기 가치반복 + 궤적 샘플링 + Skellam/Poisson 전이로 **명시적 lookahead**. **작은 문제(5~7정류소·2h) 전용.** → **휴리스틱을 크게 이긴 엔진.**
- **A2C** (model-free actor-critic, ANN): **큰 문제(31정류소) 전용**(테이블 불가). 여기선 휴리스틱과 직접 비교 없음 + **constraint-free RL(행동을 RL이 통째로=우리 방식)이 제일 나쁨**.

### RTDP는 RL인가
- DP↔RL **경계**. 엄밀히는 **model-based DP/플래닝**(전이모델 필요, Bellman 기대 백업). 우리가 하던 model-free RL(DQN/PPO)과 다름.
- 단 가치함수·Bellman·trajectory sampling을 쓰고 Sutton&Barto RL 교재에 실림 → **넓은 의미의 강화학습으로 인정 가능**. 논문도 제목에 "...and Reinforcement Learning".

### 논문 핵심 수치
- Table 2 (5~7정류소, RTDP, 미충족수요↓): No-reb 10.6 / **STR(반응형) 8.5** / SLA 9.6 / **RTDP 3.8→3.5→2.3** (안전재고 z=1.0/1.65/2.33). z↑일수록 좋아짐.
- Table 3 (7정류소, 탐색전략): **S1 전체탐색 13.67(리워드 최선)** / S2 인접 19.67 / **S3 예측오차 focus 14.00(계산 28.5%↓)**.
- Table 4 (31정류소, A2C 확률): S3는 수요 150%(큰 surprise)일 때만 최선, 50%엔 최악. constraint-free가 전반 최악.
- **즉 예측오차 focus(S3)는 리워드 개선이 아니라 "계산 절감"용**이며 큰 surprise에서만 이득.

---

## 2. 아이디어 ① — 예측오차 보정/타게팅 (배포형 단순화) → ❌

`ForecastErrorPolicy` ([src/agents/baselines.py](../src/agents/baselines.py)) + [scripts/eval_forecast_error.py](../scripts/eval_forecast_error.py).
- forecast(292일평균)로 미래를 깔되, 최근 W step **관측−forecast 잔차**를 정류소별로 추정해 앞 H step 예측을 실시간 보정(drift 가산 / scale 승산). + 잔차 큰 상위 K개만 후보(focus).

결과 (7일, 공정 metric) — **모든 변형이 forecast 예측형(-459.65)보다 나쁨**:
| 설정 | eval |
|---|---|
| forecast 예측형 (기준) | **-459.65** |
| drift W=3 / 12 / 24 | -493 / -480.6 / -480.5 |
| scale W=24 | -477.99 |
| drift W=24 **α=0.2**(약한 보정) | -475.62 |
| focus K=30 / 50 | -525 / -491 |

왜 실패: **보정 약하게(α↓)·매끄럽게(W↑) 할수록 덜 나쁨 → 최적 α=0(보정 안 함).** 최근 잔차는 신호가 아니라 **노이즈**(292일 평균이 이미 저분산 최적, 10분·정류소 수요 스파이키, 잔차 비지속). focus는 좋은 후보를 굶겨 더 나쁨.

정정: 이건 **논문 메서드가 아닌 "잔차 외삽" 단순화**였다. 논문은 잔차를 외삽하지 않고 RTDP가 매 step 재계획하며 forecast는 Poisson 평균으로만 쓴다. 또 focus가 리워드를 악화시킨 결과는 **논문(S1 전체탐색이 최선, S3는 계산용)과 모순이 아니라 일치**.

---

## 3. RTDP 소규모 재현 (논문 정공법)

목적: model-free가 아닌 **RTDP로 휴리스틱을 실제로 넘기**를 재현. 테이블형이라 상태 2^|N| 폭발 → 정류소·트럭·시간 축소.

구현: [scripts/rtdp_small.py](../scripts/rtdp_small.py)
- 마포구 07~09시 윈도우 최번잡 6정류소 + depot, 트럭 1대, 12 step
- 수요 = **Poisson(292일 forecast 평균)** 확률수요
- 상태 V키 = (시각, 트럭위치, **정류소별 3-레벨 밴드 인덱스**) — 동역학은 exact 정수재고, V 테이블만 밴드 셀로 축약(논문 fill-rate index)
- 행동 = 다음 목적지(정류소/depot). 적재량은 목표(50%)로 자동
- RTDP: dict 가치테이블 + 궤적 샘플링 비동기 백업, V(s)=min_a E[cost+γ^τ V(s')] (M-샘플 Monte-Carlo 기대), V0=0 admissible
- 비교 휴리스틱: do-nothing / STR(반응형, 밴드 밖 최근접) / SLA(정적 lookahead)

### 결과 — 갈래 A (퍼진 정류소+적재 lever+분석 백업, iters=12000)
상세 환경·과정: [rtdp_experiment_setup.md](rtdp_experiment_setup.md).

| 정책 | 확률 Poisson(30) | 배달량 | 실제(7일) |
|---|---|---|---|
| do-nothing | 19.10 | 0 | 31.71 |
| STR (반응·최소재배치) | 8.33 | 24.2 | 17.14 |
| **SLA (예측형 lookahead)** | **5.63** ⭐ | 33.5 | **11.71** ⭐ |
| RTDP (확률적 DP) | 8.43 | 57.4 | 16.71 |

→ **RTDP ≈ STR(반응형), 예측형 SLA(5.63)엔 크게 못 미침. 추월 실패 확정.** (초기 근거리 셋업 6.03 → 퍼진+분석백업 8.43, 둘 다 SLA 못 넘음.)

### 왜 추월 실패했나
1. **상태(밴드 인덱스) 축약이 과배달을 못 벌함** → RTDP 배달 57.4(SLA 33.5의 ~1.7배)로 낭비.
2. **certainty-equivalent 백업이 확률 스파이크를 과소평가** → 과배달이 안전해 보임.
3. **예측형 SLA가 강한 baseline** — 논문 SLA는 25/50/75%·1회방문으로 제약돼 약했음(논문 RTDP 2.3 ≪ SLA 9.6). 우리 SLA는 자유 예측형 → RTDP가 넘을 여지 없음.
4. (해결한 함정) 샘플 백업의 **optimizer's curse** → 분석적 기댓값 백업으로 교정했으나 여전히 과배달·미추월.

---

## 4. 최종 결론 & 남은 레버

- **논문의 "RTDP가 휴리스틱 추월"은 ① 작은 문제(coarse state로 충분) + ② 제약된 약한 baseline(STR 1회방문·SLA 25/50/75%)의 산물.** 마포구에 충실히 재현하고 **강한 예측형 baseline**을 두면 RTDP는 반응형 수준에 머물고 예측형(SLA=우리 예측형)을 못 넘는다.
- **프로젝트 중심 발견 재확인**: 예측형 휴리스틱이 진짜 레버. **model-free RL(DQN/PPO)도, model-based RTDP도 강한 예측형을 능가하지 못한다.** "RL/DP로 휴리스틱을 넘는다"는 이 문제·이 데이터에선 성립하지 않음(예측형 설계가 천장을 올리는 유일한 레버).
- 남은(미시도) 레버: **② 안전재고 z-buffer**(상태=밴드 안/밖, z↑→개선), **③ Poisson 확률수요로 RL 학습**(강건화). 단 둘 다 예측형 추월 가능성은 낮음(위 발견상).

관련 상세 로그: [experiments_2026-06-05.md](experiments_2026-06-05.md) §14(예측오차)·§15(전체 논문 비교).
