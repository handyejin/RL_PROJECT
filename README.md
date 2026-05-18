# 따릉이 재배치 최적화 (RL)

서울시 공공자전거 "따릉이"의 정류소 간 자전거 재배치를 강화학습으로 최적화하는 프로젝트.  
트럭이 자전거를 적재/하차하며 대여 실패(stockout)와 반납 실패(full)를 최소화하는 정책을 학습한다.

본 프로젝트는:

> **Replay-based Simulator + PPO(Proximal Policy Optimization)**

구조를 기반으로 구현한다.

---

# 1. 문제 정의

- **목표**
  - 자치구 단위 권역에서 N대의 트럭이 자전거를 재배치하여
  - 24시간 동안의 누적 실패(대여 실패 + 반납 실패)를 최소화

- **데이터**
  - 서울시 공공자전거 대여 이력
  - 정류소 마스터
  - 날씨
  - 공휴일

- **접근**
  - Gymnasium 커스텀 환경
  - Replay-based demand simulator
  - Stable-Baselines3 PPO

---

# 2. Replay-based Simulator

## 개념

실제 서울시 따릉이 대여/반납 데이터를 시간 순으로 replay(재생)하는 방식.

예시:

```text
08:10 강남역 대여 → bikes -1
08:12 여의도역 반납 → bikes +1
```

강화학습 Agent는:
> 실제 도시 수요가 재생되는 환경 속에서 트럭 재배치를 학습한다.

---

## 특징

- 실제 수요 패턴 기반
- 출퇴근 rush 반영 가능
- 현실적인 시뮬레이션 가능
- synthetic random demand 대비 현실성 우수

---

# 3. PPO(Proximal Policy Optimization)

## 개념

PPO는:
> “정책(policy)을 너무 급격하게 바꾸지 않으면서 안정적으로 학습하는 강화학습 알고리즘”

이다.

---

## PPO 선택 이유

| 항목 | 이유 |
|---|---|
| 학습 안정성 | policy update clipping |
| stochastic 환경 대응 | 교통/수요 변동 대응 가능 |
| multi-truck 확장 | parameter sharing 구조와 궁합 우수 |
| large action space 대응 | DQN 대비 안정적 |

---

# 4. Environment Design

학습을 시작하기 전에 아래 항목들을 확정해야 한다.

현재 `config/default.yaml`의 값은 초안이며,
EDA 이후 조정 필요.

---

# 4.1 권역

- [ ] 서울 전체 따릉이 정류소 약 2,700개
- [ ] 전체 사용 시 state/action explosion 발생 가능

따라서:
- 강남구
- 종로구
- 마포구

등 권역 단위로 나누어 시뮬레이션 수행

---

# 4.2 시간 해상도

> PPO가 결정을 내리는 step과 episode를 정의하는 항목.

## 결정 사항

| 결정 | 현재값 | 의미 |
|---|---|---|
| 1 step | 10분 | 시뮬레이터 처리 단위 |
| 1 episode | 24시간 | PPO 학습 단위 |
| Episode 시작 시각 | 00:00 고정 | 평가 일관성 |
| 학습/평가 분할 | 시간순 9:3 | 앞 9개월 학습, 뒤 3개월 평가 |

---

## 결과

```text
24시간 / 10분
→ 1 episode = 144 step
```

즉:
> PPO는 하루 동안 총 144번 재배치 결정을 수행한다.

---

# 4.3 트럭 설정

## 기본 파라미터

| 항목 | 현재값 | 비고 |
|---|---|---|
| 트럭 수 N | 3대 | config 변경 가능 |
| 적재 용량 | 20대 | 1회 운반 가능 자전거 수 |
| 평균 속도 | 25 km/h | 이동 시간 계산용 |

---

## 다중 트럭 제어 방식

### Parameter Sharing 방식 채택

> 하나의 PPO policy를 모든 트럭이 공유

---

## 동작 방식

```text
트럭 A → PPO policy 호출
트럭 B → 동일 PPO policy 호출
트럭 C → 동일 PPO policy 호출
```

---

## 장점

| 항목 | 설명 |
|---|---|
| 메모리 효율 | policy 하나만 학습 |
| 안정성 | independent multi-agent보다 안정적 |
| 확장성 | 트럭 수 증가 대응 가능 |

---

# 4.4 상태(State) 표현

## 포함 요소

- 정류소별 현재 자전거 수
- 트럭별 현재 위치
- 트럭별 현재 적재량
- 목적지까지 이동 잔여 step
- 시간 정보
  - 시각
  - 요일
  - 공휴일 여부
- 날씨
  - 기온
  - 강수량
  - 풍속
  - 습도

---

## 추후 고려 사항

### 수요 예측 feature 추가 여부

예:

```text
다음 H step 예상 대여/반납량
```

현재 상태 + 미래 예측을 함께 사용하면:
- 선제적 재배치 가능
- rush 대응 성능 향상 가능

단:
- 별도 예측 모델 필요

---

# 4.5 행동(Action) 공간

## PPO Action

- 다음 이동할 정류소 선택

---

## 적재/하차 수량

- rule-based 자동 처리

예:
- deficit station → 자동 하차
- surplus station → 자동 적재

---

## action mask 적용

탐색 효율 향상을 위해:
- 불가능한 행동 제거

예:
- 이동 중인 트럭
- 현재 위치 재선택
- 트럭 간 충돌 가능 위치

---

## PPO와 action mask

SB3 PPO는 기본적으로 discrete action 가능.

단:
- custom masking 필요 가능성 있음

---

# 4.6 보상(Reward)

## Reward 항목

| 항목 | 현재값 | 의미 | 발생 조건 |
|---|---|---|---|
| stockout | -1.0 | 대여 실패 | 빈 정류소 대여 시도 |
| full | -0.8 | 반납 실패 | 가득 찬 정류소 반납 시도 |
| 이동 거리 비용 | -0.01/km | 연료/운영 비용 | 이동 거리 비례 |
| 이동 시간 비용 | -0.005/step | 이동 중 비효율 | 이동 step마다 |

---

## stockout > full 이유

- 대여 실패:
  - 시민이 교통수단 자체를 잃음
- 반납 실패:
  - 근처 정류소 이동 가능

따라서:
- stockout penalty를 더 크게 설정

---

## PPO 관점 중요 포인트

PPO는:
> trajectory 전체 return 기반으로 policy를 업데이트

하므로:
- step reward
- cumulative reward

둘 다 중요하다.

---

# 4.7 시뮬레이터(Demand Model)

## 수요 생성 방식

### (a) Replay 방식 채택

서울시 공공자전거 대여 이력 replay

---

## 특징

- 현실 기반 simulator
- deterministic replay 가능
- stochastic extension 가능

---

## 추후 확장

### (b) Poisson sampling

OD(Origin-Destination) 분포 기반 수요 생성 가능

---

# 5. 프로젝트 진행 단계

---

# Phase 1. 데이터 준비 (Week 1)

1. 원본 CSV 수집
2. EDA 수행
3. 권역 선정
4. 전처리

---

# Phase 2. Environment 구현 (Week 2)

1. Gymnasium Env 구현
2. replay simulator 구현
3. 트럭 이동 로직 구현
4. reward 계산 구현

---

# Phase 3. 베이스라인 구현 (Week 3)

## NO-OP

트럭이 아예 이동하지 않는 정책

---

## 휴리스틱 정책

기본 규칙 기반 정책

### 기본 규칙

```text
most_imbalanced
```

- 트럭 비어있음
  → 가장 surplus 큰 정류소 이동

- 트럭 가득 참
  → 가장 deficit 큰 정류소 이동

---

## 비교 지표

- stockout
- full
- 이동 거리
- 운영 비용

---

## 완료 조건

- NO-OP 평가 완료
- heuristic 평가 완료
- baseline 비교표 작성 완료

---

# Phase 4. PPO 학습 (Week 4-5)

## 단계

1. Stable-Baselines3 PPO 적용
2. TensorBoard 학습 곡선 모니터링
3. 하이퍼파라미터 튜닝
4. baseline 대비 성능 비교

---

## 주요 PPO 하이퍼파라미터

| 항목 | 의미 |
|---|---|
| learning rate | 학습 속도 |
| gamma | 미래 reward 반영 정도 |
| clip range | 정책 변경 제한 |
| entropy coef | exploration 강도 |

---

## 완료 조건

heuristic baseline 대비:
- stockout 감소
- full 감소
- total reward 개선

---

# Phase 5. 알고리즘 비교 (Week 6)

## 비교 대상

| 알고리즘 | 계열 | 특징 |
|---|---|---|
| PPO | Policy-based | 안정적 policy optimization |
| A2C | Actor-Critic | lightweight baseline |
| SAC (확장 가능) | entropy-based RL | stochastic policy |

---

## 완료 조건

- 최소 3개 알고리즘 비교
- 동일 환경 / 동일 데이터 기반 평가
- 최적 알고리즘 선정 및 분석

---

# 6. 결정사항

| # | 결정 사항 | 옵션 | 영향 |
|---|---|---|---|
| 1 | 권역 선택 | 강남구 / 종로구 / 마포구 | state/action 크기 결정 |
| 2 | 다중 트럭 제어 | Parameter Sharing | PPO policy 구조 결정 |
| 3 | 시간 해상도 | 5분 / 10분 / 15분 | episode step 수 결정 |
| 4 | Episode 길이 | 24h / 12h / 6h | horizon 길이 결정 |
| 5 | Action 정의 | 정류소 선택 | action space 크기 |
| 6 | 적재량 처리 | rule-based | action 차원 감소 |
| 7 | 수요 모델 | Replay / Poisson / Hybrid | simulator 구조 결정 |

---

# 7. 핵심 설계 철학

본 프로젝트의 핵심은:

> “실제 서울시 따릉이 수요를 replay하는 simulator 위에서,
PPO가 안정적으로 재배치 정책을 학습하도록 만드는 것”

이다.

강화학습 알고리즘 자체보다 중요한 것은:
- 현실적인 simulator
- reward 설계
- state representation
- stochastic dynamics

이며,
실제 도시 운영과 유사한 환경을 구성하는 것이 최종 목표이다.