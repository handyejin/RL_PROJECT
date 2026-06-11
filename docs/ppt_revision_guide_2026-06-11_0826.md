# PPT 통합본 페이지별 수정 지시서

작성 기준: `따릉이_강화학습_redesign.pdf` 16쪽 초안 검토  
작성 시각: 2026-06-11 08:26  
목적: 팀 통합 발표자료를 최신 실험 기준으로 정리하기 위한 페이지별 수정 가이드

---

## 0. 전체 수정 방향

현재 초안은 디자인과 큰 흐름은 좋지만, **예전 마포구/7일/BC 중심 결과**와 **최신 서울 25개 구/73일 holdout 결과**가 섞여 있다. 최종 장표에서는 실험 기준을 반드시 하나로 통일해야 한다.

### 발표 서사 권장안

실험 결과를 바로 보여주기보다, 발표 초반은 아래 흐름으로 잡는 것이 좋다. 이 흐름이 있어야 뒤쪽의 Top-K, 수요예측, seed 실험이 “그냥 많이 해본 실험”이 아니라 문제 해결 과정으로 보인다.

```text
1. 따릉이 재배치는 왜 어려운가?
   - 정류소별 수요가 시간과 지역에 따라 계속 변한다.
   - 현재 자전거 수만 보고 움직이면 1시간 뒤 부족/포화를 놓칠 수 있다.
   - 트럭 이동은 즉시 끝나지 않고, 현재 선택이 다음 상태와 미래 reward를 바꾼다.

2. 왜 강화학습 문제인가?
   - 정답 행동 라벨이 없다.
   - 매 step의 action이 다음 state를 바꾸는 순차 의사결정이다.
   - 목표는 하루 전체 episode reward를 최대화하는 것이다.

3. 그런데 왜 기본 RL만으로 어려웠나?
   - action 후보가 너무 많다.
   - reward가 지연되어 어떤 이동이 좋은 선택이었는지 credit assignment가 어렵다.
   - 구별 수요 규모와 정류소 밀도가 달라 seed와 지역 편차가 크다.

4. 그래서 무엇을 개선했나?
   - State: 현재 재고만 보지 않고 1시간 수요예측 feature를 추가했다.
   - Action: 전체 정류소 선택 대신 Top-K 후보 rank 선택으로 줄였다.
   - Evaluation: 73일 holdout, MostImbalanced 대비 Delta, seed 반복으로 검증했다.

5. 그 결과 무엇을 배웠나?
   - A2C는 REINFORCE보다 안정적이었다.
   - PPO는 clipping이 있어도 Top-K 후보 수에 민감했다.
   - Bandit/VAE는 보조 실험으로 의미는 있었지만 주 모델을 대체하지는 못했다.
```

### 냉정한 코멘트

팀원 초안의 좋은 점은 **문제 정의 → MDP 설계 → 알고리즘 비교 → 시연**의 큰 발표 흐름이 이미 잡혀 있다는 점이다. 특히 State/Action/Reward를 한 장에 놓은 구성과 Replay Viewer 시연 장표는 발표 설득력에 도움이 된다.

다만 현재 초안의 가장 큰 약점은 **실험 기준이 섞여 있다는 점**이다. 마포구 단일 실험, 7일 평가, BC 실험, 서울 25개 구 73일 실험이 같은 결론처럼 배치되면 재현성과 신뢰도가 떨어진다. 따라서 최종본은 “최신 실험 기준 하나”를 중심축으로 삼고, 예전 실험은 삭제하거나 초기 탐색으로만 언급해야 한다.

### 외부 연구와 연결할 핵심 문장

장표 본문에는 논문 설명을 길게 넣지 말고, 아래처럼 각 설계의 정당성을 짧게 연결한다.

| 우리 설계 | 외부 연구와의 연결 | 장표에 넣을 짧은 문장 |
|---|---|---|
| Dynamic rebalancing | 자전거 재배치는 하루 중 수요와 재고가 계속 바뀌는 dynamic repositioning 문제로 다뤄진다. | “따릉이 재배치는 현재 선택이 미래 재고에 영향을 주는 동적 재배치 문제다.” |
| Demand forecast state | 정류소별 대여/반납 예측은 재배치 의사결정의 핵심 입력으로 연구되어 왔다. | “1시간 수요예측은 현재 재고만으로 보이지 않는 미래 부족/포화 위험을 state에 제공한다.” |
| Top-K action | 거대한 이산 action space는 탐색 난이도를 키우므로 후보 행동 축소가 필요하다. | “Top-K는 정답을 주는 것이 아니라, RL이 탐색할 행동공간을 현실적인 후보로 줄이는 구조다.” |
| Seed/variance reporting | RL 결과는 seed와 환경 stochasticity에 민감하므로 반복 실험과 분산 보고가 중요하다. | “평균 reward만 보지 않고 seed std와 Best-Final gap으로 안정성을 함께 평가했다.” |
| PPO clipping | PPO는 clipped objective로 policy update가 급격히 변하는 것을 제한한다. | “PPO는 old/new policy ratio를 clipping해 update 안정성을 확보하려는 알고리즘이다.” |

### 실험 결과 장표로 넘어가기 전 필요한 메시지

8~10p 사이에 아래 메시지가 자연스럽게 보여야 한다.

> 본 프로젝트의 핵심은 알고리즘을 단순히 바꿔 끼운 것이 아니라, 따릉이 재배치를 MDP로 정의하고 State와 Action을 문제 특성에 맞게 재구성한 뒤 여러 RL 알고리즘이 이 구조에서 어떻게 반응하는지 비교한 것이다.

### 활용방안으로 연결할 메시지

발표 후반에는 “이 실험을 해서 무엇에 쓸 수 있는가?”가 보여야 한다. 아래 활용방안은 결과를 과장하지 않으면서도 프로젝트 의미를 잘 보여준다.

| 활용방안 | 설명 | 발표용 문장 |
|---|---|---|
| 재배치 의사결정 보조 | 실제 운영자가 모든 정류소를 직접 판단하기 어렵기 때문에, 모델이 위험 정류소 후보를 추천할 수 있다. | “RL agent를 완전 자동 운영자가 아니라, 재배치 우선순위 추천 도구로 활용할 수 있다.” |
| 운영 시뮬레이터 | 특정 날짜/구/트럭 수/Top-K 값을 바꿔 stockout/full 변화를 비교할 수 있다. | “정책을 현장에 적용하기 전 시뮬레이터에서 비용과 실패 건수를 비교할 수 있다.” |
| 수요예측 기반 선제 대응 | 현재 재고가 아니라 1시간 뒤 예상 부족/포화를 기준으로 미리 움직일 수 있다. | “수요예측 feature는 사후 대응이 아니라 선제적 재배치를 가능하게 한다.” |
| 정책 비교 플랫폼 | MostImbalanced, REINFORCE, A2C, DQN, PPO, Bandit을 같은 holdout에서 비교할 수 있다. | “같은 데이터와 같은 reward에서 알고리즘별 장단점을 비교하는 실험 플랫폼이 된다.” |
| 자치구별 운영 분석 | 구별 수요 규모와 정류소 밀도에 따라 어떤 알고리즘/Top-K가 유리한지 확인할 수 있다. | “구별 운영 난이도를 정량화해 지역별 재배치 전략을 다르게 설계할 수 있다.” |

단, 최종 발표에서는 “즉시 실서비스 적용 가능”이라고 단정하지 않는다. 현재는 공공데이터 기반 시뮬레이션 실험이므로, 실제 배포 전에는 실시간 재고 API, 차량 운영 제약, 인력 스케줄, 안전/교통 규칙을 추가해야 한다.

### 데이터 출처와 학습자료 링크

장표 5p 또는 Appendix에 아래 표를 넣는 것을 권장한다. 공개 데이터 URL과 프로젝트 내부 산출물을 구분해서 보여주면 재현성이 좋아진다.

| 구분 | 데이터 | 공식/프로젝트 링크 | 본 프로젝트에서의 용도 |
|---|---|---|---|
| 공개 원천 | 서울시 공공자전거 따릉이 대여이력 정보 | https://data.seoul.go.kr/dataList/OA-15182/F/1/datasetView.do | 월별 대여/반납 이력, demand replay 생성 |
| 공개 원천 | 서울시 공공자전거 따릉이 대여소 정보 | https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do | 정류소 위치, 관리번호, 거치대수/capacity |
| 공개 원천 | 서울시 공공자전거 실시간 대여정보 API | https://www.data.go.kr/data/15051891/openapi.do | 향후 실시간 운영 적용 시 현재 재고 입력 후보 |
| 공개 원천 | 기상청 ASOS 시간자료 | https://data.kma.go.kr/data/grnd/selectAsosRltmList.do | 날씨 feature 또는 수요예측 보조 변수 |
| 프로젝트 산출물 | 전처리된 서울 전체 episode 데이터 | `data/processed_seoul_all.zip` 또는 `data/processed_seoul_all/` | 학습/평가 episode 생성 |
| 프로젝트 산출물 | 구별 1시간 수요예측 파일 | `data/forecast_by_gu.zip` 또는 `data/forecast_by_gu/` | forecast state feature |
| 프로젝트 산출물 | 정류소 capacity 보정 테이블 | `data/processed/station_capacity.csv` | 실제 거치대 수 기반 capacity |
| 프로젝트 산출물 | episode cache | `data/episode_cache/` | 반복 학습 로딩 속도 개선 |

발표용 짧은 문장:

> 원천 데이터는 서울 열린데이터광장의 대여이력/대여소 정보와 기상청 ASOS 자료를 사용했고, 학습에는 이를 10분 단위 demand replay와 구별 1시간 수요예측 feature로 전처리해 사용했다.

### 반드시 통일할 기준

| 항목 | 최종 기준 |
|---|---|
| 공간 범위 | 서울 25개 구 |
| 학습/평가 분할 | Chronological split |
| 학습 기간 | 2025년 앞 80%, 총 292일 |
| 평가 기간 | 2025-10-20 ~ 2025-12-31, 총 73일 holdout |
| 주 지표 | `Delta = Model reward - MostImbalanced reward` |
| 좋은 값 | reward는 덜 음수일수록 좋고, Delta는 양수일수록 좋음 |
| baseline | MostImbalanced 규칙 기반 정책 |
| 최종 핵심 | State/Action 재설계, Top-K 후보 행동, Seed 안정성, PPO clipping 진단 |

### 삭제 또는 격하할 표현

아래 표현은 최신 결과와 충돌하므로 삭제하거나 “초기 탐색 실험”으로만 작게 언급한다.

- `7개 평가일`
- `Baseline -448.3`
- `마포구 환경에 국한`
- `K=12 고정`
- `BC 안정성`
- `BC 직후`, `BC 이후 RL Fine-tuning`
- `ablation 미수행`
- `모든 알고리즘이 무조건 baseline 초과`

### 장표 전체 결론 문장

최종 발표의 핵심 결론은 다음처럼 잡는 것이 안전하다.

> 수요예측 기반 State와 Top-K 후보 Action 구조는 따릉이 재배치 문제에서 RL 학습 가능성을 높였다. 다만 성능은 알고리즘, 후보 수, 지역 특성, seed에 따라 달랐으며, A2C는 REINFORCE보다 안정적이고 PPO는 후보 수를 강하게 줄였을 때 안정성이 개선되었다.

---

## 1p. 표지

### 현재 문제

- 제목은 좋지만 “마포구” 문구와 서울 전체 실험이 충돌할 수 있다.
- 최신 실험은 서울 25개 구 기준이므로 제목/부제에서 마포구 중심 표현을 제거해야 한다.

### 수정 지시

**제목 교체**

> 수요예측과 후보 행동 구조를 이용한 서울 따릉이 재배치 강화학습

**부제 교체**

> 서울 25개 구, 73일 holdout 기반 REINFORCE·A2C·DQN·PPO 비교 실험

**표지 하단 추가**

- 데이터: 2025년 서울 따릉이 대여/반납, 정류소, 기상, 수요예측
- 평가: 2025-10-20 ~ 2025-12-31, 73일 holdout

---

## 2p. 팀 구성 및 역할분담

### 현재 문제

- 역할이 실제 최신 담당 범위와 일부 다르게 보인다.
- 박제영 담당 범위가 DQN/PPO 전담처럼 보이면 팀 내 역할과 충돌할 수 있다.

### 수정 지시

박제영 역할은 다음처럼 수정한다.

| 팀원 | 수정 권장 역할 |
|---|---|
| 박제영 | REINFORCE, A2C 구현 및 실험 / VAE latent feature 실험 / Contextual Bandit 비교 / PPO 검산 및 clipping 진단 일부 |
| 손예진 | 환경, 데이터 전처리, State/Action/Reward 설계, 시각화/Replay Viewer |
| 이형진 | DQN/PPO/QRDQN 실험, seed 실험, 통합 결과 정리 |

### 주의

역할 장표에서는 특정 팀원의 알고리즘을 과장하지 말고, 최종 제출 범위에 맞춰 **각자 담당 실험과 산출물**만 적는다.

---

## 3p. 문제 정의: 왜 강화학습인가?

### 현재 장점

- “정답 행동 데이터가 없다”는 설명은 좋다.
- 트럭이 순차적으로 정류소를 방문한다는 설명도 좋다.

### 보강할 내용

이 문제는 단순 예측 문제가 아니라 **순차 의사결정 문제**다. 현재 선택한 정류소가 다음 재고 상태와 미래 대여/반납 실패에 영향을 준다.

### 장표 구조 제안

3p는 “왜 이 문제가 RL인가?”에 집중한다. 다음 3박스 구성을 권장한다.

| 박스 | 내용 |
|---|---|
| 문제 상황 | 출퇴근/주말/날씨에 따라 정류소별 대여·반납이 달라져 stockout/full 발생 |
| 순차성 | 트럭이 지금 어디로 가는지가 이후 정류소 재고와 다음 선택지에 영향을 줌 |
| RL 적합성 | 정답 행동 라벨이 없고, 시행착오를 통해 누적 reward가 큰 정책을 학습해야 함 |

### 추가 문장

> 따릉이 재배치 문제는 현재 정류소 하나를 고르는 문제가 아니라, 트럭의 이동과 적재가 이후 여러 시간대의 재고 상태를 바꾸는 순차 의사결정 문제다. 따라서 State, Action, Reward를 정의하고 episode 누적 보상을 최대화하는 강화학습 문제로 모델링했다.

### 피해야 할 표현

- “수요를 예측하면 최적 정답을 알 수 있다”처럼 쓰면 안 된다. 수요예측은 state 보조 정보일 뿐, action의 정답 라벨이 아니다.
- “RL이 항상 휴리스틱보다 좋다”도 피한다. 본 실험은 조건과 알고리즘에 따라 결과가 달랐다.

### RL 근거

동적 자전거 재배치 연구에서는 하루 중 변하는 수요와 재고를 반영하는 dynamic repositioning 관점이 중요하게 다뤄진다. 참고: [A Reinforcement Learning Approach for Dynamic Rebalancing in Bike Sharing Systems](https://arxiv.org/html/2402.03589v1).

---

## 4p. 연구목표

### 현재 문제

- `Baseline -448.3`은 예전 마포구/7일 기준 숫자로 보인다.
- 최신 실험에서는 구별 baseline scale이 다르므로 단일 baseline 숫자를 큰 목표로 쓰면 위험하다.

### 수정 지시

`SUCCESS CRITERIA -448.3` 박스는 삭제하고, 아래 세 질문으로 교체한다.

| 연구 질문 | 확인 방법 |
|---|---|
| State/Action 재설계가 MostImbalanced baseline 초과 가능성을 만드는가? | 25개 구 73일 holdout Delta |
| TD 기반 A2C가 MC 기반 REINFORCE보다 안정적인가? | Seed std, Best-Final gap, 학습곡선 |
| PPO는 Top-K와 clipping 조건에서 안정적인가? | Top-K ablation, `approx_kl`, `clip_fraction`, Final 성능 |

### 권장 핵심 문장

> 절대 reward는 구별 수요 규모에 따라 크게 달라지므로, 본 실험은 각 구의 MostImbalanced baseline 대비 개선폭인 Delta를 중심 지표로 사용했다.

### 취지 문장

> 본 프로젝트의 목표는 “최고 점수 하나”를 찾는 것이 아니라, 따릉이 재배치 문제에서 어떤 State/Action 설계가 RL 학습을 가능하게 하고, 어떤 알고리즘이 더 안정적으로 작동하는지 확인하는 것이다.

---

## 5p. 환경 및 데이터셋

### 현재 문제

- `7일 평가`와 `73일 평가`가 섞여 있다.
- `trian` 오타가 있다.
- 데이터 분할을 더 명확하게 보여줘야 한다.

### 수정 지시

데이터 분할 표를 아래처럼 교체한다.

| 구분 | 기간 | 일수 | 목적 |
|---|---:|---:|---|
| Train | 2025-01-01 ~ 2025-10-19 | 292일 | 정책 학습 |
| Eval holdout | 2025-10-20 ~ 2025-12-31 | 73일 | 최종 평가 |

데이터셋 표는 유지하되, 아래 설명을 추가한다.

> 평가 데이터는 학습 이후의 시간 구간으로 분리했다. 이는 미래 시점 일반화 성능을 확인하기 위한 chronological split이다.

### 수요예측 근거

정류소별 대여/반납 예측은 재배치 의사결정의 핵심 입력이다. KDD 2018 station-level demand prediction 연구도 시간/날씨 feature를 사용해 정류소별 수요를 예측한다. 참고: [KDD 2018 station-level demand prediction](https://www.kdd.org/kdd2018/accepted-papers/view/towards-station-level-demand-prediction-for-effective-rebalancing-in-bike-s).

### 데이터 설명 추가

수요예측 feature 설명은 너무 모델 세부로 들어가지 말고, 아래 정도로 발표한다.

```text
정류소별 과거 대여/반납 패턴, 시간 정보, 기상 정보를 이용해
앞으로 1시간 동안의 예상 대여량과 반납량을 만든다.
이 예측값은 RL agent가 현재 state를 볼 때 함께 관측하는 보조 feature다.
```

### 누수 방지 설명

> 평가 구간의 실제 미래 대여/반납을 직접 읽는 oracle 방식이 아니라, 학습 데이터 기반 예측 feature를 사용해 미래 불균형 가능성을 제공했다.

### 데이터 출처 링크 추가

5p 하단 또는 Appendix에 다음 링크를 넣는다.

| 데이터 | 링크 |
|---|---|
| 따릉이 대여이력 | https://data.seoul.go.kr/dataList/OA-15182/F/1/datasetView.do |
| 따릉이 대여소 정보 | https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do |
| 기상청 ASOS | https://data.kma.go.kr/data/grnd/selectAsosRltmList.do |
| 실시간 대여정보 API | https://www.data.go.kr/data/15051891/openapi.do |

프로젝트 학습자료는 다음처럼 표기한다.

```text
data/processed_seoul_all.zip
data/forecast_by_gu.zip
data/processed/station_capacity.csv
data/episode_cache/
```

> 공개 원천 데이터와 프로젝트 전처리 산출물을 분리해 제시하면, 어떤 데이터가 외부에서 받을 수 있는 자료이고 어떤 데이터가 우리가 만든 학습 입력인지 명확해진다.

---

## 6p. MDP 설계: State, Action, Reward

### 현재 장점

- State/Action/Reward를 한 장에 담은 구조는 좋다.
- 수요예측 feature와 Top-K 후보 구조를 연결한 점도 좋다.

### 반드시 보강할 내용

MDP 설계는 과제 평가에서 매우 중요하므로, 이 장표는 가장 튼실해야 한다.

### State 설명

| State 구성 | 의미 |
|---|---|
| 정류소 현재 재고 비율 | 현재 자전거 수 / capacity |
| Capacity | 정류소별 거치 가능 수 |
| 트럭 상태 | 트럭 위치, 적재량, 이동 중 여부 |
| 시간 정보 | 시각, 요일, 주말/공휴일 |
| 기상 정보 | 수요 변동에 영향을 주는 외부 요인 |
| 1시간 수요예측 | 향후 rentals/returns 및 예상 재고 편차 |
| Top-K 후보 feature | 후보 정류소의 불균형 점수, 거리 penalty, 권역 penalty |

### Action 설명

전체 정류소를 직접 고르는 대신, 매 step마다 점수가 높은 후보 정류소 Top-K를 만들고 agent는 그중 **rank**를 선택한다.

```text
원래 action: 전체 정류소 id 선택
개선 action: 현재 상태에서 계산된 Top-K 후보 중 rank 선택
```

### Top-K 알고리즘 설명

Top-K는 별도의 강화학습 알고리즘이 아니라 **action wrapper / candidate generator**다. 환경의 모든 정류소 중에서 현재 상태상 의미가 큰 후보를 먼저 고르고, RL agent는 그 후보 안에서 선택한다.

```text
1. 각 정류소의 예상 불균형 계산
   forecast_imbalance = |projected_bikes - target_bikes|

2. 이동 비용과 권역 penalty 반영
   candidate_score =
       forecast_imbalance
     - travel_coef * travel_distance
     - zone_penalty

3. candidate_score 상위 K개 정류소 선택

4. RL action은 정류소 id가 아니라 Top-K 후보의 rank
   action = 0 ... K-1
```

### Top-K를 쓴 이유

| 어려움 | Top-K의 역할 |
|---|---|
| 정류소 수가 많아 action space가 큼 | 후보 수를 K개로 줄여 탐색 난이도 감소 |
| 대부분의 정류소는 현재 상황에서 중요하지 않음 | 수요예측/거리 기반으로 의미 있는 후보 우선 |
| policy가 초기에 엉뚱한 정류소를 고르기 쉬움 | 후보 후보군을 현실적인 선택지로 제한 |
| K가 너무 작으면 좋은 후보 누락 | K=3,6,9,12,15 ablation으로 확인 |

### Reward 설명

평가 reward는 원본 환경 reward를 사용한다.

```text
r_t =
  - 1.0 * stockout
  - 0.8 * full
  - 0.008 * travel_km
  - 0.002 * travel_step
```

### 장표에 넣을 핵심 문장

> 수요예측 feature는 reward를 바꾼 것이 아니라, 같은 reward를 더 잘 얻기 위해 agent가 보는 state를 보강한 것이다.

> Top-K는 reward를 직접 높이는 장치가 아니라, 너무 큰 이산 action space를 줄여 RL이 학습 가능한 후보 공간에서 탐색하도록 만드는 장치다.

---

## 7p. Reward 발생 예시

### 현재 상태

초안에는 reward 예시가 부족하다. 이 장표를 새로 만들거나 6p 아래쪽을 분리해 7p로 구성하는 것을 권장한다.

### 추가할 예시

| 상황 | Reward 변화 | 의미 |
|---|---:|---|
| 자전거가 없어 대여 실패 발생 | `-stockout penalty` | 서비스 실패 |
| 거치대가 꽉 차 반납 실패 발생 | `-full penalty` | 서비스 실패 |
| 먼 정류소로 이동 | `-travel distance cost` | 운영 비용 증가 |
| 이동 시간이 길어짐 | `-travel step cost` | 재배치 지연 |
| 부족/포화 정류소를 미리 완화 | penalty 감소 | episode reward 개선 |

### 예시 문장

> 예를 들어 퇴근 시간에 특정 정류소의 대여 수요가 몰릴 것으로 예측되면, 트럭이 미리 자전거를 공급해 stockout을 줄일 수 있다. 이 경우 직접적인 양수 보상보다는 실패 penalty가 줄어 episode 누적 reward가 개선된다.

---

## 8p. 강화학습 알고리즘 개요

### 현재 문제

- 알고리즘 수식은 좋지만, 각 알고리즘의 학습 포인트를 더 명확히 분리해야 한다.
- PPO clipping 설명은 유지하되, old policy/new policy 설명을 한 줄 넣으면 좋다.

### 수정 표

| 알고리즘 | 분류 | 학습 포인트 |
|---|---|---|
| REINFORCE | On-policy, Monte Carlo PG | episode 종료 후 reward-to-go로 policy update |
| A2C | On-policy, Actor-Critic | 1-step TD advantage로 더 자주 update |
| Double DQN | Off-policy, Value-based | replay buffer와 target network로 Q-value 학습 |
| PPO | On-policy, Clipped Actor-Critic | old policy와 new policy 차이를 clipping으로 제한 |

### PPO 설명 보강

```text
ratio = pi_new(a|s) / pi_old(a|s)
L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
```

> PPO는 좋은 행동의 확률을 높이되, 한 번의 update에서 policy가 너무 크게 변하지 않도록 ratio를 제한한다.

PPO 근거: [Schulman et al., 2017 PPO](https://arxiv.org/abs/1707.06347), [OpenAI Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

---

## 9p. 실험 설계 및 하이퍼파라미터

### 현재 문제

- `7개 평가일`, `Baseline -448.3` 표현을 제거해야 한다.
- BC epoch 중심 설정은 최종 핵심이 아니므로 삭제하거나 부록으로 이동한다.

### 실험 설계 흐름

장표를 아래 흐름으로 바꾼다.

| 단계 | 목적 | 실험 |
|---|---|---|
| 1. Full baseline run | 전체 경향 확인 | 25개 구, Top-K 12, seed 42 |
| 2. Best/Worst 선정 | 어려운 구와 쉬운 구 분리 | Delta 기준 상/하위 구 |
| 3. Top-K ablation | 후보 action 수 영향 확인 | K=3,6,9,12,15 |
| 4. Confirmation | 선택 K가 우연인지 확인 | 500 episode 또는 170k step |
| 5. Seed validation | 안정성 확인 | seed 42,123,777 |
| 6. Final run | 최종 설정 검증 | 선택 K로 전체 재실험 |

### 하이퍼파라미터 표 수정

`BC Epochs` 행은 제거하거나 `초기 탐색 실험`으로 따로 뺀다. 대신 아래를 넣는다.

| 항목 | REINFORCE | A2C | PPO |
|---|---:|---:|---:|
| 평가 기간 | 73일 holdout | 73일 holdout | 73일 holdout |
| seed 검증 | 42,123,777 | 42,123,777 | 42,123,777 |
| Top-K ablation | 3,6,9,12,15 | 3,6,9,12,15 | 3,6,9,12,15 |
| 안정성 지표 | seed std | seed std | Best-Final gap, KL, clip fraction |

---

## 10p. 평가 지표

### 현재 문제

- `7개 평가일 평균 Return`을 `73일 holdout 평균 Return`으로 바꿔야 한다.
- PPO 진단 지표를 추가해야 한다.

### 수정 표

| 지표 | 정의 | 해석 |
|---|---|---|
| Mean reward | 73일 holdout 평균 reward | 덜 음수일수록 좋음 |
| Delta | Model reward - MostImbalanced reward | 양수면 baseline 초과 |
| Best checkpoint | 학습 중 가장 좋은 평가 성능 | 대표 성능 |
| Final checkpoint | 학습 종료 시점 성능 | 후반 안정성 |
| Best-Final gap | Best Delta - Final Delta | 작을수록 안정적 |
| Seed std | seed별 Delta 표준편차 | 작을수록 재현성 높음 |
| PPO approx_kl | old/new policy 차이 | 작으면 update가 보수적 |
| PPO clip_fraction | clipping이 적용된 sample 비율 | PPO clipping 작동 여부 |

### MostImbalanced 설명

> MostImbalanced는 학습하지 않는 규칙 기반 baseline이다. 현재 트럭 적재 상태와 정류소 목표 재고를 보고, 자전거가 가장 부족하거나 과잉인 정류소를 우선 방문한다.

---

## 11p. 전체 결과 표

### 현재 문제

- BC 중심 결과표는 최종 핵심 실험과 맞지 않는다.
- 예전 마포구 숫자와 73일 서울 전체 숫자가 섞여 있다.

### 수정 지시

BC 표는 삭제하고 최신 73일 기준 전체 결과 표를 넣는다.

| 실험군 | 알고리즘 | 구 수 | Best Δ 평균 | Best 승리 구 | Final Δ 평균 | Final 승리 구 |
|---|---|---:|---:|---:|---:|---:|
| Full TopK12 | A2C | 25 | +13.0 | 17 | +3.2 | 14 |
| Full TopK12 | REINFORCE | 25 | -8.4 | 8 | -35.5 | 6 |
| Final TopK9 | A2C | 25 | +11.6 | 16 | -0.4 | 12 |
| Final TopK9 | REINFORCE | 25 | +0.2 | 13 | -24.0 | 9 |
| Bandit TopK12 | Contextual Bandit | 25 | -260.4 | 0 | -283.6 | 0 |

PPO/DQN은 팀원 최신 결과와 기준이 맞는 경우에만 추가한다. 기준이 다르면 별도 표로 분리한다.

### 핵심 해석 문장

> A2C는 전체 25개 구에서 REINFORCE보다 더 높은 평균 Delta와 더 많은 baseline 초과 구를 보였다. 이는 1-step TD 기반 critic이 Monte Carlo 기반 REINFORCE보다 안정적인 학습 신호를 제공했기 때문으로 해석된다.

---

## 12p. REINFORCE/A2C 결과

### 장표 목적

박제영 담당 알고리즘의 핵심 결과를 보여주는 장표다. 단순 성능표보다 **MC vs TD 차이**를 강조해야 한다.

### 넣을 내용

| 비교 | REINFORCE | A2C |
|---|---|---|
| 학습 신호 | episode 전체 reward-to-go | 1-step TD advantage |
| update 시점 | episode 종료 후 | transition batch 단위 |
| 장점 | 구조가 명확하고 policy gradient 설명에 적합 | critic으로 분산 감소, 안정적 |
| 약점 | seed와 초기 sampling에 민감 | critic 품질에 의존 |

### Seed 안정성 문장

> Seed 반복 실험에서 A2C의 Best seed std 중앙값은 1.0, REINFORCE는 24.4였다. 이는 대부분 구에서 A2C가 seed 변화에 덜 민감했음을 의미한다.

### 그래프 추천

- REINFORCE/A2C 학습곡선 평균 + IQR band
- seed별 Delta scatter 또는 errorbar
- Best/Worst 3구 학습곡선

RL 실험에서 seed와 분산 보고가 중요한 이유는 [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), [Empirical Design in Reinforcement Learning](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf)에 맞춰 짧게 언급한다.

---

## 13p. PPO 결과 및 clipping 진단

### 장표 목적

PPO는 단순히 “성능이 좋다/나쁘다”보다 **policy update 안정화**를 보여줘야 한다.

### 최신 PPO 결과 요약

| 실험 | 결과 |
|---|---:|
| PPO Top-K12 전체 25구 Best 승리 | 14/25 |
| PPO Top-K12 전체 25구 Final 승리 | 8/25 |
| PPO Top-K12 Best Δ 평균 | +2.7 |
| PPO Top-K12 Final Δ 평균 | -23.5 |
| PPO Top-K3 Best/Worst 6구 Best 승리 | 6/6 |
| PPO Top-K3 Best/Worst 6구 Final 승리 | 6/6 |
| PPO Top-K3 seed validation Best 승리 | 18/18 |

### 해석

> PPO는 Top-K12에서는 후반 Final 성능이 떨어지는 구가 있었지만, Top-K3로 후보를 강하게 줄이면 Best와 Final 모두 안정적으로 baseline을 초과했다. 이는 PPO의 clipped update 안정성만으로 충분한 것이 아니라, action 후보 구조가 함께 맞아야 함을 보여준다.

### Clipping 진단

현재 확인된 PPO 진단 파일은 `ppo_diagnostics.csv` 13개다. 18개가 모두 생성되면 아래 지표를 그래프로 넣는다.

| 진단 지표 | 현재 확인된 평균 | 해석 |
|---|---:|---|
| `approx_kl` | 0.00011 | old/new policy 차이가 작음 |
| `clip_fraction` | 0.00069 | clipping 발생 비율이 낮음 |
| `entropy_loss` | -0.738 | policy 탐색성 참고 지표 |
| `explained_variance` | 0.412 | critic 설명력 참고 지표 |

### 그래프 추천

- PPO Top-K별 Best/Final Delta bar chart
- PPO K=3 seed별 Delta errorbar
- `approx_kl` over timesteps
- `clip_fraction` over timesteps
- `entropy_loss` over timesteps

### 주의 문장

> PPO clipping은 policy update를 보수적으로 만드는 장치이지, 좋은 action 후보를 자동으로 찾아주는 장치는 아니다. 본 실험에서는 Top-K 후보 수가 PPO 성능과 안정성에 큰 영향을 주었다.

---

## 14p. DQN/PPO 또는 팀원 알고리즘 결과

### 현재 문제

- DQN/PPO 결과가 팀원 최신 기준인지, 이전 실험 기준인지 섞일 수 있다.
- 기준이 다르면 같은 표에 넣으면 안 된다.

### 수정 지시

팀원 최신 결과를 받은 경우에만 아래 조건을 만족하는 표로 넣는다.

| 확인 항목 | 필요 조건 |
|---|---|
| 평가 기간 | 73일 holdout |
| baseline | MostImbalanced |
| Delta 계산 | Model - Baseline |
| Top-K | 표에 명시 |
| seed | 표에 명시 |
| Best/Final | 둘 다 표시 |

### 현재 검증되지 않은 경우

“통합 예정” 같은 문구를 크게 쓰기보다, 장표를 **알고리즘별 해석**으로 바꾼다.

| 알고리즘 | 관찰 포인트 |
|---|---|
| DQN | Q-value 기반 off-policy 학습은 큰/동적인 action 후보에서 불안정할 수 있음 |
| PPO | clipped objective로 update 안정성을 추구하지만 Top-K 후보 품질에 민감 |

---

## 15p. Replay Viewer 시연

### 현재 장점

시각화 장표는 발표에서 강점이 된다. 실제 트럭이 어디로 이동하는지 보여주면 State/Action/Reward 설명이 직관적으로 연결된다.

### 수정 지시

영상 삽입 위치에 다음 설명을 붙인다.

| 화면 요소 | 의미 |
|---|---|
| 정류소 점 | 현재 재고 비율과 capacity |
| Top-K 후보 | 현재 state에서 agent가 고려하는 후보 행동 |
| 선택된 정류소 | policy가 고른 action |
| 트럭 이동선 | action 이후 이동 경로 |
| 누적 reward | stockout/full/travel cost가 반영된 episode 성능 |

### 발표 멘트

> 이 뷰어는 학습된 agent가 매 step에서 어떤 후보 정류소를 보고, 그중 어느 정류소를 선택하며, 그 선택이 episode reward에 어떻게 반영되는지 보여준다.

---

## 16p. 결론 및 향후 과제

### 현재 문제

- “모든 알고리즘이 baseline 초과”는 최신 결과 기준으로 위험하다.
- 한계에 “마포구 국한”, “단일 seed”, “ablation 미수행”이 있으면 최신 실험과 충돌한다.

### 수정 결론

| 결론 | 근거 |
|---|---|
| State/Action 재설계는 필요했다 | 수요예측 feature와 Top-K 후보 구조로 RL 학습 가능성 증가 |
| A2C는 REINFORCE보다 안정적이었다 | 25구 평균 Delta, seed std, 학습곡선 |
| PPO는 후보 수에 민감했다 | Top-K12보다 Top-K3에서 안정성 개선 |
| Bandit은 장기 return 최적화 한계를 보였다 | 즉시 보상 중심이라 재고 변화 누적 효과 반영 어려움 |
| VAE는 흥미롭지만 현재 설정에서는 선택적이다 | 일부 개선은 있으나 일관된 개선은 아님 |

### 향후 과제

- 더 많은 seed와 confidence interval
- 구별 수요 규모를 정규화한 상대 Delta 분석
- PPO 내부 지표(`approx_kl`, `clip_fraction`) 전체 18개 로그 완성 후 그래프화
- Top-K 후보 점수식의 계수 ablation
- 다중 트럭 간 협력 정책 또는 multi-agent 확장

### 활용방안 추가

결론 장표에는 아래 활용방안 박스를 넣는 것을 권장한다.

| 활용방안 | 의미 |
|---|---|
| 운영 의사결정 보조 | 위험 정류소와 재배치 후보를 추천 |
| 정책 사전 검증 | 트럭 수, Top-K, reward 가중치 변경을 시뮬레이터에서 비교 |
| 자치구별 전략 수립 | 구별 수요 규모와 정류소 밀도에 따라 다른 재배치 전략 설계 |
| 실시간 시스템 확장 | 향후 실시간 대여정보 API와 연결해 온라인 재배치 의사결정으로 확장 |

발표 문장:

> 본 프로젝트는 바로 실서비스에 투입하는 모델이라기보다, 공공자전거 재배치 정책을 데이터 기반으로 비교하고 검증할 수 있는 시뮬레이션·의사결정 보조 프레임워크로 활용할 수 있다.

### 참고문헌 정리

아래 5개 정도만 본문에 넣고, 나머지는 부록으로 보낸다.

1. Sutton and Barto, *Reinforcement Learning: An Introduction*, 2018.
2. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning”, 1992.
3. Schulman et al., “Proximal Policy Optimization Algorithms”, 2017.
4. Henderson et al., “Deep Reinforcement Learning that Matters”, 2018.
5. Bike-sharing demand/rebalancing 관련 최신 연구: dynamic rebalancing, station-level demand prediction.

---

## 최종 점검 체크리스트

| 체크 | 기준 |
---|---|
| 73일 holdout으로 통일했는가? | `2025-10-20 ~ 2025-12-31` |
| 7일 평가 표현을 제거했는가? | 모든 장표 검색 |
| `-448.3` 단일 baseline을 제거했는가? | 구별 baseline/Delta 중심 |
| BC 중심 장표를 제거했는가? | 최종 핵심은 seed/top-k/holdout |
| State/Action/Reward가 명확한가? | 과제 핸드아웃 필수 항목 |
| Reward 발생 예시가 있는가? | stockout/full/travel |
| PPO 고유 특성이 들어갔는가? | ratio, clip, KL, clip_fraction |
| seed 반복과 분산 해석이 있는가? | REINFORCE vs A2C 안정성 |
| Top-K ablation이 있는가? | 후보 action 수의 영향 |
| 결론이 과장되지 않았는가? | “가능성/조건부 개선”으로 표현 |

---

## 팀원에게 전달할 짧은 요약

현재 PPT 초안은 레이아웃은 좋지만 예전 마포구/7일/BC 결과와 최신 서울 25구/73일 결과가 섞여 있습니다. 최종본은 73일 holdout, MostImbalanced 대비 Delta, Top-K ablation, seed 안정성, PPO clipping 진단 중심으로 통일하는 것이 좋습니다. 특히 MDP 설계 장표에서는 State/Action/Reward와 reward 발생 예시를 강화하고, 결과 장표에서는 BC 표를 제거한 뒤 최신 Best/Final Delta와 seed variance를 넣는 방향을 권장합니다.
