# REINFORCE와 A2C를 이용한 서울 따릉이 재배치 강화학습 최종 보고서

**수요예측 기반 상태 보강과 Top-K 후보 행동 구조에서 REINFORCE와 A2C를 비교한 결과**

작성일: 2026-06-09 06:59

---

## Abstract

본 보고서는 서울 25개 구 따릉이 재배치 문제를 **REINFORCE with Value Baseline**과 **A2C(Advantage Actor-Critic)** 로 학습한 최종 결과를 정리한다. DQN과 PPO는 팀원 별도 실험 범위로 분리했기 때문에, 본문 결과표와 결론에서는 제외한다.

문제의 목표는 재배치 트럭이 하루 동안 방문할 정류소를 순차적으로 선택해 **자전거 부족(stockout)**, **거치 공간 부족(full)**, **이동 비용**을 줄이는 것이다. Reward는 실패와 비용을 음수로 부여하므로 **0에 가까울수록 좋은 성능**이다. 최종 평가는 시간순 split을 적용해 `2025-10-20`부터 `2025-10-26`까지 7일 평균 reward로 수행했다.

핵심 결과는 다음과 같다. **A2C는 25개 구 중 Best 기준 18개 구, Final 기준 16개 구에서 MostImbalanced baseline을 넘었고, 평균 Best Delta는 +24.9였다.** 반면 REINFORCE는 Best 기준 10개 구에서 baseline을 넘었지만 평균 Best Delta는 -4.4, 평균 Final Delta는 -38.1로 더 불안정했다.

---

## 1. 문제 정의와 최적화 목표

따릉이 재배치 문제는 정류소별 자전거 재고가 시간에 따라 변하는 상황에서, 재배치 트럭이 다음에 방문할 정류소를 반복적으로 선택하는 순차 의사결정 문제다. 어떤 정류소는 자전거가 부족해 대여 실패가 발생하고, 어떤 정류소는 거치 공간이 부족해 반납 실패가 발생한다.

강화학습 관점에서 목표는 하루 episode의 누적 reward를 최대화하는 것이다. 본 환경의 reward는 실패와 이동 비용을 음수로 계산하므로, 실제 운영 목표는 다음과 같이 해석된다.

```text
maximize episode_reward
= minimize(stockout penalty + full penalty + travel cost)
```

즉, 좋은 정책은 **재고 부족과 포화를 줄이면서도 불필요한 이동을 줄이는 정책**이다.

---

## 2. State, Action, Reward 설계

### 2.1 State

State는 현재 재고만 보지 않고, 1시간 뒤 수요 변화를 예측할 수 있도록 구성했다.

| 범주 | 주요 정보 |
|---|---|
| 정류소 재고 | 현재 자전거 수, capacity, 목표 재고 대비 편차 |
| 수요예측 | 1시간 예측 대여량, 반납량, 순수요, 예측 후 재고 편차 |
| 트럭 상태 | 현재 위치, 적재량, 이동 상태 |
| 시간 정보 | 10분 단위 step, 날짜와 시간 흐름 |
| 후보 정보 | Top-K 후보별 불균형 점수, 이동거리 penalty, 권역 penalty |

1시간 예측 수요는 다음처럼 현재 재고와 결합했다.

```text
pred_net_1h = pred_returns_1h - pred_rentals_1h
projected_bikes = current_bikes + pred_net_1h
projected_deviation = (projected_bikes - target_bikes) / capacity
```

`projected_deviation`이 음수이면 1시간 뒤 자전거 부족 가능성이 크고, 양수이면 거치 공간 포화 가능성이 크다.

### 2.2 Action

원래 action은 전체 정류소 중 다음 방문 정류소를 직접 고르는 것이다. 하지만 구별 정류소 수가 많기 때문에 전체 정류소를 그대로 action space로 두면 탐색이 어렵다.

그래서 매 step마다 수요예측과 거리 정보를 이용해 **Top-K 후보 정류소 12개**를 만들고, agent는 이 12개 후보 중 하나를 선택하도록 했다.

```text
candidate_score =
    forecast_imbalance
  - candidate_travel_coef * travel_distance
  - zone_penalty
```

### 2.3 Reward

Reward는 원본 환경의 평가 reward를 그대로 사용했다. 핵심은 stockout/full과 이동 비용을 줄이는 것이다.

```text
r_t = -1.0 * stockout
      -0.8 * full
      -0.008 * travel_km
      -0.002 * travel_step
```

평가에서는 `MostImbalanced` 규칙 정책을 baseline으로 두고, 아래 Delta를 주 지표로 사용했다.

```text
Delta = model_eval_reward - MostImbalanced_eval_reward
```

Delta가 양수이면 모델이 baseline보다 좋고, 음수이면 baseline보다 나쁘다.

---

## 3. 알고리즘

### 3.1 REINFORCE with Value Baseline

REINFORCE는 episode가 끝난 뒤 reward-to-go를 계산해 policy를 업데이트하는 Monte Carlo policy gradient 알고리즘이다. 본 구현에서는 Value Network를 baseline으로 사용해 advantage 분산을 줄였다.

```python
returns = discounted_reward_to_go(rewards, gamma)
advantages = returns - value_net(states)

policy_loss = -(log_probs * advantages.detach()).mean()
value_loss = mse_loss(value_net(states), returns)
```

REINFORCE는 구현이 직관적이고 policy gradient의 기본 구조를 설명하기 좋지만, episode 전체 return에 의존하므로 구별 수요 패턴과 seed에 민감할 수 있다.

### 3.2 A2C

A2C는 Actor가 action 확률분포를 만들고, Critic이 현재 state의 value를 추정한다. 매 step 또는 batch 단위로 TD target을 만들 수 있어 REINFORCE보다 더 자주 학습 신호를 받을 수 있다.

```python
target = reward + gamma * (1 - done) * value(next_state)
advantage = target - value(state)

actor_loss = -(log_prob(action) * advantage.detach()).mean()
critic_loss = mse_loss(value(state), target)
```

이번 실험에서는 A2C가 REINFORCE보다 평균 성능과 seed 안정성 모두에서 더 좋은 결과를 보였다.

---

## 4. 실험 설정

| 항목 | 설정 |
|---|---|
| 대상 지역 | 서울 25개 구 |
| 학습/평가 분할 | 시간순 chronological split |
| 평가 날짜 | 2025-10-20 ~ 2025-10-26, 총 7일 |
| 학습 길이 | 500 episodes |
| 평가 주기 | 50 episodes |
| 공통 seed | 42 |
| 추가 seed 실험 | Best/Worst 일부 구에서 123, 777 추가 |
| Top-K | 12 |
| BC 사용 | 사용하지 않음 |
| rollback | 사용하지 않음. 단, Best checkpoint는 저장 후 평가에 사용 |

Best checkpoint는 학습 중 평가 reward가 가장 좋았던 시점이고, Final checkpoint는 학습 종료 시점이다. 본 보고서는 Best를 성능 가능성, Final을 학습 안정성으로 해석한다.

---

## 5. 최종 결과

### 5.1 알고리즘별 요약

| Algorithm | 구 수 | Best 평균 Reward | Final 평균 Reward | Best Δ 평균 | Best Δ 중앙값 | Final Δ 평균 | Best 승리 구 | Final 승리 구 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2C | 25.0 | -815.6 | -827.4 | 24.9 | 18.5 | 13.1 | 18.0 | 16.0 |
| REINFORCE | 25.0 | -844.9 | -878.6 | -4.4 | -7.0 | -38.1 | 10.0 | 7.0 |

![REINFORCE/A2C summary](figures/reinforce_a2c_chronological_summary_2026-06-09_065917.png)

결과적으로 A2C가 더 안정적이다. A2C는 Best 기준 18개 구, Final 기준 16개 구에서 baseline을 넘었다. REINFORCE는 Best 기준 10개 구에서 baseline을 넘었지만 Final 기준으로는 7개 구만 baseline을 넘었다.

### 5.2 구별 비교

| 구 | Baseline | A2C Best Δ | A2C Final Δ | A2C Best ep | REINFORCE Best Δ | REINFORCE Final Δ | REINFORCE Best ep | Best 승자 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 강남구 | -689.9 | 50.6 | 50.6 | 50.0 | -5.7 | -38.7 | 100.0 | A2C |
| 강동구 | -615.2 | 37.4 | 28.1 | 1.0 | 0.7 | -38.2 | 100.0 | A2C |
| 강북구 | -24.0 | -2.7 | -6.4 | 50.0 | -2.7 | -2.7 | 350.0 | A2C |
| 강서구 | -4172.9 | 53.9 | 53.9 | 50.0 | -70.6 | -116.6 | 150.0 | A2C |
| 관악구 | -147.8 | -38.8 | -40.4 | 400.0 | -40.4 | -40.4 | 100.0 | A2C |
| 광진구 | -896.5 | 14.2 | 14.2 | 100.0 | 14.2 | -297.2 | 100.0 | A2C |
| 구로구 | -1037.6 | 37.9 | -21.9 | 50.0 | -91.0 | -107.1 | 100.0 | A2C |
| 금천구 | -436.0 | 10.0 | 10.0 | 200.0 | -41.8 | -41.8 | 500.0 | A2C |
| 노원구 | -720.8 | 72.0 | 64.3 | 50.0 | 72.0 | 72.0 | 50.0 | A2C |
| 도봉구 | -149.1 | -7.9 | -7.9 | 50.0 | -47.6 | -47.6 | 150.0 | A2C |
| 동대문구 | -541.3 | 27.3 | 27.3 | 50.0 | 28.8 | 27.3 | 150.0 | REINFORCE |
| 동작구 | -225.2 | -2.3 | -2.9 | 100.0 | -10.3 | -56.1 | 300.0 | A2C |
| 마포구 | -911.8 | 85.6 | 85.6 | 50.0 | 85.6 | -4.1 | 400.0 | A2C |
| 서대문구 | -204.0 | -15.0 | -15.2 | 50.0 | -31.6 | -54.9 | 1.0 | A2C |
| 서초구 | -349.7 | 39.3 | 39.3 | 500.0 | 39.3 | 39.3 | 450.0 | A2C |
| 성동구 | -975.1 | 47.3 | 24.6 | 150.0 | -34.1 | -59.1 | 50.0 | A2C |
| 성북구 | -205.3 | -9.6 | -9.6 | 100.0 | -7.8 | -41.7 | 250.0 | REINFORCE |
| 송파구 | -2115.7 | 62.2 | 62.2 | 50.0 | 63.5 | 62.2 | 1.0 | REINFORCE |
| 양천구 | -1789.4 | 29.0 | 29.0 | 50.0 | 71.6 | 35.7 | 100.0 | REINFORCE |
| 영등포구 | -3139.1 | 76.5 | -112.9 | 1.0 | -27.5 | -53.6 | 1.0 | A2C |
| 용산구 | -163.2 | 17.0 | 17.0 | 250.0 | 17.0 | 17.0 | 200.0 | A2C |
| 은평구 | -283.2 | -9.8 | -9.8 | 50.0 | -7.0 | -90.0 | 250.0 | REINFORCE |
| 종로구 | -516.8 | 18.5 | 18.5 | 100.0 | -31.7 | -31.7 | 200.0 | A2C |
| 중구 | -380.1 | 14.0 | 14.0 | 100.0 | -68.9 | -100.5 | 50.0 | A2C |
| 중랑구 | -322.7 | 16.7 | 16.7 | 100.0 | 16.7 | 16.7 | 500.0 | A2C |

구별 Best 승자 수는 A2C 20개 구, REINFORCE 5개 구였다.

### 5.3 Best/Worst 구

![Best/Worst districts](figures/reinforce_a2c_chronological_best_worst_2026-06-09_065917.png)

| 알고리즘 | Best 3 | Worst 3 |
|---|---|---|
| A2C | 마포구 +85.6, 영등포구 +76.5, 노원구 +72.0 | 은평구 -9.8, 서대문구 -15.0, 관악구 -38.8 |
| REINFORCE | 마포구 +85.6, 노원구 +72.0, 양천구 +71.6 | 중구 -68.9, 강서구 -70.6, 구로구 -91.0 |

### 5.4 학습곡선

![Learning curves](figures/reinforce_a2c_chronological_curves_2026-06-09_065917.png)

A2C는 초반 평가에서 baseline을 넘는 구가 많고 이후 비교적 안정적으로 유지된다. REINFORCE는 일부 구에서 크게 개선되지만, 구별 편차가 크고 Final 성능이 Best보다 떨어지는 경우가 많다. 이는 Monte Carlo return 기반 업데이트가 reward 분산에 더 민감하기 때문으로 해석된다.

---

## 6. Seed 안정성 분석

Best/Worst 일부 구에 대해 seed 42, 123, 777을 비교했다. seed는 neural network 초기값, action sampling, 학습 데이터 순서 등 난수 요소를 결정하는 값이다. 같은 알고리즘이라도 seed를 바꾸면 학습 초반의 탐색 경로가 달라질 수 있다.

| Algorithm | Seed 평균 Best Δ | Seed 표준편차 평균 |
| --- | --- | --- |
| A2C | 17.7 | 2.0 |
| REINFORCE | -8.0 | 53.3 |

seed 결과는 A2C와 REINFORCE의 차이를 잘 보여준다. A2C는 seed별 Best Delta 표준편차 평균이 약 2.0으로 매우 작았다. 반면 REINFORCE는 약 53.3으로 컸다. 따라서 이번 문제에서는 A2C가 더 재현성 있는 선택이고, REINFORCE는 성능 가능성은 있지만 seed 반복 평가가 꼭 필요하다.

원본 seed 상세 파일: `docs/rl_seed_sensitivity_a2c_reinforce_2026-06-09_003500.detail.csv`

---

## 7. 토의

첫째, 단순히 강화학습 알고리즘을 적용하는 것만으로는 강한 규칙 기반 baseline을 넘기 어렵다. 본 실험에서 성능 개선이 나타난 핵심은 **1시간 수요예측을 state에 넣고, Top-K 후보 구조로 action space를 줄인 점**이다.

둘째, REINFORCE와 A2C의 차이는 알고리즘 특성과 연결된다. REINFORCE는 episode 전체 reward-to-go를 사용하므로 delayed reward 문제를 직접 다루지만, 그만큼 gradient variance가 크다. A2C는 TD target과 value critic을 사용해 더 자주 보정하므로 이번 환경에서는 더 안정적이었다.

셋째, Best와 Final을 분리해서 보는 것이 중요하다. Best만 보면 특정 시점의 가능성을 볼 수 있지만, Final을 함께 봐야 학습이 끝까지 안정적으로 유지되는지 확인할 수 있다. 이번 결과에서는 A2C가 Best와 Final 모두에서 REINFORCE보다 안정적이었다.

---

## 8. 결론

최종 chronological split 실험에서 **A2C가 REINFORCE보다 평균 성능과 안정성 모두에서 우수했다.** A2C는 25개 구 중 Best 기준 18개 구에서 MostImbalanced baseline을 넘었고, 평균 Best Delta는 +24.9였다. REINFORCE는 일부 구에서 A2C와 비슷하거나 더 좋은 결과를 냈지만 평균적으로는 baseline보다 낮았고 seed 민감도가 컸다.

따라서 본 담당 범위의 최종 결론은 다음과 같다.

1. 따릉이 재배치에서는 state에 미래 수요 정보를 넣고 action 후보를 줄이는 설계가 중요하다.
2. REINFORCE는 수업 프로젝트 관점에서 policy gradient 기본 구조를 설명하기 좋지만, 성능 안정성은 낮다.
3. A2C는 TD 기반 critic 덕분에 같은 환경에서 더 안정적으로 baseline을 넘었다.
4. 최종 제출에서는 A2C를 주 모델, REINFORCE를 비교 모델로 제시하는 것이 가장 설득력 있다.

---

## Appendix A. 전체 결과 CSV

상세 수치는 `docs/chronological_a2c_reinforce_comparison_current.csv`에 저장했다.

## Appendix B. 재현 명령

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.run_a2c_reinforce_interactive
```

메뉴에서 `4. 최종 chronological 전체 실험 실행`을 선택하면 A2C와 REINFORCE의 서울 25개 구 실험 및 seed 반복 실험을 순차적으로 실행한다.
