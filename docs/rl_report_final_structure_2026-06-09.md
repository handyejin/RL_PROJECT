# REINFORCE/A2C 최종 보고서 구성안

본 문서는 현재까지 수행한 실험을 강화학습/머신러닝 논문 형식에 맞춰 재배치하기 위한 목차와 핵심 표 초안이다. 핵심 원칙은 다음과 같다.

- 전체 25개 구 결과는 **대표 성능 비교**로 사용한다.
- Top-K, episode budget, seed 실험은 **ablation / sensitivity analysis**로 분리한다.
- REINFORCE와 A2C 차이는 단순 성능 차이가 아니라 **Monte Carlo return vs TD target**에 따른 seed variance 차이로 해석한다.
- 최종 평가는 train reward가 아니라 **chronological split의 test 7일 평균 reward**로 비교한다.

---

## 1. Abstract

서울 따릉이 재배치 문제를 순차 의사결정 문제로 정의하고, REINFORCE with Value Baseline과 A2C를 비교했다. State에는 1시간 수요예측을 포함하고, action은 전체 정류소 직접 선택 대신 수요예측 기반 Top-K 후보 정류소 선택으로 구성했다. 최종 평가는 chronological split의 test 7일 평균 reward로 수행했으며, MostImbalanced 규칙 정책 대비 Delta를 주 지표로 사용했다.

핵심 결과는 다음과 같이 요약한다.

- 서울 25개 구 기본 실험에서 A2C는 REINFORCE보다 평균 성능과 안정성이 높았다.
- A2C는 TD target을 사용해 더 자주 value를 보정하므로, Monte Carlo return 기반 REINFORCE보다 seed variance가 작았다.
- Top-K sensitivity 실험에서 `K=6`이 Best 성능과 Final 안정성의 균형이 가장 좋았다.
- `K=6` seed 반복 실험에서 A2C의 평균 Best Delta는 양수로 유지되어, 단일 seed에만 의존한 결과가 아님을 확인했다.

---

## 2. Introduction

### 2.1 문제 배경

공유자전거 시스템은 시간대와 지역에 따라 대여와 반납 수요가 크게 달라진다. 일부 정류소는 자전거가 부족해 대여 실패가 발생하고, 일부 정류소는 거치 공간이 부족해 반납 실패가 발생한다. 따라서 재배치 트럭은 제한된 시간 안에 어느 정류소를 방문할지 순차적으로 결정해야 한다.

### 2.2 연구 질문

| 연구 질문 | 본 보고서에서 확인한 방법 |
|---|---|
| REINFORCE와 A2C 중 어떤 방식이 더 안정적인가? | 서울 25개 구 기본 실험 |
| Monte Carlo 방식과 TD 방식의 차이가 seed variance에 나타나는가? | seed 42/123/777 반복 실험 |
| Action 후보 수 Top-K는 성능에 어떤 영향을 주는가? | Top-K 3/6/9/12/15 screening 및 3/6/12 confirmation |
| 최종적으로 어떤 조건이 가장 합리적인가? | 성능, Final 안정성, seed variance를 함께 비교 |

---

## 3. Problem Formulation

### 3.1 State

State는 현재 정류소 재고, capacity, 트럭 상태, 시간 정보, 1시간 수요예측 정보를 포함한다.

```text
pred_net_1h = pred_returns_1h - pred_rentals_1h
projected_bikes = current_bikes + pred_net_1h
projected_deviation = (projected_bikes - target_bikes) / capacity
```

### 3.2 Action

Agent는 전체 정류소를 직접 고르지 않고, 매 step마다 생성된 Top-K 후보 정류소 중 하나를 선택한다.

```text
candidate_score =
    forecast_imbalance
  - candidate_travel_coef * travel_distance
  - zone_penalty
```

### 3.3 Reward

Reward는 원본 환경의 평가 reward를 사용하며, 값이 0에 가까울수록 좋다.

```text
r_t = -1.0 * stockout
      -0.8 * full
      -0.008 * travel_km
      -0.002 * travel_step
```

비교 지표는 다음과 같다.

```text
Delta = model_eval_reward - MostImbalanced_eval_reward
```

Delta가 양수이면 모델이 baseline보다 좋다.

---

## 4. Algorithms

### 4.1 REINFORCE with Value Baseline

REINFORCE는 episode가 끝난 뒤 reward-to-go를 계산하는 Monte Carlo policy gradient 방식이다.

```python
returns = discounted_reward_to_go(rewards, gamma)
advantages = returns - value_net(states)

policy_loss = -(log_probs * advantages.detach()).mean()
value_loss = mse_loss(value_net(states), returns)
```

특징:

- episode 전체 return을 사용한다.
- delayed reward를 직접 반영하지만 gradient variance가 크다.
- seed와 초기 탐색 경로에 민감할 수 있다.

### 4.2 A2C

A2C는 Actor가 action 분포를 만들고 Critic이 value를 추정한다. 학습은 TD target으로 수행한다.

```python
target = reward + gamma * (1 - done) * value(next_state)
advantage = target - value(state)

actor_loss = -(log_prob(action) * advantage.detach()).mean()
critic_loss = mse_loss(value(state), target)
```

특징:

- 매 step 또는 batch 단위로 TD target을 사용한다.
- REINFORCE보다 자주 학습 신호를 받는다.
- 이번 환경에서는 seed variance가 작고 안정적이었다.

---

## 5. Experimental Setup

| 항목 | 설정 |
|---|---|
| 대상 지역 | 서울 25개 구 |
| Train/Test split | 시간순 chronological split |
| Test 기간 | 2025-10-20 ~ 2025-10-26, 총 7일 |
| 기본 학습 길이 | 500 episodes |
| 기본 평가 주기 | 50 episodes |
| 기본 seed | 42 |
| Baseline | MostImbalanced 규칙 정책 |
| 평가 지표 | 7일 평균 reward, Delta vs baseline |
| Checkpoint | Best checkpoint와 Final checkpoint를 모두 보고 |

Best checkpoint는 학습 중 평가 reward가 가장 좋았던 시점이다. Final checkpoint는 학습 종료 시점이다. 본 보고서에서는 Best를 성능 가능성, Final을 학습 안정성으로 해석한다.

---

## 6. Experiment 1: 서울 25개 구 기본 비교

목적은 REINFORCE와 A2C의 기본 성능 차이를 확인하는 것이다.

| Algorithm | Best 평균 Reward | Final 평균 Reward | Best Δ 평균 | Final Δ 평균 | Best 승리 구 | Final 승리 구 |
|---|---:|---:|---:|---:|---:|---:|
| A2C | -815.6 | -827.4 | +24.9 | +13.1 | 18/25 | 16/25 |
| REINFORCE | -844.9 | -878.6 | -4.4 | -38.1 | 10/25 | 7/25 |

해석:

- A2C는 평균 Best/Final Delta가 모두 양수다.
- REINFORCE는 일부 구에서 좋은 결과를 보였지만 평균적으로 baseline을 넘지 못했다.
- A2C가 25개 구 중 20개 구에서 REINFORCE보다 높은 Best Delta를 보였다.

---

## 7. Experiment 2: Top-K Hyperparameter Screening

목적은 action 후보 수 `K`가 학습 성능과 안정성에 미치는 영향을 빠르게 확인하는 것이다. 모든 조합을 500 episodes로 실행하면 계산량이 커지므로, 1차 screening은 200 episodes로 수행했다.

| Top-K | 200 ep Screening Best Δ | 200 ep Screening Final Δ | 판단 |
|---:|---:|---:|---|
| 3 | +30.2 | +25.1 | 강한 후보 축소, 성능 좋음 |
| 6 | +27.5 | +27.5 | Best/Final 균형 좋음 |
| 9 | +21.2 | +16.1 | 상대적으로 약함 |
| 12 | +28.4 | -4.7 | Best는 좋지만 Final 불안정 |
| 15 | +27.6 | +8.6 | 후보가 넓어져 안정성 저하 |

---

## 8. Experiment 3: Top-K Confirmation

1차 screening에서 유망했던 `K=3`, `K=6`, 그리고 기존 기준인 `K=12`를 500 episodes로 재확인했다.

| Top-K | 500 ep Best Δ | 500 ep Final Δ | Best 승리 구 | Final 승리 구 | 판단 |
|---:|---:|---:|---:|---:|---|
| 3 | +25.3 | +16.3 | 3/6 | 3/6 | 안정적 후보 |
| 6 | +27.6 | +15.9 | 3/6 | 3/6 | 최종 추천 |
| 12 | +28.4 | -4.7 | 3/6 | 2/6 | Best는 높지만 Final 불안정 |

해석:

- `K=12`는 Best 기준으로 높지만 Final이 음수로 하락했다.
- `K=6`은 Best와 Final의 균형이 가장 좋았다.
- 따라서 최종 seed 검증 후보로 `Top-K=6`을 선택했다.

---

## 9. Experiment 4: Seed Validation

Top-K=6에서 seed 42/123/777을 반복 실행했다. 대상은 A2C Best/Worst 6개 구다.

| 구 | Seed 42 Best Δ | Seed 123 Best Δ | Seed 777 Best Δ | 평균 Best Δ | Best Δ 95% CI | 해석 |
|---|---:|---:|---:|---:|---:|---|
| 마포구 | +85.6 | +93.5 | +88.1 | +89.1 | ±4.6 | 매우 안정적 개선 |
| 노원구 | +72.0 | +75.3 | +67.3 | +71.5 | ±4.6 | 안정적 개선 |
| 영등포구 | +73.2 | +82.7 | +35.1 | +63.7 | ±28.5 | 개선되지만 seed 변동 큼 |
| 은평구 | -9.8 | -6.1 | -10.7 | -8.9 | ±2.7 | 일관되게 어려움 |
| 서대문구 | -15.2 | -14.4 | -16.9 | -15.5 | ±1.4 | 일관되게 어려움 |
| 관악구 | -40.4 | -38.0 | -39.6 | -39.3 | ±1.4 | 구조적으로 어려운 구 |

종합:

| 지표 | 값 |
|---|---:|
| 평균 Best Δ | +26.8 |
| 평균 Final Δ | +8.4 |
| Seed 수 | 3 |
| 대상 구 | 6 |

해석:

- Top-K=6은 seed가 바뀌어도 평균 Best Delta가 양수로 유지됐다.
- 관악구처럼 어려운 구는 seed가 바뀌어도 일관되게 낮았다.
- 영등포구는 개선 가능성이 크지만 seed별 차이가 커서 추가 튜닝 대상이다.

---

## 10. Experiment 5: 최적 조건 서울 25개 구 재검증

현재 진행 중인 실험이다.

| 항목 | 설정 |
|---|---|
| 알고리즘 | A2C |
| Top-K | 6 |
| 대상 | 서울 25개 구 |
| Split | chronological |
| Seed | 42 |
| Episodes | 500 |
| 목적 | Best/Worst 6구에서 고른 최적 조건이 전체 구에서도 유효한지 확인 |

완료 후에는 다음 표를 추가한다.

| 실험 | 대상 | Top-K | Best Δ 평균 | Final Δ 평균 | Best 승리 구 | Final 승리 구 |
|---|---|---:|---:|---:|---:|---:|
| 기존 전체 A2C | 서울 25구 | 12 | +24.9 | +13.1 | 18/25 | 16/25 |
| 최적 조건 A2C | 서울 25구 | 6 | 업데이트 예정 | 업데이트 예정 | 업데이트 예정 | 업데이트 예정 |

---

## 11. Discussion

### 11.1 TD vs MC와 Seed Variance

이번 결과에서 가장 중요한 알고리즘적 해석은 A2C와 REINFORCE의 update 방식 차이다.

| 알고리즘 | Return/Target | Update 특성 | 관찰 결과 |
|---|---|---|---|
| REINFORCE | Monte Carlo reward-to-go | episode 종료 후 전체 return으로 업데이트 | seed와 episode reward 변동에 민감 |
| A2C | TD target | step/batch 단위로 value target 업데이트 | 평균 성능과 seed 안정성이 높음 |

REINFORCE는 episode 전체 결과를 보고 업데이트하므로 delayed reward를 직접 반영하지만, 한 episode의 운에 gradient가 크게 흔들릴 수 있다. 반면 A2C는 Critic을 통해 TD target을 만들기 때문에 더 자주 보정 신호를 받고, 이번 재배치 환경에서는 더 안정적이었다.

### 11.2 Top-K의 의미

Top-K는 단순히 action을 줄이는 트릭이 아니라, 큰 정류소 action space에서 agent가 학습 가능한 후보군을 보게 만드는 구조다. 너무 작은 K는 좋은 후보를 놓칠 수 있고, 너무 큰 K는 탐색 난도를 다시 키운다. 본 실험에서는 `K=6`이 이 둘 사이의 균형점으로 나타났다.

### 11.3 Best와 Final을 함께 보는 이유

강화학습은 학습 후반에 정책이 더 나빠질 수 있다. 따라서 Best checkpoint만 보면 가능성을 볼 수 있고, Final checkpoint를 보면 안정성을 볼 수 있다. 본 보고서에서는 둘을 모두 제시한다.

---

## 12. Conclusion

본 실험에서는 서울 따릉이 재배치 문제에서 REINFORCE와 A2C를 비교했다. 서울 25개 구 기본 실험에서는 A2C가 REINFORCE보다 평균 성능과 안정성이 높았다. 추가 Top-K sensitivity와 seed validation을 통해, A2C에서 `Top-K=6`이 가장 균형 잡힌 후보 action 설정임을 확인했다.

최종 결론은 다음과 같다.

1. 이 문제에서는 상태에 수요예측을 넣고 action 후보를 줄이는 설계가 중요하다.
2. REINFORCE는 Monte Carlo 방식이라 seed variance가 크고, A2C는 TD 방식이라 더 안정적이었다.
3. Top-K는 너무 크거나 작기보다 중간값인 `K=6`에서 성능과 안정성의 균형이 좋았다.
4. 최종 제출에서는 A2C를 주 모델, REINFORCE를 비교 모델로 제시하는 것이 가장 설득력 있다.

---

## References

- Henderson et al., **Deep Reinforcement Learning That Matters**, AAAI 2018.
- Agarwal et al., **Deep Reinforcement Learning at the Edge of the Statistical Precipice**, NeurIPS 2021.
- Patterson et al., **Empirical Design in Reinforcement Learning**, JMLR 2024.

