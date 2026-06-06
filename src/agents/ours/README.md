# 사용자 RL Agent 구조

이 폴더는 따릉이 재배치 실험에서 추가 구현한 agent 코드만 모아둔 위치다.
공통 환경과 기존 DQN 파일은 그대로 두고, 동일한 평가 방식으로 여러 알고리즘을 비교할 수 있게 구성했다.

## 폴더 구성

| 폴더 | 용도 |
|---|---|
| `common/` | REINFORCE, A2C, DQN, PPO의 공통 학습 루프와 helper |
| `original_state/` | 원본 observation만 사용하는 비교 실험 |
| `modified_state/` | capacity와 1시간 수요예측 feature를 추가한 실험 |
| `experiments/` | 최종 보고서에 사용한 Top-K, 안정화, PBRS 추가 실험 |

## 핵심 비교 세트

| 세트 | State | 보조 학습 | 목적 |
|---|---|---|---|
| `original_state/pure` | 원본 state | 없음 | 팀원 환경 기준 순수 RL 성능 확인 |
| `original_state/guarded` | 원본 state | BC/rollback 등 | 같은 보호장치를 줬을 때 알고리즘 비교 |
| `modified_state/pure` | capacity + forecast state | 없음 | state 보강만으로 RL이 개선되는지 확인 |
| `modified_state/guarded` | capacity + forecast state | BC/rollback 등 | 좋은 초기 policy를 보존하며 fine-tuning |

## 최종 보고서 실험 파일

| 실행 파일 | 알고리즘 | BC | 설명 |
|---|---|---:|---|
| `experiments.reinforce_topk_forecast_pure` | REINFORCE | 없음 | 수요예측 기반 Top-K 후보 action을 사용한 순수 policy gradient |
| `experiments.a2c_topk_forecast_plus` | A2C | 없음 | Top-K 후보, 이동거리 penalty, 권역 penalty를 함께 사용 |
| `experiments.dqn_topk_forecast_plus_stable_no_bc` | DQN | 없음 | 낮은 탐색률과 n-step으로 안정화한 DQN |
| `experiments.ppo_topk_forecast_plus_conservative_no_bc` | PPO | 없음 | 작은 clip/KL 제한으로 보수적으로 update하는 PPO |
| `experiments.reinforce_topk_forecast_plus` | REINFORCE | 있음 | BC 이후 REINFORCE fine-tuning 효과 확인 |
| `experiments.dqn_topk_forecast_plus_stable_bc` | DQN | 있음 | BC policy 유지 여부 확인 |
| `experiments.ppo_topk_forecast_plus_conservative_bc` | PPO | 있음 | BC 이후 PPO fine-tuning 효과 확인 |
| `experiments.dqn_topk_forecast_plus_pbrs_no_bc` | DQN | 없음 | PBRS shaping 추가 효과 확인 |
| `experiments.ppo_topk_forecast_plus_pbrs_no_bc` | PPO | 없음 | PBRS shaping 추가 효과 확인 |

## 주요 알고리즘 식

REINFORCE는 episode 전체 reward-to-go를 사용한다.

```text
G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
A_t = G_t - V(s_t)
policy_loss = -log pi(a_t | s_t) * A_t
value_loss = MSE(V(s_t), G_t)
```

A2C는 1-step TD target으로 actor와 critic을 함께 학습한다.

```text
target = r + gamma * (1 - done) * V(s')
advantage = target - V(s)
actor_loss = -log pi(a | s) * advantage
critic_loss = MSE(V(s), target)
```

DQN은 action mask를 적용한 Q-network를 학습하며, 옵션으로 Double DQN target을 사용한다.

```text
target = r + gamma * max_a Q_target(s', a)
loss = Huber(Q(s, a), target)
```

PPO는 MaskablePPO를 사용해 action mask와 clipped objective를 함께 적용한다.

```text
ratio = pi_new(a | s) / pi_old(a | s)
loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
```
