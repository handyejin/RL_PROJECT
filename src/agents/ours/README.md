# 사용자 RL Agent 구조

이 폴더는 따릉이 재배치 실험에서 추가 구현한 agent 코드만 모아둔 위치다.
공통 환경과 기존 DQN 파일은 수정하지 않고, 우리 실험용 wrapper와 실행기만 둔다.

## 현재 유지하는 구조

| 경로 | 용도 |
|---|---|
| `common/` | REINFORCE, A2C, DQN, PPO 학습 core와 공통 helper |
| `run_from_config.py` | `config/ours/*.yaml`을 읽어 실험을 실행하는 권장 실행기 |
| `run_interactive.py` | 터미널에서 알고리즘/구를 선택하는 보조 실행기 |
| `export_replay.py` | 학습된 policy의 episode replay JSON 생성 |

예전 비교용 wrapper인 `experiments/`, `original_state/`, `modified_state/`는
YAML 실행기로 대체했다. 실험 설정은 코드 파일을 새로 만들지 않고
`config/ours/*.yaml`에서 관리한다.

## 권장 실행 방식

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml
```

다른 구나 Top-K 값은 CLI에서 바로 바꿀 수 있다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.ours.run_from_config \
  --config config/ours/dqn_topk3.yaml \
  --district 영등포구 \
  --candidate-top-k 4
```

## 주요 YAML

| 파일 | 알고리즘 | 설명 |
|---|---|---|
| `config/ours/reinforce_topk12.yaml` | REINFORCE | Reward-to-Go + Value baseline |
| `config/ours/a2c_topk12.yaml` | A2C | 1-step TD actor-critic |
| `config/ours/dqn_topk3.yaml` | DQN | 후보 action을 3개로 줄인 DQN 실험 |
| `config/ours/dqn_topk12.yaml` | DQN | 기존 Top-K 12 비교용 |
| `config/ours/ppo_topk12.yaml` | PPO | MaskablePPO 보수적 update 설정 |

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

DQN은 action mask를 적용한 Q-network를 학습하며, Double DQN target을 사용한다.

```text
a* = argmax_a Q_online(s', a)
target = r + gamma * Q_target(s', a*)
loss = Huber(Q(s, a), target)
```

PPO는 MaskablePPO를 사용해 action mask와 clipped objective를 함께 적용한다.

```text
ratio = pi_new(a | s) / pi_old(a | s)
loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
```
