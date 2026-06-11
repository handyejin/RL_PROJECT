# 박제영 담당 RL Agent 구조

이 폴더는 따릉이 재배치 실험에서 추가 구현한 agent 코드와 실행 wrapper를 모아둔 위치다.
박제영 담당 보고서의 핵심 알고리즘은 **REINFORCE**, **A2C**, **VAE latent 보조 feature**,
**Contextual Bandit(LinUCB) 비교 실험**이다.

## 최종 실험 기준

최종 보고서 기준은 서울 25개 구를 각각 독립 환경으로 보고 학습한 73일 chronological holdout 실험이다.
평가는 `MostImbalanced` 휴리스틱 대비 `Delta = model reward - baseline reward`로 비교한다.

| 구분 | 최종 기준 | 비고 |
|---|---|---|
| REINFORCE/A2C | Top-K=9 | sequential screening 후 선택한 최종 후보 수 |
| PPO | Top-K=3 | PPO clipping/seed 진단까지 포함한 비교 경로 |
| VAE latent | 보조 실험 | 수요 패턴 latent를 state 뒤에 추가해 비교 |
| Contextual Bandit | 보조 baseline | 장기 return 없이 후보 선택만 보는 LinUCB 비교 |
| DQN/QRDQN | 팀 전체 비교용 | 팀원 실험과 연결될 수 있어 보존 |

`Top-K=12` 설정 파일은 초기 기준선 또는 비교 실험용으로 유지한다. 최종 결과를 재현할 때는
interactive runner에서 위 기준값을 선택하는 편이 가장 안전하다.

## 현재 유지하는 구조

| 경로 | 용도 |
|---|---|
| `algorithms/reinforce/core.py` | REINFORCE + Value baseline 학습 core |
| `algorithms/a2c/core.py` | A2C 학습 core |
| `algorithms/bandit/core.py` | Contextual Bandit LinUCB 비교 core |
| `algorithms/dqn/core.py`, `algorithms/ppo/core.py` | 팀 전체 비교용 DQN/PPO 실행 core |
| `common/runner_config.py` | interactive/YAML 실행기가 공유하는 경로, 기본값, 명령 생성 helper |
| `common/candidate_actions.py` | Top-K 후보 action wrapper |
| `common/future_demand.py`, `common/vae_latent.py` | forecast/VAE state feature wrapper |
| `run_from_config.py` | `config/ours/*.yaml`을 읽어 실험을 실행하는 권장 실행기 |
| `run_interactive.py` | 터미널에서 알고리즘/구를 선택하는 보조 실행기 |
| `run_a2c_reinforce_interactive.py` | REINFORCE/A2C 담당 실험 전용 실행기 |
| `export_replay.py` | 학습된 policy의 episode replay JSON 생성 |

예전 비교용 wrapper인 `experiments/`, `original_state/`, `modified_state/`는
YAML 실행기로 대체했다. 실험 설정은 코드 파일을 새로 만들지 않고
`config/ours/*.yaml`에서 관리한다.

## 권장 실행 방식

REINFORCE/A2C/VAE/Bandit 담당 실험은 전용 interactive runner를 권장한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.run_a2c_reinforce_interactive
```

전체 팀 비교용 DQN/PPO까지 한 화면에서 고르는 보조 runner도 유지한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.run_interactive
```

YAML 설정으로 단일 실험을 재현할 수도 있다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.run_from_config \
  --config config/ours/a2c_topk12.yaml
```

다른 구나 Top-K 값은 CLI에서 바로 바꿀 수 있다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m src.agents.run_from_config \
  --config config/ours/a2c_topk12.yaml \
  --district 영등포구 \
  --candidate-top-k 9
```

## 주요 YAML

| 파일 | 알고리즘 | 설명 |
|---|---|---|
| `config/ours/reinforce_topk12.yaml` | REINFORCE | 초기/비교 실험용 Reward-to-Go + Value baseline |
| `config/ours/a2c_topk12.yaml` | A2C | 초기/비교 실험용 1-step TD actor-critic |
| `config/ours/a2c_topk12_vae.yaml` | A2C + VAE | 수요 latent feature 추가 실험 |
| `config/ours/bandit_topk12.yaml` | Bandit | LinUCB 후보 선택 비교 모델 |
| `config/ours/dqn_topk3.yaml`, `config/ours/dqn_topk12.yaml`, `config/ours/ppo_topk12.yaml` | DQN/PPO | 팀 전체 비교용 설정 |

## Replay Viewer

최종 viewer는 `docs/ours_replay_viewer.html`이다. `docs/` 바로 아래에는 발표 녹화용 최신 replay JSON만 둔다.

| 알고리즘 | 기준 | 예시 |
|---|---|---|
| REINFORCE | Top-K=9, best/worst 각 1개 | `송파구_REINFORCE_2025-10-20.json`, `마포구_REINFORCE_2025-10-20.json` |
| A2C | Top-K=9, best/worst 각 1개 | `노원구_A2C_2025-10-20.json`, `관악구_A2C_2025-10-20.json` |
| PPO | Top-K=3, best/worst 각 1개 | `동대문구_PPO_2025-10-20.json`, `송파구_PPO_2025-10-20.json` |

예전 `2025-03-25` replay와 Top-K12 PPO replay는 혼동을 줄이기 위해 보관 대상에서 제외했다.

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
target = r + gamma * (1 - done) * Q_target(s', a*)
loss = Huber(Q(s, a), target)
```

PPO는 MaskablePPO를 사용해 action mask와 clipped objective를 함께 적용한다.

```text
ratio = pi_new(a | s) / pi_old(a | s)
L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
loss = -L_clip + value_loss - entropy_bonus
```

Contextual Bandit은 장기 return을 bootstrap하지 않고, 현재 Top-K 후보의
feature만 보고 LinUCB score가 가장 큰 후보를 고른다.

```text
theta_a = inv(A_a) b_a
score_a = theta_a^T x_a + alpha * sqrt(x_a^T inv(A_a) x_a)
A_a <- A_a + x_a x_a^T
b_a <- b_a + reward * x_a
```
