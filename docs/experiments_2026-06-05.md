# 2026-06-05 실험 로그 — 순수 DQN(BC 없음) 계열로 휴리스틱 추월 시도

> 배경: 지금까지의 추월은 **BC prior + RL fine-tune**으로만 달성됐고, RL fine-tune은 BC prior를 못 넘고 깎기만 했다([experiments_2026-06-03.md](experiments_2026-06-03.md) 결론).
> 이번 질문: **"BC 말고 순수 value-based RL을 많이 학습시켜 성능을 올릴 수 없나?"**
> 휴리스틱 baseline = **-500.02** (deterministic 7일, raw reward). 이 값은 학습량과 무관한 고정 기준선.

## 0. "학습 횟수 늘리기"가 단독으로 안 되는 이유 (기존 증거)

| 증거 | 결과 |
|---|---|
| step3: 순수 2M step | best -598 → 휴리스틱 미달 |
| BC fine-tune 100k | best는 항상 step 5~10k, 이후 -705로 **발산** |
| Vanilla DQN 14 ablation | -657 → -593 plateau |

→ 지금 세팅에선 step을 늘릴수록 **발산**. 병목은 학습량이 아니라 **value 학습 불안정성**.

## 1. 진단 — 기존 에이전트는 "DQN 최소 버전"

`masked_dqn.py`는 SB3 DQN + DDQN + action masking + lr_decay가 전부. 현대 DQN 안정화 장치 대부분 부재:
n-step ❌ / PER ❌ / Dueling ❌ / **Distributional ❌** / NoisyNet ❌ / **reward 정규화 ❌**.
즉 "RL이 BC를 못 넘는다"는 정확히는 **"이 vanilla DQN 세팅이 못 넘는다"**.

## 2. 이번 세션에 추가한 코드

| 항목 | 파일 | 목적 |
|---|---|---|
| **MaskableQRDQN** (분포 학습 + 마스킹) | `src/agents/masked_qrdqn.py` | 분위수 분포로 Q 추정 → 장기 학습 안정성 |
| `--algo qrdqn`, `--n-steps`, `--n-quantiles` | `scripts/train.py` | QR-DQN + n-step credit assignment |
| `--reward-scale` (학습만, 평가 raw) | `scripts/train.py` | TD 타깃 ~700 → ~7 축소 (발산 완화) |
| `--max-grad-norm` (기본 10) | `scripts/train.py` | QRDQN 기본 None → grad clip 복구 |
| `--n-envs` (SubprocVecEnv) | `scripts/train.py` | rollout 병렬 수집 (속도) |
| `--device` (auto/cpu/mps/cuda) | `scripts/train.py` | GPU 지원 (현재 환경 제약 있음 — §5) |
| 의존성 | `sb3-contrib==2.8.0` 설치 | QRDQN 제공 |

환경 설정은 그들의 최적 config(`config/default.yaml`) 그대로 → **알고리즘만 바꾼 순수 비교**, BC pretrain 없음.

---

## 3. Run 1 — 순수 QR-DQN(+n-step3), 안정화 없음 ❌ 발산 재현

명령: `--algo qrdqn --n-steps 3 --timesteps 1000000`  (reward_scale 1.0, grad clip OFF=QRDQN 기본 None)

### 학습 곡선 (휴리스틱 -500.02)

| step | reward | | step | reward |
|---:|---:|---|---:|---:|
| 10k | -704 | | 150k | -707 |
| 30k | -699 | | 200k | -739 |
| 50k | -653 | | 230k | **-769** (최악) |
| 70k | -634 | | 250k | -763 |
| **90k** | **-613 (peak)** | | 300k | -736 |
| 120k | -637 | | 340k | -763 (중단) |

### 결과: peak -613 (step 90k) 후 붕괴 → 휴리스틱·BC prior·vanilla DQN 모두 미달

- 분포 학습이 붕괴를 **늦췄지만(90k까지 버팀) 막지는 못함**. 90k 이후 vanilla DQN과 동일하게 발산.
- peak -613조차 휴리스틱(-500)·BC prior(-506)·vanilla DQN plateau(-593)에 못 미침.

### 발산 원인 2개 특정

1. **grad clip OFF** — QRDQN 기본 `max_grad_norm=None`. 정작 그들 vanilla DQN은 clip=10이 켜져 있었음 → 우리 QR-DQN이 **더 불안정**했던 셈 (구현 실수).
2. **거대 reward 스케일** — 보상 -500~-770 → TD 타깃 거대 → 분포 학습으로도 못 버팀.

---

## 4. Run 2 — 안정화(reward-scale 0.01 + grad clip 10 + n-step3) ✅ 발산 방지 확인

명령:
```
PYTHONUNBUFFERED=1 python -u scripts/train.py --algo qrdqn --tag stable_1M \
  --timesteps 1000000 --eval-freq 10000 --n-train-dates 60 \
  --n-steps 3 --n-envs 1 --reward-scale 0.01 --max-grad-norm 10
```

### 곡선 + Run1 직접 비교 (휴리스틱 -500.02)

| step | **Run2 (안정화)** | Run1 (불안정) | 비고 |
|---:|---:|---:|---|
| 10k | -704 | -704 | 동일 |
| 30k | -637 | -699 | |
| 50k | -656 | -653 | |
| 70k | -644 | -634 | 여기까진 유사 |
| 80k | -698 | -653 | |
| 90k | -670 | **-613 (R1 peak)** | |
| 100k | -736 | -729 | 둘 다 dip |
| **110k** | **-602 (R2 best)** | -721 | **분기 시작** |
| 120k | -627 | -637 | |
| 130k | -674 | -668 | |
| 140k | -656 | **-707** | |
| 150k | -612 | -707 | |
| 200k | (진행중) | -739 | |
| 230k | (진행중) | **-769 (R1 붕괴 최저)** | |

### 판정 (90k~250k 구간): 안정화가 **catastrophic collapse를 막았다** ✅

- **Run1**: 90k에서 peak(-613) 후 단조 붕괴 → 230k에서 **-769**. 회복 없음.
- **Run2**: 같은 구간에서 -602~-736을 **진동하되 매번 -600대로 회복**. 150k에서도 -612로 복귀. Run1처럼 -700대로 고착·붕괴하지 않음.
- 즉 **reward-scale + grad-clip = 발산의 직접 원인 2개를 막아 collapse 제거.** 진단(§3)이 정확했음.

### 그러나 — 추월은 실패, plateau

- best **-602 (Δ -102)**. 휴리스틱(-500)·BC prior(-506)·BC v7(+5.8)에 한참 못 미침.
- 붕괴는 막았지만 **휴리스틱 쪽으로 *올라가지* 못하고 -600대에서 plateau.** 순수 value-based RL의 한계가 기존 기록(-593 plateau)과 일치.

> 결론: "발산을 막느냐"(✅ 성공) 와 "휴리스틱을 넘느냐"(❌ 실패)는 별개 문제였다. 안정화로 전자는 해결했으나 후자는 BC 없이는 여전히 미달.

---

## 5. 부수 발견

### 5.1 병렬 환경(`--n-envs`) — 마이크로벤치 3배, 그러나 60일 startup 병목
- 8000 step 벤치: 단일 0.9분 vs 4-env **0.3분 (3배)**. IPC 우려와 달리 병렬 이득이 큼.
- **하지만** `n_train_dates=60`이면 SubprocVecEnv가 60 episode × 4 워커로 pickle하는 **startup이 수 분** 소요 → 1M run에서 첫 eval 전 정체. 현재는 단일 env로 진행. 개선안: 워커가 episode를 디스크에서 직접 load(closure pickle 회피).

### 5.2 Metal GPU(MPS) — macOS 버전 제약으로 현재 불가
- 하드웨어 Apple M1 Pro(arm64) ✅, torch 2.11 MPS built ✅ — 그러나 **torch 2.11 MPS는 macOS 14.0+ 필요**, 현재 macOS 13(Ventura) → 사용 불가.
- `--device` 옵션은 추가 완료(불가 시 자동 cpu fallback + 사유 출력). macOS 14 업그레이드 또는 torch 다운그레이드 시 즉시 사용 가능.
- 단, `[256,256]` 소형 MLP + CPU-bound env라 MPS를 켜도 속도 이득은 미미할 가능성(SB3도 MLP엔 CPU 권장). 실효 속도 레버는 §5.1 병렬화.

### 5.3 stdout 버퍼링
- 로그를 파일로 redirect하면 Python이 block-buffering → eval 줄이 실시간으로 안 써짐. **`python -u` / `PYTHONUNBUFFERED=1` 필수** (모니터링용).

---

## 6. 결론

1. **순수 QR-DQN + n-step (안정화 전)** → 발산 재현. peak -613(90k) 후 -769 붕괴. 분포 학습만으론 부족.
2. **발산 원인 = grad clip 부재 + 거대 reward 스케일**로 특정.
3. **안정화(reward-scale 0.01 + grad clip 10)** → **collapse 제거 확인** (Run1 -769 vs Run2 -600대 유지). 진단 정확.
4. **그러나 추월은 실패** — best -602(Δ-102), -600대 plateau. 휴리스틱(-500)·BC(+5.8)에 미달.

> 핵심 교훈: **"발산을 막는 것"과 "휴리스틱을 넘는 것"은 다른 문제.** 안정화로 전자는 해결했으나, 순수 value-based RL은 (기존 -593 plateau와 일관되게) 휴리스틱을 못 넘는다 → **추월엔 BC prior가 여전히 필수.** "학습 횟수를 늘려 성능을 올린다"는 가설은, 발산을 먼저 잡아야 성립하지만 잡아도 plateau 때문에 추월까진 못 간다는 것으로 귀결.

## 7. Run 2 후반 + replay 시각화

- Run 2를 1M까지 두니 후반에 천천히 개선 → **best -594.4 (step 350k)**. 붕괴 없이 -594~-648 진동.
- best 모델(-602@110k 시점)을 replay JSON으로 export (`export_replay.py`에 `--algo qrdqn` 추가) → 7일 eval + 휴리스틱(`--algo heuristic` 추가)도 함께 떨궈 `replay_viewer.html`에서 비교 가능.
- **replay 관찰**: QR-DQN은 이동거리 ~680km vs 휴리스틱 ~800~880km → RL이 피크에서 **덜 움직이고 소극적**. unique 방문 78곳 vs 휴 96곳. "오전 ramp 과소대응" 진단과 일치.
- **에피소드 내 reward 곡선이 "훅훅" 깎이는 이유**: 누적 reward의 기울기 = 그 순간 빈/만차 정류소 수. 새벽 평평 → 출근(10~11시)부터 급강하 → 오후·저녁 피크에 가속. 구조적(하루 수요 패턴)이며, 이 모델은 피크 대응이 약해 휴리스틱보다 더 가파름.

## 8. Run 3 — exploration_fraction 낮춤 (0.6 → 0.2) ✅ 요동 해결, 추월은 여전히 실패

동기: 사용자가 "90k~220k 발산"을 지적. 분석 결과 **발산이 아니라 high-ε(0.6 fraction → ε가 600k에 정착) 구간의 요동**. ε가 90k≈0.87, 220k≈0.69로 학습 데이터 70~87%가 무작위 → loss 스파이크(166) + eval -602~-736 진동. 250k 이후 ε 내려가며 안정.

명령: `--exploration-fraction 0.2 --exploration-final-eps 0.05` (그 외 Run2와 동일)

| 구간 | Run3 (ε↓) | Run2 (ε↑) |
|---|---|---|
| 90k~130k 변동폭 | **-662~-683 (21점)** | -602~-736 (134점) |
| best | -596.6 (250k) | -594.4 (350k) |

### 결과
- ✅ **요동 6배 감소** — high-ε 스파이크 제거. "발산"의 정체는 과탐색 노이즈였음을 확정.
- ✅ best는 동등(-596 ≈ -594), ε 정착(200k) 후 개선됨 (-610 → -596).
- ❌ **추월은 여전히 실패** — best ≈ -595 plateau 동일. **exploration은 "안정성"의 레버였지 "추월"의 레버가 아니다.**

> 종합: seed/lr/travel/reward-scale/grad-clip/exploration — **하이퍼파라미터·reward 튜닝으로는 -595 plateau를 못 깬다**(정도껏 안정화·수렴 가속만). reward 가중치도 동일 — 핵심 가중치는 휴리스틱도 재채점되고(무의미) travel은 이미 기각, shaping은 policy-invariant라 천장 못 올림. **plateau는 신호 문제가 아니라 학습 능력(action 표현·이질적 day 일반화) 문제.**

## 9. Run 4 — action 추상화 (146 → 5 의도) 🔄 진행 중

가설(사용자 제기): "action이 너무 많아서(146) 휴리스틱을 못 따라잡는 것". 측정해보니 strict_mask가 이미 매 step ~11개로 좁혀줌(146 아님)이고, RL은 오히려 휴리스틱보다 **덜 다양하게**(78<96) 소수에 고착. → "개수"보다 **"표현(추상화 수준)"**이 레버라 판단.

구현 ([src/envs/abstract_action.py](../src/envs/abstract_action.py)): `gym.ActionWrapper`로 146지선다를 5개 의도로 축소.

| action | 매핑 규칙 (휴리스틱 primitive 재사용) |
|---|---|
| 0 stay | 현재 위치 |
| 1 most_surplus | (bikes-target) 최대 → 수거 |
| 2 most_deficit | (target-bikes) 최대 → 배달 |
| 3 nearest_urgent | 가장 가까운 위급(빈/꽉찬) 정류소 |
| 4 most_imbalanced | \|bikes-target\| 최대 |

모두 자기 위치+다른 트럭 목적지 제외. `--abstract-actions` 플래그로 train/eval env 래핑. eval 콜백은 `_episodes`는 본체에서, `action_masks`/step은 최외곽(wrapper)에서 호출하도록 수정.

**왜 plateau를 깰 가능성**: ① 탐색 146→5로 단순화 ② **도달 가능 집합에 휴리스틱 포함** — "항상 most_imbalanced"면 휴리스틱 재현이므로 최소 -500 학습 가능해야 정상 ③ 상황별 의도 시퀀싱으로 단일 고정 규칙 능가 가능.
**한계**: RL 천장이 "5 primitive의 최선 조합"으로 제한 — 어휘에 없는 행동 불가, 최악엔 휴리스틱 재현에 그칠 수 있음.

명령:
```
python -u scripts/train.py --algo qrdqn --tag abstract_1M --abstract-actions \
  --timesteps 1000000 --eval-freq 10000 --n-train-dates 60 --n-steps 3 \
  --reward-scale 0.01 --max-grad-norm 10 \
  --exploration-fraction 0.3 --exploration-final-eps 0.05
```

### 결과 (1M 완주) ❌ plateau 동일 — 추월 실패

| 지표 | 값 | 비고 |
|---|---:|---|
| best | **-553.27** (step 780k) | 기존 plateau(-553~-595)와 동급 |
| 마지막 | -570.43 (1M) | |
| 휴리스틱 | -500.02 | Δ best = **-53.3** |

- ε 정착(300k) 후 -82k 구간 spike 사라지고 **-553~-620 밴드로 수렴 → 발산 아님**(예상대로 high-ε 진동이었음).
- 그러나 **best -553 = 기존 -595 plateau와 사실상 동급.** 추상화가 천장을 못 올림.
- **핵심 반증**: 도달 집합에 휴리스틱(`most_imbalanced` 고정 = action 4)이 있는데도 RL은 **-500(휴리스틱 재현)조차 못 찾음.** → action **개수(146)가 병목이 아님**(§8 진단 재확인). plateau는 action 표현 문제가 아니라 **value-based RL이 BC 없이 -500선을 못 배우는 학습 능력 한계.**

> 결론: 추상화(어휘 축소)는 발산만 줄일 뿐 추월 레버가 아니다. **추월엔 BC(demo) 신호가 필수** → §10대로 DQfD 트랙으로 전환.

## 9.5 Run 5 — DQfD (demo anchor + large-margin) 🔄 진행 중

동기: BC fine-tune의 고질병 = best가 step 5~10k(BC 직후)에 나오고 이후 **forgetting으로 하락**([experiments_2026-06-03.md](experiments_2026-06-03.md) Step0/1 — seed·LR 무관 -506.57 천장 확정). 원인은 **목적함수 불일치**: BC는 q_net을 CE-logit(크기 무의미)으로 학습하는데, DQN이 시작되면 Bellman target(누적 -500~-700 규모)과의 거대 오차가 BC argmax를 덮어씀. LR 낮춤은 *속도*만 줄임. → 하이퍼파라미터가 아니라 **목적함수를 고쳐야** 함.

구현 ([src/agents/dqfd.py](../src/agents/dqfd.py), Hester et al. 2018):
1. **demo buffer 상주** — 휴리스틱 full-transition (s,a,r,s',done,mask)을 학습 내내 보존
2. **large-margin loss** `J_E = mean(max_a[Q(s,a)+ℓ]−Q(s,a_E))` — demo 행동 Q를 margin만큼 1등으로 강제 (forgetting 직격)
3. **pre-training** — 환경 전 demo만으로 K=20k step TD 학습 → Q를 실제 return 규모로 보정(logit 크기 불일치 제거)
4. **본학습** = agent TD + demo TD + λ·margin + λ_l2·L2

명령:
```
python -u scripts/train.py --algo dqfd --tag dqfd_1M \
  --timesteps 1000000 --eval-freq 10000 --n-train-dates 60 \
  --dqfd-pretrain-steps 20000 --reward-scale 0.01 --max-grad-norm 10 \
  --exploration-fraction 0.1 --exploration-final-eps 0.02
```

**관전 포인트**: ① pretrain 직후 첫 eval이 -500 근처인가(Q 보정 성공) ② 이후 **forgetting 없이 유지·개선**되는가(margin loss 효과 = 본 실험의 본질) ③ best가 -500 초과 시 순수 RL 트랙 첫 추월.

### 구현 중 버그 2개 (수정 완료)

1. **margin loss 마스킹 → 발산.** 처음엔 margin 내부 max에 action mask를 적용했는데, 휴리스틱이 stay-폴백(자기 위치=마스킹)을 고른 샘플에서 demo 행동이 max에서 빠져 `J_E`가 음수로 폭발(td 169k, margin -1M). **표준 DQfD대로 마스킹 제거** → `J_E ≥ 0` 하한 복구.
2. **margin(hinge)만으론 모방 신호 약함.** hinge는 샘플당 max 행동 1개에만 gradient → 146지선다 모방에 BC의 CE보다 훨씬 약함. margin loss가 0.8에 고착, eval -607~-634(plateau 동급). 또 stale BC 모델(`bc_v6b`)을 init으로 쓰니 **config drift**로 -605(예전 config 모델).

→ **해결: DQfD에 CE(BC) 항 내장** (`_bc_loss`, λ_bc). Q를 logit으로 CrossEntropy 모방 → self-contained(외부 BC·config 정합성 의존 제거). loss = TD(agent+demo) + λ_m·margin + λ_bc·CE + λ_l2·L2.

### 결과 (from-scratch DQfD+CE, 100k) — ✅ forgetting 해결, ❌ 추월 미달

| | plain BC fine-tune (06-03) | **DQfD+CE (Run5)** |
|---|---|---|
| prior/pretrain 직후 | -506 (best, step 5~10k) | **-530.8** (step 5~20k) |
| 100k 시점 | **-705 붕괴** | **-554** (유지) |
| 전체 추이 | best 직후 단조 하락 | **-558~-529 좁은 밴드, 붕괴 없음** |
| best | -506.6 | **-529.45** (step 80k, Δ -29.4) |

- ✅ **forgetting 차단 성공 = 본 실험의 핵심 목표 달성.** demo buffer 상주 + margin + CE 앵커가 plain fine-tune의 -705 붕괴를 제거, 100k 내내 -530 근처 유지. "BC 직후만 최고, 이후 하락" 문제 해소.
- ❌ **추월은 실패** — best -529, 휴리스틱(-500) 바로 아래 plateau. 앵커(λ_bc=1, margin=1)가 강해 **모방 수준에 고정**, RL이 그 위로 못 밀어냄. CE도 4.0에서 정체(모방 약함)라 prior가 BC(-506)보다 낮은 -530.

> 결론: DQfD는 *"BC를 안 까먹게"*는 확실히 해냈다(✅). 그러나 *"BC 위로 추월"*은 강한 앵커 때문에 아직(❌). 추월의 다음 레버:
> 1. **앵커 annealing** — λ_bc·λ_margin을 후반에 감쇠 → RL이 휴리스틱 위로 개선 (추월 직접 레버)
> 2. **모방 강화** — pretrain 길게/CE 튜닝으로 prior를 -506까지 끌어 출발선 개선
> 3. **demo n-step returns** — credit assignment

## 9.6 Run 6 — DQfD + 앵커 annealing (λ 1.0→0.1) 1M

동기: Run5에서 강한 앵커(λ_bc=λ_margin=1)가 모방 수준(-529)에 고정 → 추월을 막음. 가설: 본 학습 동안 λ를 **선형 감쇠**하면 전반엔 forgetting 차단, 후반엔 RL이 휴리스틱 위로 개선.

구현: `DQfDDQN`에 `_eff_lambda`(progress 기반 선형 anneal) 추가, `--lambda-bc-final --lambda-margin-final`. pretrain은 풀 강도, 본 학습만 anneal.

명령:
```
python -u scripts/train.py --algo dqfd --tag dqfd_ce_anneal_1M \
  --timesteps 1000000 --eval-freq 10000 --n-train-dates 60 \
  --dqfd-pretrain-steps 20000 --dqfd-pretrain-lr 1e-3 \
  --reward-scale 0.01 --max-grad-norm 10 \
  --exploration-fraction 0.1 --exploration-final-eps 0.02 \
  --lambda-bc-final 0.1 --lambda-margin-final 0.1
```

### 결과 — 앵커 "스위트 스팟" 발견, best -515 (역대 RL 최고), 추월은 미달 ❌

| 앵커 λ | 구간 | reward | 해석 |
|---|---|---:|---|
| ~1.0 | pretrain~400k | -530 plateau | 모방 고정(over-constrained) |
| **~0.5** | **540k** | **-515.03 (best, Δ-15)** | **앵커+RL 시너지 최적점** |
| ~0.1~0.2 | 700k~1M | -522~-549 | 앵커 과도하게 풀림 → 순수 RL 노이즈 회귀 |

- best **-515.03 (540k)**, 마지막 -526.20. **순수 RL 트랙 역대 최고**(100k 고정앵커 -529, 추상화 -553, QRDQN -594 모두 능가).
- annealing 가설 **부분 적중**: λ=1은 과했고(약화하니 -515까지↑), 그러나 **λ를 더 풀자(→0.1) 오히려 악화** → **순수 RL은 여전히 휴리스틱을 못 이기고, 앵커가 성능을 떠받친다**는 결정적 증거.
- 추월(-500) 미달. 게다가 **BC(-506.6)도 못 넘음** — 원인은 **약한 모방**: DQfD의 CE가 4.0에서 정체(BC는 더 낮은 CE로 -506 달성) → prior가 -530(BC -506보다 낮음). 출발선이 낮아 천장도 낮음.

> 종합(Run4~6): **forgetting은 DQfD로 확실히 해결**(✅, plain fine-tune -705 붕괴 → DQfD -515~-530 안정). 그러나 **추월은 여전히 미달**(❌). 핵심 병목이 "forgetting"에서 **"약한 모방(prior가 BC -506에 못 미침)"**으로 이동. 순수 RL 단독은 어떤 트랙(추상화/QRDQN/anneal)에서도 휴리스틱을 못 넘음 — 성능은 demo 앵커가 떠받치고, 앵커를 풀면 무너짐.

## 10′. 다음 후보 (갱신)

1. **모방 강화 = 최우선 레버.** CE가 4.0 정체 → prior -530. (a) pretrain demo batch ↑(64→256, BC와 동일) (b) 순수 CE warmup 단계 분리 (c) **현재 config로 BC 재학습**(`pretrain_bc.py`) 후 그 가중치를 DQfD `--pretrain` init으로 + 앵커 anneal. 출발선을 -506까지 끌면 anneal sweet-spot이 -500 근처에 닿을 가능성.
2. demo n-step returns (credit assignment).
3. 그래도 미달 시 **PPO + KL-to-BC** 트랙(value-RL 부트스트랩 자체를 회피).

## 9.7 Run 7 — 모방 강화(강한 BC) + DQfD offline pretrain → ✅ 휴리스틱 첫 추월

동기: Run6에서 병목이 "약한 모방"(prior -530)으로 드러남. 현재 config로 BC를 강하게 재학습 후 DQfD init.

### 강한 BC 재학습 (`bc_strong`)
`pretrain_bc.py --n-dates 292 --epochs 150 --lr 3e-3 --lr-schedule cosine --batch-size 256`
→ **best acc 28.7%** (과거 v6b 20.2%보다 높음). 데이터(292일)+epochs+cosine lr이 모방 정확도를 끌어올림.

### 핵심 발견 — 추월은 "강한 BC + DQfD offline pretrain"에서 나온다 (online RL 아님)

`--algo dqfd --pretrain logs/bc_strong/bc_model.zip --dqfd-pretrain-steps K --learning-starts 100000`(=online RL OFF)로 K 스윕:

| DQfD pretrain K | 0(순수BC) | 4k | 6k | **8k** | **10k** | 12k | 20k |
|---|---:|---:|---:|---:|---:|---:|---:|
| eval reward | -501.97 | -508.0 | -509.0 | **-499.18** | **-499.14** | -504.5 | -518.6 |
| Δ(휴 -500.02) | -1.95 | -8.0 | -9.0 | **+0.84 ✅** | **+0.88 ✅** | -4.5 | -18.6 |

- **K=8~10k에서 재현성 있게 휴리스틱 추월** (Δ≈+0.85). 순수 BC(-502)를 DQfD pretrain의 TD 보정+margin 샤프닝이 -500 위로 살짝 밀어올림. 단 K가 과하면(12k+) BC argmax 훼손으로 다시 하락 → **명확한 sweet spot**.
- 추월 폭(+0.85)은 휴리스틱 baseline 자체 noise(±1, §06-03 5.x)와 비슷한 수준이라 **"동률~근소 추월"**이 정직한 표현. 그러나 기존 BC(-506.6, Δ-6.6)·모든 RL 트랙(-515~-594)을 **확실히 능가**.

### online RL은 추월을 오염시킨다 (재확인)
`bc_strong`을 init으로 online RL을 켜면(ε 1.0이든 0.05든 무관) eval[1] -499 → 30k부터 **-570대로 하락**. td_agent(RL 부트스트랩)가 prior(-499)를 자기 가치추정(~-570)으로 끌어내림. 앵커(λ=1)로도 못 막음. **value-RL은 좋은 prior를 개선 못 하고 깎기만 한다**는 본 프로젝트 일관된 결론 재확인.

### 결론
- ✅ **휴리스틱 첫 추월 달성**: 강한 BC(28.7%) + DQfD offline pretrain(8~10k) → **-499.1 (Δ+0.85)**. 모델: `logs/dqfd_dqfd_strongbc_overtake/best/best_model.zip`.
- **레버는 "모방 강화"였다** — RL이 아니라. 정확도 20%→28.7%가 -506→-499로 격차를 메움.
- online RL은 무익(오히려 오염). 추월 경로 = **강한 모방 + 가벼운 offline 보정**, online fine-tune 없음.

### 다음 (추월 폭 키우기)
1. BC 정확도 더↑(데이터·epochs·net 확장) → prior를 -495 이하로.
2. **PPO + KL-to-BC**로 online에서도 안 깎이게 개선 시도(value-RL의 구조적 한계 회피).

## 11. 돌파 — 예측형 휴리스틱: 천장 자체를 올린다 (Run 4~7 교훈의 결론)

Run4~7의 일관된 결론: **RL은 레버가 아니다. 모든 방법의 천장 = 휴리스틱(-500)이고, BC는 그걸 흉내, value-RL은 못 넘음.** → 리워드를 *크게* 올리려면 **흉내낼 대상(휴리스틱) 자체를 똑똑하게** 만들어야 함.

진단(§Step2, 06-03): 격차의 100%가 stockout/full, RL/휴리스틱 모두 **오전 수요 ramp 과소대응**(반응형이라 늦음).

### 예측형 휴리스틱 (`PredictiveImbalancedPolicy`)
현재 불균형 대신 **미래 H스텝 후 예상 상태**로 결정: `predicted = bikes + Σ_{t..t+H}(returns - rentals)`. "곧 빌/꽉 찰" 정류소 선제 대응. ([src/agents/baselines.py](../src/agents/baselines.py), eval: [scripts/eval_predictive.py](../scripts/eval_predictive.py))

| 정책 | reward | Δ(반응형) |
|---|---:|---|
| 반응형 most_imbalanced | -500.02 | 기준 |
| 예측형 H=2 | -395.42 | +104.6 |
| **예측형 H=3** | **-382.79** | **+117.2** |
| 예측형 H=6 | -414.71 | +85.3 |
| 예측형 H=18 | -527.16 | -27.1 (너무 멀리) |

→ **H=3(30분 선행)이 최적, -382.79 (Δ+117).** RL로 1~2점 짜내던 걸 예측 규칙이 117점 벌었다. reward는 ≈"하루 서비스실패(stockout+full) 건수의 음수" → **하루 ~117명 더 빌림/반납 성공.**

### ⚠️ oracle 단서 + 현실성 체크 (forecast)
예측형은 `env.data.rentals[t:t+H]` = **그 날 실제 미래 수요**(완벽 예지)를 씀. env가 `future_demand_horizon` obs로 제공하는 정보지만 배포엔 비현실적. → train 60일의 **시간대별 평균**으로 forecast 대체해 재평가 ([scripts/eval_forecast.py](../scripts/eval_forecast.py)):

| forecast | reward | Δ | oracle 회수 |
|---|---:|---|---|
| oracle (완벽예지) | -382.79 | +117.2 | 100% |
| 전체평균 H=3 | -472.28 | +27.7 | 24% |
| **평일/주말 H=3** | **-465.32** | **+34.7** | **30%** |

- **+117의 ~70%는 oracle 아티팩트** (그날 비주기적 스파이크), **~30%(+35)는 진짜·배포 가능**(주기적 요일·시간 패턴).
- 결정적: **배포형 forecast 예측형(-465, Δ+35)이 RL/BC 최고(-499, Δ+0.85)를 40배 압도.** 단순 예측 규칙이 모든 RL 노력보다 실질 개선 큼.

### forecast 정교화 (A) — 결과: 292일 전체평균이 최선, 회수 34% ([scripts/eval_forecast2.py](../scripts/eval_forecast2.py), [eval_forecast3.py](../scripts/eval_forecast3.py))

| forecast | reward | Δ | oracle 회수 |
|---|---:|---|---|
| 60일 평일/주말 H=3 | -465.3 | +34.7 | 30% |
| **292일 전체평균 H=3** | **-459.7** | **+40.4** | **34%** ← 최선 |
| 요일별(7) H=3 | -470.8 | +29.3 | 25% (≈40일/요일로 쪼개져 noisy) |
| 전체평균+날씨배율 H=3 | -463.7 | +36.3 | 31% (효과 없음) |

- **더 많은 데이터(292일 전체평균)만 효과** — 요일 분할·날씨 배율은 도움 안 됨.
- **날씨 배율 무효 이유**: 회귀 R²=0.64로 날씨가 *일 수요 수준*은 잘 설명하나, 예측형 결정은 *정류소 간 상대 불균형·시간 형태*에 달려 전체를 같은 배율로 키우면 argmax 불변(≈no-op). **병목 = 수준이 아니라 per-station·per-time 형태 편차.**
- 남은 66%는 그날 비주기적 스파이크 → per-station 수요예측 모델 필요(수확 체감).

> **결론(A)**: 배포형 예측 정책 천장 ≈ **-459.7 (Δ+40.4)**. oracle(-383)의 34%지만, **RL/BC 최고(-499, +0.85)를 +40 압도** = 하루 ~40명 추가 구제. 단순 예측 규칙이 모든 RL 노력보다 실질 개선 큼.

### BC clone 시도 → 실패 (이득 소실), 규칙 직접 채택으로 결론

oracle 예측형 H=3을 future_demand obs(horizon=3) 켜고 BC clone (292일·150ep, acc 28.1%) → eval **-502.15 (Δ-2.1)**. **teacher(-383)를 전혀 못 잡고 반응형(-500) 수준으로 회귀** ([scripts/eval_bc_policy.py](../scripts/eval_bc_policy.py)).

- 이유: 예측형 이득은 "곧 무너질" 정류소를 고르는 **미묘한 소수 케이스**에서 나오는데, 그게 정확히 146지선다 BC가 틀리는 72%. 학생은 지배 신호(현재 불균형=반응형)로 회귀, future_demand obs를 거의 안 씀.
- **세션 일관 패턴 재확인**: 학습(BC/RL)은 단순 규칙을 *밑돈다*. (살리려면 분류 아닌 per-station score **regression** clone 필요 — 미시도.)

> **최종 결론 (사용자 채택 A)**: **예측형 휴리스틱을 규칙 그대로 최종 정책으로 채택.** clone은 이득을 잃으므로 안 함.
> - **배포형**: forecast 예측형(292일 전체평균, H=3) = **-459.7 (Δ+40.4)** — 미래 모름, 즉시 배포 가능.
> - **상한(참고)**: oracle 예측형 H=3 = -382.79 (완벽예지 가정).
> - 핵심 교훈: 이 환경에서 reward를 올린 건 RL/딥러닝이 아니라 **"반응형 → 예측형"이라는 정책 설계(미래 수요 선반영)**였다. RL 트랙(추상화·DQfD·anneal)은 전부 휴리스틱(-500) 언저리 천장, 예측 규칙이 그 천장을 -460~-383으로 끌어내림.

### 추가 여지 (선택)
- per-station 수요예측 모델로 forecast 회수율↑ (수확 체감).
- score-regression으로 예측형을 신경망에 담기 (신경망 산출물이 필요할 때만).

## 12. RL로 예측형 학습·유지 — 추상 predictive primitive + warm-start + DQfD ✅

동기: "RL 프로젝트라 RL을 써야 한다"는 제약. RL이 예측 로직을 *학습*하는 건 약하므로(146 clone -502 실패), **예측형을 action primitive로 쥐여주고 RL은 메타 선택만** 하게 함.

구현:
- `AbstractActionWrapper`에 **predictive 의도(index 5)** 추가 — 미래 H=3 예상 불균형 큰 곳 선점. `forecast_rent/ret` 주면 forecast(배포형), 없으면 oracle. ([src/envs/abstract_action.py](../src/envs/abstract_action.py))
- **warm-start**: `ConstantIntentPolicy(5)`("항상 predictive")를 demo로 DQfD pretrain → 추상 공간이라 "항상 idx5"가 **자명하게 학습**(146 clone 실패 회피). 출발점 = 예측형.
- **prior 유지**: DQfD demo 앵커(margin+CE, λ=1 상수) + 낮은 ε(0.05) → online RL이 안 깎게.
- `--algo dqfd --abstract-actions --predictive-mode {oracle,forecast}` ([scripts/train.py](../scripts/train.py))

### 결과 (300k, n_train_dates=60) — RL이 예측형 천장을 유지 ✅

| 버전 | warm-start eval | online 300k 최종 | Δ(휴) | 배포 |
|---|---:|---:|---|---|
| oracle predictive | -382.79 | **-382.79** (완벽 수평) | +117.2 | ❌ 상한 |
| forecast predictive (60일) | -472.28 | -472.28 (110k 일시 -489 후 회복) | +27.7 | ✅ |
| **forecast predictive (292일)** | -459.65 | **-459.65** (수평 유지) | **+40.4** | ✅ ★ |

(292일 profile = standalone forecast 평가값과 정확히 일치 → RL 정책이 "항상 predictive(292일)"를 완벽 재현. profile 데이터 많을수록 예측 정확도↑: 60일 -472 → 292일 -459.)

- **이 프로젝트 최초로 online RL이 좋은 prior를 안 깎고 유지** (cf. plain RL: 강한BC -499→-570, BC fine-tune -506→-705 붕괴).
- 유지되는 이유: 추상 6지선다 + "항상 predictive" demo → CE/margin이 Q[5]를 확고히 1등 고정 → td_agent가 argmax 못 뒤집음. eval이 결정적이라 reward가 소수점까지 동일(정책 불변).
- **정직한 해석**: RL이 예측형을 *재현·유지*하는 것이지 *능가*하는 건 아님. greedy 정책 = "항상 predictive" = 예측형 휴리스틱과 동일 동작 → 동일 reward.
- forecast -472.28 = 60일 profile 기준(292일이면 -459). 앵커 강화/292일 profile로 더 단단히·높게 가능.

> **결론(12)**: "RL을 쓰면서 예측 활용"을 달성. RL의 역할 = **예측형을 학습해 배포형 -472(상한 -383)에 도달·유지**. RL이 천장을 *뚫진* 못하지만(이 환경 일관 결론), 예측 설계를 RL 프레임에 담아 휴리스틱을 안정적으로 +28 추월.

## 10. 다음 후보

1. Run 4(추상화)가 -500 근처 학습하면 → 어휘 확장(미래수요 의도 추가)으로 추월 시도.
2. 추상화도 실패 시 → **BC로 복귀**(검증된 추월 경로) 또는 PPO 트랙.
3. (속도) SubprocVecEnv startup 개선 — 워커가 episode 디스크 직접 load.

## 13. 진짜 RL — 액션=정류소, forecast는 *입력*, RL이 직접 학습 ✅ (degenerate 해소)

지적(사용자): §12의 "predictive를 *행동*으로" 방식은 RL이 규칙에 통째 위임하는 **degenerate**. 사용자 액션 = **트럭 목적지(정류소 146)**이므로, 예측은 *행동*이 아니라 **상태(입력)**로 넣고 RL이 정류소를 직접 골라야 진짜 학습.

구현:
- **forecast를 env obs에 주입** — RebalanceEnv `future_demand_horizon` obs를 oracle→forecast(과거평균)로 교체 (`forecast_rent/ret`). RL이 "앞으로 어디가 빌지"를 입력으로 받음. ([rebalance_env.py](../src/envs/rebalance_env.py))
- **score-regression warm-start** — 분류 clone(28% acc, -502 실패) 대신 예측형의 **정류소별 점수를 MSE 회귀**로 모방 → argmax(Q)=예측형. ([scripts/warmstart_scoreregress.py](../scripts/warmstart_scoreregress.py))
- **DQfD fine-tune** — score-reg init에서 raw 정류소 공간 online RL + forecast 예측형 demo 앵커 + 낮은 ε.

### 결과 (292일, forecast obs, H=3)

| 단계 | eval | Δ휴 | 비고 |
|---|---:|---|---|
| 분류 clone (이전) | -502 | -2 | 실패 |
| **score-regression warm-start** | **-467.94** | +32.1 | 회귀가 분류 우회 ✅ |
| **+ DQfD online fine-tune** | best **-457.39**(40k) / 마지막 -464.86 | **+42.6** | -457~-470 진동 |

- **score-regression이 분류 clone(-502)을 우회** → raw 정류소 공간에서 예측형을 신경망에 담음(-467.94, 모방손실 ~8).
- **DQfD online이 처음으로 prior를 *넘어섬*** — best -457.39는 score-reg init(-467.94)·forecast 예측형 휴리스틱(**-459.65**) **둘 다 추월**. 이번 세션 통틀어 RL이 규칙/prior를 능가한 첫 사례.
- raw 공간인데 **붕괴 안 함**(이전 raw 런들은 -570). 예측형 demo 앵커 + 낮은 ε 덕분. 단 진동(-457~-470) 있음.

> **결론(13)**: **degenerate 해소.** 액션=정류소(사용자 공간), 예측은 입력, RL이 직접 학습. score-regression으로 분류 clone 실패를 우회해 raw 공간 prior(-467.94) 확보 → DQfD가 forecast 예측형(-459.65)을 **넘는 -457.39**까지 학습. "RL이 예측을 활용해 규칙을 *능가*"를 처음 달성(진동 안정화는 다음 레버: 앵커 강화/lr 조정).

## 14. 서영현(2020) 논문 아이디어 ① — 예측오차 보정/타게팅 → ❌ 우리 환경선 악화

출처: 서영현(2020) 서울대 박사 "실시간 동적 계획법 및 강화학습 기반 공공자전거 동적 재배치" (지도 고승영). 받은 자료는 1p 초록 → 환경·방법 뼈대만. 핵심: MDP→DP→**ADP**→RL, 수요는 RF예측→Poisson 확률발생, **"예측오차(관측−예측)에 빠르게 대응"**, **"예측오차 큰 대여소만 탐색해도 전체탐색과 성능 유사+계산절감"**.

### 적용 ① — 예측오차 보정 예측형 (`ForecastErrorPolicy`, [baselines.py](../src/agents/baselines.py))
배포 가능한 단순화: forecast(292일 평균)로 미래를 깔되, 최근 W스텝 **관측−forecast 잔차**를 정류소별로 추정해 앞 H스텝 예측을 실시간 보정(drift 가산 / scale 승산). + 잔차 큰 상위 K개만 후보(focus). ([scripts/eval_forecast_error.py](../scripts/eval_forecast_error.py))

### 결과 (7일, 공정 metric) — 모든 변형이 forecast(-459.65)보다 **나쁨**

| 설정 | eval | Δ(forecast -459.65) |
|---|---:|---|
| 반응형 / oracle 상한 / **forecast 예측형** | -500 / -383 / **-459.65** | (기준) |
| drift W=3 / 6 / 12 / 24 | -493 / -488 / -480.6 / -480.5 | **-34 ~ -21** ❌ |
| scale W=6 / 12 / 24 | -502 / -484 / -478 | -42 ~ -18 ❌ |
| drift W=24 **α=0.2 / 0.5** (약한 보정) | -475.6 / -479.4 | -16 / -20 ❌ |
| drift focus K=30 / 50 | -525 / -491 | -65 / -31 ❌ |

### 왜 실패했나 (단조 패턴이 원인을 특정)
- **보정을 약하게(α↓)·매끄럽게(W↑) 할수록 덜 나쁨** → 최적은 **α=0(보정 안 함=그냥 forecast)**. 즉 최근 잔차는 **신호가 아니라 노이즈**.
- **292일 평균이 이미 강한 저분산 추정**. 10분·정류소 단위 수요는 희소·스파이키(Poisson) → 최근창 잔차는 표본노이즈가 지배, 앞으로 투영하면 노이즈 주입.
- **잔차 비지속**: 오전 편차가 오후 편차를 예측 못 함 → 외삽 무의미.
- **focus는 좋은 후보를 굶김**: 예측대로 흐르지만 실제 불균형인 정류소를 후보에서 제거 → 대형 실패(K=30 → -525).
- oracle(-383)의 이득은 **참 미래**의 값이라, 관측-과거로는 복원 불가.

### 단서 (논문과의 간극)
- 우리가 테스트한 건 **잔차 외삽 단순화**지 논문의 **RTDP/ADP 코어(가치함수 백업+오차 시 재계획)**가 아님. 단순화 부분은 우리 환경에 **이식되지 않음**.
- 보정으로 메울 줄 알았던 forecast→oracle 격차(-77)는, 이 데이터에선 관측-과거로 복원 불가능한 **진짜 미래정보**임을 재확인.

> **결론(14)**: 논문 아이디어 ①의 배포형 단순화(예측오차 보정·타게팅)는 **우리 환경에서 도움 안 됨(악화)**. 배포 가능 천장은 여전히 **forecast 예측형 -459.65** / RL(DQfD) -457.39. 논문의 진짜 메서드를 따르려면 ADP/RTDP 코어(본문 필요) 또는 ③ 확률수요 학습을 별도로 시험해야 함.

## 15. 서영현(2022) **전체 논문** 정독 → 비교 (§14 해석 정정)

받은 전체 논문(JAT 2022, open access). §14는 1p 초록만 보고 ①을 "리워드 개선 레버"로 오해 → **정정**.

### 그들 환경 vs 우리 (스케일이 결정적)
| | 논문 | 우리 |
|---|---|---|
| 지역 | 여의도 2.9km² | 마포구 |
| 정류소 | RTDP **5~7개** / A2C 31개 | **146개** |
| 트럭 | 1대 (cap 15) | 3대 |
| 기간 | **2시간**(07–09 or 18–20 피크) | **하루 전체**(144×10분) |
| 수요 | 확률 Poisson(RF예측 평균), Skellam 전이 | 결정론 7일 eval |
| step | 10분 | 10분 |
| 상태 | (t, 차량위치, **정류소별 안전밴드 안/밖 이진**) | raw 재고/비율 |
| 행동 | next-station + 적재량(**적재량은 안전재고 규칙으로 자동**) | 트럭 목적지 146 (적재 자동) |

### 두 개의 엔진 — 무엇이 휴리스틱을 이겼나
- **RTDP (테이블형 비동기 가치반복, Skellam 전이확률로 명시적 lookahead)** — **작은 문제(5~7정류소·2h)** 전용. **이게 휴리스틱을 크게 이김.**
  - Table 2 (미충족수요, 낮을수록 좋음): No-reb 10.6 / **STR(반응형) 8.5** / SLA(정적 lookahead) 9.6 / **RTDP 3.8(z=1.0)→3.5(1.65)→2.3(2.33)**. 안전재고 buffer z↑일수록 좋아짐.
- **A2C (model-free actor-critic, ANN)** — **큰 문제(31정류소)** 전용(테이블 불가). 여기선 **휴리스틱(STR/SLA)과의 직접 비교가 없음**. 전략끼리만 비교(Table 4), 그리고 **constraint-free(행동을 RL이 통째로=우리 방식)가 제일 나쁨**(96.5~96.9, 표준편차 2배).

> **핵심**: 논문의 "휴리스틱 추월"은 **RTDP=확률적 DP(전이모델+Bellman 백업)**, **5~7정류소·2시간**에서 나온 것. **model-free RL이 큰 문제에서 휴리스틱을 이긴 증거는 논문에도 없음** → 우리 결론(model-free는 못 넘음)을 **외부에서 보강**.

### 예측오차 focus(Strategy 3) — 실제로 논문이 말하는 것 (§14 오해 정정)
- 세 전략: S1 전체탐색 / S2 인접탐색 / S3 **예측오차 큰 정류소**.
- Table 3 (7정류소, 결정론): do-nothing 19.0 / **S1 13.67(최선)** / S2 19.67(≈무행동) / S3 14.00. **S3는 S1과 리워드 비슷, 단 계산 28.5%↓**.
- Table 4 (31, 확률): S3는 **수요 150%(큰 surprise)일 때만 최선**(90.87), 50%일 땐 최악(98.56).
- → **논문도 "전체탐색(S1)이 리워드 최선", focus(S3)는 *계산 절감*용이며 큰 surprise에서만 리워드 이득.** 따라서 **§14에서 focus가 리워드를 악화시킨 내 결과는 논문과 모순 아님 — 일치**. (우리 평가일은 큰 surprise가 아니라 focus 이득 없음.)
- 또 내 §14 "예측오차 보정(잔차 외삽)"은 **논문 메서드가 아님**. 논문은 잔차를 외삽하지 않고, **RTDP가 매 step 실현상태+확률전이모델로 재계획**하며 forecast는 Poisson 평균으로만 쓰임. 내가 만든 잔차외삽은 논문에 없는 단순화였고 실패한 것.

### 진짜로 우리에게 이식 가능한 것
1. **② 안전재고 밴드 + 적재 자동 + z 스윕** — 상태=안전밴드 안/밖, 적재량=안전재고(z·√(lead·평균))로 자동, z 키우면 좋아짐(논문 3.8→2.3). 우리 점목표(0.5) 대비 새 레버. **가장 깔끔하게 이식 가능.**
3. **③ Poisson 확률수요로 학습** — forecast 평균의 Poisson으로 흔들어 학습→강건화.
4. **RTDP 코어** — 진짜 추월 엔진이지만 **테이블형이라 146정류소·하루엔 불가**(상태 2^|N|). 적용하려면 정류소를 5~10개로 줄이고 2h 피크로 축소한 *소규모 문제*에서만 가능.

> **결론(15)**: 논문의 추월은 **소규모 확률적 DP(RTDP)**의 성과지, 대규모 model-free RL의 성과가 아니다. 우리 환경(146·하루·결정론)에 그대로 오는 레버는 **② 안전재고 z-buffer**와 **③ 확률수요 학습**. "휴리스틱을 RL로 이긴다"를 논문처럼 재현하려면 **문제를 소규모로 줄여 RTDP**를 돌리는 게 정공법.

## 16. RTDP 소규모 재현 (논문 정공법) → ❌ 강한 예측형(SLA) 못 넘음

논문(§15)의 휴리스틱 추월 엔진 = RTDP(테이블형 확률적 DP). model-free 아님. 정류소·트럭·시간 축소해 재현. 코드 [scripts/rtdp_small.py](../scripts/rtdp_small.py), 환경·과정 [docs/rtdp_experiment_setup.md](rtdp_experiment_setup.md).

셋업(갈래 A): 마포구 07~10시, 지리적으로 퍼진 6정류소+depot, 트럭1(cap30), Poisson(forecast) 확률수요, 상태=(시각,위치,3-레벨 밴드인덱스), 행동=(목표레벨{비움/중간/채움}+목적지), 비용=미충족수요. RTDP=비동기 가치반복+궤적샘플링+**분석적 기댓값 백업**(샘플백업의 optimizer's curse 교정).

### 결과 (iters=12000, 미충족수요↓)
| 정책 | 확률 Poisson(30) | 배달량 | 실제(7일) |
|---|---|---|---|
| do-nothing | 19.10 | 0 | 31.71 |
| STR (반응·최소재배치) | 8.33 | 24.2 | 17.14 |
| **SLA (예측형 lookahead)** | **5.63** ⭐ | 33.5 | **11.71** ⭐ |
| RTDP (확률적 DP) | 8.43 | 57.4 | 16.71 |

- **RTDP ≈ STR(반응형), 예측형 SLA(5.63) 못 넘음.** RTDP 과배달(57.4) 지속.
- 원인: ① 밴드인덱스 축약이 과배달 못 벌함 ② certainty-equiv 백업이 스파이크 과소평가 ③ **우리 SLA(자유 예측형)가 강함** — 논문 SLA는 25/50/75%·1회방문 제약(그래서 논문 RTDP 2.3≪SLA 9.6).

> **결론(16)**: 논문의 RTDP 추월은 *작은 문제 + 제약된 약한 baseline*의 산물. 충실 재현+강한 예측형 baseline에선 RTDP가 반응형 수준에 머물고 예측형을 못 넘음. **model-free RL도 model-based RTDP도 강한 예측형 휴리스틱을 능가 못함** → 프로젝트 중심 결론 최종 확정. 천장을 올리는 유일 레버는 *예측형 휴리스틱 설계*.

## 17. ✅✅ DQN이 휴리스틱을 *크게* 추월 — "배울 수 있는 크기"에서 (10정류소)

§16(RTDP)·이전 결론은 "146·하루에선 RL이 휴리스틱 못 넘음"이었다. **문제를 RL이 배울 수 있게 줄이니(10정류소·1트럭) DQN이 예측형을 절반으로 압도.** 코드 [scripts/dqn_small.py](../scripts/dqn_small.py).

### 환경
- 마포구 **출퇴근 불균형 압력 top-10 정류소**(idx 13,130,139,76,2,35,75,108,135,109; 채움4·비움2·혼합4, 거리 평균2.2km) + depot, **트럭 1대(cap30)**, **전일 144step**(출근·퇴근 피크 포함), Poisson(forecast) 확률수요.
- DQN(stable-baselines3): obs=재고율+적재+위치+시각+**forecast 미래순수요**, action=목적지(+재배치량 레벨), reward=−미충족수요(+potential shaping), 400k step(3.6분).

### 결과 (미충족수요↓)
| 정책 | 확률 Poisson(30) | 실제(7일) |
|---|---|---|
| do-nothing | 159.23 | 209.57 |
| STR (반응형) | 81.70 | 138.86 |
| SLA (예측형) | 73.60 | 120.71 |
| **DQN** | **36.90** ⭐ | **72.00** ⭐ |

**같은 리워드 정의(원본 RebalanceEnv = 어제와 동일: stockout-1.0/full-0.8/km-0.008/step-0.002)로 재채점** (리워드↑ 좋음): do-nothing −139.3 / STR −74.5 / SLA −67.5 / **DQN −32.6** (확률30). → DQN +34.8 vs SLA 추월 유지. (정류소 10 vs 146이라 어제 −459와 절대값 비교는 불가.)

### 귀인(어블레이션) — 추월은 견고하고 보조장치 덕 아님
| 조건 | DQN 확률 | 결론 |
|---|---|---|
| 기본 seed1 / seed2 | 36.90 / 36.80 | **재현됨** |
| shaping=0 | 30.20 | shaping 원인 아님(없어도 추월) |
| no-amount (목적지만·50%고정 = SLA와 동일 행동력) | 36.07 | **재배치량 lever 원인 아님** |
| no-forecast (+shaping0; 현재상태만) | 69.53 | **forecast가 큰 승리폭의 핵심**(+36.7→+4.07) |

→ **추월의 진짜 원인 = ① forecast 입력 + ② 순수 라우팅 학습.** 분해: SLA 73.6 / DQN(forecast無) 69.5(거의 동률) / DQN(forecast有) 36.9(압도). forecast 빼도 시각feature로 패턴 자력학습해 살짝 이기나, 큰 승리폭은 forecast+라우팅 결합 시너지. shaping·재배치량 lever 불필요.

### 결론(17) — 프로젝트 핵심 질문의 답
> **RL은 휴리스틱을 이길 수 있다. 단 "배울 수 있는 크기"에서만.** 146·하루엔 못 넘었지만(§9~16), 10정류소·1트럭으로 줄이니 DQN이 예측형을 절반으로 압도. 예측형 SLA는 *탐욕*이라 최적이 아니고, 작은 문제에선 DQN이 forecast+가치학습으로 *비탐욕 최적 경로*를 발견 → 추월. (RTDP는 §16에서 못 넘었으나 model-free DQN은 넘음 — coarse 상태축약 없이 신경망이 raw 상태를 직접 학습한 덕.)
