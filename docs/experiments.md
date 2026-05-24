# 실험 로그 — DQN 튜닝 진화

마포구 따릉이 재배치 환경에서 휴리스틱(`most_imbalanced`)을 능가하는 DQN 정책을 찾기 위한 단계적 실험 기록.

---

## 1. Reward 계산법 — 직관적 이해

### 1.1 한 줄 요약

> **"매 10분마다 시민 불편 + 트럭 운영 비용을 점수로 매기고, 24시간 동안 다 합친다."**

모든 점수는 음수(페널티)가 기본. 트럭이 멍청하게 굴면 점수가 더 떨어지고, 학습 잘 되면 덜 떨어진다.

### 1.2 6가지 점수 항목

**🔴 실제 손실 4가지 (운영 효율)**

이건 실제로 일어나는 나쁜 일들. 운영팀이 줄이고 싶은 것.

```
① stockout (대여 실패)
   "정류소가 비어서 시민이 자전거 못 빌림"
   → 1건마다 -1.0점

② full (반납 실패)
   "정류소가 가득 차서 시민이 반납 못함"
   → 1건마다 -0.8점

③ 이동 거리 비용 (연료/마모)
   "트럭이 1km 갈 때마다"
   → -0.008점

④ 이동 시간 비용 (운영 시간)
   "트럭이 1 step(10분) 동안 이동 중일 때마다"
   → -0.002점
```

**🟢 학습 도우미 2가지 (실제 운영엔 없음)**

DQN이 빨리 배우라고 인공적으로 만든 보상 신호. 실제 운영팀은 신경 안 씀.

```
⑤ urgent_bonus (위급 정류소 도착 보너스)
   "트럭이 빈 정류소(≤15%) 또는 가득 정류소(≥85%)에 도착했을 때"
   → 1회 +2.0점
   
⑥ explore_bonus (탐색 보너스)
   "트럭이 별로 안 가본 정류소에 갔을 때"
   → +0.3 / √(방문횟수)
   → 처음 방문: +0.3
     5번째 방문: +0.13
     50번째 방문: +0.04
```

### 1.3 예시로 따라가기 — 출근 시간 1시간

오전 7:00~8:00 동안 마포구에서 일어나는 일:

```
[t=07:00] 정류소 A 자전거 3대, B는 가득 (20대)
─────────────────────────────────────────
시민들 행동:
  A에서 5명 빌리려 함 → 3명만 빌림, 2명 stockout
    페널티: -1.0 × 2 = -2.0점
  B로 4명 반납 시도 → 0명만 반납 (가득), 4명 full
    페널티: -0.8 × 4 = -3.2점
  
트럭 0이 결정: "B로 가서 자전거 가져오자!" (A→B 5km, 12 step)
  즉시 거리 비용: -0.008 × 5 = -0.04점

[t=07:00 합계]: -2.0 - 3.2 - 0.04 = -5.24점


[t=07:10] 트럭 이동 중
─────────────────────────────────────────
시민들: 또 A에서 1명 stockout
  페널티: -1.0
트럭 이동: -0.002

합계: -1.002점


... (10분마다 반복) ...


[t=07:50] 트럭 0이 B에 도착!
─────────────────────────────────────────
B는 가득 상태(20/20) → 위급 정류소!
  ✨ urgent_bonus: +2.0
  ✨ explore_bonus (B 첫 방문): +0.3
트럭이 B에서 자전거 10대 적재 → B 분포 정상화 (20→10)
이 step의 reward = +2.3점

이후 B는 정상이라 stockout/full 안 생김 → 페널티 회피
```

### 1.4 1 episode 누적 (24시간 합산)

24시간 동안 144 step의 reward를 모두 더함:

```
R_episode = Σ (모든 step의 r_t)
          = (모든 stockout × -1.0)
          + (모든 full × -0.8)
          + (총 이동거리 × -0.008)
          + (총 이동시간 × -0.002)
          + (위급 곳 도착 횟수 × +2.0)
          + (탐색 보너스 누적)
```

**실제 예시 (open_v1 모델, 1/15 episode)**:
```
stockout 47건  → -47.0
full 112건    → -89.6
이동 118 km   → -0.94
이동 시간     → -약 0.5
urgent 도착   → 약 +10~30  (대략)
explore       → 약 +5~10
─────────────────
누적 reward = -약 -100 ~ -130 (실측 -139)
```

### 1.5 평가 — "7일 평균"

학습 중 모델 평가는 **7개 episode 돌리고 평균**:

```python
eval_reward = (R_1/13 + R_1/14 + R_1/15 + ... + R_1/19) / 7
            = mean([episode별 누적 reward])
```

예: open_v1 모델의 best 시점 → 7일 평균 **+166.7점**.

⚠️ 1/15 하루만 보면 -139인데 평균은 +166. 왜?
- 그날만 운 나쁘게 stockout/full 많이 발생
- 다른 6일은 운 좋게 보너스 누적
- → 평균을 봐야 정책 품질 측정 가능

### 1.6 휴리스틱과의 비교 — Δ가 진짜 지표

같은 환경에서:
```
                  휴리스틱    DQN best    Δ
mapo_open_v1:    +128.22  →  +166.7    +38.5 ✅
```

**Δ = DQN_best - 휴리스틱**

⚠️ 환경 설정(urgent_bonus, explore_bonus 등) 바뀌면 휴리스틱 reward도 같이 바뀜:

| 환경 | 휴리스틱 reward | 이유 |
|---|---|---|
| urgent_bonus 없음 | -37 | 페널티만 누적 |
| urgent_bonus=1.0 | +43 | 휴리스틱도 위급 곳 자주 감 → 보너스 누적 |
| urgent_bonus=2.0 + explore=0.3 | +128 | 보너스 더 많이 누적 |

→ **DQN best 절대값(+166)이 중요한 게 아니라 같은 환경의 휴리스틱(+128) 대비 얼마나 더 좋은가(Δ +38)가 핵심.**

### 1.7 시각화 — 한 episode 동안 reward 흐름

```
누적 reward ↑
          ┐ 시작 0
        0 ┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ (이상적: 0 유지)
          │
       -50┤    ╲                  ↗━━━━ ← 트럭 보너스 받아 회복
          │     ╲                ╱
      -100┤      ╲             ╱
          │       ╲    ↘     ╱
      -150┤        ╲     ╲╱  ← 출퇴근시간 stockout/full 폭증
          │         ╲
      -200┤          ╲___________ 24h 끝 누적
          └────────────────────────────────────→ 시각
          0h    6h     12h     18h     24h
```

평가 시 24h 끝 누적값(R_episode)을 7일 평균.

### 1.8 정리

| | 정상 운영 | 학습 신호 |
|---|---|---|
| 음수 항목 | stockout, full, 이동 비용 | — |
| 양수 항목 | — | urgent_bonus, explore_bonus |
| 운영팀 관심사 | ✅ 줄이고 싶음 | ❌ (DQN 학습용일 뿐) |
| metric 정의 | 줄여야 좋음 | 클수록 좋음 |

핵심:
- **실제 효율은 stockout + full + 이동거리로 본다**
- DQN reward 평균값(+166)은 학습 잘 됐다는 신호일 뿐
- 진짜 의미는 **휴리스틱 대비 Δ**와 **stockout/full 절대값 감소량**

---

## 2. 환경 — 한눈에

| 구성 | 값 |
|---|---|
| 권역 | 마포구 (146 정류소) |
| 트럭 | 3대 × 적재 20대 × 25km/h |
| Step | 10분 |
| Episode | 24시간 (144 step) |
| Observation | **171 dim** (정류소 146 + 트럭 12 + 시간 4 + 캘린더 5 + 날씨 4) |
| Action | `Discrete(146)` (다음 갈 정류소) |
| Train pool | 1년치 random 80/20 분할 → 60일 sample |
| Eval set | 7일 (train과 누수 없음) |

---

## 3. 실험 과정 (Timeline)

### Phase 1 — 기준선 진단

**문제**: 초기 DQN(`dqn_long`, 500k step)이 휴리스틱(-37) 대비 -125로 매우 못함. 학습이 끝까지 발산.

**원인 분석**:
1. `action_masks` 실제 구현 안 됨 (placeholder `np.ones`)
2. Double DQN 미적용 → Q값 과대추정
3. 트럭 idle 시마다 무조건 결정 요청 → 의미 없는 결정 다수
4. raw observation에서 휴리스틱 수준의 도메인 지식 학습이 어려움

### Phase 2 — Action mask 실제 구현 + Double DQN

`MaskableDQN` 구현 → ε-greedy·argmax·predict 모두에서 invalid action을 -∞로 처리. `--double-q`로 Double DQN 타깃.

`mapo_smdp_v1` (100k): 휴리스틱 -14.5, DQN best -75.6. **여전히 미달이지만 dqn_long 대비 +63점 개선**.

### Phase 3 — SMDP 트리거

`urgent_low=0.15, urgent_high=0.85` 추가. 빈/가득 정류소 1개라도 있을 때만 DQN 호출 → 결정 횟수 약 32% 감소.

```python
def _advance_until_next_decision(self):
    while True:
        if idle_truck and self._needs_decision():  # ← 임계치 체크
            return next_decision
        tick()  # 시계만 진행
```

### Phase 4 — Reward shaping + 이동 비용 조정

- `urgent_bonus = +1.0` (위급 곳 도착 시) → "거기로 가라" 학습 신호
- `w_travel_km = -0.005`, `w_travel_step = 0` → 이동 부담 ↓

`mapo_smdp_shaped_v1` (100k): 휴리스틱 +43.9, DQN best +18.8. **DQN이 처음 양수 reward 영역 진입**.

### Phase 5 — 균형 잡기 + 더 길게

이동 비용 살짝 복원 + 300k 학습.

`mapo_smdp_shaped_v3` (300k): 휴리스틱 +42.7, DQN best **+70.7**. **휴리스틱 첫 초과 (Δ +28)** ✅

하지만 best 이후 발산 (-220까지 떨어짐) → 단일 운 좋은 봉우리.

### Phase 6 — Strict mask 실패

`--strict-mask` 추가 (위급 정류소만 선택 가능) + exploration 강화 + visit count bonus.

`mapo_explore_v1` (300k): 휴리스틱 +91.7, DQN best +49.96 → **Δ = -41.7 ❌ 역효과**.

원인: 좁은 후보 안에서 Q값 collapse 재발생 → 트럭들이 3개 정류소에만 집중.

### Phase 7 — Strict mask 제거 (전환점)

mask는 다른 트럭 destination 차단만 유지, **위급 정류소 제한은 제거**. `urgent_bonus` 2.0으로 강화.

`mapo_open_v1` (300k): 휴리스틱 +128.2, DQN best **+166.7**.

**🎉 처음으로 단조 우상승 학습 곡선 + best=final + 발산 없음.**

### Phase 8 — Observation 확장 (캘린더 + 날씨)

171 dim으로 확장. 1월만으로는 효과 미미 → 1년치 데이터로 확장 필요.

`EpisodeData`에 `dayofweek, is_weekend, is_holiday, is_holiday_eve, weather (T,4)` 추가. 전처리 parquet에서 직접 가져옴 (`holidays` 패키지 의존 제거).

### Phase 9 — 1년치 데이터

`TRAIN_DATES`를 `seed=42` random shuffle → 80/20 분할. 60일 train pool에 모든 12개월 골고루 분포. 데이터 누수 0.

`mapo_year_v1` (500k): 휴리스틱 +238.3, DQN best -141.7 → **Δ = -380 ❌**.

원인: 다양성 ↑↑ → 학습 신호 dilute, timesteps 부족. 곡선이 끝에서 회복 추세지만 미달.

---

## 4. 실험 결과 표

| # | Tag | 핵심 변경 | timesteps | 휴리스틱 | DQN best | Δ |
|---|---|---|---|---|---|---|
| 0 | `dqn_long` | 원본 (mask placeholder) | 500k | -36.97 | -139.5 | -102.5 ❌ |
| 1 | `mapo_smdp_v1` | + SMDP 트리거 | 100k | -14.52 | -75.6 | -61.0 ❌ |
| 2 | `mapo_smdp_shaped_v1` | + shaping + strict + 이동비↓↓ | 100k | +43.93 | +18.8 | -25.1 ❌ |
| 3 | `mapo_smdp_shaped_v2` | 이동비 살짝 복원 | 100k | +42.69 | +37.8 | -4.9 ❌ |
| 4 | `mapo_smdp_shaped_v3` | 300k 학습 | 300k | +42.69 | **+70.7** | **+28.0 ✅** |
| 5 | `mapo_explore_v1` | + exploration↑ + visit count | 300k | +91.67 | +49.96 | -41.7 ❌ |
| 6 | **`mapo_open_v1`** ⭐ | strict_mask 제거 | 300k | +128.22 | **+166.7** | **+38.5 ✅** |
| 7 | `mapo_year_v1` | + 1년치 데이터 (60일 train) | 500k | +238.32 | -141.7 | -380 ❌ |

---

## 5. 핵심 발견

### 5.1 strict_mask는 역효과
"위급 정류소만 선택"의 직관적 좋아 보이는 제한이 실제로는 **Q값 collapse를 좁은 영역에서 더 심하게 만듦**. 학습이 진행될수록 트럭이 3-5개 정류소에만 갇힘.

### 5.2 reward shaping은 필수 + 정도 조절
- `urgent_bonus = 0` (Phase 1~2): DQN이 stockout/full sparse reward만으로 학습 어려움
- `urgent_bonus = 1.0` (Phase 4): 처음 양수 영역 진입
- `urgent_bonus = 2.0` (Phase 7): 휴리스틱 초과 달성

### 5.3 이동 비용은 절대 0이 되면 안 됨
`w_travel_km=0, w_travel_step=0`이면 DQN이 "가까운 1-2개 정류소만 왕복" 정책으로 수렴. 약하지만 살아있는 이동 비용 (`-0.008/km, -0.002/step`)이 필요.

### 5.4 단조 상승 곡선 = 좋은 신호
| Tag | 학습 곡선 패턴 | 결과 |
|---|---|---|
| `dqn_long` | 처음~끝 진동, 발산 | ❌ |
| `mapo_smdp_shaped_v3` | 한 봉우리(+70 step 230k), 이후 발산 | △ |
| **`mapo_open_v1`** | **단조 우상승, best = final** | **✅** |

best가 final과 같다 = **계속 학습하면 더 좋아질 가능성**.

### 5.5 데이터 다양성 ↑ → timesteps도 더 필요
v6에서 train 60일 (다양한 계절) + 500k step → episode당 ~8k step만 노출. v5와 같은 episode 노출 수준이 되려면 1.5M~2M step 필요할 것으로 추정.

---

## 6. 학습 곡선 분석 — `open_v1` 예시

```
step  10k → -47   ┐
step 100k → -56   │ Phase A: 혼란기 (-50 ~ -100)
step 130k →  +50  ┘
step 210k → +116  ┐
step 250k → +138  │ Phase B: 안정 우상승 (+50 ~ +160)
step 290k → +110  │
step 300k → +167  ┘ best = final = step 300k
```

→ ε-greedy 무작위 비율이 점차 줄어들면서(ε=1.0 → 0.15) policy가 안정화. urgent_bonus + explore_bonus가 위급 정류소 패턴 학습을 유도.

대조: `dqn_long` 학습 곡선
```
step 150k → -139 (best)
step 200k → -180
step 350k → -250
step 500k → -220   ← best 이후 계속 발산
```

---

## 7. Replay 분석 — 정책 다양성

같은 1/15 episode 기준 (24h 동안 트럭들이 방문한 정류소 패턴):

| Tag | 전체 unique 방문 정류소 | 같은 정류소 모임 | 1일 이동거리 |
|---|---|---|---|
| `dqn_long` | 7 | **67.6%** ❌ | 985 km |
| `mapo_smdp_v1` | 8 | 0% | 170 km |
| `mapo_smdp_shaped_v3` | 3 (strict_mask collapse) | 0% | 16 km |
| `mapo_explore_v1` | 3 (좁은 collapse) | 0% | 24 km |
| **`mapo_open_v1`** | **8** | **0%** | **118 km** |
| `mapo_year_v1` | 8 | 0% | 158 km |

### 정책 collapse의 변화

```
dqn_long (Phase 1):
  트럭들 모두 ST-2162에 67.6% 같이 모임 → 비효율
  
open_v1 (Phase 7):
  트럭 0: ST-92 62%, ST-86 31% (두 곳 왕복)
  트럭 1: 9개 정류소 두루 (top1 47%)
  트럭 2: ST-341 61% (여전히 좁음)
  → 트럭별 분담 + 일부 collapse 완화
```

### viewer로 직접 확인

```bash
# 서버 실행
python -m http.server 8765 --directory docs

# 브라우저
http://localhost:8765/replay_viewer.html

# 📁 다른 JSON 로드 → 비교
#   docs/replay_dqn_long.json     — 원본
#   docs/replay_open_v1.json      — 휴리스틱 초과 ⭐
#   docs/replay_year_v1.json      — 1년치 데이터 (현재 default)
```

---

## 8. 현재 권장 설정

```bash
python scripts/train.py \
  --algo masked_dqn --double-q \
  --urgent-low 0.15 --urgent-high 0.85 \
  --urgent-bonus 2.0 \
  --explore-bonus 0.3 \
  --w-travel-km -0.008 --w-travel-step -0.002 \
  --exploration-fraction 0.6 --exploration-final-eps 0.15 \
  --n-train-dates 60 \
  --tag <태그> --timesteps 1000000
```

| 인자 | 값 | 이유 |
|---|---|---|
| `--algo masked_dqn` | | invalid action 차단 |
| `--double-q` | | Q 과대추정 완화 |
| `--urgent-low/high` | 0.15 / 0.85 | SMDP 트리거 임계치 |
| `--urgent-bonus` | 2.0 | 위급 곳 학습 유도 |
| ❌ `--strict-mask` | (꺼짐) | 좁은 collapse 회피 |
| `--explore-bonus` | 0.3 | 다양한 정류소 시도 |
| `--w-travel-km/step` | -0.008 / -0.002 | 살아있는 이동 비용 |
| `--exploration-fraction/eps` | 0.6 / 0.15 | 학습 후반에도 탐색 유지 |
| `--n-train-dates` | 60 | 모든 계절·공휴일 포함 |
| `--timesteps` | 1M+ | 1년치 학습에는 충분히 길게 |

---

## 9. 다음 단계 후보

| 우선순위 | 항목 | 효과 |
|---|---|---|
| 🔴 높음 | `year_v1` 더 길게 (1.5M~2M step) | 학습 곡선이 우상승 추세 → 휴리스틱 추격 |
| 🟡 중간 | Stay penalty schedule (초기 활발, 후기 신중) | 학습 초반 다양한 시도 강제 |
| 🟡 중간 | Behavior Cloning warm-up | 휴리스틱 transition을 buffer에 미리 주입 → cold start 해결 |
| 🟢 낮음 | Dueling DQN / PPO 비교 | 알고리즘 ablation (Phase 5 in main README) |
| 🟢 낮음 | 다중 권역 확장 (영등포·강남) | 다른 권역에서도 작동하는지 |

---

## 10. 관련 문서

- **README.md** — 프로젝트 전체 개요 (Phase 1~5 로드맵)
- **docs/project_overview.html** — 인터랙티브 흐름 가이드
- **docs/source_guide.html** — 소스 코드 가이드 (파일별 상세)
- **docs/replay_viewer.html** — 학습된 모델 episode 재생
- **docs/training_flow.html** — 5 정류소 미니 시뮬레이터
