# 포아송(Poisson) 확률 수요 — 적용 방식과 실험 정리

> **목적:** 결정적 replay에서는 그날 수요가 고정이라 "미래를 아는" forecast의 가치가
> 작다. 매 에피소드 수요를 **Poisson(기록값)** 으로 재샘플하면 *평균은 같고 실현만 다른*
> 확률 환경이 되어, forecast 기반 선제 라우팅의 이점이 드러나는지 검증할 수 있다.
>
> 구현: [src/agents/ours/common/stochastic_env.py](../src/agents/ours/common/stochastic_env.py)
> · 코어: [src/agents/ours/common/dqn_small_core.py](../src/agents/ours/common/dqn_small_core.py)

---

## 1. 핵심 아이디어

| | 결정적 replay (기존) | Poisson 확률 수요 |
|---|---|---|
| 수요 | 그날 기록된 실제 대여/반납을 그대로 재생 | 매 에피소드 `Poisson(기록값)` 재샘플 |
| 그날 일어날 일 | 고정 (한 가지) | 매번 다른 실현(realization) |
| forecast 가치 | 작음 (미래가 확정) | 큼 (분포는 알지만 실현은 불확실) |
| 휴리스틱(반응형) | 안 늦음 | 무작위 실현에 항상 사후 대응 |

→ Poisson은 **평균(기댓값) = 기록값은 유지**하면서 **실현만 확률화**한다. 따라서
forecast(=기댓값)는 그대로 의미를 갖되, 그날 실제로 몇 대가 빠지고 들어올지는
불확실해진다. 이 불확실성 아래에서 "미리 배치할 줄 아는" 정책이 유리해진다.

---

## 2. 구현 — `StochasticRebalanceEnv`

`RebalanceEnv`를 상속한 서브클래스. **원본 env/data_loader는 일절 수정하지 않고**,
`reset()`에서 수요 배열만 Poisson 샘플로 교체한다.

```python
class StochasticRebalanceEnv(RebalanceEnv):
    def __init__(self, *args, demand_noise="poisson", demand_rate_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.demand_noise = demand_noise            # "none" | "poisson"
        self.demand_rate_scale = float(demand_rate_scale)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)   # 원본이 episode·self.data 세팅
        if self.demand_noise != "poisson":
            return obs, info
        # λ(평균) = 기록값 × rate_scale  (음수 방지 clip)
        rate_rent = np.clip(self.data.rentals.astype(float) * self.demand_rate_scale, 0, None)
        rate_ret  = np.clip(self.data.returns.astype(float) * self.demand_rate_scale, 0, None)
        sampled = copy.copy(self.data)              # 공유객체 mutate 방지(얕은 복사)
        sampled.rentals = self._rng.poisson(rate_rent).astype(self.data.rentals.dtype)
        sampled.returns = self._rng.poisson(rate_ret).astype(self.data.returns.dtype)
        self.data = sampled
        return obs, info
```

핵심 포인트
- **λ(평균) = 그 시점·정류소의 기록 대여/반납 수** → 평균적으로 실제 수요와 동일
- **대여·반납을 각각 독립 Poisson**으로 (시간 × 정류소 격자 전체)
- `reset`마다 새로 샘플 → **에피소드마다 다른 수요 실현**
- `self._rng`는 원본 env의 시드 RNG → **seed 고정 시 재현 가능**
- **`demand_rate_scale`**: λ에 곱하는 배율(수요 강도 조절, 기본 1.0; 1.5면 더 붐비는 시나리오)
- **forecast(`agent_demand_forecast` 등)는 건드리지 않음** → 얕은 복사로 동적 속성 보존.
  즉 예측 = 기댓값, 실현 = 확률.

---

## 3. 공정 평가 — paired + 다중 실현

확률 환경은 한 번의 평가로 판단하면 분산이 크다. 그래서:

```python
def _eval_seeds(args, base_seed):
    n = max(args.eval_samples, 1)
    if args.demand_noise != "poisson":
        n = 1                       # 결정적이면 매번 같으니 1회면 충분
    return [base_seed + i for i in range(n)]
```

- **날짜별 `eval_samples`개 Poisson 실현**을 평균내 그 날 점수로 사용
- **model과 heuristic이 같은 seed를 공유** → 같은 수요 실현·같은 트럭 시작을 공유하는
  **paired 비교**(분산↓, 공정). `evaluate()` / `evaluate_heuristic()`가 동일 seed 사용.

---

## 4. 실행 방법 (CLI)

```bash
PYTHONPATH=. python -m src.agents.algorithms.dqn_small.core \
  --district 마포구 --processed-dir data/processed_seoul_all \
  --forecast-path data/forecast_by_gu/demand_forecast_1h_마포구.parquet \
  --max-stations 15 --n-trucks 1 --total-timesteps 400000 \
  --demand-noise poisson --eval-samples 5        # ← 포아송 켜기 (+ 날짜별 5실현 평균)
# 선택: --demand-rate-scale 1.5  (수요 강도 ↑)
```

- `--demand-noise poisson` : 확률 수요 ON (기본 `none` = 결정적)
- `--eval-samples 5` : 평가 시 날짜별 Poisson 실현 5개 평균 (결정적이면 무시됨)
- `--demand-rate-scale` : λ 배율 (기본 1.0)

> ⚠️ **로컬 전용**: 학습·평가·로그 저장은 전부 로컬 CPU + 로컬 parquet으로 동작한다.
> 네트워크(WiFi) 없이도 돌아간다.

---

## 5. 이번 비교 실험 (대표 3구: 결정적 vs Poisson)

전 25구 **결정적 400k**를 완주한 뒤, 대표 3구를 **Poisson 400k**로 추가 실행해
같은 구·같은 학습량에서 결정성↔확률성만 바꿔 비교한다.

| 항목 | 설정 |
|---|---|
| 대표 구 | **강남(큰 구·추월), 마포(추월 기준구), 도봉(near-miss)** |
| 학습량 | 400k step (결정적과 동일) |
| 환경 | top-15 정류소 · 트럭 1 · forecast obs ON |
| Poisson | `--demand-noise poisson --eval-samples 5` |
| 비교 휴리스틱 | `most_imbalanced` (반응형) — 양쪽 동일 |
| 로그 | `logs/runs/small400_poisson/<구>.log`, 요약 `MASTER.log` |

비교 지표: **Δ(best − 휴리스틱)** 을 결정적 400k vs Poisson 400k로 나란히 본다.

### 참고 — 강남 단일 구 사전 ablation 결과(이미 측정)
| 조건 | Δ(best−휴) |
|---|---:|
| 결정적 400k +forecast | **+16.2** |
| poisson 400k +forecast | +10.4 |
| poisson 400k −forecast | +7.9 |

→ **실제 RebalanceEnv에서는 결정적이 오히려 추월폭이 컸고, forecast 기여는 +2.5로 작음.**
즉 "Poisson이면 forecast 이점이 커진다"는 어제 `SmallProblem` 결과이고, 실제 환경에선
효과가 약하다는 게 사전 관찰. 이번 3구 비교로 이 경향이 일반화되는지 확인한다.

---

## 6. 두 가지 "Poisson"의 구분 (혼동 주의)

| | 어제 overtake (`SmallProblem`) | 지금 (`StochasticRebalanceEnv`) |
|---|---|---|
| λ 출처 | **60일 forecast 평균** | **그날 기록 실제값** |
| 환경 | 단순화 env (depot 무한버퍼·재배치량 레버) | 실제 `RebalanceEnv`(트럭 cap·실제 이동) |
| forecast 가치 | 승리폭의 핵심 | +2.5 수준(작음) |

---

## 7. 주의점 (보고서용)

- **Δ의 결정적↔Poisson 직접 비교는 주의**: Poisson은 수요 실현이 바뀌어 휴리스틱
  기준선·절대 보상 스케일이 함께 이동한다(예: 강남 결정적 휴리스틱 −194 vs Poisson −246).
  → 절대값보다 **같은 regime 내 Δ**, 또는 **forecast ON/OFF 같은 조건의 within-regime 대조**가 더 신뢰성 있다.
- **단일 seed**: 본 비교는 구당 1 학습 seed. 견고성 주장을 위해서는 대표 구에 대해
  multi-seed(예: 5 seed) 평균±CI로 보강하는 것이 바람직하다.
- 평가는 seed-42 80/20 분할의 **holdout 7일**에 대해 paired로 수행(표본 작음 — 절대값보다 Δ 중심).
