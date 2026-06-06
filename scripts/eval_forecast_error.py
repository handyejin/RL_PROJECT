"""예측오차 보정 예측형 평가 — 서영현(2020) 논문 아이디어 ① 적용.

forecast 예측형(과거평균만, -459)과 oracle 예측형(실제미래, -383) 사이 격차를,
"오늘 관측이 forecast에서 벗어난 정도(예측오차)"를 실시간 보정해 배포 가능하게 좁힌다.

비교: 반응형 / oracle 예측형(상한) / forecast 예측형(기존) / **예측오차 보정**(W·mode·focus 스윕)
사용: python scripts/eval_forecast_error.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baselines import (  # noqa: E402
    MostImbalancedPolicy, PredictiveImbalancedPolicy, ForecastErrorPolicy,
)
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.eval_forecast2 import ForecastPredictivePolicy  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402


def main() -> None:
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")
    n_trucks = _get(cfg, "truck", "n_trucks", default=3)
    ek = dict(truck_capacity=20, target_fill_ratio=0.5, urgent_low_ratio=0.15, urgent_high_ratio=0.85,
              urgent_bonus=0.0, strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
              w_travel_km=-0.008, w_travel_step=-0.002, explore_bonus_scale=0.0, shaping_scale=0.0,
              w_work_per_bike=0.0, w_idle_visit=0.0, future_demand_horizon=0)

    print(f"[1/3] 전체 {len(TRAIN_DATES)}일 로드 + forecast 프로파일(전역평균)...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in TRAIN_DATES]
    gr = np.stack([e.rentals for e in tr]).mean(0).astype(np.float32)
    ge = np.stack([e.returns for e in tr]).mean(0).astype(np.float32)

    print(f"[2/3] eval {len(EVAL_DATES)}일 로드...")
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]

    def run(make_pol):
        rs = []
        for ep in ev:
            env = RebalanceEnv(ep, n_trucks=n_trucks, **ek)
            env.reset(seed=42)
            tot, done = 0.0, False
            pol = make_pol()
            while not done:
                _, r, done, _, _ = env.step(pol.act(env))
                tot += r
            rs.append(tot)
        return float(np.mean(rs))

    print(f"\n[3/3] 비교 (7일 평균, 공정 metric)\n")
    base = run(lambda: MostImbalancedPolicy())
    orac = run(lambda: PredictiveImbalancedPolicy(horizon=3))
    fc = run(lambda: ForecastPredictivePolicy(gr, ge, horizon=3))
    print(f"  {'반응형 (most_imbalanced)':<34}{base:>9.2f}")
    print(f"  {'oracle 예측형 H=3 (상한)':<34}{orac:>9.2f}  Δ{orac-base:+.1f}")
    print(f"  {'forecast 예측형 H=3 (기존 배포)':<34}{fc:>9.2f}  Δ{fc-base:+.1f}  (oracle {100*(fc-base)/(orac-base):.0f}%)")
    print()

    span = orac - fc  # 보정으로 메우려는 격차 (forecast→oracle)
    def rec(label, make_pol):
        val = run(make_pol)
        gain = val - fc
        frac = 100 * gain / span if abs(span) > 1e-6 else 0.0
        mark = "  ✅넘음" if val > fc else ""
        print(f"  {label:<34}{val:>9.2f}  Δfc{gain:+.1f}  (격차의 {frac:+.0f}%){mark}")

    print("  ── 예측오차 보정 (drift 가산) ──")
    for W in [3, 6, 12, 24]:
        rec(f"drift W={W} H=3", lambda W=W: ForecastErrorPolicy(gr, ge, horizon=3, window=W, mode="drift"))
    print("  ── 예측오차 보정 (scale 승산) ──")
    for W in [6, 12, 24]:
        rec(f"scale W={W} H=3", lambda W=W: ForecastErrorPolicy(gr, ge, horizon=3, window=W, mode="scale"))
    print("  ── 보정 강도 α 축소 (drift W=24, 노이즈 vs 신호 판별) ──")
    for a in [0.2, 0.5]:
        rec(f"drift W=24 α={a}", lambda a=a: ForecastErrorPolicy(gr, ge, horizon=3, window=24, mode="drift", alpha=a))
    print("  ── + 예측오차 focus (상위 K 정류소만 탐색) ──")
    for K in [30, 50]:
        rec(f"drift W=12 focus K={K}",
            lambda K=K: ForecastErrorPolicy(gr, ge, horizon=3, window=12, mode="drift", error_focus=True, focus_k=K))

    print(f"\n  (forecast 예측형 {fc:.2f} 대비 ✅ 넘으면 배포 가능한 개선)")


if __name__ == "__main__":
    main()
