"""forecast 정교화 — 요일(dow)·공휴일·날씨 조건부로 oracle 회수율↑.

forecast 모델은 RL의 60일 제약과 무관(데모/롤아웃이 아니라 수요 이력) → 전체 292일 사용.
키 후보: global / 평일·주말 / 요일(7) / 요일+공휴일(공휴일→일요일 취급).
사용: python scripts/eval_forecast2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baselines import BasePolicy, MostImbalancedPolicy, PredictiveImbalancedPolicy  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402


class ForecastPredictivePolicy(BasePolicy):
    name = "forecast_predictive"

    def __init__(self, profile_rent, profile_ret, horizon: int = 3):
        self.pr = profile_rent.astype(np.float32)
        self.pre = profile_ret.astype(np.float32)
        self.horizon = int(horizon)

    def act(self, env: RebalanceEnv) -> int:
        truck = env.trucks[env.current_truck]
        target = env.data.capacity.astype(np.float32) * env.target_fill_ratio
        t = env.t
        T = self.pr.shape[0]
        t_end = min(t + self.horizon, T)
        fr = self.pr[t:t_end].sum(axis=0)
        fe = self.pre[t:t_end].sum(axis=0)
        predicted = env.bikes.astype(np.float32) + (fe - fr)
        if truck.load == 0:
            scores = predicted - target
        elif truck.load >= env.truck_capacity:
            scores = target - predicted
        else:
            scores = np.abs(predicted - target).astype(np.float32)
        scores = scores.copy()
        scores[truck.location] = -np.inf
        for i, other in enumerate(env.trucks):
            if i == env.current_truck:
                continue
            if not other.is_idle:
                scores[other.destination] = -np.inf
        best = int(np.argmax(scores))
        return best if np.isfinite(scores[best]) else int(truck.location)


def daykey_dow(ep):
    return ep.dayofweek                       # 0=Mon .. 6=Sun

def daykey_dowhol(ep):
    return 6 if (ep.is_holiday or ep.dayofweek >= 5) else ep.dayofweek  # 공휴일→일요일군


def main() -> None:
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")
    n_trucks = _get(cfg, "truck", "n_trucks", default=3)
    ek = dict(truck_capacity=20, target_fill_ratio=0.5, urgent_low_ratio=0.15, urgent_high_ratio=0.85,
              urgent_bonus=0.0, strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
              w_travel_km=-0.008, w_travel_step=-0.002, explore_bonus_scale=0.0, shaping_scale=0.0,
              w_work_per_bike=0.0, w_idle_visit=0.0, future_demand_horizon=0)

    print(f"[1/3] 전체 {len(TRAIN_DATES)}일 로드 + 프로파일 구축...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in TRAIN_DATES]

    def build_profiles(keyfn):
        """key→(avg_rent, avg_ret). key별 평균."""
        buckets = {}
        for ep in tr:
            buckets.setdefault(keyfn(ep), []).append(ep)
        return {k: (np.stack([e.rentals for e in v]).mean(0),
                    np.stack([e.returns for e in v]).mean(0)) for k, v in buckets.items()}

    glob = (np.stack([e.rentals for e in tr]).mean(0), np.stack([e.returns for e in tr]).mean(0))
    prof_dow = build_profiles(daykey_dow)
    prof_dowhol = build_profiles(daykey_dowhol)
    print(f"  요일별 일수: {{dow:len}} = " + ", ".join(f"{k}:{len(v)}" for k, v in
          sorted({k: [e for e in tr if daykey_dow(e) == k] for k in range(7)}.items())))

    print(f"[2/3] eval {len(EVAL_DATES)}일 로드...")
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]

    def run(make_pol):
        rs = []
        for ep in ev:
            env = RebalanceEnv(ep, n_trucks=n_trucks, **ek)
            env.reset(seed=42)
            tot, done = 0.0, False
            pol = make_pol(ep)
            while not done:
                _, r, done, _, _ = env.step(pol.act(env))
                tot += r
            rs.append(tot)
        return float(np.mean(rs))

    print(f"\n[3/3] 비교 (7일 평균, 공정 metric)\n")
    base = run(lambda ep: MostImbalancedPolicy())
    orac = run(lambda ep: PredictiveImbalancedPolicy(horizon=3))
    print(f"  {'반응형':<28}{base:>9.2f}")
    print(f"  {'oracle 예측형 H=3':<28}{orac:>9.2f}  Δ{orac-base:+.1f}  (상한)")
    print()

    def rec(label, val):
        print(f"  {label:<28}{val:>9.2f}  Δ{val-base:+.1f}  (oracle {100*(val-base)/(orac-base):.0f}%)")

    for H in [3, 6]:
        rec(f"전체평균 H={H}", run(lambda ep, H=H: ForecastPredictivePolicy(*glob, horizon=H)))
    for H in [3, 6]:
        rec(f"요일별(7) H={H}", run(lambda ep, H=H: ForecastPredictivePolicy(*prof_dow[daykey_dow(ep)], horizon=H)))
    for H in [3, 6]:
        rec(f"요일+공휴일 H={H}", run(lambda ep, H=H: ForecastPredictivePolicy(*prof_dowhol[daykey_dowhol(ep)], horizon=H)))


if __name__ == "__main__":
    main()
