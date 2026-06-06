"""forecast 현실성 체크 — 예측형 휴리스틱의 +117 이득이 oracle 없이도 살아남나?

oracle 예측형: env.data.rentals[t:t+H] = 그 에피소드 *실제* 미래 (완벽 예지, 비현실적).
forecast 예측형: train 60일의 *시간대별 평균 수요* 프로파일로 미래 추정 (eval일 미사용=누수 없음).

전체평균 / 평일·주말 분리 두 forecast를 비교한다.
사용: python scripts/eval_forecast.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baselines import BasePolicy, MostImbalancedPolicy, PredictiveImbalancedPolicy  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402


class ForecastPredictivePolicy(BasePolicy):
    """예측형이되, 미래 수요를 oracle이 아니라 *주어진 평균 프로파일*로 추정.

    profile_rent/ret: (T, N) — 시간대별 평균 수요. env.t로 인덱싱.
    """
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
        future_rent = self.pr[t:t_end].sum(axis=0)
        future_ret = self.pre[t:t_end].sum(axis=0)
        predicted = env.bikes.astype(np.float32) + (future_ret - future_rent)

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


def is_weekend(date_str: str) -> bool:
    return datetime.strptime(date_str, "%Y-%m-%d").weekday() >= 5


def main() -> None:
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")
    n_trucks = _get(cfg, "truck", "n_trucks", default=3)
    env_kwargs = dict(
        truck_capacity=20, target_fill_ratio=0.5, urgent_low_ratio=0.15, urgent_high_ratio=0.85,
        urgent_bonus=0.0, strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
        w_travel_km=-0.008, w_travel_step=-0.002, explore_bonus_scale=0.0, shaping_scale=0.0,
        w_work_per_bike=0.0, w_idle_visit=0.0, future_demand_horizon=0,
    )

    train_dates = TRAIN_DATES[:60]
    print(f"[1/3] train {len(train_dates)}일로 수요 프로파일 구축...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in train_dates]
    rent_all = np.stack([e.rentals for e in tr])   # (D, T, N)
    ret_all = np.stack([e.returns for e in tr])
    wk = np.array([is_weekend(d) for d in train_dates])

    prof = {
        "global": (rent_all.mean(0), ret_all.mean(0)),
        "weekday": (rent_all[~wk].mean(0), ret_all[~wk].mean(0)),
        "weekend": (rent_all[wk].mean(0), ret_all[wk].mean(0)),
    }
    print(f"  평일 {(~wk).sum()}일 / 주말 {wk.sum()}일")

    print(f"[2/3] eval {len(EVAL_DATES)}일 로드...")
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]

    def eval_pol_per_day(make_pol):
        """make_pol(date_str) → policy. 일자별로 (평일/주말) 프로파일 다르게."""
        rs = []
        for ep, d in zip(ev, EVAL_DATES):
            env = RebalanceEnv(ep, n_trucks=n_trucks, **env_kwargs)
            env.reset(seed=42)
            tot, done = 0.0, False
            pol = make_pol(d)
            while not done:
                _, r, done, _, _ = env.step(pol.act(env))
                tot += r
            rs.append(tot)
        return float(np.mean(rs))

    print(f"\n[3/3] 비교 (7일 평균, 공정 metric)\n")
    base = eval_pol_per_day(lambda d: MostImbalancedPolicy())
    print(f"  {'반응형 most_imbalanced':<34}{base:>9.2f}  (기준)")
    orac = eval_pol_per_day(lambda d: PredictiveImbalancedPolicy(horizon=3))
    print(f"  {'oracle 예측형 H=3 (완벽예지)':<34}{orac:>9.2f}  Δ{orac-base:+.1f}")

    print()
    for H in [3, 6]:
        g = eval_pol_per_day(lambda d, H=H: ForecastPredictivePolicy(*prof["global"], horizon=H))
        print(f"  {'forecast(전체평균) H='+str(H):<34}{g:>9.2f}  Δ{g-base:+.1f}  "
              f"(oracle의 {100*(g-base)/(orac-base):.0f}% 회수)")
    for H in [3, 6]:
        ww = eval_pol_per_day(
            lambda d, H=H: ForecastPredictivePolicy(*prof["weekend" if is_weekend(d) else "weekday"], horizon=H))
        print(f"  {'forecast(평일/주말) H='+str(H):<34}{ww:>9.2f}  Δ{ww-base:+.1f}  "
              f"(oracle의 {100*(ww-base)/(orac-base):.0f}% 회수)")

    print(f"\n  → +이득 클수록 현실에서도 통함. oracle 대비 회수율이 핵심.")


if __name__ == "__main__":
    main()
