"""반응형 vs 예측형 휴리스틱 비교 — train.py와 동일한 공정 eval 환경(7일, shaping OFF).

예측형이 휴리스틱(-500)을 넘으면, 그걸 BC로 clone해 천장을 올릴 수 있다.
사용: python scripts/eval_predictive.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baselines import MostImbalancedPolicy, PredictiveImbalancedPolicy  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import EVAL_DATES, _load_yaml, _get  # noqa: E402


def eval_policy(policy, eval_episodes, env_kwargs, n_trucks, seed=42):
    """7일 평균 reward + stockout/full/travel 분해."""
    rewards, stockouts, fulls, travels = [], [], [], []
    for ep in eval_episodes:
        env = RebalanceEnv(ep, n_trucks=n_trucks, **env_kwargs)
        env.reset(seed=seed)
        total, so, fu, tv = 0.0, 0, 0, 0.0
        done = False
        while not done:
            _, r, done, _, info = env.step(policy.act(env))
            total += r
            so += info.get("stockout", 0)
            fu += info.get("full", 0)
            tv += info.get("travel_km", 0.0)
        rewards.append(total); stockouts.append(so); fulls.append(fu); travels.append(tv)
    return (float(np.mean(rewards)), float(np.mean(stockouts)),
            float(np.mean(fulls)), float(np.mean(travels)))


def main() -> None:
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")
    n_trucks = _get(cfg, "truck", "n_trucks", default=3)

    # train.py eval과 동일한 공정 환경: shaping/urgent_bonus OFF
    env_kwargs = dict(
        truck_capacity=_get(cfg, "truck", "capacity", default=20),
        target_fill_ratio=_get(cfg, "truck", "target_fill_ratio", default=0.5),
        urgent_low_ratio=_get(cfg, "env", "urgent_low", default=0.15),
        urgent_high_ratio=_get(cfg, "env", "urgent_high", default=0.85),
        urgent_bonus=0.0, strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
        w_travel_km=_get(cfg, "reward", "travel_km", default=-0.008),
        w_travel_step=_get(cfg, "reward", "travel_step", default=-0.002),
        explore_bonus_scale=0.0, shaping_scale=0.0,
        w_work_per_bike=_get(cfg, "env", "w_work_per_bike", default=0.0),
        w_idle_visit=_get(cfg, "env", "w_idle_visit", default=0.0),
        future_demand_horizon=0,
    )

    print(f"[1/2] loading {len(EVAL_DATES)} eval episodes ({district})...")
    eval_episodes = [load_episode("data/processed", district=district,
                                  episode_start=f"{d} 00:00") for d in EVAL_DATES]

    print(f"\n[2/2] evaluating (7일 평균, 공정 metric)\n")
    print(f"  {'policy':<26}{'reward':>10}{'Δ휴':>9}{'stockout':>10}{'full':>8}{'travel':>9}")
    t0 = time.time()

    base_r, base_so, base_fu, base_tv = eval_policy(
        MostImbalancedPolicy(), eval_episodes, env_kwargs, n_trucks)
    print(f"  {'most_imbalanced (반응형)':<26}{base_r:>10.2f}{0.0:>+9.2f}"
          f"{base_so:>10.1f}{base_fu:>8.1f}{base_tv:>9.1f}")

    for H in [2, 4, 6, 9, 12, 18]:
        r, so, fu, tv = eval_policy(
            PredictiveImbalancedPolicy(horizon=H), eval_episodes, env_kwargs, n_trucks)
        mark = " ✅" if r > base_r else ""
        print(f"  {'predictive H=' + str(H):<26}{r:>10.2f}{r-base_r:>+9.2f}"
              f"{so:>10.1f}{fu:>8.1f}{tv:>9.1f}{mark}")

    print(f"\n  (Δ휴 = 반응형 대비. 양수면 예측형이 더 좋음. {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
