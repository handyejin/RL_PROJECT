"""RebalanceEnv 동작 검증.

두 가지 정책으로 1 episode를 돌려서 환경이 정상 동작하는지 확인:
  1. NO-OP: 트럭이 자기 위치에 계속 머무름 (재배치 없음) - 재배치 자체가 의미 있는지 확인함.
  2. RANDOM: 매번 무작위 정류소 선택 - 환경이 무작위 행동에서 심각하게 망가지지 않는지 확인함.

각 정책의 누적 stockout/full/reward를 출력. NO-OP의 stockout/full이 RANDOM 대비
유의미하게 크지 않으면(랜덤이 더 나쁠 수도 있음) reward 신호 정상.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402


def run_policy(env: RebalanceEnv, policy: str, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        if policy == "noop":
            action = env.trucks[env.current_truck].location
        elif policy == "random":
            action = int(rng.integers(env.N))
        else:
            raise ValueError(policy)
        obs, r, done, truncated, info = env.step(action)
        total_reward += r
        steps += 1
        if steps > 50_000:
            raise RuntimeError("step limit exceeded — infinite loop?")

    return {
        "policy": policy,
        "decisions": steps,
        "total_reward": round(total_reward, 2),
        **info,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--date", default="2025-01-15")
    parser.add_argument("--n-trucks", type=int, default=3)
    args = parser.parse_args()

    print(f"=== Loading episode: {args.district} @ {args.date} ===")
    ep = load_episode(
        "data/processed",
        district=args.district,
        episode_start=f"{args.date} 00:00",
    )
    print(f"  stations={ep.n_stations}, steps={ep.n_steps}, "
          f"total_rentals={ep.rentals.sum()}, total_returns={ep.returns.sum()}")

    for policy in ["noop", "random"]:
        env = RebalanceEnv(ep, n_trucks=args.n_trucks)
        result = run_policy(env, policy)
        print(f"\n=== {policy.upper()} ===")
        for k, v in result.items():
            print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
