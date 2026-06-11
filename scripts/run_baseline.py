"""베이스라인 정책 비교 실행기.

여러 날짜로 episode를 돌려 각 정책의 평균 성능을 표로 출력.

사용:
    python scripts/run_baseline.py                          # 1월 첫 7일, noop+most_imbalanced
    python scripts/run_baseline.py --dates 2025-01-15 2025-02-15
    python scripts/run_baseline.py --policies noop random most_imbalanced
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.common.baselines import POLICY_REGISTRY, BasePolicy, get_policy  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402


def run_one_episode(env: RebalanceEnv, policy: BasePolicy, seed: int) -> dict:
    env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = policy.act(env)
        _, r, done, _, info = env.step(action)
        total_reward += r
    return {"total_reward": total_reward, **info}


def run_random(env: RebalanceEnv, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        _, r, done, _, info = env.step(int(rng.integers(env.N)))
        total_reward += r
    return {"total_reward": total_reward, **info}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", default="마포구")
    parser.add_argument(
        "--dates",
        nargs="+",
        default=[
            "2025-01-13", "2025-01-14", "2025-01-15",
            "2025-01-16", "2025-01-17", "2025-01-18", "2025-01-19",
        ],
        help="evaluation episode 시작 일자들",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["noop", "random", "most_imbalanced"],
        help=f"평가할 정책. choices={list(POLICY_REGISTRY) + ['random']}",
    )
    parser.add_argument("--n-trucks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--urgent-low", type=float, default=0.0,
                        help="bikes/capacity ≤ 이 값이면 빈 위급 (트리거)")
    parser.add_argument("--urgent-high", type=float, default=1.0,
                        help="bikes/capacity ≥ 이 값이면 가득 위급 (트리거)")
    args = parser.parse_args()

    # 데이터는 정책마다 같으니 1회 로드
    episodes = []
    for d in args.dates:
        ep = load_episode("data/processed", district=args.district, episode_start=f"{d} 00:00")
        episodes.append((d, ep))

    print(f"{'='*78}")
    print(f"District: {args.district}  Trucks: {args.n_trucks}  Episodes: {len(args.dates)}")
    print(f"{'='*78}")

    rows = []
    for pname in args.policies:
        stockouts, fulls, kms, rewards = [], [], [], []
        for d, ep in episodes:
            env = RebalanceEnv(
                ep, n_trucks=args.n_trucks,
                urgent_low_ratio=args.urgent_low,
                urgent_high_ratio=args.urgent_high,
            )
            if pname == "random":
                result = run_random(env, seed=args.seed)
            else:
                policy = get_policy(pname)
                result = run_one_episode(env, policy, seed=args.seed)
            stockouts.append(result["cum_stockout"])
            fulls.append(result["cum_full"])
            kms.append(result["cum_travel_km"])
            rewards.append(result["total_reward"])
        rows.append({
            "policy": pname,
            "stockout_mean": mean(stockouts),
            "stockout_std": stdev(stockouts) if len(stockouts) > 1 else 0.0,
            "full_mean": mean(fulls),
            "full_std": stdev(fulls) if len(fulls) > 1 else 0.0,
            "km_mean": mean(kms),
            "reward_mean": mean(rewards),
            "reward_std": stdev(rewards) if len(rewards) > 1 else 0.0,
        })

    # 표 출력
    hdr = f"{'policy':<18}{'stockout':>16}{'full':>16}{'travel_km':>12}{'reward':>18}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['policy']:<18}"
            f"{r['stockout_mean']:>9.1f} ± {r['stockout_std']:<4.1f}"
            f"{r['full_mean']:>9.1f} ± {r['full_std']:<4.1f}"
            f"{r['km_mean']:>12.1f}"
            f"{r['reward_mean']:>11.1f} ± {r['reward_std']:<4.1f}"
        )

    # baseline 대비 개선율 (noop이 있으면)
    noop_row = next((r for r in rows if r["policy"] == "noop"), None)
    if noop_row:
        print()
        print("vs NO-OP (음수 = 개선, 양수 = 악화):")
        for r in rows:
            if r["policy"] == "noop":
                continue
            d_stk = r["stockout_mean"] - noop_row["stockout_mean"]
            d_full = r["full_mean"] - noop_row["full_mean"]
            d_rwd = r["reward_mean"] - noop_row["reward_mean"]
            print(f"  {r['policy']:<18} Δstockout={d_stk:+.1f}  Δfull={d_full:+.1f}  Δreward={d_rwd:+.1f}")


if __name__ == "__main__":
    main()
