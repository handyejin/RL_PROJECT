"""학습된 모델을 'shaping 제거된 환경'에서 평가 — 공정한 비교.

학습 시 사용된 urgent_bonus + explore_bonus는 학습 도우미일 뿐 실제 운영 metric이 아님.
평가에서 이를 0으로 두고 측정 → 휴리스틱과 진짜 정책 품질 비교.

사용:
    python scripts/eval_pure.py                                    # 1년치 obs171 모델 모두
    python scripts/eval_pure.py --models year_v2 open_v1           # 특정 모델만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN  # noqa: E402

from src.agents.baselines import get_policy  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402

# Pure 평가 환경 — shaping 모두 OFF
PURE_ENV_KWARGS = dict(
    urgent_low_ratio=0.15, urgent_high_ratio=0.85,
    urgent_bonus=0.0,            # shaping OFF
    explore_bonus_scale=0.0,     # shaping OFF
    strict_urgent_mask=False,
    w_travel_km=-0.008,
    w_travel_step=-0.002,
)

# 평가 셋 — 두 가지 계절 분포
EVAL_WINTER = [
    "2025-01-13", "2025-01-14", "2025-01-15",
    "2025-01-16", "2025-01-17", "2025-01-18", "2025-01-19",
]
EVAL_SUMMER = [
    "2025-03-25", "2025-04-18", "2025-05-17",
    "2025-07-01", "2025-07-06", "2025-07-09", "2025-08-21",
]

# 평가 대상 모델 (obs_dim 171 — 캘린더+날씨 포함)
DEFAULT_MODELS = [
    ("year_v1",  "masked_dqn", "logs/masked_dqn_mapo_year_v1/best/best_model.zip"),
    ("year_v2",  "masked_dqn", "logs/masked_dqn_mapo_year_v2/best/best_model.zip"),
]


def _load_model(algo: str, path: str, env: RebalanceEnv):
    if algo == "masked_dqn":
        from src.agents.masked_dqn import MaskableDQN
        return MaskableDQN.load(path, env=env)
    return DQN.load(path, env=env)


def eval_model(name: str, algo: str, model_path: str, episodes: list, seed: int = 42) -> dict:
    rewards, stocks, fulls, kms = [], [], [], []
    env0 = RebalanceEnv(episodes[0], n_trucks=3, **PURE_ENV_KWARGS)
    try:
        model = _load_model(algo, model_path, env0)
    except Exception as e:
        return {"name": name, "error": str(e)}

    for ep in episodes:
        env = RebalanceEnv(ep, n_trucks=3, **PURE_ENV_KWARGS)
        obs, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            if algo == "masked_dqn":
                mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(int(action))
            total += r
        rewards.append(total)
        stocks.append(info["cum_stockout"])
        fulls.append(info["cum_full"])
        kms.append(info["cum_travel_km"])

    return {
        "name": name,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "stockout_mean": np.mean(stocks),
        "full_mean": np.mean(fulls),
        "km_mean": np.mean(kms),
    }


def eval_heuristic(episodes: list, name: str = "most_imbalanced", seed: int = 42) -> dict:
    policy = get_policy(name)
    rewards, stocks, fulls, kms = [], [], [], []
    for ep in episodes:
        env = RebalanceEnv(ep, n_trucks=3, **PURE_ENV_KWARGS)
        env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            _, r, done, _, info = env.step(policy.act(env))
            total += r
        rewards.append(total)
        stocks.append(info["cum_stockout"])
        fulls.append(info["cum_full"])
        kms.append(info["cum_travel_km"])
    return {
        "name": f"휴리스틱 ({name})",
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "stockout_mean": np.mean(stocks),
        "full_mean": np.mean(fulls),
        "km_mean": np.mean(kms),
    }


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'═'*78}")
    print(f" {title}")
    print('═'*78)
    print(f"{'정책':<22}{'reward':>13}{'stockout':>12}{'full':>10}{'km':>10}")
    print('-'*78)
    for r in rows:
        if r.get("error"):
            print(f"{r['name']:<22}  ERROR: {r['error'][:50]}")
            continue
        print(f"{r['name']:<22}"
              f"{r['reward_mean']:>8.1f} ± {r['reward_std']:<4.1f}"
              f"{r['stockout_mean']:>12.1f}"
              f"{r['full_mean']:>10.1f}"
              f"{r['km_mean']:>10.1f}")

    # 휴리스틱 대비 Δ
    h = next((r for r in rows if r["name"].startswith("휴리스틱")), None)
    if h:
        print('-'*78)
        print(f"  vs 휴리스틱 (Δreward, 음수=DQN이 부족):")
        for r in rows:
            if r.get("error") or r["name"].startswith("휴리스틱"):
                continue
            d = r["reward_mean"] - h["reward_mean"]
            ds = r["stockout_mean"] - h["stockout_mean"]
            df = r["full_mean"] - h["full_mean"]
            marker = "✅" if d > 0 else "❌"
            print(f"    {r['name']:<20} Δreward={d:+.1f}  "
                  f"Δstockout={ds:+.1f}  Δfull={df:+.1f}  {marker}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="평가할 모델 이름 (year_v1, year_v2 등). 미지정 시 default 전체")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--district", default="마포구")
    ap.add_argument("--seasons", nargs="*", default=["winter", "summer"],
                    choices=["winter", "summer"])
    args = ap.parse_args()

    models = DEFAULT_MODELS
    if args.models:
        models = [m for m in DEFAULT_MODELS if m[0] in args.models]
        if not models:
            print(f"⚠ no models matched: {args.models}")
            return

    season_sets = {
        "winter": ("겨울 (1/13~1/19) — open_v1이 학습한 환경", EVAL_WINTER),
        "summer": ("봄·여름 (3~8월 7일) — year_v1/v2가 평가받은 환경", EVAL_SUMMER),
    }

    for season_key in args.seasons:
        title, dates = season_sets[season_key]
        print(f"\n[loading {season_key} eval set: {len(dates)} days]")
        episodes = [load_episode("data/processed", district=args.district,
                                 episode_start=f"{d} 00:00") for d in dates]

        rows = [eval_heuristic(episodes, seed=args.seed)]
        for name, algo, path in models:
            if not Path(path).exists():
                rows.append({"name": name, "error": f"model not found: {path}"})
                continue
            print(f"  evaluating {name}...")
            rows.append(eval_model(name, algo, path, episodes, seed=args.seed))

        print_table(rows, f"순수 metric (shaping OFF) | {title}")

    print("\n" + "═"*78)
    print(" 해석")
    print("═"*78)
    print(" - reward = -stockout - 0.8×full - 0.008×km - 0.002×travel_steps")
    print(" - 모두 음수가 정상 (페널티만 누적)")
    print(" - 휴리스틱과 가까울수록 좋음, 초과하면 ✅")


if __name__ == "__main__":
    main()
