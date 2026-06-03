"""7일 eval set에서 모델 vs 휴리스틱 날짜별 reward 비교 (fair eval 환경).

사용: PYTHONPATH=. python scripts/eval_7day.py --model logs/.../best/best_model.zip
"""
import argparse
import numpy as np

from src.envs.data_loader import load_episode
from src.envs.rebalance_env import RebalanceEnv
from src.agents.baselines import get_policy
from src.agents.masked_dqn import MaskableDQN
from scripts.train import EVAL_DATES

ENV_KW = dict(
    n_trucks=3, truck_capacity=20, target_fill_ratio=0.5,
    urgent_low_ratio=0.15, urgent_high_ratio=0.85,
    urgent_bonus=0.0, strict_urgent_mask=True,
    w_travel_km=-0.008, w_travel_step=-0.002,
    explore_bonus_scale=0.0, shaping_scale=0.0, future_demand_horizon=0,
)


def roll_heuristic(ep):
    e = RebalanceEnv(ep, **ENV_KW)
    e.reset(seed=42)
    pol = get_policy("most_imbalanced")
    tot, done = 0.0, False
    while not done:
        _, r, done, _, _ = e.step(pol.act(e))
        tot += r
    return tot


def roll_model(ep, model):
    e = RebalanceEnv(ep, **ENV_KW)
    model.set_env(e)
    obs, _ = e.reset(seed=42)
    tot, done = 0.0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True, action_masks=e.action_masks())
        obs, r, done, _, _ = e.step(int(a))
        tot += r
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    model = MaskableDQN.load(args.model)
    print(f"=== {args.label} vs 휴리스틱 (7일) ===")
    print(f"{'날짜':12}{'휴리스틱':>10}{'모델':>10}{'Δ(M-휴)':>9}")
    hs, rs = [], []
    for d in EVAL_DATES:
        ep = load_episode("data/processed", district="마포구", episode_start=f"{d} 00:00")
        h = roll_heuristic(ep)
        r = roll_model(ep, model)
        hs.append(h); rs.append(r)
        mark = "✅" if r > h else ""
        print(f"{d:12}{h:>10.1f}{r:>10.1f}{r - h:>9.1f} {mark}")
    wins = sum(1 for h, r in zip(hs, rs) if r > h)
    print(f"{'평균':12}{np.mean(hs):>10.1f}{np.mean(rs):>10.1f}"
          f"{np.mean(rs) - np.mean(hs):>9.1f}  ({wins}/7 추월)")


if __name__ == "__main__":
    main()
