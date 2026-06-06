"""저장된 BC 모델을 7일 eval에서 greedy(masked) 실행 → 평균 reward.

future_demand_horizon은 학습 때와 같아야 obs_dim이 맞는다.
사용: python scripts/eval_bc_policy.py --model logs/bc_<tag>/bc_model.zip --future-demand-horizon 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.save_util import load_from_zip_file  # noqa: E402
from src.agents.masked_dqn import MaskableDQN  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import EVAL_DATES, _load_yaml, _get  # noqa: E402


def main() -> None:
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--future-demand-horizon", type=int, default=3)
    ap.add_argument("--district", default=_get(cfg, "district", default="마포구"))
    args = ap.parse_args()

    n_trucks = _get(cfg, "truck", "n_trucks", default=3)
    ek = dict(truck_capacity=20, target_fill_ratio=0.5, urgent_low_ratio=0.15, urgent_high_ratio=0.85,
              urgent_bonus=0.0, strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
              w_travel_km=-0.008, w_travel_step=-0.002, explore_bonus_scale=0.0, shaping_scale=0.0,
              w_work_per_bike=0.0, w_idle_visit=0.0,
              future_demand_horizon=args.future_demand_horizon, use_action_mask=True)
    net_arch = list(_get(cfg, "dqn", "net_arch", default=[256, 256]))

    ev = [load_episode("data/processed", district=args.district, episode_start=f"{d} 00:00") for d in EVAL_DATES]

    # BC q_net 로드 — dummy env로 MaskableDQN 만든 뒤 weight 주입
    dummy = RebalanceEnv(ev[0], n_trucks=n_trucks, **ek)
    model = MaskableDQN("MlpPolicy", dummy, policy_kwargs={"net_arch": net_arch}, verbose=0)
    _, params, _ = load_from_zip_file(args.model)
    state = params["policy"]
    qs = {k[len("q_net."):]: v for k, v in state.items() if k.startswith("q_net.")}
    model.policy.q_net.load_state_dict(qs)
    model.policy.q_net_target.load_state_dict(qs)
    model.policy.set_training_mode(False)

    rewards = []
    for ep in ev:
        env = RebalanceEnv(ep, n_trucks=n_trucks, **ek)
        obs, _ = env.reset(seed=42)
        tot, done = 0.0, False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, r, done, _, _ = env.step(int(action))
            tot += r
        rewards.append(tot)
    mean_r = float(np.mean(rewards))
    print(f"\n  BC clone 정책 7일 평균 reward = {mean_r:.2f}")
    print(f"    (반응형 휴리스틱 -500.02 | oracle 예측형 H=3 -382.79)")
    print(f"    Δ반응형 = {mean_r - (-500.02):+.1f}")


if __name__ == "__main__":
    main()
