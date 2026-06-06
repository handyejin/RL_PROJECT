"""학습된 모델로 1 episode를 돌려 replay JSON 생성.

HTML training_flow.html이 읽어서 트럭 이동을 재생할 수 있게 함.

사용:
    python scripts/export_replay.py                                                   # 기본값
    python scripts/export_replay.py --model logs/dqn_long/best/best_model.zip
    python scripts/export_replay.py --algo masked_dqn --model logs/masked_dqn_xx/best/best_model.zip
    python scripts/export_replay.py --date 2025-01-17 --out docs/replay_017.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN  # noqa: E402

from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402


def truck_snapshot(env: RebalanceEnv, total_steps: list[int]) -> list[dict]:
    return [
        {
            "loc": int(tr.location),
            "dest": int(tr.destination),
            "load": int(tr.load),
            "remaining": int(tr.remaining_steps),
            "total_steps": int(total_steps[i]),
        }
        for i, tr in enumerate(env.trucks)
    ]


def compute_q_values(model, obs) -> list[float] | None:
    """현재 obs에 대한 Q값 (없으면 None)."""
    if model is None:  # heuristic 등 Q-net 없는 정책
        return None
    try:
        import torch
        with torch.no_grad():
            if hasattr(model, "quantile_net"):  # QRDQN: 분위수 평균 = 기대 Q
                obs_tensor, _ = model.quantile_net.obs_to_tensor(obs)
                q = model.quantile_net(obs_tensor).mean(dim=1).cpu().numpy().flatten()
            else:
                obs_tensor, _ = model.q_net.obs_to_tensor(obs)
                q = model.q_net(obs_tensor).cpu().numpy().flatten()
        return q.tolist()
    except Exception as e:
        print(f"  [warn] Q-value extract failed: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logs/dqn_long/best/best_model.zip",
                    help="zip 경로 (예: logs/dqn_long/best/best_model.zip, dqn_final.zip)")
    ap.add_argument("--algo", choices=["dqn", "masked_dqn", "qrdqn", "heuristic"], default="dqn")
    ap.add_argument("--district", default="마포구")
    ap.add_argument("--date", default="2025-01-15")
    ap.add_argument("--n-trucks", type=int, default=3)
    ap.add_argument("--out", default="docs/replay.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-q", action="store_true", help="Q-value 추출 비활성")
    ap.add_argument("--urgent-low", type=float, default=0.0)
    ap.add_argument("--urgent-high", type=float, default=1.0)
    ap.add_argument("--strict-mask", action="store_true",
                    help="학습 환경과 일치시키기 위한 strict_urgent_mask 적용")
    ap.add_argument("--w-travel-km", type=float, default=-0.01,
                    help="학습 시 사용한 km 비용 (기본 -0.01, config에선 -0.008)")
    ap.add_argument("--w-travel-step", type=float, default=-0.005,
                    help="학습 시 사용한 step 비용 (기본 -0.005, config에선 -0.002)")
    ap.add_argument("--target-fill-ratio", type=float, default=0.5,
                    help="정류소 채움 목표 (학습 시 사용한 값)")
    ap.add_argument("--w-work", type=float, default=0.0,
                    help="적재/하차 1대당 양수 reward (학습 시 사용한 값)")
    ap.add_argument("--w-idle", type=float, default=0.0,
                    help="허탕 방문 페널티 (학습 시 사용한 값)")
    ap.add_argument("--future-demand-horizon", type=int, default=0,
                    help="학습 시 사용한 미래 demand obs horizon")
    args = ap.parse_args()

    print(f"[1/3] loading episode: {args.district} @ {args.date}")
    ep = load_episode(
        "data/processed",
        district=args.district,
        episode_start=f"{args.date} 00:00",
    )
    print(f"      stations={ep.n_stations}, steps={ep.n_steps}, "
          f"total_rentals={ep.rentals.sum()}, total_returns={ep.returns.sum()}")

    env = RebalanceEnv(
        ep, n_trucks=args.n_trucks,
        target_fill_ratio=args.target_fill_ratio,
        urgent_low_ratio=args.urgent_low,
        urgent_high_ratio=args.urgent_high,
        strict_urgent_mask=args.strict_mask,
        w_travel_km=args.w_travel_km,
        w_travel_step=args.w_travel_step,
        w_work_per_bike=args.w_work,
        w_idle_visit=args.w_idle,
        future_demand_horizon=args.future_demand_horizon,
    )

    print(f"\n[2/3] loading model: {args.model} (algo={args.algo})")
    policy = None
    model = None
    if args.algo == "heuristic":
        from src.agents.baselines import get_policy
        policy = get_policy("most_imbalanced")
        print("      heuristic = most_imbalanced (모델 로드 없음)")
    elif args.algo == "masked_dqn":
        from src.agents.masked_dqn import MaskableDQN
        model = MaskableDQN.load(args.model, env=env)
    elif args.algo == "qrdqn":
        from src.agents.masked_qrdqn import MaskableQRDQN
        model = MaskableQRDQN.load(args.model, env=env)
    else:
        model = DQN.load(args.model, env=env)

    obs, _ = env.reset(seed=args.seed)
    rl_step = 0
    total_steps_track = [0] * env.n_trucks  # 각 트럭이 현재 trip을 출발했을 때의 remaining

    cum_reward = 0.0
    snapshots: list[dict] = []

    # 초기 상태
    snapshots.append({
        "rl_step": 0,
        "t": int(env.t),
        "current_truck": int(env.current_truck),
        "bikes": env.bikes.tolist(),
        "trucks": truck_snapshot(env, total_steps_track),
        "action": None,
        "action_name": None,
        "reward": 0.0,
        "cum_reward": 0.0,
        "cum_stockout": int(env.cum_stockout),
        "cum_full": int(env.cum_full),
        "cum_km": float(env.cum_travel_km),
        "q_values": None if args.no_q else compute_q_values(model, obs),
        "mask": env.action_masks().tolist(),
    })

    print(f"\n[3/3] rolling episode...")
    done = False
    while not done:
        mask = env.action_masks()
        q_vals = None if args.no_q else compute_q_values(model, obs)

        if args.algo == "heuristic":
            action = policy.act(env)
        elif args.algo in ("masked_dqn", "qrdqn"):
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        else:
            action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # action 적용 전: 결정 차례 트럭은 idle
        cur_idx = env.current_truck

        obs, reward, done, _, _ = env.step(action)
        cum_reward += float(reward)
        rl_step += 1

        # action을 받은 트럭의 totalSteps 갱신 (방금 출발했거나 stay)
        total_steps_track[cur_idx] = env.trucks[cur_idx].remaining_steps
        # 시계 진행 도중 도착해서 idle 된 트럭은 0으로 리셋
        for i, tr in enumerate(env.trucks):
            if tr.remaining_steps == 0:
                total_steps_track[i] = 0

        snapshots.append({
            "rl_step": rl_step,
            "t": int(env.t),
            "current_truck": int(env.current_truck),
            "bikes": env.bikes.tolist(),
            "trucks": truck_snapshot(env, total_steps_track),
            "action": action,
            "action_name": ep.station_ids[action] if action < len(ep.station_ids) else str(action),
            "actor_truck": int(cur_idx),
            "reward": float(reward),
            "cum_reward": float(cum_reward),
            "cum_stockout": int(env.cum_stockout),
            "cum_full": int(env.cum_full),
            "cum_km": float(env.cum_travel_km),
            "q_values": q_vals,
            "mask": mask.tolist(),
        })

        if rl_step % 50 == 0:
            print(f"      rl_step={rl_step}, t={env.t}, reward={cum_reward:.2f}, "
                  f"stockout={env.cum_stockout}, full={env.cum_full}")

    # 메타 데이터
    output = {
        "meta": {
            "district": args.district,
            "date": args.date,
            "model": args.model,
            "algo": args.algo,
            "n_trucks": args.n_trucks,
            "n_stations": ep.n_stations,
            "station_ids": ep.station_ids,
            "station_coords": ep.station_coords.tolist(),
            "T_max": int(env.T),
            "step_minutes": 10,
            "rl_steps": rl_step,
            "total_reward": float(cum_reward),
            "total_stockout": int(env.cum_stockout),
            "total_full": int(env.cum_full),
            "total_km": float(env.cum_travel_km),
            "truck_capacity": int(env.truck_capacity),
            "station_capacities": ep.capacity.astype(int).tolist(),
        },
        "snapshots": snapshots,
    }

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n=== 결과 ===")
    print(f"  rl_steps:       {rl_step}")
    print(f"  total_reward:   {cum_reward:.2f}")
    print(f"  cum_stockout:   {env.cum_stockout}")
    print(f"  cum_full:       {env.cum_full}")
    print(f"  cum_km:         {env.cum_travel_km:.1f}")
    print(f"  output:         {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
