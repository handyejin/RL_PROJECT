"""우리 REINFORCE/A2C 모델의 episode replay JSON을 생성한다.

이 파일은 팀원용 DQN replay exporter를 수정하지 않고, 우리가 추가한
forecast state와 Top-K action wrapper를 그대로 적용한 별도 exporter다.

생성된 JSON은 ``docs/ours_replay_viewer.html``에서 열 수 있다.

실행 예:
    PYTHONPATH=. python -m src.agents.export_replay \
        --algorithm reinforce --district 강남구 --date 2025-03-25 \
        --checkpoint logs/reinforce_interactive_reinforce_강남구/reinforce_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Categorical


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.algorithms.a2c import core as a2c_core  # noqa: E402
from src.agents.algorithms.reinforce import core as reinforce_core  # noqa: E402
from src.agents.common.candidate_actions import CandidateTopKActionWrapper  # noqa: E402
from src.agents.common.data_overrides import apply_capacity_override, attach_forecast_override  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402


def _project_path(value: str | Path) -> Path:
    """상대 경로는 프로젝트 루트 기준으로 변환한다."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_forecast_path(district: str) -> Path:
    """구별 forecast parquet의 기본 위치를 반환한다."""
    return PROJECT_ROOT / "data" / "forecast_by_gu" / f"demand_forecast_1h_{district}.parquet"


def _build_env_args(args: argparse.Namespace) -> SimpleNamespace:
    """학습 때 사용한 wrapper 옵션을 make_env가 읽을 수 있는 형태로 정리한다."""
    return SimpleNamespace(
        future_mode=args.future_mode,
        future_horizon=args.future_horizon,
        history_profile=None,
        candidate_top_k=args.candidate_top_k,
        candidate_mode=args.candidate_mode,
        candidate_travel_coef=args.candidate_travel_coef,
        candidate_zone_mode=args.candidate_zone_mode,
        candidate_zone_count=args.candidate_zone_count,
        candidate_zone_penalty=args.candidate_zone_penalty,
        candidate_feature_mode=args.candidate_feature_mode,
        agent_shaping_mode="projected_imbalance",
        agent_shaping_scale=0.0,
        agent_shaping_gamma=0.99,
    )


def _make_env(ep, args: argparse.Namespace):
    """알고리즘에 맞는 core의 make_env를 호출한다."""
    env_args = _build_env_args(args)
    if args.algorithm == "a2c":
        return a2c_core.make_env(ep, env_args, seed=args.seed, for_eval=True)
    return reinforce_core.make_env(ep, env_args, seed=args.seed, for_eval=True)


def _policy_class(algorithm: str):
    """알고리즘 이름에 맞는 policy class를 반환한다."""
    if algorithm == "a2c":
        return a2c_core.PolicyNetwork
    return reinforce_core.PolicyNetwork


def _load_policy(args: argparse.Namespace, obs_dim: int, action_dim: int, device: torch.device):
    """저장된 checkpoint에서 policy network만 복원한다."""
    policy_cls = _policy_class(args.algorithm)
    policy = policy_cls(
        obs_dim,
        action_dim,
        hidden_layer_size=args.hidden,
        residual_policy=args.residual_policy,
        residual_temp=args.residual_temp,
    ).to(device)
    checkpoint = torch.load(_project_path(args.checkpoint), map_location=device)
    if "policy" not in checkpoint:
        raise ValueError(f"checkpoint does not contain policy weights: {args.checkpoint}")
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    return policy


def _truck_snapshot(env, total_steps: list[int]) -> list[dict]:
    """viewer가 읽기 쉬운 트럭 상태 목록을 만든다."""
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


def _candidate_station_ids(env, mask: np.ndarray) -> np.ndarray:
    """Top-K wrapper가 있으면 rank별 station index를, 없으면 전체 station index를 반환한다."""
    if isinstance(env, CandidateTopKActionWrapper):
        return env.candidate_station_ids()
    return np.arange(len(mask), dtype=np.int64)


def _policy_distribution(policy, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> tuple[int, np.ndarray, np.ndarray]:
    """현재 상태에서 greedy action rank, rank별 확률, rank별 logit을 계산한다."""
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logits = policy(obs_t, mask_t).squeeze(0)
        dist = Categorical(logits=logits)
        probs = dist.probs.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
    action_rank = int(np.argmax(logits_np))
    return action_rank, probs.astype(float), logits_np.astype(float)


def _station_policy_values(n_stations: int, candidate_stations: np.ndarray, probs: np.ndarray) -> list[float | None]:
    """rank별 확률을 station index 기준 배열로 펼친다."""
    values: list[float | None] = [None] * n_stations
    for rank, station_idx in enumerate(candidate_stations):
        if rank < len(probs):
            values[int(station_idx)] = float(probs[rank])
    return values


def _candidate_rows(ep, candidate_stations: np.ndarray, probs: np.ndarray, logits: np.ndarray, valid: np.ndarray) -> list[dict]:
    """viewer side panel에 표시할 Top-K 후보 목록을 만든다."""
    rows = []
    for rank, station_idx in enumerate(candidate_stations):
        station_idx = int(station_idx)
        if rank >= len(valid) or not bool(valid[rank]):
            continue
        rows.append(
            {
                "rank": rank,
                "station_index": station_idx,
                "station_id": str(ep.station_ids[station_idx]),
                "prob": float(probs[rank]),
                "logit": float(logits[rank]),
            }
        )
    return rows


def _snapshot(
    *,
    env,
    ep,
    obs: np.ndarray,
    policy,
    device: torch.device,
    rl_step: int,
    total_steps_track: list[int],
    reward: float,
    cum_reward: float,
    action_rank: int | None,
    station_action: int | None,
    actor_truck: int | None,
) -> dict:
    """현재 env 상태를 하나의 replay frame으로 저장한다."""
    mask = env.action_masks()
    greedy_rank, probs, logits = _policy_distribution(policy, obs, mask, device)
    candidate_stations = _candidate_station_ids(env, mask)
    if action_rank is None:
        action_rank = greedy_rank
    if station_action is None:
        station_action = int(candidate_stations[action_rank])

    station_mask = np.zeros(ep.n_stations, dtype=bool)
    for rank, station_idx in enumerate(candidate_stations):
        if rank < len(mask) and bool(mask[rank]):
            station_mask[int(station_idx)] = True

    return {
        "rl_step": int(rl_step),
        "t": int(env.t),
        "current_truck": int(env.current_truck),
        "bikes": env.bikes.astype(int).tolist(),
        "trucks": _truck_snapshot(env, total_steps_track),
        "action": int(station_action) if station_action is not None else None,
        "action_rank": int(action_rank) if action_rank is not None else None,
        "action_name": str(ep.station_ids[station_action]) if station_action is not None else None,
        "actor_truck": int(actor_truck) if actor_truck is not None else None,
        "reward": float(reward),
        "cum_reward": float(cum_reward),
        "cum_stockout": int(env.cum_stockout),
        "cum_full": int(env.cum_full),
        "cum_km": float(env.cum_travel_km),
        "policy_values": _station_policy_values(ep.n_stations, candidate_stations, probs),
        "q_values": _station_policy_values(ep.n_stations, candidate_stations, probs),
        "mask": station_mask.tolist(),
        "candidate_station_indices": [int(x) for x in candidate_stations],
        "candidate_rows": _candidate_rows(ep, candidate_stations, probs, logits, mask),
    }


def parse_args() -> argparse.Namespace:
    """CLI 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="Export ours REINFORCE/A2C replay JSON")
    parser.add_argument("--algorithm", choices=["reinforce", "a2c"], default="reinforce")
    parser.add_argument("--district", default="강남구")
    parser.add_argument("--date", default="2025-03-25")
    parser.add_argument("--processed-dir", default="data/processed_seoul_all")
    parser.add_argument("--capacity-path", default="data/processed/station_capacity.csv")
    parser.add_argument("--forecast-path", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--future-mode", default="forecast_projected_travel")
    parser.add_argument("--future-horizon", type=int, default=6)
    parser.add_argument("--candidate-top-k", type=int, default=12)
    parser.add_argument("--candidate-mode", choices=["imbalance", "forecast_imbalance"], default="forecast_imbalance")
    parser.add_argument("--candidate-travel-coef", type=float, default=0.20)
    parser.add_argument("--candidate-zone-mode", choices=["none", "static3"], default="static3")
    parser.add_argument("--candidate-zone-count", type=int, default=3)
    parser.add_argument("--candidate-zone-penalty", type=float, default=1.0)
    parser.add_argument("--candidate-feature-mode", choices=["none", "basic"], default="basic")
    parser.add_argument("--capacity-initial-fill-ratio", type=float, default=0.5)
    parser.add_argument("--residual-policy", action="store_true")
    parser.add_argument("--residual-temp", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """episode 로드, 모델 실행, replay JSON 저장을 순서대로 수행한다."""
    args = parse_args()
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu")
    if args.device in {"cpu", "mps"}:
        device = torch.device(args.device)

    processed_dir = _project_path(args.processed_dir)
    capacity_path = str(_project_path(args.capacity_path)) if args.capacity_path else ""
    forecast_path = str(_project_path(args.forecast_path)) if args.forecast_path else str(_default_forecast_path(args.district))
    out_path = (
        _project_path(args.out)
        if args.out
        else PROJECT_ROOT / "docs" / f"{args.district}_{args.algorithm.upper()}_{args.date}.json"
    )

    print(f"[1/4] loading episode: {args.district} {args.date}", flush=True)
    ep = load_episode(str(processed_dir), district=args.district, episode_start=f"{args.date} 00:00")
    apply_capacity_override([ep], capacity_path, initial_fill_ratio=args.capacity_initial_fill_ratio)
    attach_forecast_override([ep], forecast_path)
    print(f"      stations={ep.n_stations}, steps={ep.n_steps}, forecast={Path(forecast_path).name}", flush=True)

    print("[2/4] building env", flush=True)
    env = _make_env(ep, args)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    print(f"[3/4] loading policy: {args.checkpoint}", flush=True)
    policy = _load_policy(args, obs_dim, action_dim, device)

    snapshots: list[dict] = []
    total_steps_track = [0] * env.n_trucks
    rl_step = 0
    cum_reward = 0.0
    done = False

    snapshots.append(
        _snapshot(
            env=env,
            ep=ep,
            obs=obs,
            policy=policy,
            device=device,
            rl_step=rl_step,
            total_steps_track=total_steps_track,
            reward=0.0,
            cum_reward=0.0,
            action_rank=None,
            station_action=None,
            actor_truck=None,
        )
    )

    print("[4/4] rolling episode", flush=True)
    while not done:
        mask = env.action_masks()
        action_rank, _, _ = _policy_distribution(policy, obs, mask, device)
        candidate_stations = _candidate_station_ids(env, mask)
        station_action = int(candidate_stations[action_rank])
        actor_truck = int(env.current_truck)

        obs, reward, terminated, truncated, _ = env.step(action_rank)
        done = bool(terminated or truncated)
        cum_reward += float(reward)
        rl_step += 1

        total_steps_track[actor_truck] = int(env.trucks[actor_truck].remaining_steps)
        for i, tr in enumerate(env.trucks):
            if tr.remaining_steps == 0:
                total_steps_track[i] = 0

        snapshots.append(
            _snapshot(
                env=env,
                ep=ep,
                obs=obs,
                policy=policy,
                device=device,
                rl_step=rl_step,
                total_steps_track=total_steps_track,
                reward=float(reward),
                cum_reward=cum_reward,
                action_rank=action_rank,
                station_action=station_action,
                actor_truck=actor_truck,
            )
        )

        if rl_step % 50 == 0:
            print(
                f"      step={rl_step:3d} t={env.t:3d} reward={cum_reward:8.1f} "
                f"stockout={env.cum_stockout} full={env.cum_full}",
                flush=True,
            )

    output = {
        "meta": {
            "viewer_kind": "ours_policy_replay",
            "district": args.district,
            "date": args.date,
            "model": str(_project_path(args.checkpoint)),
            "algo": args.algorithm,
            "state": args.future_mode,
            "candidate_top_k": int(args.candidate_top_k),
            "candidate_mode": args.candidate_mode,
            "n_trucks": int(env.n_trucks),
            "n_stations": int(ep.n_stations),
            "station_ids": [str(x) for x in ep.station_ids],
            "station_coords": ep.station_coords.tolist(),
            "T_max": int(env.T),
            "step_minutes": 10,
            "rl_steps": int(rl_step),
            "total_reward": float(cum_reward),
            "total_stockout": int(env.cum_stockout),
            "total_full": int(env.cum_full),
            "total_km": float(env.cum_travel_km),
            "truck_capacity": int(env.truck_capacity),
            "station_capacities": ep.capacity.astype(int).tolist(),
            "policy_values_label": "policy probability over Top-K candidates",
        },
        "snapshots": snapshots,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    print("\n=== replay export result ===")
    print(f"  output:       {out_path} ({size_kb:.1f} KB)")
    print(f"  rl_steps:     {rl_step}")
    print(f"  total_reward: {cum_reward:.1f}")
    print(f"  stockout:     {env.cum_stockout}")
    print(f"  full:         {env.cum_full}")
    print(f"  km:           {env.cum_travel_km:.1f}")


if __name__ == "__main__":
    main()
