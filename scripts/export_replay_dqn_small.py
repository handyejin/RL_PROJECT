"""dqn_small 축소환경(top-N 정류소·소수 트럭)에서 학습된 MaskableDQN replay JSON 생성.

`scripts/export_replay.py` 는 전체 환경(N=수백)을 가정하지만,
``src.agents.algorithms.dqn_small.core`` 로 학습된 가중치는 출퇴근 압력 top-N
정류소로 축소한 부분 환경에서 학습됐기 때문에 observation 차원이 다르다.

이 스크립트는 dqn_small 학습 시점과 동일한 절차로 환경을 재구성한다.

  1. train 풀(`chronological`, seed=42, 처음 ``n_train_dates`` 일) 로딩
  2. ``select_commute_imbalance_stations`` 로 top-N 정류소 선택
  3. 평가 episode를 ``subset_episode`` 로 동일하게 슬라이스
  4. ``RebalanceEnv`` + ``CandidateTopKActionWrapper`` 등 ours wrapper 체인 구성
  5. MaskableDQN으로 deterministic 1 episode 굴리고 snapshot JSON 저장

실행 예:

    PYTHONPATH=. python scripts/export_replay_dqn_small.py \\
      --model logs/dqn_seed42_k12_dqn_small_강남구/best_model.zip \\
      --district 강남구 --date 2025-10-20 \\
      --out docs/replay_dqn_topk12_강남구_2025-10-20.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.algorithms.dqn_small.core import (  # noqa: E402
    BASE_ENV_KW,
    select_commute_imbalance_stations,
    subset_episode,
)
from src.agents.common.candidate_actions import maybe_wrap_candidate_actions  # noqa: E402
from src.agents.common.data_overrides import (  # noqa: E402
    apply_capacity_override,
    attach_forecast_override,
)
from src.agents.common.date_split import compute_split  # noqa: E402
from src.agents.common.future_demand import maybe_wrap_future_demand  # noqa: E402
from src.agents.models.masked_dqn import MaskableDQN  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402


def truck_snapshot(env: RebalanceEnv, total_steps: list[int]) -> list[dict]:
    """현재 트럭들의 위치·적재·잔여 step·이번 trip 총 step을 dict로 직렬화."""
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


def compute_q_values(model: MaskableDQN, obs: np.ndarray) -> list[float] | None:
    """현재 obs에 대한 Q값. 추출 실패 시 None."""
    try:
        import torch

        with torch.no_grad():
            obs_tensor, _ = model.q_net.obs_to_tensor(obs)
            q = model.q_net(obs_tensor).cpu().numpy().flatten()
        return q.tolist()
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] Q-value extract failed: {exc}")
        return None


def build_candidate_rows(
    q_values: list[float] | None,
    mask: np.ndarray,
    cand_to_station: list[int],
    station_ids: list[str],
    top_k: int = 12,
) -> tuple[list[dict], list[int]]:
    """Q 값을 softmax 확률로 바꿔 후보 정류소를 rank순으로 정렬한다.

    DQN은 stochastic policy가 아니지만 viewer는 ``prob`` 컬럼을 표시하므로
    masked Q에 softmax를 씌워 시각화용 확률을 만든다.

    Args:
        q_values: 현재 action space 길이의 Q 값. None이면 균등 분포로 fallback.
        mask: 길이 동일한 bool mask. True인 action만 후보로 친다.
        cand_to_station: action index → 실제 정류소 idx 매핑. candidate wrapper가
            없는 환경에서는 ``[0, 1, ..., N-1]`` 과 동일하다.
        station_ids: 정류소 id 리스트(부분 환경이면 subset).
        top_k: 표시할 최대 후보 수.

    Returns:
        ``(candidate_rows, candidate_station_indices)``.
          - ``candidate_rows``: viewer의 Top-K 표를 채우는 dict 리스트.
          - ``candidate_station_indices``: 지도에 파란 테두리로 강조할 정류소 idx 리스트.
    """
    n_actions = mask.shape[0]
    if q_values is None or len(q_values) != n_actions:
        q_arr = np.zeros(n_actions, dtype=np.float64)
    else:
        q_arr = np.asarray(q_values, dtype=np.float64)

    # masked softmax — 마스킹된 action은 확률 0으로 만든다.
    logits = np.where(mask, q_arr, -np.inf)
    if not np.isfinite(logits).any():
        # 모든 action이 막힌 비상 케이스: 균등 분포로 폴백.
        probs = np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
    else:
        shifted = logits - np.nanmax(logits[np.isfinite(logits)])
        exp = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
        total = exp.sum()
        probs = exp / total if total > 0 else np.zeros_like(exp)

    # mask=True인 action만 후보로 채우고 prob 내림차순으로 정렬.
    candidate_actions = np.flatnonzero(mask)
    candidate_actions = sorted(
        candidate_actions.tolist(), key=lambda a: -float(probs[a])
    )[: max(top_k, 1)]

    rows: list[dict] = []
    station_indices: list[int] = []
    for rank, action in enumerate(candidate_actions):
        station_idx = cand_to_station[action] if action < len(cand_to_station) else int(action)
        station_id = (
            station_ids[station_idx]
            if 0 <= station_idx < len(station_ids)
            else str(station_idx)
        )
        rows.append({
            "rank": int(rank),
            "station_index": int(station_idx),
            "station_id": station_id,
            "prob": float(probs[action]),
            "logit": float(q_arr[action]),
        })
        station_indices.append(int(station_idx))
    return rows, station_indices


def build_env_args(args: argparse.Namespace) -> SimpleNamespace:
    """dqn_small의 ``make_env`` 와 동일한 wrapper 체인을 만들기 위한 args 객체."""
    return SimpleNamespace(
        # candidate_actions wrapper
        candidate_top_k=args.candidate_top_k,
        candidate_mode=args.candidate_mode,
        candidate_travel_coef=args.candidate_travel_coef,
        candidate_zone_mode=args.candidate_zone_mode,
        candidate_zone_count=3,
        candidate_zone_penalty=args.candidate_zone_penalty,
        candidate_feature_mode=args.candidate_feature_mode,
        # future demand wrapper
        future_mode=args.future_mode,
        future_horizon=args.future_horizon,
        # VAE wrapper (off)
        vae_mode="none",
        vae_latent_path="",
        vae_latent_dim=4,
    )


def main() -> None:
    """학습 시 설정을 재구성하고 1 episode replay를 JSON으로 export."""
    ap = argparse.ArgumentParser(description="Export DQN replay on dqn_small reduced env")
    ap.add_argument("--model", required=True,
                    help="dqn_small에서 학습한 MaskableDQN best/final zip 경로")
    ap.add_argument("--district", required=True)
    ap.add_argument("--date", required=True, help="평가할 날짜(YYYY-MM-DD)")
    ap.add_argument("--processed-dir", default="data/processed_seoul_all")
    ap.add_argument("--out", default="docs/replay_dqn_small.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-q", action="store_true")
    # 축소 환경 옵션 — dqn_small 학습 시 설정과 일치시켜야 함
    ap.add_argument("--max-stations", type=int, default=15,
                    help="dqn_small --max-stations와 동일하게.")
    ap.add_argument("--n-trucks", type=int, default=1,
                    help="dqn_small --n-trucks와 동일하게.")
    ap.add_argument("--n-train-dates", type=int, default=200,
                    help="station 선택에 사용할 train 풀 크기 (학습 시와 동일하게).")
    ap.add_argument("--split-mode", choices=["random", "chronological"],
                    default="chronological",
                    help="config/ours/dqn_topk12.yaml과 동일한 chronological 기본.")
    # 보고서 기준 forecast/capacity 경로
    ap.add_argument("--capacity-path", default="data/processed/station_capacity.csv")
    ap.add_argument("--capacity-initial-fill-ratio", type=float, default=0.5)
    ap.add_argument("--forecast-path", default="")
    # Candidate Top-K (보고서 기준 dqn_topk12.yaml 값)
    ap.add_argument("--candidate-top-k", type=int, default=12)
    ap.add_argument("--candidate-mode",
                    choices=["imbalance", "forecast_imbalance"],
                    default="forecast_imbalance")
    ap.add_argument("--candidate-travel-coef", type=float, default=0.20)
    ap.add_argument("--candidate-zone-mode", choices=["none", "static3"], default="static3")
    ap.add_argument("--candidate-zone-penalty", type=float, default=1.0)
    ap.add_argument("--candidate-feature-mode", choices=["none", "basic"], default="basic")
    # Future demand (보고서 기준)
    ap.add_argument("--future-mode", default="forecast_projected_travel")
    ap.add_argument("--future-horizon", type=int, default=6)
    args = ap.parse_args()

    # 학습 시와 같은 chronological split으로 train 풀과 eval 풀을 만든 뒤,
    # train 풀에서 top-N 출퇴근 압력 정류소를 선택한다.
    forecast_path = (args.forecast_path or
                     f"data/forecast_by_gu/demand_forecast_1h_{args.district}.parquet")

    print(f"[1/5] split = {args.split_mode}, train dates를 station 선택에 사용")
    train_dates_all, _ = compute_split(args.split_mode, seed=args.seed)
    train_dates = train_dates_all[: args.n_train_dates]
    train_episodes = [
        load_episode(args.processed_dir, district=args.district,
                     episode_start=f"{d} 00:00")
        for d in train_dates
    ]
    print(f"      train pool = {len(train_episodes)}일, full N = {train_episodes[0].n_stations}")

    print(f"[2/5] target episode: {args.district} @ {args.date}")
    target_episode = load_episode(
        args.processed_dir, district=args.district,
        episode_start=f"{args.date} 00:00",
    )

    # capacity / forecast override는 학습 시와 동일하게 train + target episode 모두에 적용
    all_episodes = train_episodes + [target_episode]
    apply_capacity_override(all_episodes, args.capacity_path, args.capacity_initial_fill_ratio)
    if Path(forecast_path).exists():
        attach_forecast_override(all_episodes, forecast_path)
        print(f"      forecast attached: {forecast_path}")
    else:
        print(f"      [warn] forecast not found, skip: {forecast_path}")

    print(f"[3/5] subset to top-{args.max_stations} commute-imbalance stations")
    full_n = target_episode.n_stations
    if 0 < args.max_stations < full_n:
        sel = select_commute_imbalance_stations(train_episodes, args.max_stations)
        target_episode = subset_episode(target_episode, sel)
        print(f"      {full_n} → {target_episode.n_stations} stations, "
              f"ids = {target_episode.station_ids}")
    else:
        print(f"      축소 없음: 전체 {full_n} stations 사용")

    print(f"[4/5] build env + wrapper chain (n_trucks={args.n_trucks}, "
          f"top_k={args.candidate_top_k})")
    env_kw = dict(BASE_ENV_KW, n_trucks=args.n_trucks)
    env = RebalanceEnv(target_episode, seed=args.seed, **env_kw)
    wrap_args = build_env_args(args)
    env = maybe_wrap_future_demand(env, wrap_args)
    env = maybe_wrap_candidate_actions(env, wrap_args)
    print(f"      obs_dim = {env.observation_space.shape[0]}, "
          f"action_space = {env.action_space}")

    print(f"[5/5] loading model: {args.model}")
    model = MaskableDQN.load(args.model, env=env)
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"obs space mismatch — model expects {model.observation_space.shape}, "
            f"env produces {env.observation_space.shape}. "
            f"학습 시 --max-stations / --n-trucks / --candidate-* / --future-* 설정을 "
            f"인자로 동일하게 맞춰주세요."
        )

    obs, _ = env.reset(seed=args.seed)
    rl_step = 0
    total_steps_track = [0] * env_kw["n_trucks"]
    cum_reward = 0.0
    snapshots: list[dict] = []

    # 학습된 candidate wrapper는 action을 rank(0..K-1)로 받고 내부에서 station idx로 변환한다.
    # rank → station idx 역매핑을 snapshot에 같이 남겨 두면 viewer가 해석할 수 있다.
    def current_rank_to_station() -> list[int]:
        # CandidateTopKActionWrapper는 현재 환경에 _last_candidates를 노출하지 않을 수도 있어
        # action_masks 길이로 안전하게 대응한다.
        cands = getattr(env, "_last_candidates", None)
        if cands is None:
            return list(range(env.action_space.n))
        return [int(c) for c in cands]

    raw_env: RebalanceEnv = env.unwrapped  # type: ignore[assignment]

    init_mask = env.action_masks()
    init_q = None if args.no_q else compute_q_values(model, obs)
    init_rows, init_cand_idx = build_candidate_rows(
        init_q, init_mask, current_rank_to_station(),
        list(target_episode.station_ids), top_k=args.candidate_top_k or env.action_space.n,
    )
    snapshots.append({
        "rl_step": 0,
        "t": int(raw_env.t),
        "current_truck": int(raw_env.current_truck),
        "bikes": raw_env.bikes.tolist(),
        "trucks": truck_snapshot(raw_env, total_steps_track),
        "action": None,
        "action_rank": None,
        "action_name": None,
        "reward": 0.0,
        "cum_reward": 0.0,
        "cum_stockout": int(raw_env.cum_stockout),
        "cum_full": int(raw_env.cum_full),
        "cum_km": float(raw_env.cum_travel_km),
        "q_values": init_q,
        "policy_values": [row["prob"] for row in init_rows],
        "mask": init_mask.tolist(),
        "candidates": current_rank_to_station(),
        "candidate_rows": init_rows,
        "candidate_station_indices": init_cand_idx,
    })

    done = False
    while not done:
        mask = env.action_masks()
        q_vals = None if args.no_q else compute_q_values(model, obs)
        cands = current_rank_to_station()
        actor_idx = int(raw_env.current_truck)

        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        action = int(action)
        station_idx = cands[action] if action < len(cands) else action

        obs, reward, done, _, _ = env.step(action)
        cum_reward += float(reward)
        rl_step += 1

        total_steps_track[actor_idx] = raw_env.trucks[actor_idx].remaining_steps
        for i, tr in enumerate(raw_env.trucks):
            if tr.remaining_steps == 0:
                total_steps_track[i] = 0

        cand_rows, cand_indices = build_candidate_rows(
            q_vals, mask, cands,
            list(target_episode.station_ids),
            top_k=args.candidate_top_k or env.action_space.n,
        )
        # 선택된 action이 후보 표에서 몇 번째 rank인지 표시.
        action_rank: int | None = None
        for row in cand_rows:
            if row["station_index"] == int(station_idx):
                action_rank = int(row["rank"])
                break

        snapshots.append({
            "rl_step": rl_step,
            "t": int(raw_env.t),
            "current_truck": int(raw_env.current_truck),
            "bikes": raw_env.bikes.tolist(),
            "trucks": truck_snapshot(raw_env, total_steps_track),
            "action": action,                          # action_space index (rank 또는 station idx)
            "action_rank": action_rank,                # 후보 표 안에서의 순위
            "station_idx": int(station_idx),           # 실제 정류소 idx
            "action_name": target_episode.station_ids[station_idx]
                if station_idx < len(target_episode.station_ids) else str(station_idx),
            "actor_truck": actor_idx,
            "reward": float(reward),
            "cum_reward": float(cum_reward),
            "cum_stockout": int(raw_env.cum_stockout),
            "cum_full": int(raw_env.cum_full),
            "cum_km": float(raw_env.cum_travel_km),
            "q_values": q_vals,
            "policy_values": [row["prob"] for row in cand_rows],
            "mask": mask.tolist(),
            "candidates": cands,
            "candidate_rows": cand_rows,
            "candidate_station_indices": cand_indices,
        })

        if rl_step % 50 == 0:
            print(f"      rl_step={rl_step}, t={raw_env.t}, reward={cum_reward:.2f}, "
                  f"stockout={raw_env.cum_stockout}, full={raw_env.cum_full}")

    # viewer 헤더 라벨: "<district> · <date> · <ALGO> · <state> · Top-K <K>"
    state_label = (
        f"dqn_small N={target_episode.n_stations}·n_trucks={env_kw['n_trucks']}"
        f"·future={args.future_mode}"
    )

    output = {
        "meta": {
            "district": args.district,
            "date": args.date,
            "model": args.model,
            "algo": "DQN",
            "viewer_kind": "dqn_small",
            "env_kind": "dqn_small",
            "state": state_label,
            "n_trucks": int(env_kw["n_trucks"]),
            "max_stations": int(args.max_stations),
            "n_stations": int(target_episode.n_stations),
            "station_ids": list(target_episode.station_ids),
            "station_coords": target_episode.station_coords.tolist(),
            "T_max": int(raw_env.T),
            "step_minutes": 10,
            "rl_steps": rl_step,
            "total_reward": float(cum_reward),
            "total_stockout": int(raw_env.cum_stockout),
            "total_full": int(raw_env.cum_full),
            "total_km": float(raw_env.cum_travel_km),
            "truck_capacity": int(raw_env.truck_capacity),
            "station_capacities": target_episode.capacity.astype(int).tolist(),
            # viewer 헤더 표시용 — 실제 action space 크기를 그대로 노출한다.
            "candidate_top_k": int(env.action_space.n),
            "candidate_mode": args.candidate_mode,
            "policy_values_label": "softmax(Q)",
            "future_mode": args.future_mode,
            "future_horizon": int(args.future_horizon),
        },
        "snapshots": snapshots,
    }

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)

    size_kb = out_path.stat().st_size / 1024
    print("\n=== 결과 ===")
    print(f"  rl_steps:       {rl_step}")
    print(f"  total_reward:   {cum_reward:.2f}")
    print(f"  cum_stockout:   {raw_env.cum_stockout}")
    print(f"  cum_full:       {raw_env.cum_full}")
    print(f"  cum_km:         {raw_env.cum_travel_km:.1f}")
    print(f"  output:         {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
