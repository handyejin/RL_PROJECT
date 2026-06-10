"""저장된 dqn_small 모델 vs 휴리스틱의 reward를 stockout/full/travel로 분해 진단.

total reward = w_stockout·stockout + w_full·full + w_km·travel_km + w_step·travel_step
"동률/열세"가 어느 비용 항목에서 나오는지(특히 stockout은 이겼는데 travel에 가려진
건지)를 가린다.

기존 env/data_loader 무수정. dqn_small_core의 환경 구성 함수를 그대로 재사용해
학습 때와 동일한 축소 환경(top-N 정류소·트럭 수·demand-noise)에서 평가한다.

사용 예:
    PYTHONPATH=. python scripts/diagnose_reward_breakdown.py \\
        --district 마포구 --processed-dir data/processed_seoul_all \\
        --forecast-path data/forecast_by_gu/demand_forecast_1h_마포구.parquet \\
        --model-path logs/dqn_interactive_dqn_small_마포구/best_model \\
        --max-stations 15 --n-trucks 1 --demand-noise poisson --eval-samples 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.common.baselines import get_policy  # noqa: E402
from src.agents.models.masked_dqn import MaskableDQN  # noqa: E402
from src.agents.common.data_overrides import (  # noqa: E402
    apply_capacity_override,
    attach_forecast_override,
)
from src.agents.algorithms.dqn_small import core as core  # noqa: E402


METRIC_KEYS = ["cum_stockout", "cum_full", "cum_travel_km", "cum_travel_steps"]


def run_policy(episodes, args, env_kw, seeds, *, model=None) -> dict[str, float]:
    """모델(있으면) 또는 휴리스틱으로 전체 평가셋을 돌려 지표 평균을 낸다."""
    heuristic = None if model is not None else get_policy("most_imbalanced")
    agg = {k: [] for k in METRIC_KEYS + ["reward"]}
    for ep in episodes:
        per_sample = {k: [] for k in METRIC_KEYS + ["reward"]}
        for s in seeds:
            if model is not None:
                env = core.make_env(ep, args, env_kw, seed=s, for_eval=True)
            else:
                env = core.build_raw_env(ep, args, env_kw, seed=s)
            obs, info = env.reset(seed=s)
            done = False
            total = 0.0
            while not done:
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
                    action = int(action)
                else:
                    action = int(heuristic.act(env))
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                done = terminated or truncated
            for k in METRIC_KEYS:
                per_sample[k].append(float(info.get(k, 0.0)))
            per_sample["reward"].append(total)
        for k in per_sample:
            agg[k].append(float(np.mean(per_sample[k])))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="reward breakdown diagnosis for dqn_small")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--processed-dir", default="data/processed_seoul_all")
    parser.add_argument("--forecast-path", default="")
    parser.add_argument("--capacity-path", default="data/processed/station_capacity.csv")
    parser.add_argument("--capacity-initial-fill-ratio", type=float, default=0.5)
    parser.add_argument("--model-path", required=True, help="MaskableDQN .zip 경로(확장자 생략 가능)")
    parser.add_argument("--n-train-dates", type=int, default=200)
    parser.add_argument("--station-ids", nargs="*", default=None,
                        help="학습 때 출력된 '선택 정류소 id' 그대로 주면 선택 재계산 생략(정확·고속).")
    parser.add_argument("--max-stations", type=int, default=15)
    parser.add_argument("--split-mode", default="",
                        help="학습과 동일 분할 재현(예: chronological). 비우면 모듈 기본 80/20.")
    parser.add_argument("--n-trucks", type=int, default=1)
    parser.add_argument("--demand-noise", choices=["none", "poisson"], default="none")
    parser.add_argument("--demand-rate-scale", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-mode", default="forecast_projected_travel")
    parser.add_argument("--future-horizon", type=int, default=6)
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--candidate-mode", default="forecast_imbalance")
    parser.add_argument("--agent-shaping-scale", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env_kw = core.build_env_kw(args)

    # 학습과 동일한 날짜 분할을 재현한다. --split-mode 가 주어지면 compute_split 으로
    # train pool / eval(73일)을 학습과 똑같이 잡는다(모델이 학습한 정류소셋·평가창 일치).
    if args.split_mode:
        from src.agents.common.date_split import compute_split
        train_dates, eval_dates = compute_split(args.split_mode, seed=args.seed)
        print(f"split-mode={args.split_mode}: train pool {len(train_dates)}일, eval {len(eval_dates)}일 "
              f"({eval_dates[0]} ~ {eval_dates[-1]})")
    else:
        train_dates, eval_dates = core.TRAIN_DATES, core.EVAL_DATES

    eval_episodes = core.load_episodes(eval_dates, args.district, args.processed_dir)
    full_n = eval_episodes[0].n_stations

    if args.station_ids:
        # 학습 때 고른 정류소 id를 그대로 사용 → 선택 재계산/200일 로딩 불필요(정확·고속).
        id_to_idx = {sid: i for i, sid in enumerate(eval_episodes[0].station_ids)}
        missing = [s for s in args.station_ids if s not in id_to_idx]
        if missing:
            raise SystemExit(f"station_ids not found in {args.district}: {missing}")
        sel = np.array([id_to_idx[s] for s in args.station_ids], dtype=int)
        eval_episodes = [core.subset_episode(ep, sel) for ep in eval_episodes]
        print(f"환경 축소: {full_n} → {len(sel)} 정류소 (--station-ids 지정)")
    elif args.max_stations and 0 < args.max_stations < full_n:
        # fallback: 선택 재계산(학습과 동일 조건이어야 일치) — 200일 로딩 필요.
        train_episodes = core.load_episodes(train_dates[: args.n_train_dates], args.district, args.processed_dir)
        sel = core.select_commute_imbalance_stations(train_episodes, args.max_stations)
        eval_episodes = [core.subset_episode(ep, sel) for ep in eval_episodes]
        print(f"환경 축소: {full_n} → {len(sel)} 정류소 (선택 재계산)")

    apply_capacity_override(eval_episodes, args.capacity_path, args.capacity_initial_fill_ratio)
    attach_forecast_override(eval_episodes, args.forecast_path)

    seeds = core._eval_seeds(args, args.seed)
    mode = "Poisson 확률" if args.demand_noise == "poisson" else "결정적 replay"
    print(f"district={args.district}, 트럭={args.n_trucks}, 수요={mode}, eval_samples={len(seeds)}")

    model = MaskableDQN.load(args.model_path, device=args.device)

    heur = run_policy(eval_episodes, args, env_kw, seeds, model=None)
    dqn = run_policy(eval_episodes, args, env_kw, seeds, model=model)

    # 비용 가중치는 env 기본값과 동일.
    w = dict(stockout=-1.0, full=-0.8, km=env_kw["w_travel_km"], step=env_kw["w_travel_step"])

    def cost_rows(m: dict[str, float]) -> dict[str, float]:
        return {
            "stockout(건)": m["cum_stockout"],
            "full(건)": m["cum_full"],
            "travel_km": m["cum_travel_km"],
            "travel_step": m["cum_travel_steps"],
            "── 비용 환산 ──": float("nan"),
            "stockout_cost": w["stockout"] * m["cum_stockout"],
            "full_cost": w["full"] * m["cum_full"],
            "travel_cost": w["km"] * m["cum_travel_km"] + w["step"] * m["cum_travel_steps"],
            "reward(합)": m["reward"],
        }

    hr, dr = cost_rows(heur), cost_rows(dqn)
    print(f"\n{'항목':18}{'휴리스틱':>14}{'DQN':>14}{'Δ(DQN−휴)':>14}")
    print("-" * 60)
    for k in hr:
        if k.startswith("──"):
            print(k)
            continue
        h, d = hr[k], dr[k]
        better = ""
        if "cost" in k or "reward" in k:
            better = "  ✅" if d > h else ("  ❌" if d < h else "")
        print(f"{k:18}{h:>14.2f}{d:>14.2f}{d - h:>14.2f}{better}")

    print("\n해석 가이드:")
    print("  - stockout_cost Δ가 +면 DQN이 미충족수요를 줄였다는 뜻(옛 실험의 핵심 지표).")
    print("  - 그런데 reward(합) Δ가 −면 travel_cost 등 다른 항목에서 그만큼 더 썼다는 뜻.")


if __name__ == "__main__":
    main()
