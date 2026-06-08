"""A2C/REINFORCE Best-Worst 구 seed 반복 실험을 실행하고 요약한다.

목적:
    과제 요구사항의 "random seed 변경 및 신뢰구간" 항목을 보강하기 위해
    서울 전체는 seed 42로 학습한 기존 결과를 대표값으로 두고,
    그 결과에서 Best 3 / Worst 3 구만 seed 123/777을 추가 실행한다.

실험 범위:
    - A2C: Best 3구 + Worst 3구
    - REINFORCE: Best 3구 + Worst 3구
    - base seed: 42
    - additional seed: 123, 777

주의:
    팀원 공통 환경/DQN 코드는 수정하지 않는다. 이 스크립트는 우리 agent core를
    subprocess로 호출하고, 결과 history.npy를 읽어 CSV/MD 요약만 만든다.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_SEED = 42
DEFAULT_ADDITIONAL_SEEDS = [123, 777]
EXPERIMENTS = {
    "a2c": {
        "module": "src.agents.ours.algorithms.a2c.core",
        "log_prefix": "actor_critic",
        "best": ["강서구", "강남구", "노원구"],
        "worst": ["관악구", "은평구", "서대문구"],
    },
    "reinforce": {
        "module": "src.agents.ours.algorithms.reinforce.core",
        "log_prefix": "reinforce",
        "best": ["양천구", "강남구", "광진구"],
        "worst": ["강서구", "마포구", "강동구"],
    },
}


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """추가 패키지 없이 DataFrame을 Markdown 표 문자열로 변환한다."""
    columns = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows)) if rows else len(columns[idx])
        for idx in range(len(columns))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    header = fmt_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [fmt_row(row) for row in rows]
    return "\n".join([header, separator, *body])


def subprocess_env() -> dict[str, str]:
    """학습 subprocess용 환경변수."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    return env


def build_cmd(algorithm: str, district: str, seed: int, args: argparse.Namespace) -> list[str]:
    """알고리즘 core 실행 명령을 만든다."""
    module = EXPERIMENTS[algorithm]["module"]
    tag = f"seedci_{algorithm}_{district}_s{seed}"
    return [
        sys.executable,
        "-m",
        module,
        "--processed-dir",
        str(ROOT / "data" / "processed_seoul_all"),
        "--district",
        district,
        "--n-train-dates",
        str(args.n_train_dates),
        "--bc-epochs",
        "0",
        "--future-mode",
        "forecast_projected_travel",
        "--future-horizon",
        "6",
        "--capacity-path",
        str(ROOT / "data" / "processed" / "station_capacity.csv"),
        "--forecast-path",
        str(ROOT / "data" / "forecast_by_gu" / f"demand_forecast_1h_{district}.parquet"),
        "--candidate-top-k",
        str(args.candidate_top_k),
        "--candidate-mode",
        "forecast_imbalance",
        "--candidate-travel-coef",
        "0.20",
        "--candidate-zone-mode",
        "static3",
        "--candidate-zone-penalty",
        "1.0",
        "--candidate-feature-mode",
        "basic",
        "--tag",
        tag,
        "--device",
        args.device,
        "--episodes",
        str(args.episodes),
        "--eval-every",
        str(args.eval_every),
        "--split-mode",
        args.split_mode,
        "--normalize-advantages",
        "--seed",
        str(seed),
        "--progress",
    ]


def history_path(algorithm: str, district: str, seed: int) -> Path:
    """core가 저장하는 history.npy 경로를 계산한다."""
    prefix = EXPERIMENTS[algorithm]["log_prefix"]
    tag = f"seedci_{algorithm}_{district}_s{seed}"
    return ROOT / "logs" / f"{prefix}_{tag}" / "history.npy"


def base_history_paths(algorithm: str, district: str) -> list[Path]:
    """서울 25개 구 seed 42 full run에서 생성된 history.npy 후보 경로들."""
    prefix = EXPERIMENTS[algorithm]["log_prefix"]
    return [
        ROOT / "logs" / f"{prefix}_interactive_topk12_{algorithm}_{district}" / "history.npy",
        ROOT / "logs" / f"{prefix}_interactive_{algorithm}_{district}" / "history.npy",
        ROOT / "logs" / f"{prefix}_seedci_{algorithm}_{district}_s42" / "history.npy",
    ]


def result_path(algorithm: str, district: str, seed: int, args: argparse.Namespace) -> Path:
    """요약에 사용할 history.npy 경로를 반환한다.

    seed 42는 기본적으로 서울 전체 full run의 기존 로그를 재사용한다. 추가 seed는
    이 스크립트가 만든 `seedci_*_s{seed}` 로그를 읽는다.
    """
    if args.use_existing_seed42 and seed == args.base_seed:
        for path in base_history_paths(algorithm, district):
            if path.exists():
                return path
        return base_history_paths(algorithm, district)[0]
    return history_path(algorithm, district, seed)


def read_result(algorithm: str, group: str, district: str, seed: int, baseline: float, args: argparse.Namespace) -> dict:
    """history.npy에서 best/final 결과를 읽어 한 행으로 반환한다."""
    path = result_path(algorithm, district, seed, args)
    if not path.exists():
        raise FileNotFoundError(
            f"seed 결과 파일이 없습니다: {path.relative_to(ROOT)}\n"
            f"- seed {args.base_seed} 기존 full run을 재사용하려면 먼저 서울 전체 실험 로그가 있어야 합니다.\n"
            "- 새로 모두 실행하려면 --no-use-existing-seed42 옵션을 사용하세요."
        )
    arr = np.load(path, allow_pickle=True)
    rows = [dict(x) for x in arr.tolist()]
    values = np.array([float(row["eval_reward"]) for row in rows])
    points = np.array([float(row.get("episode", i)) for i, row in enumerate(rows)])
    best_idx = int(values.argmax())
    best_reward = float(values[best_idx])
    final_reward = float(values[-1])
    return {
        "algorithm": algorithm.upper() if algorithm == "a2c" else "REINFORCE",
        "group": group,
        "district": district,
        "seed": seed,
        "baseline": baseline,
        "best_reward": best_reward,
        "final_reward": final_reward,
        "best_delta": best_reward - baseline,
        "final_delta": final_reward - baseline,
        "best_episode": int(points[best_idx]),
        "history_path": str(path.relative_to(ROOT)),
    }


def parse_seed_list(raw: str) -> list[int]:
    """쉼표로 구분된 seed 문자열을 정수 리스트로 변환한다."""
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def summarize(rows: list[dict], out_prefix: Path, args: argparse.Namespace) -> None:
    """seed별 결과와 평균/95% CI 요약을 저장한다."""
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["algorithm", "group", "district"], as_index=False)
        .agg(
            baseline=("baseline", "first"),
            n_seed=("seed", "count"),
            best_delta_mean=("best_delta", "mean"),
            best_delta_std=("best_delta", "std"),
            final_delta_mean=("final_delta", "mean"),
            final_delta_std=("final_delta", "std"),
            best_reward_mean=("best_reward", "mean"),
            final_reward_mean=("final_reward", "mean"),
        )
        .fillna(0.0)
    )
    summary["best_delta_ci95"] = 1.96 * summary["best_delta_std"] / summary["n_seed"].map(math.sqrt)
    summary["final_delta_ci95"] = 1.96 * summary["final_delta_std"] / summary["n_seed"].map(math.sqrt)

    detail_path = out_prefix.with_suffix(".detail.csv")
    summary_path = out_prefix.with_suffix(".summary.csv")
    md_path = out_prefix.with_suffix(".md")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    display = summary.copy()
    for col in [
        "baseline",
        "best_delta_mean",
        "best_delta_ci95",
        "final_delta_mean",
        "final_delta_ci95",
        "best_reward_mean",
        "final_reward_mean",
    ]:
        display[col] = display[col].map(lambda x: f"{x:.1f}")

    md = [
        "# Seed Sensitivity 결과",
        "",
        "대상: A2C/REINFORCE의 Best 3구와 Worst 3구",
        "",
        f"Seed: {', '.join(map(str, sorted(detail['seed'].unique())))}",
        "",
        f"seed {args.base_seed}는 서울 전체 full run 로그를 재사용하고, 추가 seed만 별도 학습했다.",
        "",
        "지표: 7개 평가일 평균 reward의 MostImbalanced baseline 대비 Delta",
        "",
        dataframe_to_markdown(display),
        "",
        f"- detail: `{detail_path.relative_to(ROOT)}`",
        f"- summary: `{summary_path.relative_to(ROOT)}`",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {detail_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A2C/REINFORCE seed sensitivity experiments.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--n-train-dates", type=int, default=200)
    parser.add_argument("--candidate-top-k", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--split-mode", choices=["random", "chronological"], default="random")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--additional-seeds",
        default=",".join(map(str, DEFAULT_ADDITIONAL_SEEDS)),
        help="쉼표 구분 예: 123,777",
    )
    parser.add_argument(
        "--use-existing-seed42",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="seed 42는 서울 전체 full run 로그를 재사용하고, 추가 seed만 실행한다.",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    additional_seeds = parse_seed_list(args.additional_seeds)
    all_seeds = [args.base_seed, *additional_seeds]
    seeds_to_run = additional_seeds if args.use_existing_seed42 else all_seeds
    baselines = pd.read_csv(ROOT / "docs" / "rl_current_gu_algorithm_summary.csv")
    baseline_by_district = baselines.drop_duplicates("district").set_index("district")["baseline_reward"].to_dict()

    total = sum((len(spec["best"]) + len(spec["worst"])) * len(seeds_to_run) for spec in EXPERIMENTS.values())
    run_index = 0
    for algorithm, spec in EXPERIMENTS.items():
        for group in ["best", "worst"]:
            for district in spec[group]:
                for seed in seeds_to_run:
                    run_index += 1
                    path = history_path(algorithm, district, seed)
                    print("\n" + "=" * 80, flush=True)
                    print(f"[{run_index}/{total}] {algorithm.upper()} {group} {district} seed={seed}", flush=True)
                    if args.skip_existing and path.exists():
                        print(f"skip existing: {path.relative_to(ROOT)}", flush=True)
                        continue
                    cmd = build_cmd(algorithm, district, seed, args)
                    print(" ".join(cmd), flush=True)
                    if args.dry_run:
                        continue
                    code = subprocess.run(cmd, cwd=ROOT, env=subprocess_env()).returncode
                    if code != 0:
                        raise SystemExit(code)

    if args.dry_run:
        return

    rows = []
    for algorithm, spec in EXPERIMENTS.items():
        for group in ["best", "worst"]:
            for district in spec[group]:
                baseline = float(baseline_by_district[district])
                for seed in all_seeds:
                    rows.append(read_result(algorithm, group, district, seed, baseline, args))

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summarize(rows, ROOT / "docs" / f"rl_seed_sensitivity_a2c_reinforce_{ts}", args)


if __name__ == "__main__":
    main()
