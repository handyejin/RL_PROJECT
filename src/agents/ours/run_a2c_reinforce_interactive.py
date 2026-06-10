"""REINFORCE/A2C 담당 실험만 실행하는 전용 interactive runner.

이 파일은 발표/보고서에서 사용한 담당 알고리즘을 팀원 공용 runner와 분리해서
실행하기 위한 얇은 진입점이다. 실제 학습 코드는 각 알고리즘 core에 있고, 이
파일은 터미널 선택값을 명령어로 바꿔 호출한다.

지원 작업:
    1. 단일 학습 실행: REINFORCE, A2C, Contextual Bandit
    2. VAE demand latent 파일 생성
    3. 73일 최종 프로토콜: 전체 실험, Top-K, seed, final run
    4. VAE ablation: 같은 조건에서 VAE feature만 추가 비교
    5. Contextual Bandit baseline 비교

실행 예:
    PYTHONPATH=. python -m src.agents.ours.run_a2c_reinforce_interactive
    PYTHONPATH=. python -m src.agents.ours.run_a2c_reinforce_interactive --mode final73_protocol
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.agents.ours.common.runner_config import (
    DEFAULT_RUNNER_VALUES,
    DISTRICTS,
    PROJECT_ROOT,
    build_training_command,
    build_vae_command,
    ensure_training_inputs,
    ensure_vae_inputs,
    project_path,
    selected_districts,
    subprocess_env,
)


LOG_PREFIX = {"a2c": "actor_critic", "reinforce": "reinforce"}


def choose_mode() -> str:
    """담당 실험에서 실행할 작업 종류를 선택한다."""
    print("\nREINFORCE/A2C 실험 실행 메뉴")
    print("  1. 단일 학습 실행 (REINFORCE / A2C / Contextual Bandit)")
    print("  2. VAE latent 파일 생성")
    print("  3. Final 73-day Protocol (전체 -> Top-K -> Seed -> 최종)")
    print("  4. VAE Ablation (Best/Worst 구, VAE 추가 효과 확인)")
    print("  5. Contextual Bandit baseline 비교")
    choice = input("선택 [1/2/3/4/5, Enter=1]: ").strip()
    return {
        "2": "vae",
        "3": "final73_protocol",
        "4": "vae_ablation",
        "5": "bandit_compare",
    }.get(
        choice, "train"
    )


def choose_algorithm() -> str:
    """담당 알고리즘 중 하나를 선택한다."""
    print("\n알고리즘을 선택하세요.")
    print("  1. REINFORCE")
    print("  2. A2C")
    print("  3. Contextual Bandit (LinUCB)")
    choice = input("선택 [1/2/3, Enter=2]: ").strip()
    if choice == "1":
        return "reinforce"
    if choice == "3":
        return "bandit"
    return "a2c"


def choose_district() -> str:
    """ALL 또는 자주 쓰는 구, 직접 입력 중 실행 지역을 선택한다."""
    print("\n실행 지역을 선택하세요.")
    print("  1. ALL (25개 구 전체 순차 실행)")
    print("  2. 영등포구")
    print("  3. 마포구")
    print("  4. 관악구")
    print("  5. A2C Best 3구 (마포구, 영등포구, 노원구)")
    print("  6. A2C Worst 3구 (은평구, 서대문구, 관악구)")
    print("  7. A2C Best/Worst 6구")
    print("  8. 직접 입력")
    choice = input("선택 [1/2/3/4/5/6/7/8, Enter=1]: ").strip()
    if choice in {"", "1"}:
        return "ALL"
    if choice == "2":
        return "영등포구"
    if choice == "3":
        return "마포구"
    if choice == "4":
        return "관악구"
    if choice == "5":
        return "A2C_BEST3"
    if choice == "6":
        return "A2C_WORST3"
    if choice == "7":
        return "A2C_BEST_WORST"
    typed = input("구 이름 입력 예: 양천구 또는 A2C_BEST_WORST: ").strip()
    return typed or "ALL"


def choose_top_k() -> int:
    """Top-K 후보 action 개수를 선택한다."""
    print("\nTop-K 후보 개수를 선택하세요.")
    print("  1. Top-K = 3  (강한 후보 축소)")
    print("  2. Top-K = 6  (중간 절충)")
    print("  3. Top-K = 9  (넓은 후보군)")
    print("  4. Top-K = 12 (보고서 기준)")
    print("  5. Top-K = 15 (더 넓은 후보군)")
    print("  6. 직접 입력")
    choice = input("선택 [1/2/3/4/5/6, Enter=12]: ").strip()
    if choice in {"1", "2", "3", "4", "5"}:
        return {"1": 3, "2": 6, "3": 9, "4": 12, "5": 15}[choice]
    if choice == "6":
        typed = input("Top-K 숫자 입력 예: 12: ").strip()
        try:
            return max(2, int(typed))
        except ValueError:
            print("숫자가 아니어서 기본값 12를 사용합니다.")
    return 12


def choose_vae_mode() -> str:
    """학습 state에 VAE latent를 붙일지 선택한다."""
    print("\nVAE 수요 latent feature를 사용할까요?")
    print("  1. 사용 안 함 (기본)")
    print("  2. 사용함: Forecast + VAE latent")
    choice = input("선택 [1/2, Enter=1]: ").strip()
    return "demand_latent" if choice == "2" else "none"


def choose_final73_stage() -> str:
    """73일 최종 평가 프로토콜의 세부 단계를 선택한다."""
    print("\nFinal 73-day Protocol 단계를 선택하세요.")
    print("  1. Full baseline: REINFORCE/A2C, ALL, Top-K=12, seed=42")
    print("  2. Top-K ablation: 73일 결과 Best/Worst 구, K=3/6/9/12/15, 200 ep")
    print("  3. Confirmation: 선택 Top-K, 500 ep")
    print("  4. Seed validation: 선택 Top-K, seed 42/123/777")
    print("  5. Final full run: 선택 Top-K, REINFORCE/A2C, ALL")
    choice = input("선택 [1/2/3/4/5, Enter=1]: ").strip()
    return {"2": "topk", "3": "confirm", "4": "seed", "5": "final_full"}.get(choice, "baseline")


def choose_ablation_algorithms() -> list[str]:
    """VAE ablation에서 비교할 policy-gradient 알고리즘을 고른다."""
    print("\nVAE ablation 알고리즘을 선택하세요.")
    print("  1. REINFORCE + A2C")
    print("  2. REINFORCE만")
    print("  3. A2C만")
    choice = input("선택 [1/2/3, Enter=1]: ").strip()
    if choice == "2":
        return ["reinforce"]
    if choice == "3":
        return ["a2c"]
    return ["reinforce", "a2c"]


def parse_args() -> argparse.Namespace:
    """대화형/명령형 실행 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="REINFORCE/A2C 전용 interactive runner.")
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "vae",
            "final73_protocol",
            "vae_ablation",
            "bandit_compare",
        ],
        default="",
    )
    parser.add_argument("--algorithm", choices=["reinforce", "a2c", "bandit"], default="")
    parser.add_argument("--district", default="")
    parser.add_argument("--processed-dir", default=DEFAULT_RUNNER_VALUES["processed_dir"])
    parser.add_argument("--forecast-dir", default=DEFAULT_RUNNER_VALUES["forecast_dir"])
    parser.add_argument("--vae-latent-dir", default=DEFAULT_RUNNER_VALUES["vae_latent_dir"])
    parser.add_argument("--capacity-path", default=DEFAULT_RUNNER_VALUES["capacity_path"])
    parser.add_argument("--episodes", type=int, default=DEFAULT_RUNNER_VALUES["episodes"])
    parser.add_argument("--eval-every", type=int, default=DEFAULT_RUNNER_VALUES["eval_every"])
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_RUNNER_VALUES["total_timesteps"])
    parser.add_argument("--eval-every-timesteps", type=int, default=DEFAULT_RUNNER_VALUES["eval_every_timesteps"])
    parser.add_argument("--n-train-dates", type=int, default=DEFAULT_RUNNER_VALUES["n_train_dates"])
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological"],
        default="chronological",
        help="팀 최종 기준은 chronological train/test split이다.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RUNNER_VALUES["seed"])
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--additional-seeds", default="123,777")
    parser.add_argument(
        "--ablation-top-ks",
        default="3,6,9,12,15",
        help="Top-K ablation study에서 순차 실행할 K 목록이다. 예: 3,6,9,12,15",
    )
    parser.add_argument(
        "--ablation-district-group",
        default="A2C_BEST_WORST",
        help="Top-K ablation 대상 구 묶음. 기본값은 A2C Best/Worst 6구다.",
    )
    parser.add_argument(
        "--final73-stage",
        choices=["baseline", "topk", "confirm", "seed", "final_full"],
        default="",
        help="Final 73-day Protocol 단계.",
    )
    parser.add_argument(
        "--final73-source-tag",
        default="final_73d_topk12",
        help="Best/Worst를 뽑을 기준 full-run tag.",
    )
    parser.add_argument(
        "--final73-top-ks",
        default="3,6,9,12,15",
        help="Final 73-day Top-K screening 후보 목록.",
    )
    parser.add_argument(
        "--final73-confirm-top-ks",
        default="",
        help="Final 73-day confirmation/final run 후보 Top-K. 예: 6,12",
    )
    parser.add_argument("--use-existing-seed42", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--future-mode", default=DEFAULT_RUNNER_VALUES["future_mode"])
    parser.add_argument("--future-horizon", type=int, default=DEFAULT_RUNNER_VALUES["future_horizon"])
    parser.add_argument("--vae-mode", choices=["none", "demand_latent"], default="")
    parser.add_argument("--vae-latent-dim", type=int, default=DEFAULT_RUNNER_VALUES["vae_latent_dim"])
    parser.add_argument("--vae-epochs", type=int, default=DEFAULT_RUNNER_VALUES["vae_epochs"])
    parser.add_argument("--vae-hidden", type=int, default=DEFAULT_RUNNER_VALUES["vae_hidden"])
    parser.add_argument("--vae-batch-size", type=int, default=DEFAULT_RUNNER_VALUES["vae_batch_size"])
    parser.add_argument("--vae-lr", type=float, default=DEFAULT_RUNNER_VALUES["vae_lr"])
    parser.add_argument("--vae-beta", type=float, default=DEFAULT_RUNNER_VALUES["vae_beta"])
    parser.add_argument("--candidate-top-k", type=int, default=None)
    parser.add_argument("--candidate-mode", default=DEFAULT_RUNNER_VALUES["candidate_mode"])
    parser.add_argument("--candidate-travel-coef", type=float, default=DEFAULT_RUNNER_VALUES["candidate_travel_coef"])
    parser.add_argument("--candidate-zone-mode", default=DEFAULT_RUNNER_VALUES["candidate_zone_mode"])
    parser.add_argument("--candidate-zone-penalty", type=float, default=DEFAULT_RUNNER_VALUES["candidate_zone_penalty"])
    parser.add_argument("--candidate-feature-mode", default=DEFAULT_RUNNER_VALUES["candidate_feature_mode"])
    parser.add_argument("--bandit-alpha", type=float, default=DEFAULT_RUNNER_VALUES["bandit_alpha"])
    parser.add_argument("--bandit-l2", type=float, default=DEFAULT_RUNNER_VALUES["bandit_l2"])
    parser.add_argument("--bandit-reward-scale", type=float, default=DEFAULT_RUNNER_VALUES["bandit_reward_scale"])
    parser.add_argument("--tag", default=DEFAULT_RUNNER_VALUES["tag"])
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default=DEFAULT_RUNNER_VALUES["device"])
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=DEFAULT_RUNNER_VALUES["progress"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], dry_run: bool) -> None:
    """명령어를 출력하고 dry-run이 아니면 실행한다."""
    print("명령:", flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    code = subprocess.run(cmd, cwd=PROJECT_ROOT, env=subprocess_env()).returncode
    if code != 0:
        raise SystemExit(code)


def run_training(args: argparse.Namespace, should_prompt: bool) -> None:
    """REINFORCE/A2C 학습을 선택한 구에 대해 실행한다."""
    if not args.algorithm:
        args.algorithm = choose_algorithm()
    if not args.district:
        args.district = choose_district()
    if args.candidate_top_k is None:
        args.candidate_top_k = choose_top_k()
    if not args.vae_mode:
        args.vae_mode = choose_vae_mode() if should_prompt else "none"

    districts = selected_districts(args.district)
    print(f"\n실행 알고리즘: {args.algorithm.upper()}", flush=True)
    print(f"실행 지역: {', '.join(districts)}", flush=True)
    print(f"seed={args.seed}, split={args.split_mode}, episodes={args.episodes}, eval_every={args.eval_every}", flush=True)
    print(f"top_k={args.candidate_top_k}, VAE={'사용' if args.vae_mode != 'none' else '미사용'}", flush=True)

    for index, district in enumerate(districts, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[{index}/{len(districts)}] {district} 실행", flush=True)
        if not ensure_training_inputs(args, district):
            continue
        run_command(build_training_command(args, district), args.dry_run)


def run_vae(args: argparse.Namespace) -> None:
    """선택한 구 또는 25개 구 전체에 대해 VAE latent 파일을 생성한다."""
    if not args.district:
        args.district = choose_district()
    districts = selected_districts(args.district)
    print("\n실행 작업: VAE latent 생성", flush=True)
    print(f"실행 지역: {', '.join(districts)}", flush=True)
    print(f"latent_dim={args.vae_latent_dim}, epochs={args.vae_epochs}", flush=True)
    print(f"out_dir={project_path(args.vae_latent_dir)}", flush=True)
    if not ensure_vae_inputs(args):
        return
    for index, district in enumerate(districts, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[{index}/{len(districts)}] {district} VAE latent 생성", flush=True)
        run_command(build_vae_command(args, district), args.dry_run)


def run_seed_sensitivity(args: argparse.Namespace) -> None:
    """Best/Worst 3구에 seed 123/777을 추가 실행하고 요약 파일을 만든다."""
    if args.split_mode == "chronological" and args.use_existing_seed42:
        print(
            "\nchronological split 기준 seed 실험입니다.\n"
            "seed 42는 chronological full run 로그만 재사용하며, 기존 random split 로그는 사용하지 않습니다.",
            flush=True,
        )
    seed_algorithms = getattr(args, "seed_algorithms", "")
    seed_districts = getattr(args, "seed_districts", "")
    seed_experiment_label = getattr(args, "seed_experiment_label", "")
    print("\n실행 작업: Seed sensitivity 반복 실험", flush=True)
    if seed_algorithms or seed_districts:
        print(f"대상 알고리즘: {seed_algorithms or 'a2c,reinforce'}", flush=True)
        print(f"대상 구: {seed_districts or '기본 Best/Worst 3구'}", flush=True)
    else:
        print("대상: A2C Best/Worst 3구 + REINFORCE Best/Worst 3구", flush=True)
    if args.use_existing_seed42:
        print(f"base_seed={args.base_seed} 기존 full run 재사용, additional_seeds={args.additional_seeds}", flush=True)
    else:
        print(f"seeds={args.base_seed},{args.additional_seeds} 모두 새 태그로 학습", flush=True)
    print(f"episodes={args.episodes}, eval_every={args.eval_every}, top_k={args.candidate_top_k or 12}", flush=True)

    cmd = [
        sys.executable,
        "scripts/run_seed_sensitivity.py",
        "--episodes",
        str(args.episodes),
        "--eval-every",
        str(args.eval_every),
        "--n-train-dates",
        str(args.n_train_dates),
        "--candidate-top-k",
        str(args.candidate_top_k or 12),
        "--device",
        "mps" if args.device == "mps" else "cpu",
        "--split-mode",
        args.split_mode,
        "--base-seed",
        str(args.base_seed),
        "--additional-seeds",
        args.additional_seeds,
    ]
    if seed_algorithms:
        cmd.extend(["--algorithms", seed_algorithms])
    if seed_districts:
        cmd.extend(["--districts", seed_districts])
    if seed_experiment_label:
        cmd.extend(["--experiment-label", seed_experiment_label, "--include-topk-in-tag"])
    if getattr(args, "recompute_seed_baseline", False):
        cmd.append("--recompute-baseline")
    if not args.use_existing_seed42:
        cmd.append("--no-use-existing-seed42")
    if not args.skip_existing:
        cmd.append("--no-skip-existing")
    if args.dry_run:
        cmd.append("--dry-run")
    # 이 경우 wrapper의 dry-run도 seed 스크립트를 --dry-run으로 실행해야
    # Best/Worst 3구 x 추가 seed 실행 목록까지 확인할 수 있다.
    run_command(cmd, False)


def run_topk_ablation(args: argparse.Namespace, should_prompt: bool) -> None:
    """Best/Worst 6구에서 Top-K 후보 개수 민감도 실험을 실행한다.

    논문식 표현으로는 ablation study 또는 hyperparameter sensitivity
    analysis에 해당한다. 한 번에 하나의 알고리즘을 고르고, Top-K만
    3/6/9/12/15로 바꿔가며 같은 6개 구를 반복 학습한다.
    """
    if not args.algorithm:
        args.algorithm = choose_algorithm()
    if not args.vae_mode:
        args.vae_mode = choose_vae_mode() if should_prompt and sys.stdin.isatty() else "none"

    top_ks: list[int] = []
    for raw in str(args.ablation_top_ks).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            top_ks.append(max(2, int(raw)))
        except ValueError:
            print(f"Top-K 값 {raw!r}은 숫자가 아니어서 건너뜁니다.", flush=True)
    if not top_ks:
        top_ks = [3, 6, 9, 12, 15]

    districts = selected_districts(args.ablation_district_group)
    print("\n실행 작업: Top-K Ablation Study / Hyperparameter Sensitivity Analysis", flush=True)
    print(f"알고리즘: {args.algorithm.upper()}", flush=True)
    print(f"대상 구: {', '.join(districts)}", flush=True)
    print(f"Top-K 목록: {', '.join(map(str, top_ks))}", flush=True)
    print(f"seed={args.seed}, split={args.split_mode}, episodes={args.episodes}, eval_every={args.eval_every}", flush=True)
    print(f"VAE={'사용' if args.vae_mode != 'none' else '미사용'}", flush=True)

    total = len(top_ks) * len(districts)
    done = 0
    for top_k in top_ks:
        for district in districts:
            done += 1
            print("\n" + "=" * 80, flush=True)
            print(f"[{done}/{total}] Top-K={top_k} | {district} 실행", flush=True)
            run_args = argparse.Namespace(**vars(args))
            run_args.district = district
            run_args.candidate_top_k = top_k
            run_args.tag = f"topk_ablation_k{top_k}"
            if not ensure_training_inputs(run_args, district):
                continue
            run_command(build_training_command(run_args, district), args.dry_run)


def _history_best_reward(history_path: Path) -> float | None:
    """history.npy에서 가장 좋은 eval reward를 읽는다."""
    if not history_path.exists():
        return None
    rows = np.load(history_path, allow_pickle=True)
    values = [float(row["eval_reward"]) for row in rows if isinstance(row, dict) and "eval_reward" in row]
    return max(values) if values else None


def _log_history_path(algorithm: str, tag: str, district: str) -> Path:
    """runner_config가 만드는 로그 경로의 history.npy 위치를 반환한다."""
    return PROJECT_ROOT / "logs" / f"{LOG_PREFIX[algorithm]}_{tag}_chronological_{algorithm}_{district}" / "history.npy"


def _baseline_73day(district: str, args: argparse.Namespace) -> float:
    """현재 73일 chronological holdout에서 MostImbalanced baseline을 계산한다."""
    from src.agents.ours.algorithms.a2c.core import evaluate_heuristic, load_episodes
    from src.agents.ours.common.data_overrides import apply_capacity_override, attach_forecast_override
    from src.agents.ours.common.date_split import compute_split

    _, eval_dates = compute_split("chronological", seed=42)
    cache_dir = None if getattr(args, "no_episode_cache", False) else getattr(args, "episode_cache_dir", "data/episode_cache")
    episodes = load_episodes(
        eval_dates,
        district,
        args.processed_dir,
        cache_dir,
        f"baseline {district} load eval" if args.progress else None,
    )
    apply_capacity_override(episodes, str(project_path(args.capacity_path)), 0.5)
    forecast_path = project_path(args.forecast_dir) / f"demand_forecast_1h_{district}.parquet"
    attach_forecast_override(episodes, str(forecast_path))
    mean, _ = evaluate_heuristic(episodes, args.seed)
    return float(mean)


def infer_best_worst_districts(args: argparse.Namespace, algorithms: list[str], source_tag: str) -> list[str]:
    """full-run 로그에서 알고리즘별 Best/Worst 3구를 Delta 기준으로 추론한다."""
    selected: list[str] = []
    for algorithm in algorithms:
        rows = []
        for district in DISTRICTS:
            history_path = _log_history_path(algorithm, source_tag, district)
            best_reward = _history_best_reward(history_path)
            if best_reward is None:
                continue
            baseline = _baseline_73day(district, args)
            rows.append((district, best_reward - baseline))
        if len(rows) < len(DISTRICTS):
            missing = len(DISTRICTS) - len(rows)
            print(f"{algorithm.upper()} {source_tag} 로그가 {missing}개 부족합니다. 먼저 baseline full run을 완료하세요.", flush=True)
            continue
        rows.sort(key=lambda item: item[1], reverse=True)
        best = rows[:3]
        worst = rows[-3:]
        print(f"\n{algorithm.upper()} 73일 기준 Best 3: " + ", ".join(f"{d}({v:+.1f})" for d, v in best), flush=True)
        print(f"{algorithm.upper()} 73일 기준 Worst 3: " + ", ".join(f"{d}({v:+.1f})" for d, v in worst), flush=True)
        selected.extend([district for district, _ in best + worst])
    # 순서를 유지하면서 중복 제거
    return list(dict.fromkeys(selected))


def _parse_top_k_list(raw_values: str, fallback: list[int]) -> list[int]:
    """쉼표로 입력한 Top-K 목록을 정수 리스트로 변환한다."""
    top_ks: list[int] = []
    for raw in str(raw_values).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            top_ks.append(max(2, int(raw)))
        except ValueError:
            print(f"Top-K 값 {raw!r}은 숫자가 아니어서 건너뜁니다.", flush=True)
    return top_ks or fallback


def _final73_topks(args: argparse.Namespace, field: str, fallback: list[int]) -> list[int]:
    """Final 73-day protocol용 Top-K 목록을 읽는다."""
    text = getattr(args, field)
    if not text and sys.stdin.isatty():
        text = input(f"Top-K 목록 입력 예: {','.join(map(str, fallback))}: ").strip()
    return _parse_top_k_list(text, fallback=fallback)


def run_final73_protocol(args: argparse.Namespace) -> None:
    """73일 holdout 기준 최종 실험 프로토콜을 단계별로 실행한다.

    단계:
        1. REINFORCE/A2C 전체 25개 구 baseline 재학습
        2. 73일 결과에서 Best/Worst 구 자동 선정 후 Top-K ablation
        3. 유망 Top-K confirmation
        4. 선택 Top-K seed validation
        5. 선택 Top-K 전체 25개 구 final full run
    """
    if not args.final73_stage:
        args.final73_stage = choose_final73_stage() if sys.stdin.isatty() else "baseline"
    args.split_mode = "chronological"
    if not args.vae_mode:
        args.vae_mode = "none"

    print("\n실행 작업: Final 73-day Protocol", flush=True)
    print("평가 기준: chronological holdout 전체 2025-10-20~2025-12-31 (73일)", flush=True)
    print(f"stage={args.final73_stage}", flush=True)

    if args.final73_stage == "baseline":
        print("\n[1단계] REINFORCE/A2C 전체 25개 구 baseline 재학습", flush=True)
        for algorithm in ["reinforce", "a2c"]:
            run_args = argparse.Namespace(**vars(args))
            run_args.algorithm = algorithm
            run_args.district = "ALL"
            run_args.candidate_top_k = 12
            run_args.seed = args.base_seed
            run_args.episodes = 500
            run_args.eval_every = 50
            run_args.tag = args.final73_source_tag
            run_training(run_args, should_prompt=False)
        return

    algorithms = ["reinforce", "a2c"]
    districts = infer_best_worst_districts(args, algorithms, args.final73_source_tag)
    if not districts:
        print("\nBest/Worst 구를 계산할 수 없습니다. 먼저 1단계 baseline full run을 완료하세요.", flush=True)
        return
    district_group = ",".join(districts)

    if args.final73_stage == "topk":
        print("\n[2단계] 73일 기준 Best/Worst 구 Top-K ablation", flush=True)
        for algorithm in algorithms:
            run_args = argparse.Namespace(**vars(args))
            run_args.algorithm = algorithm
            run_args.district = district_group
            run_args.ablation_district_group = district_group
            run_args.ablation_top_ks = args.final73_top_ks
            run_args.episodes = 200
            run_args.eval_every = 20
            run_args.seed = args.base_seed
            run_args.tag = "final73_topk_ablation"
            run_topk_ablation(run_args, should_prompt=False)
        return

    if args.final73_stage == "confirm":
        top_ks = _final73_topks(args, "final73_confirm_top_ks", fallback=[12])
        print(f"\n[3단계] Confirmation: Top-K={top_ks}, 500 episodes", flush=True)
        for algorithm in algorithms:
            run_args = argparse.Namespace(**vars(args))
            run_args.algorithm = algorithm
            run_args.district = district_group
            run_args.ablation_district_group = district_group
            run_args.ablation_top_ks = ",".join(map(str, top_ks))
            run_args.episodes = 500
            run_args.eval_every = 50
            run_args.seed = args.base_seed
            run_args.tag = "final73_confirm"
            run_topk_ablation(run_args, should_prompt=False)
        return

    if args.final73_stage == "seed":
        top_ks = _final73_topks(args, "final73_confirm_top_ks", fallback=[12])
        chosen_top_k = top_ks[0]
        print(f"\n[4단계] Seed validation: Top-K={chosen_top_k}, seed 42/123/777", flush=True)
        seed_args = argparse.Namespace(**vars(args))
        seed_args.candidate_top_k = chosen_top_k
        seed_args.episodes = 500
        seed_args.eval_every = 50
        seed_args.use_existing_seed42 = False
        seed_args.additional_seeds = args.additional_seeds
        seed_args.seed_algorithms = "reinforce,a2c"
        seed_args.seed_districts = district_group
        seed_args.seed_experiment_label = f"final73_seedci"
        seed_args.recompute_seed_baseline = True
        run_seed_sensitivity(seed_args)
        return

    top_ks = _final73_topks(args, "final73_confirm_top_ks", fallback=[12])
    chosen_top_k = top_ks[0]
    print(f"\n[5단계] 최종 전체 재학습: REINFORCE/A2C, ALL, Top-K={chosen_top_k}", flush=True)
    for algorithm in algorithms:
        run_args = argparse.Namespace(**vars(args))
        run_args.algorithm = algorithm
        run_args.district = "ALL"
        run_args.candidate_top_k = chosen_top_k
        run_args.seed = args.base_seed
        run_args.episodes = 500
        run_args.eval_every = 50
        run_args.tag = f"final73_topk{chosen_top_k}"
        run_training(run_args, should_prompt=False)


def run_vae_ablation(args: argparse.Namespace) -> None:
    """Best/Worst 구에서 VAE latent feature 추가 효과만 비교한다.

    이 실험은 알고리즘을 새로 바꾸는 것이 아니라 state 표현만 바꾼다.
    따라서 보고서에서는 같은 구, 같은 Top-K, 같은 seed, 같은 episode에서
    `VAE 없음` 결과와 `VAE 있음` 결과를 비교하는 ablation으로 해석한다.
    """
    args.split_mode = "chronological"
    algorithms = choose_ablation_algorithms() if sys.stdin.isatty() and not args.algorithm else [args.algorithm or "a2c"]
    top_ks = _final73_topks(args, "final73_confirm_top_ks", fallback=[9])
    chosen_top_k = top_ks[0]

    districts = infer_best_worst_districts(args, ["reinforce", "a2c"], args.final73_source_tag)
    if not districts:
        print("\nBest/Worst 구를 계산할 수 없습니다. 먼저 Final 73-day 1단계 full run을 완료하세요.", flush=True)
        return

    print("\n실행 작업: VAE Ablation", flush=True)
    print("목적: 같은 조건에서 VAE demand latent feature 추가 효과만 확인", flush=True)
    print(f"대상 알고리즘: {', '.join(a.upper() for a in algorithms)}", flush=True)
    print(f"대상 구: {', '.join(districts)}", flush=True)
    print(f"Top-K={chosen_top_k}, seed={args.seed}, episodes={args.episodes}, split=chronological", flush=True)

    for algorithm in algorithms:
        for district in districts:
            print("\n" + "=" * 80, flush=True)
            print(f"{algorithm.upper()} + VAE | {district} 실행", flush=True)
            run_args = argparse.Namespace(**vars(args))
            run_args.algorithm = algorithm
            run_args.district = district
            run_args.candidate_top_k = chosen_top_k
            run_args.vae_mode = "demand_latent"
            run_args.tag = f"final73_vae_ablation_topk{chosen_top_k}"
            if not ensure_training_inputs(run_args, district):
                print(f"[{district}] VAE latent 파일이 없으면 2번 메뉴에서 먼저 생성하세요.", flush=True)
                continue
            run_command(build_training_command(run_args, district), args.dry_run)


def run_bandit_compare(args: argparse.Namespace) -> None:
    """Contextual Bandit을 같은 73일 holdout과 baseline 기준으로 비교한다.

    Bandit은 장기 return을 학습하는 RL agent가 아니라, 현재 context에서 Top-K
    후보 중 어떤 rank를 고를지 학습하는 단기 의사결정 대조군이다.
    core 출력에서 MostImbalanced baseline, Best/Final reward, Delta가 함께
    나오므로 REINFORCE/A2C와 같은 표에 넣을 수 있다.
    """
    args.split_mode = "chronological"
    if args.candidate_top_k is None:
        args.candidate_top_k = choose_top_k() if sys.stdin.isatty() else 9
    if not args.district:
        print("\nBandit 비교 대상 지역을 선택하세요.")
        print("  1. Final 73일 Best/Worst 구")
        print("  2. ALL (25개 구 전체)")
        print("  3. 직접 선택")
        choice = input("선택 [1/2/3, Enter=1]: ").strip() if sys.stdin.isatty() else "1"
        if choice == "2":
            args.district = "ALL"
        elif choice == "3":
            args.district = choose_district()
        else:
            districts = infer_best_worst_districts(args, ["reinforce", "a2c"], args.final73_source_tag)
            args.district = ",".join(districts) if districts else "A2C_BEST_WORST"

    districts = selected_districts(args.district)
    print("\n실행 작업: Contextual Bandit baseline 비교", flush=True)
    print(f"대상 구: {', '.join(districts)}", flush=True)
    print(
        f"Top-K={args.candidate_top_k}, timesteps={args.total_timesteps}, "
        f"eval_every={args.eval_every_timesteps}, split=chronological",
        flush=True,
    )
    print(f"LinUCB alpha={args.bandit_alpha}, l2={args.bandit_l2}, reward_scale={args.bandit_reward_scale}", flush=True)

    for index, district in enumerate(districts, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[{index}/{len(districts)}] BANDIT {district} 실행", flush=True)
        run_args = argparse.Namespace(**vars(args))
        run_args.algorithm = "bandit"
        run_args.district = district
        run_args.vae_mode = "none"
        run_args.tag = f"final73_bandit_topk{args.candidate_top_k}"
        if not ensure_training_inputs(run_args, district):
            continue
        run_command(build_training_command(run_args, district), args.dry_run)


def main() -> None:
    """선택한 담당 실험을 실행한다."""
    args = parse_args()
    if not args.mode:
        has_cli_args = bool(args.algorithm or args.district or args.candidate_top_k is not None)
        args.mode = "train" if has_cli_args else choose_mode()
    should_prompt = not args.algorithm or not args.district or args.candidate_top_k is None

    if args.mode == "vae":
        run_vae(args)
    elif args.mode == "final73_protocol":
        run_final73_protocol(args)
    elif args.mode == "vae_ablation":
        run_vae_ablation(args)
    elif args.mode == "bandit_compare":
        run_bandit_compare(args)
    else:
        run_training(args, should_prompt)


if __name__ == "__main__":
    main()
