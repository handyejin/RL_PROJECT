"""REINFORCE/A2C 담당 실험만 실행하는 전용 interactive runner.

이 파일은 발표/보고서에서 사용한 담당 알고리즘을 팀원 공용 runner와 분리해서
실행하기 위한 얇은 진입점이다. 실제 학습 코드는 각 알고리즘 core에 있고, 이
파일은 터미널 선택값을 명령어로 바꿔 호출한다.

지원 작업:
    1. REINFORCE 또는 A2C 학습
    2. VAE demand latent 파일 생성
    3. Best/Worst 3개 구 seed sensitivity 실험
    4. 최종 chronological split 실험 전체 실행

실행 예:
    PYTHONPATH=. python -m src.agents.ours.run_a2c_reinforce_interactive
    PYTHONPATH=. python -m src.agents.ours.run_a2c_reinforce_interactive --mode seed --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.agents.ours.common.runner_config import (
    DEFAULT_RUNNER_VALUES,
    PROJECT_ROOT,
    build_training_command,
    build_vae_command,
    ensure_training_inputs,
    ensure_vae_inputs,
    project_path,
    selected_districts,
    subprocess_env,
)


def choose_mode() -> str:
    """담당 실험에서 실행할 작업 종류를 선택한다."""
    print("\nREINFORCE/A2C 전용 실행 메뉴")
    print("  1. REINFORCE/A2C 학습 실행")
    print("  2. VAE latent 파일 생성")
    print("  3. Best/Worst 3구 seed 반복 실험")
    print("  4. 최종 chronological 전체 실험 실행")
    choice = input("선택 [1/2/3/4, Enter=1]: ").strip()
    return {"2": "vae", "3": "seed", "4": "final"}.get(choice, "train")


def choose_algorithm() -> str:
    """담당 알고리즘 중 하나를 선택한다."""
    print("\n알고리즘을 선택하세요.")
    print("  1. REINFORCE")
    print("  2. A2C")
    choice = input("선택 [1/2, Enter=2]: ").strip()
    return "reinforce" if choice == "1" else "a2c"


def choose_district() -> str:
    """ALL 또는 자주 쓰는 구, 직접 입력 중 실행 지역을 선택한다."""
    print("\n실행 지역을 선택하세요.")
    print("  1. ALL (25개 구 전체 순차 실행)")
    print("  2. 영등포구")
    print("  3. 마포구")
    print("  4. 관악구")
    print("  5. 직접 입력")
    choice = input("선택 [1/2/3/4/5, Enter=1]: ").strip()
    if choice in {"", "1"}:
        return "ALL"
    if choice == "2":
        return "영등포구"
    if choice == "3":
        return "마포구"
    if choice == "4":
        return "관악구"
    typed = input("구 이름 입력 예: 양천구: ").strip()
    return typed or "ALL"


def choose_top_k() -> int:
    """Top-K 후보 action 개수를 선택한다."""
    print("\nTop-K 후보 개수를 선택하세요.")
    print("  1. Top-K = 3  (강한 후보 축소)")
    print("  2. Top-K = 6  (중간 절충)")
    print("  3. Top-K = 9  (넓은 후보군)")
    print("  4. Top-K = 12 (보고서 기준)")
    print("  5. 직접 입력")
    choice = input("선택 [1/2/3/4/5, Enter=12]: ").strip()
    if choice in {"1", "2", "3"}:
        return {"1": 3, "2": 6, "3": 9}[choice]
    if choice == "5":
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


def parse_args() -> argparse.Namespace:
    """대화형/명령형 실행 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="REINFORCE/A2C 전용 interactive runner.")
    parser.add_argument("--mode", choices=["train", "vae", "seed", "final"], default="")
    parser.add_argument("--algorithm", choices=["reinforce", "a2c"], default="")
    parser.add_argument("--district", default="")
    parser.add_argument("--processed-dir", default=DEFAULT_RUNNER_VALUES["processed_dir"])
    parser.add_argument("--forecast-dir", default=DEFAULT_RUNNER_VALUES["forecast_dir"])
    parser.add_argument("--vae-latent-dir", default=DEFAULT_RUNNER_VALUES["vae_latent_dir"])
    parser.add_argument("--capacity-path", default=DEFAULT_RUNNER_VALUES["capacity_path"])
    parser.add_argument("--episodes", type=int, default=DEFAULT_RUNNER_VALUES["episodes"])
    parser.add_argument("--eval-every", type=int, default=DEFAULT_RUNNER_VALUES["eval_every"])
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
    print("\n실행 작업: Best/Worst 3구 seed 반복 실험", flush=True)
    print("대상: A2C Best/Worst 3구 + REINFORCE Best/Worst 3구", flush=True)
    print(f"base_seed={args.base_seed} 기존 full run 재사용, additional_seeds={args.additional_seeds}", flush=True)
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
    if not args.use_existing_seed42:
        cmd.append("--no-use-existing-seed42")
    if not args.skip_existing:
        cmd.append("--no-skip-existing")
    if args.dry_run:
        cmd.append("--dry-run")
    # 이 경우 wrapper의 dry-run도 seed 스크립트를 --dry-run으로 실행해야
    # Best/Worst 3구 x 추가 seed 실행 목록까지 확인할 수 있다.
    run_command(cmd, False)


def run_final_chronological(args: argparse.Namespace) -> None:
    """최종 기준 split으로 A2C/REINFORCE 전체 학습 후 seed 반복 실험을 실행한다."""
    if args.candidate_top_k is None:
        args.candidate_top_k = 12
    if not args.vae_mode:
        args.vae_mode = "none"
    args.split_mode = "chronological"
    args.seed = args.base_seed
    args.district = "ALL"

    print("\n실행 작업: 최종 chronological 전체 실험", flush=True)
    print("1. A2C seed 42 전체 25개 구 학습", flush=True)
    print("2. REINFORCE seed 42 전체 25개 구 학습", flush=True)
    print("3. Best/Worst 3구 seed 123/777 추가 학습 및 요약", flush=True)
    print(f"episodes={args.episodes}, eval_every={args.eval_every}, top_k={args.candidate_top_k}", flush=True)

    for algorithm in ["a2c", "reinforce"]:
        stage_args = argparse.Namespace(**vars(args))
        stage_args.algorithm = algorithm
        stage_args.district = "ALL"
        run_training(stage_args, should_prompt=False)

    seed_args = argparse.Namespace(**vars(args))
    seed_args.use_existing_seed42 = True
    run_seed_sensitivity(seed_args)


def main() -> None:
    """선택한 담당 실험을 실행한다."""
    args = parse_args()
    if not args.mode:
        has_cli_args = bool(args.algorithm or args.district or args.candidate_top_k is not None or args.dry_run)
        args.mode = "train" if has_cli_args else choose_mode()
    should_prompt = not args.algorithm or not args.district or args.candidate_top_k is None

    if args.mode == "vae":
        run_vae(args)
    elif args.mode == "seed":
        run_seed_sensitivity(args)
    elif args.mode == "final":
        run_final_chronological(args)
    else:
        run_training(args, should_prompt)


if __name__ == "__main__":
    main()
