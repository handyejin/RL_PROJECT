"""우리 RL 실험을 터미널에서 선택 실행하는 wrapper.

이 파일은 팀원이 긴 학습 명령을 직접 외우지 않아도 되도록 만든 얇은
진입점이다. 실제 알고리즘은 `common/*_core.py`에 있고, 공통 경로/명령
생성 로직은 `common.runner_config`에 둔다.

실행 예:
    PYTHONPATH=. python -m src.agents.ours.run_interactive
    PYTHONPATH=. python -m src.agents.ours.run_interactive --task vae --district ALL
    PYTHONPATH=. python -m src.agents.ours.run_interactive --algorithm dqn --district ALL --candidate-top-k 3
"""

from __future__ import annotations

import argparse
import subprocess

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


def choose_task() -> str:
    """터미널에서 VAE 생성 또는 RL 학습 중 실행할 작업을 선택한다."""
    print("\n실행 작업을 선택하세요.")
    print("  1. RL 학습 실행")
    print("  2. VAE latent 파일 생성")
    choice = input("선택 [1/2, Enter=1]: ").strip()
    return "vae" if choice == "2" else "rl"


def choose_algorithm() -> str:
    """터미널에서 실행할 알고리즘을 선택한다."""
    print("\n알고리즘을 선택하세요.")
    print("  1. REINFORCE")
    print("  2. A2C")
    print("  3. DQN (Double DQN)")
    print("  4. PPO")
    print("  5. Contextual Bandit (LinUCB)")
    choice = input("선택 [1/2/3/4/5]: ").strip()
    return {"1": "reinforce", "3": "dqn", "4": "ppo", "5": "bandit"}.get(choice, "a2c")


def choose_district() -> str:
    """ALL 또는 자주 쓰는 구, 직접 입력 중 실행 지역을 선택한다."""
    print("\n실행 지역을 선택하세요.")
    print("  1. ALL (25개 구 전체 순차 실행)")
    print("  2. 영등포구")
    print("  3. 마포구")
    print("  4. 관악구")
    print("  5. 직접 입력")
    choice = input("선택 [1/2/3/4/5]: ").strip()
    if choice == "1":
        return "ALL"
    if choice == "2":
        return "영등포구"
    if choice == "3":
        return "마포구"
    if choice == "4":
        return "관악구"
    typed = input("구 이름 입력 예: 양천구: ").strip()
    return typed or "영등포구"


def choose_top_k() -> int:
    """터미널에서 Top-K 후보 action 개수를 선택한다."""
    print("\nTop-K 후보 개수를 선택하세요.")
    print("  1. Top-K = 3  (강한 후보 축소, 안정성 확인용)")
    print("  2. Top-K = 6  (중간 절충)")
    print("  3. Top-K = 9  (넓은 후보군)")
    print("  4. Top-K = 12 (기존 기준)")
    print("  5. 직접 입력")
    choice = input("선택 [1/2/3/4/5, Enter=12]: ").strip()
    if choice in {"1", "2", "3"}:
        return {"1": 3, "2": 6, "3": 9}[choice]
    if choice == "5":
        typed = input("Top-K 숫자 입력 예: 3: ").strip()
        try:
            return max(2, int(typed))
        except ValueError:
            print("숫자가 아니어서 기본값 12를 사용합니다.")
    return 12


def choose_vae_mode() -> str:
    """터미널에서 VAE latent state feature 사용 여부를 선택한다."""
    print("\nVAE 수요 latent feature를 사용할까요?")
    print("  1. 사용 안 함 (기본)")
    print("  2. 사용함: Forecast + VAE latent")
    choice = input("선택 [1/2, Enter=1]: ").strip()
    return "demand_latent" if choice == "2" else "none"


def parse_args() -> argparse.Namespace:
    """대화형 실행과 명령형 실행을 모두 지원하는 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="Interactive runner for our RL experiments.")
    parser.add_argument("--task", choices=["rl", "vae"], default="")
    parser.add_argument("--algorithm", choices=["reinforce", "a2c", "dqn", "ppo", "bandit"], default="")
    parser.add_argument("--district", default="")
    parser.add_argument("--processed-dir", default=DEFAULT_RUNNER_VALUES["processed_dir"])
    parser.add_argument("--forecast-dir", default=DEFAULT_RUNNER_VALUES["forecast_dir"])
    parser.add_argument("--vae-latent-dir", default=DEFAULT_RUNNER_VALUES["vae_latent_dir"])
    parser.add_argument("--capacity-path", default=DEFAULT_RUNNER_VALUES["capacity_path"])
    parser.add_argument("--episodes", type=int, default=DEFAULT_RUNNER_VALUES["episodes"])
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_RUNNER_VALUES["total_timesteps"])
    parser.add_argument("--eval-every", type=int, default=DEFAULT_RUNNER_VALUES["eval_every"])
    parser.add_argument("--eval-every-timesteps", type=int, default=DEFAULT_RUNNER_VALUES["eval_every_timesteps"])
    parser.add_argument("--n-train-dates", type=int, default=DEFAULT_RUNNER_VALUES["n_train_dates"])
    parser.add_argument("--bc-epochs", type=int, default=DEFAULT_RUNNER_VALUES["bc_epochs"])
    parser.add_argument(
        "--future-mode",
        choices=["none", "oracle_net", "oracle_inout", "history_net", "forecast_projected_travel"],
        default=DEFAULT_RUNNER_VALUES["future_mode"],
    )
    parser.add_argument("--future-horizon", type=int, default=DEFAULT_RUNNER_VALUES["future_horizon"])
    parser.add_argument("--vae-mode", choices=["none", "demand_latent"], default="")
    parser.add_argument("--vae-latent-dim", type=int, default=DEFAULT_RUNNER_VALUES["vae_latent_dim"])
    parser.add_argument("--vae-epochs", type=int, default=DEFAULT_RUNNER_VALUES["vae_epochs"])
    parser.add_argument("--vae-hidden", type=int, default=DEFAULT_RUNNER_VALUES["vae_hidden"])
    parser.add_argument("--vae-batch-size", type=int, default=DEFAULT_RUNNER_VALUES["vae_batch_size"])
    parser.add_argument("--vae-lr", type=float, default=DEFAULT_RUNNER_VALUES["vae_lr"])
    parser.add_argument("--vae-beta", type=float, default=DEFAULT_RUNNER_VALUES["vae_beta"])
    parser.add_argument("--candidate-top-k", type=int, default=None)
    parser.add_argument("--candidate-mode", choices=["imbalance", "forecast_imbalance"], default=DEFAULT_RUNNER_VALUES["candidate_mode"])
    parser.add_argument("--candidate-travel-coef", type=float, default=DEFAULT_RUNNER_VALUES["candidate_travel_coef"])
    parser.add_argument("--candidate-zone-mode", choices=["none", "static3"], default=DEFAULT_RUNNER_VALUES["candidate_zone_mode"])
    parser.add_argument("--candidate-zone-penalty", type=float, default=DEFAULT_RUNNER_VALUES["candidate_zone_penalty"])
    parser.add_argument("--candidate-feature-mode", choices=["none", "basic"], default=DEFAULT_RUNNER_VALUES["candidate_feature_mode"])
    parser.add_argument("--ppo-learning-rate", type=float, default=DEFAULT_RUNNER_VALUES["ppo_learning_rate"])
    parser.add_argument("--ppo-ent-coef", type=float, default=DEFAULT_RUNNER_VALUES["ppo_ent_coef"])
    parser.add_argument("--ppo-target-kl", type=float, default=DEFAULT_RUNNER_VALUES["ppo_target_kl"])
    parser.add_argument("--ppo-clip-range", type=float, default=DEFAULT_RUNNER_VALUES["ppo_clip_range"])
    parser.add_argument("--ppo-n-epochs", type=int, default=DEFAULT_RUNNER_VALUES["ppo_n_epochs"])
    parser.add_argument("--ppo-n-steps", type=int, default=DEFAULT_RUNNER_VALUES["ppo_n_steps"])
    parser.add_argument("--ppo-batch-size", type=int, default=DEFAULT_RUNNER_VALUES["ppo_batch_size"])
    parser.add_argument("--dqn-reward-scale", type=float, default=DEFAULT_RUNNER_VALUES["dqn_reward_scale"])
    parser.add_argument("--dqn-exploration-initial-eps", type=float, default=DEFAULT_RUNNER_VALUES["dqn_exploration_initial_eps"])
    parser.add_argument("--dqn-exploration-fraction", type=float, default=DEFAULT_RUNNER_VALUES["dqn_exploration_fraction"])
    parser.add_argument("--dqn-exploration-final-eps", type=float, default=DEFAULT_RUNNER_VALUES["dqn_exploration_final_eps"])
    parser.add_argument("--bandit-alpha", type=float, default=DEFAULT_RUNNER_VALUES["bandit_alpha"])
    parser.add_argument("--bandit-l2", type=float, default=DEFAULT_RUNNER_VALUES["bandit_l2"])
    parser.add_argument("--bandit-reward-scale", type=float, default=DEFAULT_RUNNER_VALUES["bandit_reward_scale"])
    parser.add_argument("--tag", default=DEFAULT_RUNNER_VALUES["tag"])
    parser.add_argument("--device", default=DEFAULT_RUNNER_VALUES["device"], choices=["auto", "cpu", "mps"])
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=DEFAULT_RUNNER_VALUES["progress"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], dry_run: bool) -> None:
    """명령을 출력하고, dry-run이 아니면 프로젝트 루트에서 실행한다."""
    print("명령:", flush=True)
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    code = subprocess.run(cmd, cwd=PROJECT_ROOT, env=subprocess_env()).returncode
    if code != 0:
        raise SystemExit(code)


def run_vae(args: argparse.Namespace) -> None:
    """선택한 구 또는 25개 구 전체에 대해 VAE latent 파일을 생성한다."""
    if not args.district:
        args.district = choose_district()
    districts = selected_districts(args.district)
    print("\n실행 작업: VAE latent 생성", flush=True)
    print(f"실행 지역: {', '.join(districts)}", flush=True)
    print(f"latent_dim={args.vae_latent_dim}, epochs={args.vae_epochs}, out_dir={project_path(args.vae_latent_dir)}", flush=True)
    if not ensure_vae_inputs(args):
        return
    for index, district in enumerate(districts, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[{index}/{len(districts)}] {district} VAE latent 생성", flush=True)
        run_command(build_vae_command(args, district), args.dry_run)


def run_training(args: argparse.Namespace, should_prompt: bool) -> None:
    """선택한 알고리즘을 선택한 구 또는 25개 구 전체에 대해 순차 실행한다."""
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
    print(f"VAE latent: {'사용' if args.vae_mode != 'none' else '미사용'}", flush=True)
    if args.algorithm in {"reinforce", "a2c"}:
        print(f"episodes={args.episodes}, eval_every={args.eval_every}, top_k={args.candidate_top_k}", flush=True)
    else:
        print(f"timesteps={args.total_timesteps}, eval_every_timesteps={args.eval_every_timesteps}, top_k={args.candidate_top_k}", flush=True)

    for index, district in enumerate(districts, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[{index}/{len(districts)}] {district} 실행", flush=True)
        if not ensure_training_inputs(args, district):
            continue
        run_command(build_training_command(args, district), args.dry_run)


def main() -> None:
    """선택한 작업을 실행한다."""
    args = parse_args()
    if not args.task:
        has_rl_cli_args = bool(args.algorithm or args.district or args.candidate_top_k is not None or args.dry_run)
        args.task = "rl" if has_rl_cli_args else choose_task()
    should_prompt = not args.algorithm or not args.district or args.candidate_top_k is None
    if args.task == "vae":
        run_vae(args)
    else:
        run_training(args, should_prompt)


if __name__ == "__main__":
    main()
