"""REINFORCE/A2C/PPO/PPO_V4/QRDQN 실험을 쉽게 실행하는 터미널 wrapper.

목적:
    팀원이 내부 core 옵션을 모두 외우지 않아도, 알고리즘과 구만 선택해서
    실험을 실행할 수 있게 한다.

알고리즘 → core 모듈 매핑 (모두 src.agents.ours.common.* 아래):
    - REINFORCE  → reinforce_core
    - A2C        → a2c_core
    - PPO        → ppo_core         (--algo ppo)
    - PPO_V4     → ppo_core         (--algo ppo_v4, KL-to-BC)
    - QRDQN      → qrdqn_core

5개 알고리즘이 모두 동일한 ours 환경(forecast + capacity + Top-K 후보)을 사용하므로,
실행 지역 'ALL' 선택 시 25개 구를 순차적으로 학습한다. 알고리즘만 swap 되고
state/action/reward 정의·data 경로·wrapper 구성은 동일하다.

데이터 경로:
    - processed_dir:   data/processed_seoul_all
    - forecast_path:   data/forecast_by_gu/demand_forecast_1h_{구}.parquet
    - capacity_path:   data/processed/station_capacity.csv

실행 예:
    PYTHONPATH=. python -m src.agents.ours.run_interactive
    PYTHONPATH=. python -m src.agents.ours.run_interactive --algorithm a2c --district 영등포구
    PYTHONPATH=. python -m src.agents.ours.run_interactive --algorithm ppo_v4 --district ALL
    PYTHONPATH=. python -m src.agents.ours.run_interactive --algorithm qrdqn --district 강남구 --timesteps 50000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DISTRICTS = [
    "강남구",
    "강동구",
    "강북구",
    "강서구",
    "관악구",
    "광진구",
    "구로구",
    "금천구",
    "노원구",
    "도봉구",
    "동대문구",
    "동작구",
    "마포구",
    "서대문구",
    "서초구",
    "성동구",
    "성북구",
    "송파구",
    "양천구",
    "영등포구",
    "용산구",
    "은평구",
    "종로구",
    "중구",
    "중랑구",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 알고리즘별 ours core 모듈 매핑
ALGO_MODULE = {
    "reinforce": "src.agents.ours.common.reinforce_core",
    "a2c": "src.agents.ours.common.a2c_core",
    "ppo": "src.agents.ours.common.ppo_core",
    "ppo_v4": "src.agents.ours.common.ppo_core",
    "qrdqn": "src.agents.ours.common.qrdqn_core",
}

# SB3 기반 알고리즘 (timesteps 단위로 학습)
SB3_ALGOS = {"ppo", "ppo_v4", "qrdqn"}

# 5개 알고리즘 메뉴 옵션
ALGO_MENU = [
    ("1", "reinforce", "REINFORCE (ours core)"),
    ("2", "a2c", "A2C (ours core)"),
    ("3", "ppo", "PPO (MaskablePPO via ours core)"),
    ("4", "ppo_v4", "PPO_V4 KL-to-BC (ours core)"),
    ("5", "qrdqn", "QRDQN (MaskableQRDQN via ours core)"),
]
ALGO_CHOICES = [algo for _, algo, _ in ALGO_MENU]


def project_path(path: str | Path) -> Path:
    """상대경로를 프로젝트 루트 기준 절대경로로 변환한다."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def choose_algorithm() -> str:
    """터미널에서 실행할 알고리즘을 선택한다."""
    print("\n알고리즘을 선택하세요.")
    for key, _, label in ALGO_MENU:
        print(f"  {key}. {label}")
    valid_keys = "/".join(key for key, _, _ in ALGO_MENU)
    choice = input(f"선택 [{valid_keys}]: ").strip()
    mapping = {key: algo for key, algo, _ in ALGO_MENU}
    return mapping.get(choice, "a2c")


def choose_district() -> str:
    """ALL, 영등포구, 직접 입력 중 실행 지역을 선택한다."""
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


def choose_split_mode() -> str:
    """train/eval 데이터 분할 방식을 선택한다.

    두 모드 모두 seed=42 / 80:20 비율 / EVAL_DATES 7일을 그대로 사용한다.
        1. random        — 기존과 동일. seed=42 셔플 후 앞 80% train / 뒤 20% eval pool.
        2. chronological — 시간순 분할. 2025-01-01 ~ ~10-19 (앞 80%) train,
                                        2025-10-20 ~ 12-31 (뒤 20%) eval pool (계절 OOD).
    """
    print("\ntrain/eval 데이터 분할 방식을 선택하세요.")
    print("  1. Random (seed=42 셔플, 기존 방식)")
    print("  2. Chronological (시간순: train 1월~10월 중순, eval pool 10월 후반~12월)")
    choice = input("선택 [1/2]: ").strip()
    if choice == "2":
        return "chronological"
    return "random"


def build_command(args: argparse.Namespace, district: str) -> list[str]:
    """선택한 알고리즘/구에 맞는 실행 명령을 만든다.

    모든 알고리즘은 ours/common core 모듈을 -m 모드로 실행하며,
    forecast/capacity/Top-K 후보 wrapper 옵션을 동일하게 전달한다.
    REINFORCE/A2C 는 episode 단위, PPO/PPO_V4/QRDQN 은 timestep 단위로 학습 길이를 설정한다.

    로그 디렉토리:
        - reinforce/a2c: logs/actor_critic_{tag}_{algo}_{district}
        - ppo:           logs/ppo_{tag}_{district}
        - ppo_v4:        logs/ppo_v4_{tag}_{district}
        - qrdqn:         logs/qrdqn_{tag}_{district}
    """
    module = ALGO_MODULE[args.algorithm]

    forecast_path = project_path(args.forecast_dir) / f"demand_forecast_1h_{district}.parquet"
    processed_dir = project_path(args.processed_dir)
    capacity_path = project_path(args.capacity_path)

    if args.algorithm in SB3_ALGOS:
        # SB3 알고리즘은 timestep 단위 — 공통 wrapper 옵션은 reinforce/a2c 와 동일.
        tag = f"{args.tag}_{district}"
        cmd = [
            sys.executable,
            "-m",
            module,
            "--processed-dir",
            str(processed_dir),
            "--district",
            district,
            "--total-timesteps",
            str(args.timesteps),
            "--eval-every",
            str(args.eval_freq),
            "--n-train-dates",
            str(args.n_train_dates),
            "--seed",
            str(args.seed),
            "--bc-epochs",
            "0",
            "--future-mode",
            "forecast_projected_travel",
            "--future-horizon",
            "6",
            "--capacity-path",
            str(capacity_path),
            "--forecast-path",
            str(forecast_path),
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
            "--split-mode",
            args.split_mode,
        ]
        # PPO 계열: Top-K 후보 환경에서 policy 가 과도하게 흔들리지 않도록
        # 보수적 하이퍼파라미터를 적용한다 (실험 보고서 PPO 스펙).
        if args.algorithm in {"ppo", "ppo_v4"}:
            cmd += [
                "--algo", args.algorithm,
                "--learning-rate", "0.0001",
                "--clip-range", "0.1",
                "--target-kl", "0.03",
                "--ent-coef", "0.003",
                "--n-epochs", "5",
                "--n-steps", "256",
                "--batch-size", "128",
            ]
        return cmd

    # reinforce / a2c → episode 단위
    tag = f"{args.tag}_{args.algorithm}_{district}"
    cmd = [
        sys.executable,
        "-m",
        module,
        "--processed-dir",
        str(processed_dir),
        "--district",
        district,
        "--episodes",
        str(args.episodes),
        "--eval-every",
        str(args.eval_every),
        "--n-train-dates",
        str(args.n_train_dates),
        "--bc-epochs",
        "0",
        "--future-mode",
        "forecast_projected_travel",
        "--future-horizon",
        "6",
        "--capacity-path",
        str(capacity_path),
        "--forecast-path",
        str(forecast_path),
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
        "--normalize-advantages",
        "--tag",
        tag,
        "--device",
        args.device,
        "--split-mode",
        args.split_mode,
    ]
    if args.algorithm == "a2c":
        cmd += ["--bc-val-dates", "0", "--anchor-coef", "0.0"]
    if args.progress:
        cmd.append("--progress")
    return cmd


def ensure_inputs(args: argparse.Namespace, district: str) -> bool:
    """학습에 필요한 전처리/forecast 파일이 있는지 확인한다.

    5개 알고리즘이 모두 동일한 ours env(processed_dir + forecast + capacity)를
    사용하므로 검사 로직도 동일하다.
    """
    missing = []
    processed_dir = project_path(args.processed_dir)
    forecast_path = project_path(args.forecast_dir) / f"demand_forecast_1h_{district}.parquet"
    if not (processed_dir / "stations.parquet").exists():
        missing.append(str(processed_dir / "stations.parquet"))
    if not (processed_dir / "trips.parquet").exists():
        missing.append(str(processed_dir / "trips.parquet"))
    if not forecast_path.exists():
        missing.append(str(forecast_path))

    if missing:
        print(f"\n[{district}] 필요한 파일이 없습니다.")
        for path in missing:
            print(f"  - {path}")
        print("전처리/수요예측 파일 생성 후 다시 실행하세요.")
        return False
    return True


def parse_args() -> argparse.Namespace:
    """대화형 실행과 명령형 실행을 모두 지원하는 옵션을 정의한다."""
    parser = argparse.ArgumentParser(
        description="Interactive runner for REINFORCE/A2C/PPO/PPO_V4/QRDQN experiments."
    )
    parser.add_argument("--algorithm", choices=ALGO_CHOICES, default="")
    parser.add_argument("--district", default="")
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological"],
        default="",
        help="train/eval 분할 방식. 빈값이면 대화형 prompt 로 묻는다.",
    )
    # 모든 알고리즘이 공유하는 ours env 데이터 경로
    parser.add_argument("--processed-dir", default="data/processed_seoul_all")
    parser.add_argument("--forecast-dir", default="data/forecast_by_gu")
    parser.add_argument("--capacity-path", default="data/processed/station_capacity.csv")
    # reinforce/a2c 학습 길이 (episode 단위)
    parser.add_argument("--episodes", type=int, default=500,
                        help="reinforce/a2c 학습 episode 수")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="reinforce/a2c 평가 주기 (episodes)")
    # ppo/ppo_v4/qrdqn 학습 길이 (env step 단위)
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="ppo/ppo_v4/qrdqn 학습 timesteps")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="ppo/ppo_v4/qrdqn 평가 주기 (env steps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="ppo/ppo_v4/qrdqn 학습 seed")
    parser.add_argument("--n-train-dates", type=int, default=200)
    parser.add_argument("--candidate-top-k", type=int, default=12)
    parser.add_argument("--tag", default="interactive")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps"])
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """선택한 알고리즘을 선택한 구 또는 25개 구 전체에 대해 순차 실행한다."""
    args = parse_args()
    if not args.algorithm:
        args.algorithm = choose_algorithm()
    if not args.district:
        args.district = choose_district()
    if not args.split_mode:
        args.split_mode = choose_split_mode()

    districts = DISTRICTS if args.district.upper() == "ALL" else [args.district]
    print(f"\n실행 알고리즘: {args.algorithm.upper()}")
    print(f"실행 지역: {', '.join(districts)}")
    print(f"데이터 분할: {args.split_mode}")
    if args.algorithm in SB3_ALGOS:
        print(
            f"timesteps={args.timesteps}, eval_freq={args.eval_freq}, "
            f"n_train_dates={args.n_train_dates}, top_k={args.candidate_top_k}"
        )
    else:
        print(
            f"episodes={args.episodes}, eval_every={args.eval_every}, "
            f"top_k={args.candidate_top_k}"
        )

    for index, district in enumerate(districts, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(districts)}] {district} 실행")
        if not ensure_inputs(args, district):
            continue
        cmd = build_command(args, district)
        print("명령:")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        # Windows의 기본 콘솔 코덱(cp949 등)이 학습 스크립트의 비-ASCII 문자(em-dash 등)
        # 출력에서 UnicodeEncodeError를 일으키는 것을 방지한다.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        code = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
