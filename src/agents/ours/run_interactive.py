"""REINFORCE/A2C 실험을 쉽게 실행하는 터미널 wrapper.

목적:
    팀원이 내부 core 옵션을 모두 외우지 않아도, 알고리즘과 구만 선택해서
    수정 state + forecast + Top-K 실험을 실행할 수 있게 한다.

기본 설정:
    - processed data: data/processed_seoul_all
    - forecast data: data/forecast_by_gu/demand_forecast_1h_{구}.parquet
    - capacity data: data/processed/station_capacity.csv
    - action 후보: forecast imbalance 기준 Top-K 12
    - BC 없음, rollback 없음
    - 진행률 표시: --progress 사용

실행 예:
    PYTHONPATH=. python -m src.agents.ours.run_interactive
    PYTHONPATH=. python -m src.agents.ours.run_interactive --algorithm a2c --district 영등포구
    PYTHONPATH=. python -m src.agents.ours.run_interactive --algorithm reinforce --district ALL
"""

from __future__ import annotations

import argparse
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


def choose_algorithm() -> str:
    """터미널에서 실행할 알고리즘을 선택한다."""
    print("\n알고리즘을 선택하세요.")
    print("  1. REINFORCE")
    print("  2. A2C")
    choice = input("선택 [1/2]: ").strip()
    return "reinforce" if choice == "1" else "a2c"


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


def build_command(args: argparse.Namespace, district: str) -> list[str]:
    """선택한 알고리즘/구에 맞는 core 실행 명령을 만든다."""
    module = "src.agents.ours.common.a2c_core"
    if args.algorithm == "reinforce":
        module = "src.agents.ours.common.reinforce_core"

    forecast_path = Path(args.forecast_dir) / f"demand_forecast_1h_{district}.parquet"
    tag = f"{args.tag}_{args.algorithm}_{district}"

    cmd = [
        sys.executable,
        "-m",
        module,
        "--processed-dir",
        args.processed_dir,
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
        args.capacity_path,
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
    ]
    if args.algorithm == "a2c":
        cmd += ["--bc-val-dates", "0", "--anchor-coef", "0.0"]
    if args.progress:
        cmd.append("--progress")
    return cmd


def ensure_inputs(args: argparse.Namespace, district: str) -> bool:
    """학습에 필요한 전처리/forecast 파일이 있는지 확인한다."""
    processed_dir = Path(args.processed_dir)
    forecast_path = Path(args.forecast_dir) / f"demand_forecast_1h_{district}.parquet"
    missing = []
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
        print("전처리/수요예측 생성 후 다시 실행하세요.")
        return False
    return True


def parse_args() -> argparse.Namespace:
    """대화형 실행과 명령형 실행을 모두 지원하는 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="Interactive runner for our REINFORCE/A2C experiments.")
    parser.add_argument("--algorithm", choices=["reinforce", "a2c"], default="")
    parser.add_argument("--district", default="")
    parser.add_argument("--processed-dir", default="data/processed_seoul_all")
    parser.add_argument("--forecast-dir", default="data/forecast_by_gu")
    parser.add_argument("--capacity-path", default="data/processed/station_capacity.csv")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
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

    districts = DISTRICTS if args.district.upper() == "ALL" else [args.district]
    print(f"\n실행 알고리즘: {args.algorithm.upper()}")
    print(f"실행 지역: {', '.join(districts)}")
    print(f"episodes={args.episodes}, eval_every={args.eval_every}, top_k={args.candidate_top_k}")

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
        code = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
