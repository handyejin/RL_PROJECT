"""전처리 진입점.

사용:
    python scripts/run_preprocess.py                       # 마포구, 10분 step
    python scripts/run_preprocess.py --gu 강남구 마포구    # 여러 자치구
    python scripts/run_preprocess.py --gu all              # 서울 전체
    python scripts/run_preprocess.py --step 5min
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="따릉이 데이터 전처리")
    p.add_argument(
        "--gu",
        nargs="+",
        default=["마포구"],
        help="필터링할 자치구 (예: 마포구 강남구). 'all' 입력 시 서울 전체.",
    )
    p.add_argument("--step", default="10min", help="수요/날씨 resample 주기 (기본 10min)")
    p.add_argument("--out", type=Path, default=None, help="출력 디렉토리 (기본 data/processed)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    keep_gu = None if args.gu == ["all"] else args.gu
    paths = run(keep_gu=keep_gu, step_freq=args.step, out_dir=args.out)
    print("\n=== 출력 파일 ===")
    for name, p in paths.items():
        print(f"  {name:10s} {p}")


if __name__ == "__main__":
    main()
