"""수정 state PPO + BC 실행 파일.

State:
    원본 observation 뒤에 capacity/forecast 기반 feature를 추가한다.

보조장치:
    공평 비교를 위해 기본 teacher는 original guarded와 같은
    most_imbalanced 기반 masked_heuristic을 사용한다.
"""

from __future__ import annotations

import sys

from src.agents.ours.common import ppo_core


DEFAULT_ARGS = [
    "--future-mode",
    "forecast_projected_travel",
    "--future-horizon",
    "6",
    "--capacity-path",
    "data/processed/station_capacity.csv",
    "--forecast-path",
    "data/processed/demand_forecast_1h_rlholdout_seed42.parquet",
    "--bc-epochs",
    "30",
    "--bc-policy",
    "masked_heuristic",
    "--tag",
    "ppo_modified_state_bc",
]


def main() -> None:
    """수정 state guarded PPO 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    ppo_core.main()


if __name__ == "__main__":
    main()
