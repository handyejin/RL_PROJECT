"""수정 state REINFORCE pure 실행 파일.

알고리즘:
    REINFORCE + Reward-to-Go + Value Network baseline

State:
    원본 observation 뒤에 capacity/forecast 기반 feature를 추가한다.

보조장치:
    BC, residual policy, rollback 없이 순수 policy gradient만 학습한다.
"""

from __future__ import annotations

import sys

from src.agents.ours.common import reinforce_core


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
    "0",
    "--tag",
    "reinforce_modified_state_pure",
]


def main() -> None:
    """수정 state pure REINFORCE 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    reinforce_core.main()


if __name__ == "__main__":
    main()
