"""수정 state + Forecast Top-K action PPO pure 실험."""

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
    "--candidate-top-k",
    "12",
    "--candidate-mode",
    "forecast_imbalance",
    "--bc-epochs",
    "0",
    "--tag",
    "topk_forecast_pure_ppo",
]


def main() -> None:
    """Forecast Top-K no-BC PPO 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    ppo_core.main()


if __name__ == "__main__":
    main()
