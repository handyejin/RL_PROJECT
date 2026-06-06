"""수정 state + Forecast Top-K Plus REINFORCE guarded 실험."""

from __future__ import annotations

import sys

from src.agents.ours.common import reinforce_core


DEFAULT_ARGS = [
    "--n-train-dates",
    "200",
    "--episodes",
    "500",
    "--eval-every",
    "50",
    "--bc-dates",
    "200",
    "--bc-epochs",
    "20",
    "--bc-policy",
    "masked_heuristic",
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
    "--candidate-travel-coef",
    "0.20",
    "--candidate-zone-mode",
    "static3",
    "--candidate-zone-penalty",
    "1.0",
    "--candidate-feature-mode",
    "basic",
    "--normalize-advantages",
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "8",
    "--tag",
    "topk_forecast_plus_reinforce",
]


def main() -> None:
    """Forecast Top-K Plus REINFORCE 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    reinforce_core.main()


if __name__ == "__main__":
    main()
