"""수정 state + Forecast Top-K Plus A2C guarded 실험."""

from __future__ import annotations

import sys

from src.agents.ours.common import a2c_core


DEFAULT_ARGS = [
    "--episodes",
    "500",
    "--eval-every",
    "50",
    "--n-train-dates",
    "200",
    "--bc-dates",
    "200",
    "--bc-val-dates",
    "30",
    "--bc-epochs",
    "20",
    "--bc-patience",
    "5",
    "--bc-policy",
    "masked_heuristic",
    "--lr-policy",
    "0.0001",
    "--lr-value",
    "0.0003",
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
    "--anchor-coef",
    "0.01",
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "8",
    "--tag",
    "topk_forecast_plus_a2c",
]


def main() -> None:
    """Forecast Top-K Plus A2C 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    a2c_core.main()


if __name__ == "__main__":
    main()
