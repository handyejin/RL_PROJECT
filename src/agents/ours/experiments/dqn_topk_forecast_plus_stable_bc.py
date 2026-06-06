"""Forecast Top-K Plus DQN BC 안정화 실험.

목표:
    BC 직후의 좋은 policy가 RL fine-tuning 중 급격히 무너지지 않게 한다.

핵심 설정:
    - BC 이후 낮은 epsilon으로 조심스럽게 탐색
    - n-step target으로 delayed reward 완화
    - 평가 악화 시 best checkpoint rollback
"""

from __future__ import annotations

import sys

from src.agents.ours.common import dqn_core


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
    "--candidate-travel-coef",
    "0.20",
    "--candidate-zone-mode",
    "static3",
    "--candidate-zone-penalty",
    "1.0",
    "--candidate-feature-mode",
    "basic",
    "--bc-epochs",
    "10",
    "--bc-policy",
    "masked_heuristic",
    "--learning-rate",
    "0.00003",
    "--learning-starts",
    "5000",
    "--n-steps",
    "3",
    "--train-freq",
    "1",
    "--gradient-steps",
    "1",
    "--target-update-interval",
    "500",
    "--exploration-initial-eps",
    "0.05",
    "--exploration-fraction",
    "0.20",
    "--exploration-final-eps",
    "0.005",
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "2",
    "--tag",
    "topk_forecast_plus_stable_bc_dqn",
]


def main() -> None:
    """Forecast Top-K Plus BC DQN 안정화 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    dqn_core.main()


if __name__ == "__main__":
    main()
