"""Forecast Top-K Plus DQN no-BC 안정화 실험.

목표:
    Behavior Cloning 없이 DQN이 MostImbalanced baseline을 안정적으로 넘는지 확인한다.

핵심 설정:
    - forecast 기반 Top-K 후보 action
    - 후보별 feature observation
    - n-step target으로 지연 reward를 조금 더 빠르게 반영
    - 평가가 나빠지면 best checkpoint로 rollback
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
    "0",
    "--learning-rate",
    "0.00005",
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
    "0.30",
    "--exploration-fraction",
    "0.60",
    "--exploration-final-eps",
    "0.02",
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "8",
    "--tag",
    "topk_forecast_plus_stable_no_bc_dqn",
]


def main() -> None:
    """Forecast Top-K Plus no-BC DQN 안정화 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    dqn_core.main()


if __name__ == "__main__":
    main()
