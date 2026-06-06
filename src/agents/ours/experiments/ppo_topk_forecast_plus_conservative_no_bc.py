"""Forecast Top-K Plus PPO no-BC conservative 실험.

목표:
    Behavior Cloning 없이 PPO update 폭을 줄여 baseline 이상 성능을 안정적으로 찾는다.

핵심 설정:
    - 낮은 learning rate
    - 작은 clip range
    - 적은 epoch
    - target KL로 policy update 제한
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
    "--clip-range",
    "0.05",
    "--ent-coef",
    "0.003",
    "--n-epochs",
    "3",
    "--target-kl",
    "0.01",
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "8",
    "--tag",
    "topk_forecast_plus_conservative_no_bc_ppo",
]


def main() -> None:
    """Forecast Top-K Plus no-BC PPO conservative 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    ppo_core.main()


if __name__ == "__main__":
    main()
