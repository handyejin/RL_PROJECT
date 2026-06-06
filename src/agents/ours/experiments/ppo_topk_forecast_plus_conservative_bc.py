"""Forecast Top-K Plus PPO BC conservative 실험.

목표:
    BC로 만든 policy를 PPO fine-tuning이 크게 망가뜨리지 않으면서 개선한다.

핵심 설정:
    - 작은 PPO update
    - target KL
    - 평가 악화 시 best checkpoint rollback
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
    "10",
    "--bc-policy",
    "masked_heuristic",
    "--learning-rate",
    "0.00003",
    "--clip-range",
    "0.05",
    "--ent-coef",
    "0.0",
    "--n-epochs",
    "3",
    "--target-kl",
    "0.01",
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "4",
    "--tag",
    "topk_forecast_plus_conservative_bc_ppo",
]


def main() -> None:
    """Forecast Top-K Plus BC PPO conservative 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    ppo_core.main()


if __name__ == "__main__":
    main()
