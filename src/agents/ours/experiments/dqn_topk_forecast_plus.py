"""수정 state + Forecast Top-K Plus DQN guarded 실험.

Plus 구성:
    - forecast projected imbalance 기준 Top-K 후보
    - 이동 step penalty
    - 마포구 3권역 soft penalty
    - 후보별 feature observation

평가:
    학습 중 보조 wrapper를 쓰더라도 평가는 원본 reward 기준으로 수행한다.
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
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "8",
    "--tag",
    "topk_forecast_plus_dqn",
]


def main() -> None:
    """Forecast Top-K Plus DQN 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    dqn_core.main()


if __name__ == "__main__":
    main()
