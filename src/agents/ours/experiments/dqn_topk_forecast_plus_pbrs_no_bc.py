"""Forecast Top-K Plus + PBRS DQN no-BC 실험.

목표:
    BC 없이도 delayed reward 문제를 줄였을 때 DQN이 baseline을 넘는지 확인한다.

주의:
    PBRS는 학습 reward에만 적용하고, 평가는 원본 reward로 수행한다.
"""

from __future__ import annotations

import sys

from src.agents.ours.experiments import dqn_topk_forecast_plus_stable_no_bc


DEFAULT_ARGS = [
    "--agent-shaping-scale",
    "5.0",
    "--agent-shaping-gamma",
    "0.99",
    "--tag",
    "topk_forecast_plus_pbrs_no_bc_dqn",
]


def main() -> None:
    """Forecast Top-K Plus + PBRS no-BC DQN 설정으로 실행한다."""
    sys.argv[1:1] = dqn_topk_forecast_plus_stable_no_bc.DEFAULT_ARGS + DEFAULT_ARGS
    dqn_topk_forecast_plus_stable_no_bc.dqn_core.main()


if __name__ == "__main__":
    main()
