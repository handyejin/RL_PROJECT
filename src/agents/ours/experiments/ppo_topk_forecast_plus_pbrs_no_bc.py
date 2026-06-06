"""Forecast Top-K Plus + PBRS PPO no-BC 실험.

목표:
    BC 없이 PBRS만 추가해 delayed reward를 완화했을 때 PPO가 baseline을 넘는지 확인한다.

주의:
    PBRS는 학습 reward에만 적용하고, 평가는 원본 reward로 수행한다.
"""

from __future__ import annotations

import sys

from src.agents.ours.experiments import ppo_topk_forecast_plus_conservative_no_bc


DEFAULT_ARGS = [
    "--agent-shaping-scale",
    "5.0",
    "--agent-shaping-gamma",
    "0.99",
    "--tag",
    "topk_forecast_plus_pbrs_no_bc_ppo",
]


def main() -> None:
    """Forecast Top-K Plus + PBRS no-BC PPO 설정으로 실행한다."""
    sys.argv[1:1] = ppo_topk_forecast_plus_conservative_no_bc.DEFAULT_ARGS + DEFAULT_ARGS
    ppo_topk_forecast_plus_conservative_no_bc.ppo_core.main()


if __name__ == "__main__":
    main()
