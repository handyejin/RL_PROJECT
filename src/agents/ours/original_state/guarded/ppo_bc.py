"""원본 state PPO + BC 실행 파일.

State:
    팀 공통 RebalanceEnv의 원본 observation만 사용한다.

보조장치:
    most_imbalanced teacher를 Behavior Cloning으로 먼저 모방한 뒤
    MaskablePPO 학습을 진행한다.
"""

from __future__ import annotations

import sys

from src.agents.ours.common import ppo_core


DEFAULT_ARGS = [
    "--future-mode",
    "none",
    "--capacity-path",
    "",
    "--forecast-path",
    "",
    "--bc-epochs",
    "30",
    "--bc-policy",
    "masked_heuristic",
    "--tag",
    "ppo_original_state_bc",
]


def main() -> None:
    """원본 state guarded PPO 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    ppo_core.main()


if __name__ == "__main__":
    main()
