"""REINFORCE 기본형 실행 파일.

알고리즘:
    REINFORCE + Reward-to-Go + Value Network baseline

State:
    팀 공통 RebalanceEnv의 원본 observation만 사용한다.

Action:
    현재 트럭이 이동할 정류소 index를 선택한다.

Reward:
    팀 공통 환경 reward를 그대로 사용한다.
"""

from __future__ import annotations

import sys

from src.agents.ours.common import reinforce_core as _reinforce_core


DEFAULT_ARGS = [
    "--future-mode",
    "none",
    "--bc-epochs",
    "0",
    "--tag",
    "reinforce_basic",
]


def main() -> None:
    """기본 REINFORCE 설정을 CLI 기본값으로 넣고 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    _reinforce_core.main()


if __name__ == "__main__":
    main()
