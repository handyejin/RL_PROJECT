"""DQN 기본형 실행 파일.

State:
    팀 공통 RebalanceEnv의 원본 observation만 사용한다.

보조장치:
    capacity override와 forecast feature를 사용하지 않는다.
"""

from __future__ import annotations

import sys

from src.agents.ours.common import dqn_core


DEFAULT_ARGS = [
    "--future-mode",
    "none",
    "--capacity-path",
    "",
    "--forecast-path",
    "",
    "--tag",
    "dqn_original_state",
]


def main() -> None:
    """원본 state DQN 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    dqn_core.main()


if __name__ == "__main__":
    main()
