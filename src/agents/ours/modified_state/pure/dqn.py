"""DQN + capacity/forecast state 보강 실행 파일.

State:
    원본 observation 뒤에 실제 capacity와 1시간 예측 수요 기반 feature를 추가한다.

보조장치:
    BC/rollback 없이 MaskableDQN을 학습한다.
"""

from __future__ import annotations

from src.agents.ours.common.dqn_core import main


if __name__ == "__main__":
    main()
