"""DQN 소규모 환경 에이전트 — top-N 불균형 정류소 + 트럭 1대 + forecast 기반.

원본 src/agents/dqn_small.py 가 untracked 상태에서 소실되어,
현재 maintained 버전인 src/agents/ours/common/dqn_small_core.py 를 그대로 노출한다.
구현·CLI·하이퍼파라미터는 dqn_small_core 와 100% 동일하다.

특징:
    - 출퇴근 불균형 압력 top-N(default 15) 정류소만 선택
    - 트럭 1대로 축소하여 신용할당 단순화
    - forecast 1시간 예상 수요로 state 보강
    - 실제 거치대 수(capacity) override
    - Masked DQN + Double DQN target

실행 예:
    PYTHONPATH=. python -m src.agents.dqn_small \\
        --district 영등포구 --processed-dir data/processed_seoul_all \\
        --forecast-path data/forecast_by_gu/demand_forecast_1h_영등포구.parquet \\
        --max-stations 15 --n-trucks 1 --total-timesteps 400000
"""

from __future__ import annotations

from src.agents.ours.common.dqn_small_core import *  # noqa: F401,F403
from src.agents.ours.common.dqn_small_core import main


if __name__ == "__main__":
    main()
