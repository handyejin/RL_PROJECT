"""DQN 에이전트 — forecast/capacity 보강 + Masked DQN.

원본 src/agents/dqn.py 가 untracked 상태에서 소실되어,
현재 maintained 버전인 src/agents/ours/algorithms/dqn/core.py 를 그대로 노출한다.
구현·CLI·하이퍼파라미터는 ours/algorithms/dqn/core 와 100% 동일하다.

특징:
    - Masked DQN (action mask 적용)
    - Double DQN target 옵션 (--double-q)
    - Dueling Q-network 옵션 (--dueling-q)
    - forecast_projected_travel state feature
    - 실제 거치대 수(capacity) override

실행 예:
    PYTHONPATH=. python -m src.agents.dqn \\
        --district 영등포구 --processed-dir data/processed_seoul_all \\
        --forecast-path data/forecast_by_gu/demand_forecast_1h_영등포구.parquet \\
        --total-timesteps 400000 --double-q --dueling-q
"""

from __future__ import annotations

from src.agents.ours.algorithms.dqn.core import *  # noqa: F401,F403
from src.agents.ours.algorithms.dqn.core import main


if __name__ == "__main__":
    main()
