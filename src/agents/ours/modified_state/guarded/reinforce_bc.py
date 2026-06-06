"""REINFORCE + 공공데이터 예측/거치대 수 보강 실행 파일.

알고리즘:
    REINFORCE + Reward-to-Go
    + Value Network baseline
    + Behavior Cloning pretraining
    + forecast_projected_travel state feature

State 보강:
    정류소별 1시간 예상 대여/반납을 사용해 예상 재고 방향을 만든다.
    이 값은 실제 미래 수요 oracle이 아니라, 별도 예측 모델의 출력이다.

데이터 보강:
    팀원 env 파일은 수정하지 않고, agent 실행 중 EpisodeData에만
    실제 거치대 수(capacity)와 forecast grid를 붙인다.
"""

from __future__ import annotations

import sys

from src.agents.ours.common import reinforce_core as _reinforce_core


DEFAULT_ARGS = [
    "--episodes",
    "500",
    "--eval-every",
    "50",
    "--n-train-dates",
    "200",
    "--bc-dates",
    "200",
    "--bc-epochs",
    "30",
    "--bc-policy",
    "masked_heuristic",
    "--lr-policy",
    "0.0003",
    "--lr-value",
    "0.001",
    "--future-mode",
    "forecast_projected_travel",
    "--future-horizon",
    "6",
    "--capacity-path",
    "data/processed/station_capacity.csv",
    "--forecast-path",
    "data/processed/demand_forecast_1h_rlholdout_seed42.parquet",
    "--tag",
    "reinforce_forecast_capacity",
]


def main() -> None:
    """공공데이터 forecast/capacity 보강 REINFORCE 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    _reinforce_core.main()


if __name__ == "__main__":
    main()
