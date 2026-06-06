"""A2C + 공공데이터 예측/거치대 수 보강 실행 파일.

알고리즘:
    A2C + Behavior Cloning validation
    + advantage normalization
    + BC anchor KL
    + best checkpoint rollback
    + forecast_projected_travel state feature

State 보강:
    이전 프로젝트에서 만든 정류소별 1시간 예상 대여/반납을 사용한다.
    현재 재고에 pred_returns_1h - pred_rentals_1h를 더해
    1시간 뒤 예상 불균형을 feature로 제공한다.

데이터 보강:
    팀원 env 파일은 수정하지 않고, agent 실행 중 EpisodeData에만
    실제 거치대 수(capacity)와 forecast grid를 붙인다.
"""

from __future__ import annotations

import sys

from src.agents.ours.common import a2c_core as _a2c_core


DEFAULT_ARGS = [
    "--episodes",
    "500",
    "--eval-every",
    "50",
    "--n-train-dates",
    "200",
    "--bc-dates",
    "200",
    "--bc-val-dates",
    "30",
    "--bc-epochs",
    "30",
    "--bc-patience",
    "5",
    "--bc-policy",
    "masked_heuristic",
    "--lr-policy",
    "0.0001",
    "--lr-value",
    "0.0003",
    "--future-mode",
    "forecast_projected_travel",
    "--future-horizon",
    "6",
    "--capacity-path",
    "data/processed/station_capacity.csv",
    "--forecast-path",
    "data/processed/demand_forecast_1h_rlholdout_seed42.parquet",
    "--normalize-advantages",
    "--anchor-coef",
    "0.01",
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "6",
    "--tag",
    "a2c_forecast_capacity_guarded",
]


def main() -> None:
    """공공데이터 forecast/capacity 보강 A2C 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    _a2c_core.main()


if __name__ == "__main__":
    main()
