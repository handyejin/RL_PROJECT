"""원본 state A2C + BC guard 실행 파일.

State:
    팀 공통 RebalanceEnv의 원본 observation만 사용한다.

보조장치:
    Behavior Cloning, validation early stopping, advantage normalization,
    BC anchor KL, best checkpoint rollback을 사용한다.
"""

from __future__ import annotations

import sys

from src.agents.ours.common import a2c_core


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
    "--future-mode",
    "none",
    "--capacity-path",
    "",
    "--forecast-path",
    "",
    "--normalize-advantages",
    "--anchor-coef",
    "0.01",
    "--rollback-to-best-on-eval",
    "--finetune-patience",
    "6",
    "--tag",
    "a2c_original_bc_guarded",
]


def main() -> None:
    """원본 state A2C guarded 설정으로 core 학습 루프를 실행한다."""
    sys.argv[1:1] = DEFAULT_ARGS
    a2c_core.main()


if __name__ == "__main__":
    main()
