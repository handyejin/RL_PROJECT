"""학습/평가용 train/eval 날짜 분할 helper.

ours 알고리즘 core들이 동일한 분할 변수
(seed=42, 80/20 비율, 1년 = 365일, EVAL_DATES = eval pool 전체)
를 공유한다. 분할 mode 만 외부에서 선택할 수 있도록 함수로 분리한다.

지원 mode:
    random
        기존 동작. 365 일을 seed=42 로 셔플한 뒤 앞 80% 를 train,
        뒤 20% 를 eval pool 로 사용한다. 계절·요일·공휴일이 train/eval
        양쪽에 균등하게 분포한다.
    chronological
        시간순 분할. 1월 1일 ~ 약 10월 19일 (앞 80%, 292일) 을 train,
        약 10월 20일 ~ 12월 31일 (뒤 20%, 73일) 을 eval pool 로 사용한다.
        train/eval 사이에 계절 분포가 다르므로 OOD 일반화를 측정한다.

두 mode 모두 다음 변수는 동일하게 유지한다:
    seed = 42
    train_ratio = 0.8 → train 292일, eval pool 73일
    EVAL_DATES = sorted(eval_pool)  → eval pool 전체(73일)를 실제 평가 holdout으로 사용
"""

from __future__ import annotations

import datetime
import random
from typing import Literal

SplitMode = Literal["random", "chronological"]


def date_range(start: str, end: str) -> list[str]:
    """yyyy-mm-dd 문자열 리스트 (start~end 포함)."""
    d = datetime.date.fromisoformat(start)
    end_d = datetime.date.fromisoformat(end)
    out: list[str] = []
    while d <= end_d:
        out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


def compute_split(
    mode: SplitMode = "random",
    seed: int = 42,
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    train_ratio: float = 0.8,
    n_eval: int | None = None,
) -> tuple[list[str], list[str]]:
    """선택된 mode 로 (TRAIN_DATES, EVAL_DATES) 를 만든다.

    EVAL_DATES 는 두 mode 모두 eval pool 을 정렬해 사용한다. n_eval 이 None
    이면 eval pool 전체(73일; chronological 기준 2025-10-20~2025-12-31)를
    평가에 쓰고, 정수면 상위 n_eval 일만 평가에 쓴다.
    """
    dates = date_range(start, end)
    n_train = int(len(dates) * train_ratio)

    if mode == "random":
        rng = random.Random(seed)
        rng.shuffle(dates)
        train_dates = dates[:n_train]
        eval_pool = dates[n_train:]
    elif mode == "chronological":
        # 시간순 정렬 그대로 사용. 앞쪽 80% = train, 뒤쪽 20% = eval pool.
        train_dates = dates[:n_train]
        eval_pool = dates[n_train:]
    else:
        raise ValueError(f"unknown split mode: {mode!r}")

    eval_dates = sorted(eval_pool if n_eval is None else eval_pool[:n_eval])
    return train_dates, eval_dates
