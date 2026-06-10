"""우리 실험 agent들이 공통으로 사용하는 날짜, 로딩, 평가 유틸리티.

알고리즘 파일에는 policy/value/Q/PPO update처럼 강화학습 핵심 로직만
잘 보이도록 두고, 평가 날짜와 episode 로딩 같은 반복 코드는 여기로 모았다.
"""

from __future__ import annotations

import datetime
import random

import numpy as np

from src.agents.common.baselines import get_policy
from src.agents.common.episode_cache import load_episodes_cached
from src.envs.data_loader import load_episode
from src.envs.rebalance_env import RebalanceEnv


def date_range(start: str, end: str) -> list[str]:
    """시작일부터 종료일까지 날짜 문자열 목록을 만든다."""
    d = datetime.date.fromisoformat(start)
    end_d = datetime.date.fromisoformat(end)
    dates = []
    while d <= end_d:
        dates.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return dates


RNG = random.Random(42)
ALL_DATES = date_range("2025-01-01", "2025-12-31")
RNG.shuffle(ALL_DATES)
N_TRAIN = int(len(ALL_DATES) * 0.8)
TRAIN_DATES = ALL_DATES[:N_TRAIN]
EVAL_DATES = sorted(ALL_DATES[N_TRAIN:])  # eval pool 전체(73일)


# 모든 ours agent가 같은 평가 reward를 쓰도록 환경 기본값을 한 곳에 둔다.
ENV_KW = dict(
    n_trucks=3,
    truck_capacity=20,
    target_fill_ratio=0.5,
    urgent_low_ratio=0.15,
    urgent_high_ratio=0.85,
    urgent_bonus=0.0,
    strict_urgent_mask=True,
    w_travel_km=-0.008,
    w_travel_step=-0.002,
    explore_bonus_scale=0.0,
    shaping_scale=0.0,
    future_demand_horizon=0,
)


def load_rebalance_episodes(
    dates: list[str],
    district: str,
    processed_dir: str = "data/processed",
    cache_dir: str | None = "data/episode_cache",
    progress_label: str | None = None,
) -> list:
    """날짜 목록을 RebalanceEnv episode 데이터로 변환한다."""
    return load_episodes_cached(
        dates,
        district,
        processed_dir,
        lambda root, gu, date: load_episode(root, district=gu, episode_start=f"{date} 00:00"),
        cache_dir=cache_dir,
        progress_label=progress_label,
    )


def evaluate_most_imbalanced(episodes: list, seed: int) -> tuple[float, list[float]]:
    """같은 데이터 기준에서 MostImbalanced baseline reward를 계산한다."""
    heuristic = get_policy("most_imbalanced")
    rewards = []
    for ep in episodes:
        env = RebalanceEnv(ep, seed=seed, **ENV_KW)
        env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            _, reward, terminated, truncated, _ = env.step(heuristic.act(env))
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), rewards


def print_eval_table(label: str, heuristic_rewards: list[float], model_rewards: list[float]) -> None:
    """평가 결과를 baseline과 나란히 출력한다."""
    print(f"\n=== {label} vs 휴리스틱 ({len(EVAL_DATES)}일) ===")
    print(f"{'날짜':12}{'휴리스틱':>10}{'모델':>10}{'Δ(M-휴)':>9}")
    for date, h, r in zip(EVAL_DATES, heuristic_rewards, model_rewards):
        print(f"{date:12}{h:>10.1f}{r:>10.1f}{r - h:>9.1f}")
    print(
        f"{'평균':12}{np.mean(heuristic_rewards):>10.1f}{np.mean(model_rewards):>10.1f}"
        f"{np.mean(model_rewards) - np.mean(heuristic_rewards):>9.1f}"
    )
