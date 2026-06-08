"""Behavior Cloning 데이터 수집 공통 유틸리티.

이 파일은 DQN, PPO, REINFORCE, A2C가 같은 방식으로 teacher action을
수집할 수 있게 만든다. 팀원 env 코드는 수정하지 않고, 각 agent가 만든
환경 인스턴스에서 state, action, action mask만 읽는다.
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np


def masked_heuristic_action(env) -> int:
    """현재 action mask 안에서 가장 불균형한 정류소를 고르는 teacher."""
    truck = env.trucks[env.current_truck]
    bikes = env.bikes.astype(np.float32)
    target = env.data.capacity.astype(np.float32) * env.target_fill_ratio
    if truck.load == 0:
        scores = bikes - target
    elif truck.load >= env.truck_capacity:
        scores = target - bikes
    else:
        scores = np.abs(bikes - target).astype(np.float32)
    return _masked_argmax(scores, env.action_masks())


def future_heuristic_action(env, horizon: int) -> int:
    """실제 미래 H step 수요를 반영한 oracle teacher action."""
    truck = env.trucks[env.current_truck]
    bikes = env.bikes.astype(np.float32)
    t_end = min(env.t + horizon, env.T)
    if t_end > env.t:
        rentals = env.data.rentals[env.t:t_end].sum(axis=0).astype(np.float32)
        returns = env.data.returns[env.t:t_end].sum(axis=0).astype(np.float32)
        bikes = np.clip(bikes + returns - rentals, 0.0, env.data.capacity.astype(np.float32))
    target = env.data.capacity.astype(np.float32) * env.target_fill_ratio
    if truck.load == 0:
        scores = bikes - target
    elif truck.load >= env.truck_capacity:
        scores = target - bikes
    else:
        scores = np.abs(bikes - target).astype(np.float32)
    return _masked_argmax(scores, env.action_masks())


def forecast_heuristic_action(env) -> int:
    """예측 수요 feature를 반영한 projected imbalance teacher action."""
    truck = env.trucks[env.current_truck]
    bikes = env.bikes.astype(np.float32)
    forecast = getattr(env.data, "agent_demand_forecast", None)
    if forecast is not None and len(forecast) > 0:
        idx = min(int(env.t), len(forecast) - 1)
        net = forecast[idx, :, 2].astype(np.float32)
        bikes = np.clip(bikes + net, 0.0, env.data.capacity.astype(np.float32))
    target = env.data.capacity.astype(np.float32) * env.target_fill_ratio
    if truck.load == 0:
        scores = bikes - target
    elif truck.load >= env.truck_capacity:
        scores = target - bikes
    else:
        scores = np.abs(bikes - target).astype(np.float32)
    return _masked_argmax(scores, env.action_masks())


def collect_bc_data(
    episodes: list,
    args: argparse.Namespace,
    make_env_fn: Callable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """teacher policy가 만든 state-action-mask 데이터를 수집한다."""
    states, actions, masks = [], [], []
    for i, ep in enumerate(episodes[: args.bc_dates]):
        env = make_env_fn(ep, args, seed=args.seed + i)
        state, _ = env.reset(seed=args.seed + i)
        done = False
        while not done:
            mask = env.action_masks()
            action = select_teacher_action(env, args)
            states.append(state.copy())
            actions.append(action)
            masks.append(mask.copy())
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.asarray(states, np.float32), np.asarray(actions, np.int64), np.asarray(masks, bool)


def select_teacher_action(env, args: argparse.Namespace) -> int:
    """CLI의 bc_policy 이름에 맞는 teacher action을 고른다."""
    if hasattr(env, "teacher_action"):
        return int(env.teacher_action(args.bc_policy, getattr(args, "future_horizon", 6)))
    if args.bc_policy == "future_heuristic":
        return future_heuristic_action(env, args.future_horizon)
    if args.bc_policy == "forecast_heuristic":
        return forecast_heuristic_action(env)
    return masked_heuristic_action(env)


def _masked_argmax(scores: np.ndarray, mask: np.ndarray) -> int:
    """불가능한 action을 제외하고 가장 큰 score의 index를 반환한다."""
    masked_scores = scores.astype(np.float32, copy=True)
    masked_scores[~mask] = -np.inf
    best = int(np.argmax(masked_scores))
    if not np.isfinite(masked_scores[best]):
        return int(np.flatnonzero(mask)[0])
    return best
