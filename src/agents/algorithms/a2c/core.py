"""A2C 기반 따릉이 재배치 agent.

기본 알고리즘(2):
    A2C / Actor-Critic

고도화 알고리즘(2'):
    A2C + future-demand state + Behavior Cloning pretraining

RL 정의:
    State:
        원본 RebalanceEnv observation
        + 선택적으로 향후 H step의 미래 수요 feature
    Action:
        현재 선택된 트럭이 이동할 정류소 index
    Reward:
        원본 환경 reward를 그대로 사용한다.
        r_t = - imbalance_cost - travel_distance_cost - travel_time_cost

A2C update:
    target = r + gamma * (1 - done) * V(s')
    advantage = target - V(s)
    actor_loss = -log pi(a|s) * advantage
    critic_loss = MSE(target, V(s))

Fine-tuning 보호:
    BC로 얻은 좋은 policy가 RL update 중 무너지지 않도록
    validation early stopping, best rollback, anchor KL loss를 선택적으로 사용한다.

실행 예:
    PYTHONPATH=. python -m src.agents.actor_critic --episodes 200
    PYTHONPATH=. python -m src.agents.actor_critic --bc-epochs 30 --bc-dates 200 \
        --future-mode oracle_net --future-horizon 6 --bc-policy future_heuristic
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.agents.baselines import get_policy
from src.agents.common.candidate_actions import maybe_wrap_candidate_actions
from src.agents.common.data_overrides import apply_capacity_override, attach_forecast_override
from src.agents.common.date_split import compute_split
from src.agents.common.future_demand import build_history_net_profile, maybe_wrap_future_demand
from src.agents.common.reward_shaping import maybe_wrap_agent_reward_shaping
from src.envs.data_loader import load_episode
from src.envs.rebalance_env import RebalanceEnv


def date_range(start: str, end: str) -> list[str]:
    """시작일부터 종료일까지 날짜 문자열 목록을 만든다."""
    import datetime

    d = datetime.date.fromisoformat(start)
    end_d = datetime.date.fromisoformat(end)
    dates = []
    while d <= end_d:
        dates.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return dates


# TRAIN_DATES / EVAL_DATES 는 main() 에서 --split-mode 에 따라 compute_split 으로 생성한다.


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


class PolicyNetwork(nn.Module):
    """상태를 각 정류소 action logit으로 변환하는 actor network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_size: int = 256,
        residual_policy: bool = False,
        residual_temp: float = 1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.residual_policy = residual_policy
        self.residual_temp = float(residual_temp)
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        obs = x
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        if self.residual_policy:
            # observation 마지막 N개를 heuristic residual score로 사용한다.
            base_score = x.new_zeros(logits.shape)
            if logits.shape[-1] <= obs.shape[-1]:
                base_score = obs[..., -self.output_size :]
            logits = logits + base_score / max(self.residual_temp, 1e-6)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits

    def select_action(self, state: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
        """action mask를 적용한 뒤 Categorical policy에서 action을 샘플링한다."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            action = Categorical(logits=self.forward(state_t, mask_t)).sample()
        return int(action.item())

    def greedy_action(self, state: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
        """평가 시 가장 확률이 높은 action을 선택한다."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            logits = self.forward(state_t, mask_t)
        return int(torch.argmax(logits, dim=-1).item())


class ValueNetwork(nn.Module):
    """상태 가치 V(s)를 추정하는 critic network."""

    def __init__(self, input_size: int, hidden_layer_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class Memory:
    """A2C batch update에 사용할 transition memory."""

    def __init__(self, memory_size: int) -> None:
        self.buffer = deque(maxlen=memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            start = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(start, start + batch_size)]
        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indexes]

    def clear(self) -> None:
        self.buffer.clear()


@dataclass
class TrainStats:
    """한 episode 학습 결과 요약."""

    reward: float
    actor_loss: float
    critic_loss: float


def make_env(ep, args: argparse.Namespace, seed: int | None = None, for_eval: bool = False):
    """공통 환경을 만들고, 필요하면 agent-local wrapper를 적용한다."""
    env = RebalanceEnv(ep, seed=seed, **ENV_KW)
    env = maybe_wrap_future_demand(env, args)
    if not for_eval:
        env = maybe_wrap_agent_reward_shaping(env, args)
    return maybe_wrap_candidate_actions(env, args)


def load_episodes(
    dates: list[str],
    district: str,
    processed_dir: str = "data/processed",
    progress_label: str | None = None,
) -> list:
    """날짜 목록을 RebalanceEnv episode 데이터로 변환한다."""
    date_iter = tqdm(dates, desc=progress_label, unit="day") if progress_label else dates
    return [
        load_episode(processed_dir, district=district, episode_start=f"{date} 00:00")
        for date in date_iter
    ]


def update_a2c(
    policy: PolicyNetwork,
    value: ValueNetwork,
    policy_optim: torch.optim.Optimizer,
    value_optim: torch.optim.Optimizer,
    experiences: list,
    gamma: float,
    device: torch.device,
    normalize_advantages: bool,
    anchor_policy: PolicyNetwork | None = None,
    anchor_coef: float = 0.0,
) -> tuple[float, float]:
    """TD target과 advantage를 이용해 actor/critic을 업데이트한다."""
    # batch_state: s_t, batch_next_state: s_{t+1}
    # batch_done은 terminal transition이면 1, 아니면 0이다.
    states, next_states, actions, rewards, dones, masks = zip(*experiences)
    batch_state = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    batch_next_state = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device)
    batch_action = torch.as_tensor(actions, dtype=torch.long, device=device)
    batch_reward = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    batch_done = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    batch_mask = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)

    with torch.no_grad():
        # Critic target:
        #   target = r + gamma * (1 - done) * V(s')
        # done이면 다음 상태 가치 V(s')를 더하지 않는다.
        value_target = batch_reward + gamma * (1.0 - batch_done) * value(batch_next_state)

        # Advantage:
        #   A(s,a) = target - V(s)
        # actor update에는 critic gradient가 섞이지 않도록 no_grad 안에서 계산한다.
        advantage = value_target - value(batch_state)
        if normalize_advantages and advantage.numel() > 1:
            # Advantage normalization은 batch별 scale 차이를 줄여 policy gradient 분산을 낮춘다.
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

    # Actor loss:
    #   좋은 advantage를 만든 action의 log probability는 키우고,
    #   나쁜 advantage를 만든 action의 log probability는 줄인다.
    logits = policy(batch_state, batch_mask)
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(batch_action).unsqueeze(1)
    rl_actor_loss = -(log_prob * advantage).mean()

    # Anchor KL loss:
    #   BC 직후 policy(pi_anchor)에서 너무 멀어지지 않도록 KL(pi_anchor || pi_current)를 더한다.
    #   BC policy가 좋은 출발점일 때 RL fine-tuning이 이를 망가뜨리는 현상을 완화한다.
    anchor_loss = torch.tensor(0.0, device=device)
    if anchor_policy is not None and anchor_coef > 0.0:
        with torch.no_grad():
            anchor_logits = anchor_policy(batch_state, batch_mask)
            anchor_probs = torch.softmax(anchor_logits, dim=-1)
            anchor_log_probs = torch.log_softmax(anchor_logits, dim=-1)
        current_log_probs = torch.log_softmax(logits, dim=-1)
        anchor_loss = (anchor_probs * (anchor_log_probs - current_log_probs)).sum(dim=-1).mean()

    actor_loss = rl_actor_loss + anchor_coef * anchor_loss
    policy_optim.zero_grad()
    actor_loss.backward()
    policy_optim.step()

    # Critic loss:
    #   V(s)가 TD target을 따라가도록 MSE를 최소화한다.
    critic_loss = F.mse_loss(value(batch_state), value_target)
    value_optim.zero_grad()
    critic_loss.backward()
    value_optim.step()

    return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu())


def train_episode(
    env,
    policy: PolicyNetwork,
    value: ValueNetwork,
    policy_optim: torch.optim.Optimizer,
    value_optim: torch.optim.Optimizer,
    memory: Memory,
    batch_size: int,
    gamma: float,
    device: torch.device,
    seed: int,
    normalize_advantages: bool,
    anchor_policy: PolicyNetwork | None,
    anchor_coef: float,
) -> TrainStats:
    """한 episode를 실행하면서 batch_size마다 A2C update를 수행한다."""
    state, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    actor_loss = 0.0
    critic_loss = 0.0

    while not done:
        # 현재 환경에서 허용되는 정류소만 action 후보로 둔다.
        mask = env.action_masks()
        action = policy.select_action(state, mask, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # A2C update에 필요한 transition (s, s', a, r, done, mask)을 저장한다.
        memory.add((state, next_state, action, float(reward), done, mask.copy()))

        if memory.size() >= batch_size:
            batch = memory.sample(batch_size, continuous=True)
            actor_loss, critic_loss = update_a2c(
                policy,
                value,
                policy_optim,
                value_optim,
                batch,
                gamma,
                device,
                normalize_advantages,
                anchor_policy,
                anchor_coef,
            )
            memory.clear()

        total += float(reward)
        state = next_state

    return TrainStats(total, actor_loss, critic_loss)


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
    scores = scores.copy()
    mask = env.action_masks()
    scores[~mask] = -np.inf
    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]):
        return int(np.flatnonzero(mask)[0])
    return best


def future_heuristic_action(env, horizon: int) -> int:
    """향후 수요를 반영한 projected imbalance 기준 teacher action."""
    truck = env.trucks[env.current_truck]
    bikes = env.bikes.astype(np.float32)
    t_end = min(env.t + horizon, env.T)
    if t_end > env.t:
        # 미래 H step의 대여/반납을 현재 재고에 반영해 예상 재고를 만든다.
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
    scores = scores.copy()
    mask = env.action_masks()
    scores[~mask] = -np.inf
    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]):
        return int(np.flatnonzero(mask)[0])
    return best


def forecast_heuristic_action(env) -> int:
    """예측된 1시간 대여/반납을 반영한 projected imbalance teacher action."""
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
    scores = scores.copy()
    mask = env.action_masks()
    scores[~mask] = -np.inf
    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]):
        return int(np.flatnonzero(mask)[0])
    return best


def collect_bc_data(episodes: list, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """teacher policy가 만든 state-action pair를 수집한다."""
    states, actions, masks = [], [], []
    for i, ep in enumerate(episodes[: args.bc_dates]):
        env = make_env(ep, args, seed=args.seed + i)
        state, _ = env.reset(seed=args.seed + i)
        done = False
        while not done:
            mask = env.action_masks()
            if hasattr(env, "teacher_action"):
                action = int(env.teacher_action(args.bc_policy, args.future_horizon))
            elif args.bc_policy == "future_heuristic":
                action = future_heuristic_action(env, args.future_horizon)
            elif args.bc_policy == "forecast_heuristic":
                action = forecast_heuristic_action(env)
            else:
                action = masked_heuristic_action(env)
            states.append(state.copy())
            actions.append(action)
            masks.append(mask.copy())
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.asarray(states, np.float32), np.asarray(actions, np.int64), np.asarray(masks, bool)


def pretrain_behavior_cloning(
    policy: PolicyNetwork,
    train_episodes: list,
    val_episodes: list,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | dict[str, torch.Tensor]]:
    """teacher action을 CrossEntropy로 모방해 actor를 먼저 초기화한다."""
    states, actions, masks = collect_bc_data(train_episodes, args)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(masks)),
        batch_size=args.bc_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.bc_lr)
    last_loss = 0.0
    last_acc = 0.0
    best_val_reward = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(policy.state_dict())
    patience_left = args.bc_patience
    for epoch in range(args.bc_epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y, m in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            m = m.to(device, dtype=torch.bool)

            # BC loss:
            #   teacher action y를 정답 label로 보고 actor를 먼저 지도학습한다.
            #   불가능한 action은 mask로 제거한다.
            logits = policy(x, m)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += len(y)
        last_loss = total_loss / max(total, 1)
        last_acc = correct / max(total, 1)
        val_reward = None
        if val_episodes:
            # BC validation은 teacher label 정확도 대신 실제 환경 reward로 본다.
            val_reward, _ = evaluate(policy, val_episodes, args, device, args.seed)
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_epoch = epoch + 1
                best_state = copy.deepcopy(policy.state_dict())
                patience_left = args.bc_patience
            else:
                patience_left -= 1
        if epoch == 0 or (epoch + 1) % max(args.bc_log_every, 1) == 0:
            msg = f"  BC epoch {epoch+1}/{args.bc_epochs}: loss={last_loss:.4f}, acc={last_acc:.3f}"
            if val_reward is not None:
                msg += f", val_reward={val_reward:.2f}"
            print(msg)
        if val_episodes and args.bc_patience > 0 and patience_left <= 0:
            print(f"  BC early stop: best_epoch={best_epoch}, best_val_reward={best_val_reward:.2f}")
            break
    if val_episodes:
        # validation reward가 가장 좋았던 BC policy를 fine-tuning 시작점으로 사용한다.
        policy.load_state_dict(best_state)
    return {
        "bc_samples": float(len(actions)),
        "bc_loss": last_loss,
        "bc_acc": last_acc,
        "bc_best_val_reward": float(best_val_reward),
        "bc_best_epoch": float(best_epoch),
        "bc_state": copy.deepcopy(policy.state_dict()),
    }


def evaluate(policy: PolicyNetwork, episodes: list, args: argparse.Namespace, device: torch.device, seed: int) -> tuple[float, list[float]]:
    """고정 7일 평가셋에서 greedy policy의 평균 reward를 계산한다."""
    rewards = []
    for ep in episodes:
        env = make_env(ep, args, for_eval=True)
        state, _ = env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            action = policy.greedy_action(state, env.action_masks(), device)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), rewards


def evaluate_heuristic(episodes: list, seed: int) -> tuple[float, list[float]]:
    """기존 most_imbalanced 휴리스틱의 7일 평균 reward를 계산한다."""
    heuristic = get_policy("most_imbalanced")
    rewards = []
    for ep in episodes:
        env = RebalanceEnv(ep, **ENV_KW)
        env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            _, reward, terminated, truncated, _ = env.step(heuristic.act(env))
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), rewards


def print_eval_table(
    label: str,
    heuristic_rewards: list[float],
    model_rewards: list[float],
    eval_dates: list[str],
) -> None:
    """7일 평가 결과를 표로 출력한다."""
    print(f"\n=== {label} vs 휴리스틱 (7일) ===")
    print(f"{'날짜':12}{'휴리스틱':>10}{'모델':>10}{'Δ(M-휴)':>9}")
    for date, h, r in zip(eval_dates, heuristic_rewards, model_rewards):
        print(f"{date:12}{h:>10.1f}{r:>10.1f}{r - h:>9.1f}")
    print(
        f"{'평균':12}{np.mean(heuristic_rewards):>10.1f}{np.mean(model_rewards):>10.1f}"
        f"{np.mean(model_rewards) - np.mean(heuristic_rewards):>9.1f}"
    )


def parse_args() -> argparse.Namespace:
    """A2C 실험에 필요한 CLI 옵션을 정의한다.

    원본 state와 수정 state 실행 파일이 같은 core를 사용하므로
    forecast/capacity 보강, 후보 action, BC, anchor KL, rollback 옵션을 함께 둔다.
    """
    parser = argparse.ArgumentParser(description="A2C actor-critic agent")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--n-train-dates", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--tag", default="run1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--lr-value", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-dates", type=int, default=60)
    parser.add_argument("--bc-val-dates", type=int, default=0)
    parser.add_argument("--bc-patience", type=int, default=0)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-log-every", type=int, default=5)
    parser.add_argument("--bc-only", action="store_true")
    parser.add_argument(
        "--bc-policy",
        choices=["masked_heuristic", "future_heuristic", "forecast_heuristic"],
        default="masked_heuristic",
    )
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--anchor-coef", type=float, default=0.0)
    parser.add_argument("--rollback-to-best-on-eval", action="store_true")
    parser.add_argument("--finetune-patience", type=int, default=0)
    parser.add_argument(
        "--future-mode",
        choices=[
            "none",
            "oracle_net",
            "oracle_inout",
            "history_net",
            "history_projected_travel",
            "forecast_net",
            "forecast_inout",
            "forecast_projected_travel",
        ],
        default="none",
    )
    parser.add_argument("--future-horizon", type=int, default=6)
    parser.add_argument("--capacity-path", default="")
    parser.add_argument("--capacity-initial-fill-ratio", type=float, default=0.5)
    parser.add_argument("--forecast-path", default="")
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--candidate-mode", choices=["imbalance", "forecast_imbalance"], default="imbalance")
    parser.add_argument("--candidate-travel-coef", type=float, default=0.0)
    parser.add_argument("--candidate-zone-mode", choices=["none", "static3"], default="none")
    parser.add_argument("--candidate-zone-count", type=int, default=3)
    parser.add_argument("--candidate-zone-penalty", type=float, default=0.0)
    parser.add_argument("--candidate-feature-mode", choices=["none", "basic"], default="none")
    parser.add_argument("--agent-shaping-mode", choices=["projected_imbalance"], default="projected_imbalance")
    parser.add_argument("--agent-shaping-scale", type=float, default=0.0)
    parser.add_argument("--agent-shaping-gamma", type=float, default=0.99)
    parser.add_argument("--residual-policy", action="store_true")
    parser.add_argument("--residual-temp", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological"],
        default="random",
        help="random: seed=42 셔플 후 80/20, chronological: 시간순 80/20 (계절 OOD 평가)",
    )
    return parser.parse_args()


def main() -> None:
    """데이터 준비부터 BC 선택 적용, A2C 학습, best/final 평가까지 수행한다."""
    args = parse_args()
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu")
    if args.device in {"cpu", "mps"}:
        device = torch.device(args.device)

    train_dates_all, eval_dates = compute_split(args.split_mode, seed=42)
    train_dates = train_dates_all[: args.n_train_dates]
    val_start = args.n_train_dates
    val_end = args.n_train_dates + args.bc_val_dates
    val_dates = train_dates_all[val_start:val_end]
    print(
        f"[A2C:{args.district}] split={args.split_mode} loading episodes "
        f"(train={len(train_dates)}, val={len(val_dates)}, eval={len(eval_dates)})...",
        flush=True,
    )
    train_episodes = load_episodes(
        train_dates,
        args.district,
        args.processed_dir,
        f"A2C {args.district} load train" if args.progress else None,
    )
    bc_val_episodes = (
        load_episodes(
            val_dates,
            args.district,
            args.processed_dir,
            f"A2C {args.district} load val" if args.progress else None,
        )
        if args.bc_val_dates > 0
        else []
    )
    eval_episodes = load_episodes(
        eval_dates,
        args.district,
        args.processed_dir,
        f"A2C {args.district} load eval" if args.progress else None,
    )
    all_episodes = train_episodes + bc_val_episodes + eval_episodes
    print(f"[A2C:{args.district}] applying capacity/forecast data...", flush=True)
    capacity_stats = apply_capacity_override(
        all_episodes,
        args.capacity_path,
        args.capacity_initial_fill_ratio,
    )
    forecast_stats = attach_forecast_override(all_episodes, args.forecast_path)
    if args.future_mode in {"history_net", "history_projected_travel"}:
        args.history_profile = build_history_net_profile(train_episodes)
    sample_env = make_env(eval_episodes[0], args)
    obs_dim = int(sample_env.observation_space.shape[0])
    n_actions = int(sample_env.action_space.n)
    print(f"[A2C:{args.district}] setup complete. starting training...", flush=True)

    policy = PolicyNetwork(
        obs_dim,
        n_actions,
        args.hidden,
        residual_policy=args.residual_policy,
        residual_temp=args.residual_temp,
    ).to(device)
    value = ValueNetwork(obs_dim, args.hidden).to(device)
    policy_optim = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    value_optim = torch.optim.Adam(value.parameters(), lr=args.lr_value)
    memory = Memory(args.memory_size)
    rng = np.random.default_rng(args.seed)

    out_dir = Path("logs") / f"actor_critic_{args.tag}"
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    heuristic_mean, heuristic_rewards = evaluate_heuristic(eval_episodes, args.seed)
    history = []
    best_reward = -np.inf
    best_episode = 0
    best_state = {
        "policy": copy.deepcopy(policy.state_dict()),
        "value": copy.deepcopy(value.state_dict()),
    }
    anchor_policy = None
    patience_left = args.finetune_patience

    print(f"=== A2C | tag={args.tag} ===")
    print(f"device={device}, obs_dim={obs_dim}, n_actions={n_actions}")
    if capacity_stats:
        print(
            "capacity override: "
            f"matched={int(capacity_stats['capacity_matched'])}/{int(capacity_stats['capacity_total'])}, "
            f"mean_capacity={capacity_stats['capacity_mean']:.2f}"
        )
    if forecast_stats:
        print(
            "forecast override: "
            f"matched={int(forecast_stats['forecast_matched'])}/{int(forecast_stats['forecast_total'])}"
        )
    print(f"heuristic mean reward: {heuristic_mean:.2f}")
    if bc_val_episodes:
        print(f"BC validation dates: {len(bc_val_episodes)}")

    if args.bc_epochs > 0:
        bc_stats = pretrain_behavior_cloning(policy, train_episodes, bc_val_episodes, args, device)
        print(
            f"BC done: samples={int(bc_stats['bc_samples'])}, "
            f"loss={bc_stats['bc_loss']:.4f}, acc={bc_stats['bc_acc']:.3f}"
        )
        if bc_val_episodes:
            print(
                f"BC best validation: epoch={int(bc_stats['bc_best_epoch'])}, "
                f"reward={bc_stats['bc_best_val_reward']:.2f}"
            )
        if args.anchor_coef > 0.0:
            # anchor_policy는 BC 직후 policy를 고정 복사한 것이다.
            anchor_policy = PolicyNetwork(
                obs_dim,
                n_actions,
                args.hidden,
                residual_policy=args.residual_policy,
                residual_temp=args.residual_temp,
            ).to(device)
            anchor_policy.load_state_dict(policy.state_dict())
            anchor_policy.eval()
            for param in anchor_policy.parameters():
                param.requires_grad_(False)
        eval_reward, _ = evaluate(policy, eval_episodes, args, device, args.seed)
        history.append({"episode": 0, "eval_reward": eval_reward, "stage": "bc"})
        best_reward = eval_reward
        best_state = {
            "policy": copy.deepcopy(policy.state_dict()),
            "value": copy.deepcopy(value.state_dict()),
        }
        torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, best_dir / "best_model.pt")
        print(f"episode={0:4d} eval={eval_reward:8.2f} stage=BC")

    if args.bc_only or args.episodes <= 0:
        final_mean, final_rewards = evaluate(policy, eval_episodes, args, device, args.seed)
        torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, out_dir / "actor_critic_final.pt")
        np.save(out_dir / "history.npy", np.asarray(history or [{"episode": 0, "eval_reward": final_mean}], dtype=object))
        print_eval_table("a2c_bc_only", heuristic_rewards, final_rewards, eval_dates)
        return

    episode_iter = range(1, args.episodes + 1)
    progress_bar = None
    if args.progress:
        progress_bar = tqdm(
            episode_iter,
            total=args.episodes,
            desc=f"A2C {args.district}",
            unit="episode",
        )
        episode_iter = progress_bar

    for episode in episode_iter:
        ep = train_episodes[int(rng.integers(len(train_episodes)))]
        env = make_env(ep, args, seed=args.seed + episode)
        stats = train_episode(
            env,
            policy,
            value,
            policy_optim,
            value_optim,
            memory,
            args.batch_size,
            args.gamma,
            device,
            args.seed + episode,
            args.normalize_advantages,
            anchor_policy,
            args.anchor_coef,
        )

        if episode == 1 or episode % args.eval_every == 0:
            eval_reward, _ = evaluate(policy, eval_episodes, args, device, args.seed)
            history.append(
                {
                    "episode": episode,
                    "eval_reward": eval_reward,
                    "train_return": stats.reward,
                    "policy_loss": stats.actor_loss,
                    "value_loss": stats.critic_loss,
                }
            )
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_episode = episode
                best_state = {
                    "policy": copy.deepcopy(policy.state_dict()),
                    "value": copy.deepcopy(value.state_dict()),
                }
                patience_left = args.finetune_patience
                torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, best_dir / "best_model.pt")
            else:
                if args.rollback_to_best_on_eval:
                    # 평가 성능이 나빠지면 best checkpoint로 되돌려 BC policy 붕괴를 막는다.
                    policy.load_state_dict(best_state["policy"])
                    value.load_state_dict(best_state["value"])
                if args.finetune_patience > 0:
                    patience_left -= 1
            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "eval": f"{eval_reward:.1f}",
                        "base": f"{heuristic_mean:.1f}",
                        "delta": f"{eval_reward - heuristic_mean:+.1f}",
                        "best": f"{best_reward - heuristic_mean:+.1f}",
                    }
                )
            print(
                f"episode={episode:4d} eval={eval_reward:8.2f} "
                f"train={stats.reward:8.2f} actor_loss={stats.actor_loss:7.3f} "
                f"critic_loss={stats.critic_loss:7.3f}"
            )
            if args.finetune_patience > 0 and patience_left <= 0:
                print(f"fine-tuning early stop: best_episode={best_episode}, best_reward={best_reward:.2f}")
                break

    final_mean, final_rewards = evaluate(policy, eval_episodes, args, device, args.seed)
    if not history or abs(float(history[-1]["eval_reward"]) - final_mean) > 1e-9:
        # rollback을 사용하면 마지막 주기 평가값과 실제 final policy 평가값이 다를 수 있다.
        history.append({"episode": episode, "eval_reward": final_mean, "stage": "final"})
    torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, out_dir / "actor_critic_final.pt")
    np.save(out_dir / "history.npy", np.asarray(history, dtype=object))
    print(f"best reward: {best_reward:.2f} at episode {best_episode}")
    print(f"final reward: {final_mean:.2f}")
    print_eval_table("a2c_final", heuristic_rewards, final_rewards, eval_dates)


if __name__ == "__main__":
    main()
