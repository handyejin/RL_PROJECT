"""REINFORCE 기반 따릉이 재배치 agent.

기본 알고리즘(1):
    REINFORCE + Reward-to-Go + Value Network baseline

고도화 알고리즘(1'):
    REINFORCE + future-demand state + Behavior Cloning pretraining

RL 정의:
    State:
        원본 RebalanceEnv observation
        + 선택적으로 향후 H step의 미래 수요 feature
    Action:
        현재 선택된 트럭이 이동할 정류소 index
    Reward:
        원본 환경 reward를 그대로 사용한다.
        r_t = - imbalance_cost - travel_distance_cost - travel_time_cost

REINFORCE update:
    G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
    A_t = G_t - V(s_t)
    policy_loss = -log pi(a_t|s_t) * A_t
    value_loss = MSE(V(s_t), G_t)

실행 예:
    PYTHONPATH=. python -m src.agents.reinforce --episodes 200
    PYTHONPATH=. python -m src.agents.reinforce --bc-epochs 30 --bc-dates 200 \
        --future-mode oracle_net --future-horizon 6 --bc-policy future_heuristic
"""

from __future__ import annotations

import argparse
import copy
import random
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
from src.agents.ours.common.candidate_actions import maybe_wrap_candidate_actions
from src.agents.ours.common.data_overrides import apply_capacity_override, attach_forecast_override
from src.agents.ours.common.date_split import compute_split
from src.agents.ours.common.episode_cache import load_episodes_cached
from src.agents.ours.common.future_demand import build_history_net_profile, maybe_wrap_future_demand
from src.agents.ours.common.reward_shaping import maybe_wrap_agent_reward_shaping
from src.agents.ours.common.vae_latent import attach_vae_latent_override, maybe_wrap_vae_latent
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
# (random 분할: 기존 동작과 bit-identical, chronological 분할: 시간순 80/20 분할)


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
    """상태를 각 정류소 action logit으로 변환하는 policy network."""

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

    def get_action_and_logp(self, state: np.ndarray, mask: np.ndarray, device: torch.device) -> tuple[int, torch.Tensor]:
        """Categorical policy에서 action을 샘플링하고 log probability를 반환한다."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logits = self.forward(state_t, mask_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action).squeeze(0)

    def act(self, state: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
        """평가 시 가장 확률이 높은 action을 선택한다."""
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            logits = self.forward(state_t, mask_t)
        return int(torch.argmax(logits, dim=-1).item())


class ValueNetwork(nn.Module):
    """REINFORCE baseline으로 사용하는 V(s) value network."""

    def __init__(self, input_size: int, hidden_layer_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


@dataclass
class Trajectory:
    """REINFORCE update에 사용하는 한 episode trajectory."""

    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    masks: list[np.ndarray]
    logp: list[torch.Tensor]
    total_reward: float


def make_env(ep, args: argparse.Namespace, seed: int | None = None, for_eval: bool = False):
    """공통 환경을 만들고, 필요하면 agent-local future wrapper를 적용한다."""
    env = RebalanceEnv(ep, seed=seed, **ENV_KW)
    env = maybe_wrap_future_demand(env, args)
    env = maybe_wrap_vae_latent(env, args)
    if not for_eval:
        env = maybe_wrap_agent_reward_shaping(env, args)
    return maybe_wrap_candidate_actions(env, args)


def load_episodes(
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


def collect_trajectory(
    env,
    policy: PolicyNetwork,
    device: torch.device,
    seed: int,
    max_num_steps: int,
) -> Trajectory:
    """한 episode를 실행해 REINFORCE update에 필요한 trajectory를 수집한다."""
    state_list, action_list, reward_list, mask_list, logp_list = [], [], [], [], []
    state, _ = env.reset(seed=seed)
    done = False
    steps = 0

    while not done and steps < max_num_steps:
        # mask=True인 정류소만 현재 트럭이 선택할 수 있다.
        mask = env.action_masks()
        action, logp = policy.get_action_and_logp(state, mask, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state_list.append(state)
        action_list.append(action)
        reward_list.append(float(reward))
        mask_list.append(mask.copy())
        logp_list.append(logp)

        state = next_state
        steps += 1

    return Trajectory(state_list, action_list, reward_list, mask_list, logp_list, float(np.sum(reward_list)))


def calc_returns(rewards: list[float], gamma: float) -> list[float]:
    """Reward-to-Go G_t를 계산한다."""
    returns = []
    running = 0.0
    # 뒤에서 앞으로 누적하면 각 시점의 G_t를 한 번에 계산할 수 있다.
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    return list(reversed(returns))


def update_reinforce(
    traj: Trajectory,
    returns: list[float],
    policy: PolicyNetwork,
    value: ValueNetwork,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    device: torch.device,
    normalize_advantages: bool,
) -> dict[str, float]:
    """Reward-to-Go와 V(s) baseline을 사용해 policy/value network를 업데이트한다."""
    # states: (T, obs_dim), returns_t: (T,)
    states = torch.as_tensor(np.asarray(traj.states), dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    # values = V(s_t), baseline 역할을 하며 policy gradient의 분산을 줄인다.
    values = value(states)

    # advantage A_t = G_t - V(s_t)
    # policy update에는 critic 쪽 gradient가 섞이지 않도록 values.detach()를 사용한다.
    advantages = returns_t - values.detach()
    if normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    # REINFORCE loss:
    #   J(theta)를 최대화하려면 -log pi(a_t|s_t) * A_t 를 최소화한다.
    policy_loss_terms = [-logp * advantages[i] for i, logp in enumerate(traj.logp)]
    policy_loss = torch.stack(policy_loss_terms).mean()
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Value loss:
    #   critic이 reward-to-go G_t를 예측하도록 MSE로 학습한다.
    value_loss = F.mse_loss(values, returns_t)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "return": traj.total_reward,
    }


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
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """teacher action을 CrossEntropy로 모방해 policy를 먼저 초기화한다."""
    states, actions, masks = collect_bc_data(train_episodes, args)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(masks)),
        batch_size=args.bc_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.bc_lr)
    last_loss = 0.0
    last_acc = 0.0
    for epoch in range(args.bc_epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y, m in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            m = m.to(device, dtype=torch.bool)

            # BC loss:
            #   teacher가 선택한 정류소 y를 정답 label로 보고 CrossEntropy를 최소화한다.
            #   action mask를 적용해 불가능한 정류소는 확률을 거의 0으로 만든다.
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
        if epoch == 0 or (epoch + 1) % max(args.bc_log_every, 1) == 0:
            print(f"  BC epoch {epoch+1}/{args.bc_epochs}: loss={last_loss:.4f}, acc={last_acc:.3f}")
    return {"bc_samples": float(len(actions)), "bc_loss": last_loss, "bc_acc": last_acc}


def evaluate(policy: PolicyNetwork, episodes: list, args: argparse.Namespace, device: torch.device, seed: int) -> tuple[float, list[float]]:
    """고정 7일 평가셋에서 greedy policy의 평균 reward를 계산한다."""
    rewards = []
    for ep in episodes:
        env = make_env(ep, args, seed=seed, for_eval=True)
        state, _ = env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            action = policy.act(state, env.action_masks(), device)
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
    """REINFORCE 실험에 필요한 CLI 옵션을 정의한다.

    같은 core를 원본 state, 수정 state, BC 포함 실험에서 재사용하므로
    state 보강, 후보 action, BC, rollback 옵션을 모두 여기서 받는다.
    """
    parser = argparse.ArgumentParser(description="REINFORCE reward-to-go agent")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--episode-cache-dir", default="data/episode_cache")
    parser.add_argument("--no-episode-cache", action="store_true")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-num-steps", type=int, default=500)
    parser.add_argument("--n-train-dates", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--tag", default="run1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr-policy", type=float, default=3e-4)
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-dates", type=int, default=60)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-log-every", type=int, default=5)
    parser.add_argument("--bc-only", action="store_true")
    parser.add_argument(
        "--bc-policy",
        choices=["masked_heuristic", "future_heuristic", "forecast_heuristic"],
        default="masked_heuristic",
    )
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
    parser.add_argument("--vae-mode", choices=["none", "demand_latent"], default="none")
    parser.add_argument("--vae-latent-path", default="")
    parser.add_argument("--vae-latent-dim", type=int, default=4)
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
    parser.add_argument("--rollback-to-best-on-eval", action="store_true")
    parser.add_argument("--finetune-patience", type=int, default=0)
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
    """데이터 로드, agent-local 보강, 학습, 7일 평가를 순서대로 실행한다."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu")
    if args.device in {"cpu", "mps"}:
        device = torch.device(args.device)

    train_dates_all, eval_dates = compute_split(args.split_mode, seed=42)
    print(
        f"[REINFORCE:{args.district}] split={args.split_mode} "
        f"loading episodes (train={args.n_train_dates}, eval={len(eval_dates)})...",
        flush=True,
    )
    train_episodes = load_episodes(
        train_dates_all[: args.n_train_dates],
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"REINFORCE {args.district} load train" if args.progress else None,
    )
    eval_episodes = load_episodes(
        eval_dates,
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"REINFORCE {args.district} load eval" if args.progress else None,
    )
    all_episodes = train_episodes + eval_episodes
    print(f"[REINFORCE:{args.district}] applying capacity/forecast data...", flush=True)
    capacity_stats = apply_capacity_override(
        all_episodes,
        args.capacity_path,
        args.capacity_initial_fill_ratio,
    )
    forecast_stats = attach_forecast_override(all_episodes, args.forecast_path)
    vae_stats = attach_vae_latent_override(all_episodes, args.vae_latent_path)
    if args.future_mode in {"history_net", "history_projected_travel"}:
        args.history_profile = build_history_net_profile(train_episodes)
    sample_env = make_env(eval_episodes[0], args)
    obs_dim = int(sample_env.observation_space.shape[0])
    n_actions = int(sample_env.action_space.n)
    print(f"[REINFORCE:{args.district}] setup complete. starting training...", flush=True)

    policy = PolicyNetwork(
        obs_dim,
        n_actions,
        args.hidden,
        residual_policy=args.residual_policy,
        residual_temp=args.residual_temp,
    ).to(device)
    value = ValueNetwork(obs_dim, args.hidden).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=args.lr_value)
    rng = np.random.default_rng(args.seed)

    out_dir = Path("logs") / f"reinforce_{args.tag}"
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
    patience_left = args.finetune_patience

    print(f"=== REINFORCE | tag={args.tag} ===")
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
    if vae_stats:
        print(
            "VAE latent override: "
            f"matched={int(vae_stats['vae_matched'])}/{int(vae_stats['vae_total'])}, "
            f"latent_dim={int(vae_stats['vae_latent_dim'])}"
        )
    print(f"heuristic mean reward: {heuristic_mean:.2f}")

    if args.bc_epochs > 0:
        bc_stats = pretrain_behavior_cloning(policy, train_episodes, args, device)
        print(
            f"BC done: samples={int(bc_stats['bc_samples'])}, "
            f"loss={bc_stats['bc_loss']:.4f}, acc={bc_stats['bc_acc']:.3f}"
        )
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
        torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, out_dir / "reinforce_final.pt")
        np.save(out_dir / "history.npy", np.asarray(history or [{"episode": 0, "eval_reward": final_mean}], dtype=object))
        print_eval_table("reinforce_bc_only", heuristic_rewards, final_rewards, eval_dates)
        return

    episode_iter = range(1, args.episodes + 1)
    progress_bar = None
    if args.progress:
        progress_bar = tqdm(
            episode_iter,
            total=args.episodes,
            desc=f"REINFORCE {args.district}",
            unit="episode",
        )
        episode_iter = progress_bar

    for episode in episode_iter:
        ep = train_episodes[int(rng.integers(len(train_episodes)))]
        env = make_env(ep, args, seed=args.seed + episode)
        traj = collect_trajectory(env, policy, device, args.seed + episode, args.max_num_steps)
        returns = calc_returns(traj.rewards, args.gamma)
        stats = update_reinforce(
            traj,
            returns,
            policy,
            value,
            policy_optimizer,
            value_optimizer,
            device,
            args.normalize_advantages,
        )

        if episode == 1 or episode % args.eval_every == 0:
            eval_reward, _ = evaluate(policy, eval_episodes, args, device, args.seed)
            history.append(
                {
                    "episode": episode,
                    "eval_reward": eval_reward,
                    "train_return": stats["return"],
                    "policy_loss": stats["policy_loss"],
                    "value_loss": stats["value_loss"],
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
                    # 평가 성능이 나빠지면 best REINFORCE checkpoint로 되돌린다.
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
                f"train={stats['return']:8.2f} policy_loss={stats['policy_loss']:7.3f} "
                f"value_loss={stats['value_loss']:7.3f}"
            )
            if args.finetune_patience > 0 and patience_left <= 0:
                print(f"fine-tuning early stop: best_episode={best_episode}, best_reward={best_reward:.2f}")
                break

    final_mean, final_rewards = evaluate(policy, eval_episodes, args, device, args.seed)
    torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, out_dir / "reinforce_final.pt")
    np.save(out_dir / "history.npy", np.asarray(history, dtype=object))
    print(f"best reward: {best_reward:.2f} at episode {best_episode}")
    print(f"final reward: {final_mean:.2f}")
    print_eval_table("reinforce_final", heuristic_rewards, final_rewards, eval_dates)


if __name__ == "__main__":
    main()
