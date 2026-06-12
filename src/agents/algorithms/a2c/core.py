####################
# 작성자 : 박제영
# 설명   : A2C(Actor-Critic, 1-step TD advantage) 기반 따릉이 재배치 agent.
#          Actor는 이동 정류소 정책을, Critic은 현재 상태의 가치 V(s)를 학습한다.
####################

"""A2C 기반 따릉이 재배치 agent.

기본 알고리즘(2):
    A2C / Actor-Critic

최종 실험 설정:
    A2C + 수요예측 state + Top-K 후보 action

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

실행 예:
    PYTHONPATH=. python -m src.agents.algorithms.a2c.core \
        --episodes 500 --split-mode chronological --candidate-top-k 9
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
from tqdm.auto import tqdm

from src.agents.common.candidate_actions import maybe_wrap_candidate_actions
from src.agents.common.data_overrides import apply_capacity_override, attach_forecast_override
from src.agents.common.date_split import compute_split
from src.agents.common.episode_cache import load_episodes_cached
from src.agents.common.experiment_utils import (
    evaluate_most_imbalanced as evaluate_heuristic,
    print_eval_table,
)
from src.agents.common.future_demand import build_history_net_profile, maybe_wrap_future_demand
from src.agents.common.vae_latent import attach_vae_latent_override, maybe_wrap_vae_latent
from src.envs.data_loader import EpisodeData, load_episode
from src.envs.rebalance_env import RebalanceEnv


# TRAIN_DATES / EVAL_DATES 는 main() 에서 --split-mode 에 따라 compute_split 으로 생성한다.
# 최종 보고서 기준은 chronological split(2025-10-20~2025-12-31 holdout)이다.


# A2C 학습에서 한 transition을 표현하는 6-튜플 형식.
#   (state, next_state, action, reward, done, mask)
Transition = tuple[np.ndarray, np.ndarray, int, float, bool, np.ndarray]


# 최종 보고서 기준 RebalanceEnv 생성 설정.
# 환경 자체의 default 와 다른 보고서 기준값만 명시한다(strict_urgent_mask 등).
ENV_KW: dict[str, Any] = dict(
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
    """A2C batch update에 사용하는 짧은 transition buffer.

    DQN의 long-horizon replay와 달리 A2C는 on-policy에 가까운 짧은 window를
    사용하므로, ``memory_size`` 가 차면 가장 오래된 transition부터 잘려나간다.
    """

    def __init__(self, memory_size: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=memory_size)

    def add(self, experience: Transition) -> None:
        """transition 한 건을 buffer에 추가한다."""
        self.buffer.append(experience)

    def size(self) -> int:
        """현재 buffer에 쌓여 있는 transition 수."""
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True) -> list[Transition]:
        """학습용 transition batch를 추출한다.

        Args:
            batch_size: 원하는 batch 크기. buffer보다 크면 buffer 크기로 줄인다.
            continuous: True면 연속 구간(시간순)을 추출, False면 무작위 index.

        Returns:
            transition 리스트.
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            start = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(start, start + batch_size)]
        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indexes]

    def clear(self) -> None:
        """buffer를 비운다 (batch update 직후 호출)."""
        self.buffer.clear()


@dataclass
class TrainStats:
    """한 episode 학습 결과 요약."""

    reward: float
    actor_loss: float
    critic_loss: float


def make_env(
    ep: EpisodeData | list[EpisodeData],
    args: argparse.Namespace,
    seed: int | None = None,
    for_eval: bool = False,
) -> gym.Env:
    """RebalanceEnv를 만들고 future-demand · VAE · Top-K wrapper를 차례로 적용한다.

    Args:
        ep: 단일 episode 또는 리스트. 리스트면 환경 내부에서 회전 사용된다.
        args: CLI / YAML 로 합쳐진 학습 설정.
        seed: 환경 RNG 시드.
        for_eval: True여도 A2C는 evaluation 전용 wrapper가 따로 없어
            현재는 동작에 영향을 주지 않는다(시그니처 호환용).

    Returns:
        wrapper가 적용된 ``gym.Env``.
    """
    env = RebalanceEnv(ep, seed=seed, **ENV_KW)
    env = maybe_wrap_future_demand(env, args)
    env = maybe_wrap_vae_latent(env, args)
    return maybe_wrap_candidate_actions(env, args)


def load_episodes(
    dates: list[str],
    district: str,
    processed_dir: str = "data/processed",
    cache_dir: str | None = "data/episode_cache",
    progress_label: str | None = None,
) -> list[EpisodeData]:
    """날짜 목록을 RebalanceEnv용 ``EpisodeData`` 리스트로 변환한다.

    Args:
        dates: ``YYYY-MM-DD`` 문자열 리스트.
        district: 자치구 이름(예: ``"강남구"``).
        processed_dir: 전처리된 데이터 루트 디렉토리.
        cache_dir: episode 로딩 캐시 위치. ``None`` 이면 캐시를 끈다.
        progress_label: tqdm 진행바 라벨. ``None`` 이면 진행바를 표시하지 않는다.
    """
    return load_episodes_cached(
        dates,
        district,
        processed_dir,
        lambda root, gu, date: load_episode(root, district=gu, episode_start=f"{date} 00:00"),
        cache_dir=cache_dir,
        progress_label=progress_label,
    )


def update_a2c(
    policy: PolicyNetwork,
    value: ValueNetwork,
    policy_optim: torch.optim.Optimizer,
    value_optim: torch.optim.Optimizer,
    experiences: list[Transition],
    gamma: float,
    device: torch.device,
    normalize_advantages: bool,
) -> tuple[float, float]:
    """TD target과 advantage를 사용해 actor/critic을 한 번 업데이트한다.

    Args:
        policy: 업데이트 대상 actor.
        value: 업데이트 대상 critic.
        policy_optim: actor optimizer.
        value_optim: critic optimizer.
        experiences: 학습 transition batch.
        gamma: 미래 보상 할인율.
        device: torch device.
        normalize_advantages: True면 batch advantage를 mean/std로 정규화.

    Returns:
        ``(actor_loss, critic_loss)`` 스칼라 튜플.
    """
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
    actor_loss = -(log_prob * advantage).mean()
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
    env: gym.Env,
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
) -> TrainStats:
    """한 episode를 굴리면서 ``batch_size`` step마다 A2C update를 수행한다.

    매 step의 transition을 ``memory`` 에 쌓고, 가득 차면 한 번에 update→clear
    한다. 마지막에 누적 reward와 마지막 update의 loss를 리턴한다.
    """
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
            )
            memory.clear()

        total += float(reward)
        state = next_state

    return TrainStats(total, actor_loss, critic_loss)


def evaluate(
    policy: PolicyNetwork,
    episodes: list[EpisodeData],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[float, list[float]]:
    """평가 holdout에서 greedy policy의 평균 reward와 day-별 reward 리스트를 계산한다."""
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


def parse_args() -> argparse.Namespace:
    """A2C 실험에 필요한 CLI 옵션을 정의한다.

    최종 보고서 기준인 1-step TD actor-critic, 수요예측 state, Top-K 후보
    action, 73일 holdout 평가만 남겨 흐름을 단순하게 유지한다.
    """
    parser = argparse.ArgumentParser(description="A2C actor-critic agent")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--episode-cache-dir", default="data/episode_cache")
    parser.add_argument("--no-episode-cache", action="store_true")
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
    parser.add_argument("--normalize-advantages", action="store_true")
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
    parser.add_argument("--residual-policy", action="store_true")
    parser.add_argument("--residual-temp", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological"],
        default="chronological",
        help="chronological: 시간순 80/20 holdout, random: seed=42 셔플 후 80/20.",
    )
    return parser.parse_args()


def main() -> None:
    """데이터 준비, A2C 학습, best/final 평가를 순서대로 수행한다."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu")
    if args.device in {"cpu", "mps"}:
        device = torch.device(args.device)

    train_dates_all, eval_dates = compute_split(args.split_mode, seed=42)
    train_dates = train_dates_all[: args.n_train_dates]
    print(
        f"[A2C:{args.district}] split={args.split_mode} loading episodes "
        f"(train={len(train_dates)}, eval={len(eval_dates)})...",
        flush=True,
    )
    train_episodes = load_episodes(
        train_dates,
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"A2C {args.district} load train" if args.progress else None,
    )
    eval_episodes = load_episodes(
        eval_dates,
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"A2C {args.district} load eval" if args.progress else None,
    )
    all_episodes = train_episodes + eval_episodes
    print(f"[A2C:{args.district}] applying capacity/forecast data...", flush=True)
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
    if vae_stats:
        print(
            "VAE latent override: "
            f"matched={int(vae_stats['vae_matched'])}/{int(vae_stats['vae_total'])}, "
            f"latent_dim={int(vae_stats['vae_latent_dim'])}"
        )
    print(f"heuristic mean reward: {heuristic_mean:.2f}")

    if args.episodes <= 0:
        final_mean, final_rewards = evaluate(policy, eval_episodes, args, device, args.seed)
        torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, out_dir / "actor_critic_final.pt")
        np.save(out_dir / "history.npy", np.asarray(history or [{"episode": 0, "eval_reward": final_mean}], dtype=object))
        print_eval_table("a2c_final", heuristic_rewards, final_rewards, eval_dates)
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
                torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, best_dir / "best_model.pt")
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
    final_mean, final_rewards = evaluate(policy, eval_episodes, args, device, args.seed)
    if not history or abs(float(history[-1]["eval_reward"]) - final_mean) > 1e-9:
        history.append({"episode": episode, "eval_reward": final_mean, "stage": "final"})
    torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, out_dir / "actor_critic_final.pt")
    np.save(out_dir / "history.npy", np.asarray(history, dtype=object))
    print(f"best reward: {best_reward:.2f} at episode {best_episode}")
    print(f"final reward: {final_mean:.2f}")
    print_eval_table("a2c_final", heuristic_rewards, final_rewards, eval_dates)


if __name__ == "__main__":
    main()
