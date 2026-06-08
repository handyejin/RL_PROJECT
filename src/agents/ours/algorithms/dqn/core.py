"""DQN + 공공데이터 예측/거치대 수 보강 실행 파일.

목적:
    팀원이 만든 DQN 소스(`masked_dqn.py`)는 수정하지 않고,
    우리 수정 데이터 기준에서 DQN도 REINFORCE/A2C와 비교한다.

알고리즘:
    Masked DQN
    + Double DQN target 옵션
    + Dueling Q-network 옵션
    + forecast_projected_travel state feature
    + 실제 거치대 수(capacity) agent-local 반영

네트워크와 loss:
    Q-network가 Q(s, a)를 출력한다. 학습은 replay buffer에서 transition을 뽑아
    TD target을 만들고, 현재 Q값과 target Q값 사이의 Huber loss를 줄인다.

    Double DQN:
        a* = argmax_a Q_online(s', a)
        y = r + gamma * Q_target(s', a*)
        loss = Huber(Q_online(s, a), y)

State 보강:
    정류소별 1시간 예상 대여/반납을 사용해 예상 재고 방향을 만든다.
    현재 재고에 pred_returns_1h - pred_rentals_1h를 더해
    1시간 뒤 예상 불균형을 feature로 제공한다.

Action:
    현재 선택된 트럭이 이동할 정류소 index.
    action mask를 적용해 불가능한 정류소 선택을 막는다.

Reward:
    평가는 원본 RebalanceEnv reward를 그대로 사용한다.
    학습에서는 선택적으로 reward scale만 줄여 DQN TD target의 크기를 안정화한다.

실행 예:
    PYTHONPATH=. python -m src.agents.ours.run_from_config \
        --config config/ours/dqn_topk12.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset

from src.agents.ours.common.bc_utils import collect_bc_data
from src.agents.ours.common.candidate_actions import maybe_wrap_candidate_actions
from src.agents.ours.common.data_overrides import apply_capacity_override, attach_forecast_override
from src.agents.ours.common.date_split import compute_split
from src.agents.ours.common.experiment_utils import (
    ENV_KW,
    evaluate_most_imbalanced as evaluate_heuristic,
    load_rebalance_episodes as load_episodes,
    print_eval_table,
)
from src.agents.ours.common.future_demand import maybe_wrap_future_demand
from src.agents.ours.common.vae_latent import attach_vae_latent_override, maybe_wrap_vae_latent
from src.agents.masked_dqn import MaskableDQN
from src.envs.rebalance_env import RebalanceEnv


class DuelingQNetwork(torch.nn.Module):
    """Dueling DQN Q-network.

    Q(s, a)를 상태 가치 V(s)와 행동 advantage A(s, a)로 나누어 추정한다.
    Top-K 후보 구조에서는 "현재 상태 자체가 좋은지"와 "후보 rank 중 무엇이 좋은지"가
    섞이기 쉬우므로, DQN 안정화 실험에서 선택적으로 사용한다.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.net_arch = net_arch or [64, 64]
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images
        action_dim = int(action_space.n)

        value_net = create_mlp(features_dim, 1, self.net_arch, activation_fn)
        advantage_net = create_mlp(features_dim, action_dim, self.net_arch, activation_fn)
        self.value_net = nn.Sequential(*value_net)
        self.advantage_net = nn.Sequential(*advantage_net)

    @property
    def device(self):
        """SB3 policy 저장/로드 호환을 위한 device property."""
        return next(self.parameters()).device

    def extract_features(self, obs, features_extractor):
        """SB3 QNetwork와 같은 방식으로 feature extractor를 통과시킨다."""
        return features_extractor(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Q(s,a)=V(s)+A(s,a)-mean_a A(s,a)를 계산한다."""
        features = self.extract_features(obs, self.features_extractor)
        value = self.value_net(features)
        advantage = self.advantage_net(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def set_training_mode(self, mode: bool) -> None:
        """SB3 BasePolicy 인터페이스와 맞춘 training mode 설정."""
        self.train(mode)


class DuelingDQNPolicy(DQNPolicy):
    """SB3 DQNPolicy의 Q-network만 Dueling 구조로 바꾼 정책 클래스."""

    def make_q_net(self):
        """online/target Q-network 생성 시 DuelingQNetwork를 사용한다."""
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


class DQNRewardScaleWrapper(gym.Wrapper):
    """DQN 학습 reward만 일정 비율로 줄이는 agent-local wrapper.

    평가 환경에는 적용하지 않는다. 원본 reward 식과 평가 reward는 그대로 두고,
    TD target의 scale만 낮춰 Q-learning 불안정을 줄이기 위한 용도다.
    """

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = float(scale)

    def __getattr__(self, name):
        """wrapper에 없는 속성은 원본 env로 위임한다."""
        return getattr(self.env, name)

    def step(self, action):
        """학습에 전달되는 reward만 scale한다."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(reward)
        info["dqn_reward_scale"] = self.scale
        return obs, float(reward) * self.scale, terminated, truncated, info


class TopKMaskableDQN(MaskableDQN):
    """Top-K 후보 action에 맞게 next-state mask를 TD target에도 적용하는 DQN.

    팀원 `MaskableDQN`은 현재 action을 고를 때 mask를 적용한다. 하지만 TD target의
    `max_a Q(s', a)`에는 다음 state의 mask가 들어가지 않는다. Top-K wrapper에서는
    state마다 유효한 rank 수가 달라질 수 있으므로, invalid rank가 target에 섞이지
    않도록 우리 실험 코드 안에서만 보정한다.
    """

    def __init__(
        self,
        *args,
        next_mask_top_k: int = 0,
        candidate_feature_dim: int = 8,
        masked_target_q: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.next_mask_top_k = int(next_mask_top_k)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.masked_target_q = bool(masked_target_q)

    def _next_action_mask_from_obs(self, next_observations: torch.Tensor) -> torch.Tensor | None:
        """observation 뒤에 붙은 candidate feature에서 다음 state valid mask를 복원한다."""
        if not self.masked_target_q or self.next_mask_top_k <= 0:
            return None
        tail_dim = self.next_mask_top_k * self.candidate_feature_dim
        if next_observations.shape[1] < tail_dim:
            return None
        tail = next_observations[:, -tail_dim:]
        features = tail.reshape(-1, self.next_mask_top_k, self.candidate_feature_dim)
        # CandidateTopKActionWrapper의 feature 0번은 valid flag다.
        mask = features[:, :, 0] > 0.5
        if mask.shape[1] != self.action_space.n:
            return None
        # 안전장치: 혹시 전부 False면 rank 0만 허용한다.
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return mask

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Double DQN target에 next-state Top-K mask를 적용해 Q update를 수행한다."""
        if not self.masked_target_q or self.next_mask_top_k <= 0:
            return super().train(gradient_steps, batch_size)

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            next_mask = self._next_action_mask_from_obs(replay_data.next_observations)

            with torch.no_grad():
                if self.double_q:
                    # Double DQN: online net으로 다음 action을 고르되 invalid rank는 제외한다.
                    next_q_online = self.q_net(replay_data.next_observations)
                    if next_mask is not None:
                        next_q_online = next_q_online.masked_fill(~next_mask, -1e9)
                    next_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_target = self.q_net_target(replay_data.next_observations)
                    next_q_values = next_q_target.gather(1, next_actions)
                else:
                    next_q_target = self.q_net_target(replay_data.next_observations)
                    if next_mask is not None:
                        next_q_target = next_q_target.masked_fill(~next_mask, -1e9)
                    next_q_values, _ = next_q_target.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)

                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def make_env(episodes, args: argparse.Namespace, seed: int | None = None, for_eval: bool = False):
    """공통 환경을 만들고 agent-local forecast wrapper를 적용한다."""
    env = RebalanceEnv(episodes, seed=seed, **ENV_KW)
    env = maybe_wrap_future_demand(env, args)
    env = maybe_wrap_vae_latent(env, args)
    if not for_eval:
        reward_scale = float(getattr(args, "dqn_reward_scale", 1.0) or 1.0)
        if reward_scale != 1.0:
            env = DQNRewardScaleWrapper(env, reward_scale)
    return maybe_wrap_candidate_actions(env, args)


def evaluate(model: MaskableDQN, episodes: list, args: argparse.Namespace, seed: int) -> tuple[float, list[float]]:
    """고정 7일 평가셋에서 greedy DQN policy의 평균 reward를 계산한다."""
    rewards = []
    for ep in episodes:
        env = make_env(ep, args, seed=seed, for_eval=True)
        obs, _ = env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            # DQN Q값 argmax에도 action mask를 적용한다.
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards)), rewards


def pretrain_behavior_cloning(model: MaskableDQN, train_episodes: list, args: argparse.Namespace) -> dict[str, float]:
    """teacher action을 CrossEntropy로 모방해 DQN Q-network를 먼저 초기화한다."""
    states, actions, masks = collect_bc_data(train_episodes, args, make_env)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(masks)),
        batch_size=args.bc_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.q_net.parameters(), lr=args.bc_lr)
    last_loss = 0.0
    last_acc = 0.0
    model.policy.set_training_mode(True)
    for epoch in range(args.bc_epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y, m in loader:
            x = x.to(model.device, dtype=torch.float32)
            y = y.to(model.device, dtype=torch.long)
            m = m.to(model.device, dtype=torch.bool)

            # DQN BC loss:
            #   Q-network 출력을 action logit처럼 보고 teacher action y를 맞춘다.
            #   action mask는 불가능한 정류소의 logit을 매우 작게 만들어 제외한다.
            q_values = model.q_net(x)
            masked_q = q_values.masked_fill(~m, -1e9)
            loss = F.cross_entropy(masked_q, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.q_net.parameters(), args.max_bc_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            correct += int((masked_q.argmax(dim=1) == y).sum().item())
            total += len(y)
        last_loss = total_loss / max(total, 1)
        last_acc = correct / max(total, 1)
        if epoch == 0 or (epoch + 1) % max(args.bc_log_every, 1) == 0:
            print(f"  BC epoch {epoch+1}/{args.bc_epochs}: loss={last_loss:.4f}, acc={last_acc:.3f}")
    # target network도 BC로 학습된 Q-network와 맞춰 DQN 시작점을 일관되게 만든다.
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    return {"bc_samples": float(len(actions)), "bc_loss": last_loss, "bc_acc": last_acc}


def parse_args() -> argparse.Namespace:
    """DQN 비교 실험을 위한 CLI 옵션을 정의한다.

    팀원 DQN 구현은 수정하지 않고, 이 core에서 state 보강과 candidate action,
    BC 초기화, Double DQN, n-step 같은 실험 옵션만 조합한다.
    """
    parser = argparse.ArgumentParser(description="DQN forecast/capacity comparison agent")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--episode-cache-dir", default="data/episode_cache")
    parser.add_argument("--no-episode-cache", action="store_true")
    parser.add_argument("--n-train-dates", type=int, default=200)
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological"],
        default="random",
        help="random: seed=42 셔플 후 80/20, chronological: 시간순 80/20 (계절 OOD 평가)",
    )
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--tag", default="dqn_forecast_capacity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--double-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dueling-q", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dqn-reward-scale", type=float, default=1.0)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-dates", type=int, default=200)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-log-every", type=int, default=5)
    parser.add_argument("--max-bc-grad-norm", type=float, default=10.0)
    parser.add_argument("--bc-only", action="store_true")
    parser.add_argument(
        "--bc-policy",
        choices=["masked_heuristic", "future_heuristic", "forecast_heuristic"],
        default="masked_heuristic",
    )
    parser.add_argument(
        "--future-mode",
        choices=["none", "forecast_net", "forecast_inout", "forecast_projected_travel"],
        default="forecast_projected_travel",
    )
    parser.add_argument("--future-horizon", type=int, default=6)
    parser.add_argument("--vae-mode", choices=["none", "demand_latent"], default="none")
    parser.add_argument("--vae-latent-path", default="")
    parser.add_argument("--vae-latent-dim", type=int, default=4)
    parser.add_argument("--capacity-path", default="data/processed/station_capacity.csv")
    parser.add_argument("--capacity-initial-fill-ratio", type=float, default=0.5)
    parser.add_argument("--forecast-path", default="data/processed/demand_forecast_1h_rlholdout_seed42.parquet")
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--candidate-mode", choices=["imbalance", "forecast_imbalance"], default="imbalance")
    parser.add_argument("--candidate-travel-coef", type=float, default=0.0)
    parser.add_argument("--candidate-zone-mode", choices=["none", "static3"], default="none")
    parser.add_argument("--candidate-zone-count", type=int, default=3)
    parser.add_argument("--candidate-zone-penalty", type=float, default=0.0)
    parser.add_argument("--candidate-feature-mode", choices=["none", "basic"], default="none")
    parser.add_argument("--masked-target-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-update-steps", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    """MaskableDQN을 생성하고 주기적 7일 평가로 best/final 모델을 저장한다."""
    args = parse_args()
    train_dates_all, eval_dates = compute_split(args.split_mode, seed=42)
    print(
        f"[DQN:{args.district}] split={args.split_mode} loading episodes "
        f"(train={args.n_train_dates}, eval={len(eval_dates)})...",
        flush=True,
    )
    train_episodes = load_episodes(
        train_dates_all[: args.n_train_dates],
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"DQN {args.district} load train" if args.progress else None,
    )
    eval_episodes = load_episodes(
        eval_dates,
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"DQN {args.district} load eval" if args.progress else None,
    )
    all_episodes = train_episodes + eval_episodes

    capacity_stats = apply_capacity_override(
        all_episodes,
        args.capacity_path,
        args.capacity_initial_fill_ratio,
    )
    forecast_stats = attach_forecast_override(all_episodes, args.forecast_path)
    vae_stats = attach_vae_latent_override(all_episodes, args.vae_latent_path)

    train_env = DummyVecEnv([lambda: make_env(train_episodes, args, seed=args.seed)])
    sample_env = make_env(eval_episodes[0], args, seed=args.seed)
    obs_dim = int(sample_env.observation_space.shape[0])
    n_actions = int(sample_env.action_space.n)

    out_dir = Path("logs") / f"dqn_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    heuristic_mean, heuristic_rewards = evaluate_heuristic(eval_episodes, args.seed)
    print(f"=== DQN | tag={args.tag} ===")
    print(f"device={args.device}, obs_dim={obs_dim}, n_actions={n_actions}")
    print(
        "dqn target: "
        f"double_q={args.double_q}, "
        f"dueling_q={args.dueling_q}, "
        f"masked_target_q={args.masked_target_q and args.candidate_feature_mode == 'basic' and args.candidate_top_k > 0}, "
        f"reward_scale={args.dqn_reward_scale}"
    )
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

    policy_class = DuelingDQNPolicy if args.dueling_q else "MlpPolicy"
    model = TopKMaskableDQN(
        policy_class,
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        n_steps=args.n_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs={"net_arch": [args.hidden, args.hidden]},
        seed=args.seed,
        verbose=0,
        device=args.device,
        double_q=args.double_q,
        next_mask_top_k=args.candidate_top_k if args.candidate_feature_mode == "basic" else 0,
        masked_target_q=args.masked_target_q,
    )

    history = []
    best_reward = -np.inf
    best_step = 0
    best_policy_state = copy.deepcopy(model.policy.state_dict())
    if args.bc_epochs > 0:
        bc_stats = pretrain_behavior_cloning(model, train_episodes, args)
        print(
            f"BC done: samples={int(bc_stats['bc_samples'])}, "
            f"loss={bc_stats['bc_loss']:.4f}, acc={bc_stats['bc_acc']:.3f}"
        )
        eval_reward, _ = evaluate(model, eval_episodes, args, args.seed)
        history.append({"timesteps": 0, "eval_reward": eval_reward, "stage": "bc"})
        best_reward = eval_reward
        best_policy_state = copy.deepcopy(model.policy.state_dict())
        model.save(out_dir / "best_model")
        print(f"timesteps={0:7d} eval={eval_reward:8.2f} stage=BC")

    if args.bc_only or args.total_timesteps <= 0:
        final_mean, final_rewards = evaluate(model, eval_episodes, args, args.seed)
        model.save(out_dir / "final_model")
        np.save(out_dir / "history.npy", np.asarray(history or [{"timesteps": 0, "eval_reward": final_mean}], dtype=object))
        print_eval_table("dqn_bc_only", heuristic_rewards, final_rewards, eval_dates)
        return

    steps_done = 0
    progress_bar = None
    if args.progress:
        progress_bar = tqdm(
            total=args.total_timesteps,
            desc=f"DQN {args.district}",
            unit="step",
            dynamic_ncols=True,
        )
    try:
        while steps_done < args.total_timesteps:
            next_eval_step = min(
                ((steps_done // args.eval_every) + 1) * args.eval_every,
                args.total_timesteps,
            )
            chunk = min(next_eval_step - steps_done, args.total_timesteps - steps_done)
            if progress_bar is not None and args.progress_update_steps > 0:
                chunk = min(chunk, args.progress_update_steps)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
            steps_done += chunk
            if progress_bar is not None:
                progress_bar.update(chunk)
            if steps_done < next_eval_step and steps_done < args.total_timesteps:
                continue
            eval_reward, eval_rewards = evaluate(model, eval_episodes, args, args.seed)
            history.append({"timesteps": steps_done, "eval_reward": eval_reward})
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_step = steps_done
                best_policy_state = copy.deepcopy(model.policy.state_dict())
                model.save(out_dir / "best_model")
            delta = eval_reward - heuristic_mean
            if progress_bar is not None:
                progress_bar.set_postfix(
                    eval=f"{eval_reward:.1f}",
                    base=f"{heuristic_mean:.1f}",
                    delta=f"{delta:+.1f}",
                    best=f"{best_reward - heuristic_mean:+.1f}",
                )
                tqdm.write(f"timesteps={steps_done:7d} eval={eval_reward:8.2f} delta={delta:+8.2f}")
            else:
                print(f"timesteps={steps_done:7d} eval={eval_reward:8.2f}")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    final_mean, final_rewards = evaluate(model, eval_episodes, args, args.seed)
    model.save(out_dir / "final_model")
    if not history or abs(float(history[-1]["eval_reward"]) - final_mean) > 1e-9:
        history.append({"timesteps": steps_done, "eval_reward": final_mean, "stage": "final"})
    np.save(out_dir / "history.npy", np.asarray(history, dtype=object))
    model.policy.load_state_dict(best_policy_state)
    best_mean, best_rewards = evaluate(model, eval_episodes, args, args.seed)

    print(f"best reward: {best_reward:.2f} at timesteps {best_step}")
    print(f"final reward: {final_mean:.2f}")
    print_eval_table("dqn_best", heuristic_rewards, best_rewards, eval_dates)


if __name__ == "__main__":
    main()
