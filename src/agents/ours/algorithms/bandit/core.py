"""Contextual Bandit 기반 따릉이 재배치 agent.

알고리즘:
    LinUCB Contextual Bandit

목적:
    강화학습이 장기 보상을 학습하는 동안 불안정할 수 있으므로, 현재 step에서
    Top-K 후보 중 어느 후보를 고를지만 더 단순한 bandit 문제로 본다.

RL과의 차이:
    - REINFORCE/A2C/PPO/DQN은 현재 action이 이후 state와 reward에 미치는
      장기 효과를 학습한다.
    - Contextual Bandit은 현재 후보 feature와 바로 받은 reward만 사용해
      다음 선택을 개선한다.

State:
    원본 RebalanceEnv observation
    + forecast 수요 feature
    + Top-K 후보별 feature

Action:
    Top-K 후보 rank 중 하나.
    action=0은 현재 후보 중 가장 높은 휴리스틱 후보, action=1은 두 번째 후보다.

Reward:
    원본 환경 reward를 그대로 사용하되, bandit update에는 숫자 안정성을 위해
    reward_scale을 곱한다. 평가 reward는 원본 reward 그대로 출력한다.

LinUCB update:
    각 action a마다 A_a, b_a를 유지한다.

        theta_a = inv(A_a) b_a
        score_a = theta_a^T x_a + alpha * sqrt(x_a^T inv(A_a) x_a)
        A_a <- A_a + x_a x_a^T
        b_a <- b_a + reward * x_a

    첫 항은 exploitation, 두 번째 항은 uncertainty 기반 exploration이다.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.agents.ours.common.candidate_actions import maybe_wrap_candidate_actions
from src.agents.ours.common.data_overrides import apply_capacity_override, attach_forecast_override
from src.agents.ours.common.date_split import compute_split
from src.agents.ours.common.experiment_utils import (
    ENV_KW,
    evaluate_most_imbalanced as evaluate_heuristic,
    load_rebalance_episodes as load_episodes,
    print_eval_table,
)
from src.agents.ours.common.future_demand import build_history_net_profile, maybe_wrap_future_demand
from src.agents.ours.common.vae_latent import attach_vae_latent_override, maybe_wrap_vae_latent
from src.envs.rebalance_env import RebalanceEnv


@dataclass
class LinUCBPolicy:
    """Top-K rank action을 선택하는 disjoint LinUCB policy."""

    n_actions: int
    feature_dim: int
    alpha: float = 0.5
    l2: float = 1.0

    def __post_init__(self) -> None:
        """각 action마다 독립적인 선형 모델 통계량을 초기화한다."""
        eye = np.eye(self.feature_dim, dtype=np.float64)
        self.A = np.stack([self.l2 * eye.copy() for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, self.feature_dim), dtype=np.float64)

    def select_action(self, features: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        """UCB score가 가장 높은 후보 rank를 선택한다."""
        scores = np.full(self.n_actions, -np.inf, dtype=np.float64)
        bonus_coef = self.alpha if explore else 0.0

        for action in np.flatnonzero(mask):
            x = features[action].astype(np.float64)
            inv_a = np.linalg.inv(self.A[action])
            theta = inv_a @ self.b[action]
            mean = float(theta @ x)
            uncertainty = float(np.sqrt(max(x @ inv_a @ x, 0.0)))
            scores[action] = mean + bonus_coef * uncertainty

        if not np.isfinite(scores).any():
            return int(np.flatnonzero(mask)[0])
        return int(np.argmax(scores))

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        """선택한 action의 선형 reward 모델만 갱신한다."""
        x = x.astype(np.float64)
        self.A[action] += np.outer(x, x)
        self.b[action] += float(reward) * x

    def state_dict(self) -> dict[str, np.ndarray]:
        """npz 저장용 state를 반환한다."""
        return {"A": self.A, "b": self.b}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """저장된 npz state를 복원한다."""
        self.A = np.asarray(state["A"], dtype=np.float64)
        self.b = np.asarray(state["b"], dtype=np.float64)


def make_env(ep, args: argparse.Namespace, seed: int | None = None, for_eval: bool = False):
    """공통 환경을 만들고 agent-local state/action wrapper를 적용한다."""
    env = RebalanceEnv(ep, seed=seed, **ENV_KW)
    env = maybe_wrap_future_demand(env, args)
    env = maybe_wrap_vae_latent(env, args)
    return maybe_wrap_candidate_actions(env, args)


def candidate_features_from_obs(obs: np.ndarray, args: argparse.Namespace, n_actions: int) -> np.ndarray:
    """observation 끝에 붙은 Top-K 후보 feature를 bandit feature로 변환한다."""
    top_k = int(getattr(args, "candidate_top_k", n_actions))
    feature_mode = getattr(args, "candidate_feature_mode", "none")

    if feature_mode == "basic":
        dim = 8
        needed = top_k * dim
        if len(obs) < needed:
            raise ValueError("observation is shorter than Top-K candidate feature block")
        return np.asarray(obs[-needed:], dtype=np.float64).reshape(top_k, dim)

    # feature를 붙이지 않은 경우에도 실행은 가능하게 rank 기반 최소 feature를 만든다.
    rows = []
    for rank in range(top_k):
        rank_norm = rank / max(top_k - 1, 1)
        rows.append([1.0, rank_norm])
    return np.asarray(rows, dtype=np.float64)


def run_episode(
    env,
    policy: LinUCBPolicy,
    args: argparse.Namespace,
    seed: int,
    train: bool,
) -> float:
    """한 episode를 실행하고, train=True이면 step마다 LinUCB를 갱신한다."""
    state, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    steps = 0

    while not done and steps < args.max_num_steps:
        mask = env.action_masks()
        features = candidate_features_from_obs(state, args, policy.n_actions)
        action = policy.select_action(features, mask, explore=train)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)

        if train:
            # 학습 안정성을 위해 update에만 reward scale을 적용한다.
            policy.update(action, features[action], float(reward) * args.bandit_reward_scale)

        state = next_state
        done = terminated or truncated
        steps += 1
    return total


def evaluate(policy: LinUCBPolicy, episodes: list, args: argparse.Namespace, seed: int) -> tuple[float, list[float]]:
    """고정 평가셋에서 exploration bonus 없이 greedy bandit policy를 평가한다."""
    rewards = []
    for ep in episodes:
        env = make_env(ep, args, seed=seed, for_eval=True)
        rewards.append(run_episode(env, policy, args, seed=seed, train=False))
    return float(np.mean(rewards)), rewards


def parse_args() -> argparse.Namespace:
    """Contextual Bandit 실험 CLI 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="LinUCB contextual bandit agent")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--episode-cache-dir", default="data/episode_cache")
    parser.add_argument("--no-episode-cache", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=170_000)
    parser.add_argument("--max-num-steps", type=int, default=500)
    parser.add_argument("--n-train-dates", type=int, default=60)
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological"],
        default="random",
        help="random: seed=42 셔플 후 80/20, chronological: 시간순 80/20 (계절 OOD 평가)",
    )
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--tag", default="bandit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bandit-alpha", type=float, default=0.5)
    parser.add_argument("--bandit-l2", type=float, default=1.0)
    parser.add_argument("--bandit-reward-scale", type=float, default=0.01)
    parser.add_argument("--bc-epochs", type=int, default=0, help="runner 호환용. bandit에서는 사용하지 않는다.")
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
    parser.add_argument("--candidate-top-k", type=int, default=12)
    parser.add_argument("--candidate-mode", choices=["imbalance", "forecast_imbalance"], default="forecast_imbalance")
    parser.add_argument("--candidate-travel-coef", type=float, default=0.20)
    parser.add_argument("--candidate-zone-mode", choices=["none", "static3"], default="static3")
    parser.add_argument("--candidate-zone-count", type=int, default=3)
    parser.add_argument("--candidate-zone-penalty", type=float, default=1.0)
    parser.add_argument("--candidate-feature-mode", choices=["none", "basic"], default="basic")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps"], help="runner 호환용. bandit은 CPU만 사용한다.")
    return parser.parse_args()


def main() -> None:
    """데이터 로드, LinUCB 온라인 학습, Best/Final 평가를 실행한다."""
    args = parse_args()
    if args.bc_epochs:
        print("Contextual Bandit은 BC를 사용하지 않으므로 --bc-epochs 값은 무시합니다.")

    train_dates_all, eval_dates = compute_split(args.split_mode, seed=42)
    print(
        f"[BANDIT:{args.district}] split={args.split_mode} loading episodes "
        f"(train={args.n_train_dates}, eval={len(eval_dates)})...",
        flush=True,
    )
    train_episodes = load_episodes(
        train_dates_all[: args.n_train_dates],
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"BANDIT {args.district} load train" if args.progress else None,
    )
    eval_episodes = load_episodes(
        eval_dates,
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"BANDIT {args.district} load eval" if args.progress else None,
    )
    all_episodes = train_episodes + eval_episodes

    print(f"[BANDIT:{args.district}] applying capacity/forecast data...", flush=True)
    capacity_stats = apply_capacity_override(all_episodes, args.capacity_path, args.capacity_initial_fill_ratio)
    forecast_stats = attach_forecast_override(all_episodes, args.forecast_path)
    vae_stats = attach_vae_latent_override(all_episodes, args.vae_latent_path)
    if args.future_mode in {"history_net", "history_projected_travel"}:
        args.history_profile = build_history_net_profile(train_episodes)

    sample_env = make_env(eval_episodes[0], args)
    n_actions = int(sample_env.action_space.n)
    feature_dim = candidate_features_from_obs(sample_env.reset(seed=args.seed)[0], args, n_actions).shape[1]
    policy = LinUCBPolicy(n_actions, feature_dim, alpha=args.bandit_alpha, l2=args.bandit_l2)
    rng = np.random.default_rng(args.seed)

    out_dir = Path("logs") / f"bandit_{args.tag}"
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    heuristic_mean, heuristic_rewards = evaluate_heuristic(eval_episodes, args.seed)
    print(f"=== Contextual Bandit LinUCB | tag={args.tag} ===")
    print(f"obs_dim={sample_env.observation_space.shape[0]}, n_actions={n_actions}, feature_dim={feature_dim}")
    print(
        f"alpha={args.bandit_alpha}, l2={args.bandit_l2}, "
        f"reward_scale={args.bandit_reward_scale}"
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

    history = []
    best_reward = -np.inf
    best_step = 0
    best_state = copy.deepcopy(policy.state_dict())
    elapsed_steps = 0
    progress_bar = tqdm(
        total=args.total_timesteps,
        desc=f"BANDIT {args.district}",
        unit="step",
        disable=not args.progress,
    )

    while elapsed_steps < args.total_timesteps:
        ep = train_episodes[int(rng.integers(len(train_episodes)))]
        env = make_env(ep, args, seed=args.seed + elapsed_steps)
        before = elapsed_steps
        run_episode(env, policy, args, seed=args.seed + elapsed_steps, train=True)
        elapsed_steps += args.max_num_steps
        elapsed_steps = min(elapsed_steps, args.total_timesteps)
        progress_bar.update(elapsed_steps - before)

        if elapsed_steps == args.total_timesteps or elapsed_steps % args.eval_every == 0:
            eval_reward, _ = evaluate(policy, eval_episodes, args, args.seed)
            history.append({"timesteps": elapsed_steps, "eval_reward": eval_reward})
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_step = elapsed_steps
                best_state = copy.deepcopy(policy.state_dict())
                np.savez(best_dir / "best_model.npz", **best_state)
            progress_bar.set_postfix(
                {
                    "eval": f"{eval_reward:.1f}",
                    "base": f"{heuristic_mean:.1f}",
                    "delta": f"{eval_reward - heuristic_mean:+.1f}",
                    "best": f"{best_reward - heuristic_mean:+.1f}",
                }
            )
            message = (
                f"timesteps={elapsed_steps:7d} eval={eval_reward:8.2f} "
                f"delta={eval_reward - heuristic_mean:+8.2f}"
            )
            if args.progress:
                progress_bar.clear()
                tqdm.write(message)
                progress_bar.refresh()
            else:
                print(message)

    progress_bar.close()
    final_mean, final_rewards = evaluate(policy, eval_episodes, args, args.seed)
    np.savez(out_dir / "bandit_final.npz", **policy.state_dict())
    np.save(out_dir / "history.npy", np.asarray(history, dtype=object))

    policy.load_state_dict(best_state)
    best_mean, best_rewards = evaluate(policy, eval_episodes, args, args.seed)
    print(f"best reward: {best_mean:.2f} at timesteps {best_step}")
    print(f"final reward: {final_mean:.2f}")
    print_eval_table("bandit_best", heuristic_rewards, best_rewards, eval_dates)
    print_eval_table("bandit_final", heuristic_rewards, final_rewards, eval_dates)


if __name__ == "__main__":
    main()
