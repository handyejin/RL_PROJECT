"""QR-DQN + 공공데이터 예측/거치대 수 보강 실행 파일.

목적:
    REINFORCE/A2C/PPO 와 동일한 ours env (forecast + capacity + top-k 후보) 위에서
    Quantile Regression DQN 만 알고리즘으로 swap 한 실험을 돌리기 위함.

알고리즘:
    Maskable QR-DQN
    + Double DQN target 옵션
    + forecast_projected_travel state feature
    + 실제 거치대 수(capacity) agent-local 반영

State / Action / Reward 정의는 dqn_core 와 동일하며,
다른 점은 Q-값 대신 quantile distribution 을 학습한다는 것뿐이다.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, TensorDataset

from src.agents.baselines import get_policy
from src.agents.ours.common.bc_utils import collect_bc_data
from src.agents.ours.common.candidate_actions import maybe_wrap_candidate_actions
from src.agents.ours.common.data_overrides import apply_capacity_override, attach_forecast_override
from src.agents.ours.common.date_split import compute_split
from src.agents.ours.common.future_demand import maybe_wrap_future_demand
from src.agents.ours.common.reward_shaping import maybe_wrap_agent_reward_shaping
from src.agents.qrdqn import MaskableQRDQN
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


def load_episodes(dates: list[str], district: str, processed_dir: str = "data/processed") -> list:
    """날짜 목록을 RebalanceEnv episode 데이터로 변환한다."""
    return [
        load_episode(processed_dir, district=district, episode_start=f"{date} 00:00")
        for date in dates
    ]


def make_env(episodes, args: argparse.Namespace, seed: int | None = None, for_eval: bool = False):
    """공통 환경을 만들고 agent-local forecast wrapper를 적용한다."""
    env = RebalanceEnv(episodes, seed=seed, **ENV_KW)
    env = maybe_wrap_future_demand(env, args)
    if not for_eval:
        env = maybe_wrap_agent_reward_shaping(env, args)
    return maybe_wrap_candidate_actions(env, args)


def evaluate(model: MaskableQRDQN, episodes: list, args: argparse.Namespace, seed: int) -> tuple[float, list[float]]:
    """고정 7일 평가셋에서 greedy QR-DQN policy의 평균 reward를 계산한다."""
    rewards = []
    for ep in episodes:
        env = make_env(ep, args, seed=seed, for_eval=True)
        obs, _ = env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
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


def evaluate_heuristic(episodes: list, seed: int) -> tuple[float, list[float]]:
    """같은 데이터 기준에서 most_imbalanced 휴리스틱 reward를 계산한다."""
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
    """QR-DQN 비교 실험을 위한 CLI 옵션을 정의한다.

    dqn_core 와 동일한 wrapper/state 옵션을 그대로 받고, QR-DQN 고유 옵션
    (n_quantiles, kappa) 만 추가한다.
    """
    parser = argparse.ArgumentParser(description="MaskableQRDQN forecast/capacity comparison agent")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--n-train-dates", type=int, default=200)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--tag", default="qrdqn_forecast_capacity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--double-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-quantiles", type=int, default=200,
                        help="QR-DQN quantile 개수")
    parser.add_argument("--kappa", type=float, default=1.0,
                        help="Huber loss threshold")
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological"],
        default="random",
        help="random: seed=42 셔플 후 80/20, chronological: 시간순 80/20 (계절 OOD 평가)",
    )
    return parser.parse_args()


def main() -> None:
    """MaskableQRDQN 을 생성하고 주기적 7일 평가로 best/final 모델을 저장한다."""
    args = parse_args()
    train_dates_all, eval_dates = compute_split(args.split_mode, seed=42)
    train_episodes = load_episodes(train_dates_all[: args.n_train_dates], args.district, args.processed_dir)
    eval_episodes = load_episodes(eval_dates, args.district, args.processed_dir)
    all_episodes = train_episodes + eval_episodes

    capacity_stats = apply_capacity_override(
        all_episodes,
        args.capacity_path,
        args.capacity_initial_fill_ratio,
    )
    forecast_stats = attach_forecast_override(all_episodes, args.forecast_path)

    train_env = DummyVecEnv([lambda: make_env(train_episodes, args, seed=args.seed)])
    sample_env = make_env(eval_episodes[0], args, seed=args.seed)
    obs_dim = int(sample_env.observation_space.shape[0])
    n_actions = int(sample_env.action_space.n)

    out_dir = Path("logs") / f"qrdqn_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    heuristic_mean, heuristic_rewards = evaluate_heuristic(eval_episodes, args.seed)
    print(f"=== QR-DQN | tag={args.tag} ===")
    print(f"device={args.device}, obs_dim={obs_dim}, n_actions={n_actions}, n_quantiles={args.n_quantiles}")
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

    model = MaskableQRDQN(
        "MlpPolicy",
        train_env,
        n_quantiles=args.n_quantiles,
        kappa=args.kappa,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs={"net_arch": [args.hidden, args.hidden]},
        seed=args.seed,
        verbose=0,
        device=args.device,
        double_q=args.double_q,
    )

    history = []
    best_reward = -np.inf
    best_step = 0
    best_policy_state = copy.deepcopy(model.policy.state_dict())
    patience_left = args.finetune_patience

    if args.bc_only or args.total_timesteps <= 0:
        final_mean, final_rewards = evaluate(model, eval_episodes, args, args.seed)
        model.save(out_dir / "final_model")
        np.save(out_dir / "history.npy", np.asarray([{"timesteps": 0, "eval_reward": final_mean}], dtype=object))
        print_eval_table("qrdqn_bc_only", heuristic_rewards, final_rewards, eval_dates)
        return

    steps_done = 0
    while steps_done < args.total_timesteps:
        chunk = min(args.eval_every, args.total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        steps_done += chunk
        eval_reward, eval_rewards = evaluate(model, eval_episodes, args, args.seed)
        history.append({"timesteps": steps_done, "eval_reward": eval_reward})
        if eval_reward > best_reward:
            best_reward = eval_reward
            best_step = steps_done
            best_policy_state = copy.deepcopy(model.policy.state_dict())
            patience_left = args.finetune_patience
            model.save(out_dir / "best_model")
        else:
            if args.rollback_to_best_on_eval:
                model.policy.load_state_dict(best_policy_state)
            if args.finetune_patience > 0:
                patience_left -= 1
        print(f"timesteps={steps_done:7d} eval={eval_reward:8.2f}")
        if args.finetune_patience > 0 and patience_left <= 0:
            print(f"fine-tuning early stop: best_step={best_step}, best_reward={best_reward:.2f}")
            break

    final_mean, final_rewards = evaluate(model, eval_episodes, args, args.seed)
    model.save(out_dir / "final_model")
    if not history or abs(float(history[-1]["eval_reward"]) - final_mean) > 1e-9:
        history.append({"timesteps": steps_done, "eval_reward": final_mean, "stage": "final"})
    np.save(out_dir / "history.npy", np.asarray(history, dtype=object))

    print(f"best reward: {best_reward:.2f} at timesteps {best_step}")
    print(f"final reward: {final_mean:.2f}")
    print_eval_table("qrdqn_final", heuristic_rewards, final_rewards, eval_dates)


if __name__ == "__main__":
    main()
