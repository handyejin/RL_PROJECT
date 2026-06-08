"""MaskablePPO 기반 따릉이 재배치 agent 공통 실행 루프.

알고리즘:
    PPO(Proximal Policy Optimization)
    + GAE advantage estimate
    + clipped policy objective
    + action mask

구현 방식:
    PPO update는 sb3-contrib의 MaskablePPO를 사용한다. 이 파일은 공통 데이터,
    action mask, 수요예측 state, 주기적 7일 평가, best/final 저장을 연결한다.

PPO 목적함수:
    r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
    L_clip = min(
        r_t(theta) * A_t,
        clip(r_t(theta), 1 - eps, 1 + eps) * A_t
    )
    최종 loss에는 policy loss, value loss, entropy bonus가 함께 들어간다.

State:
    기본형은 팀 공통 RebalanceEnv의 원본 observation만 사용한다.
    수정형은 observation 뒤에 forecast/capacity 기반 feature를 추가한다.

Action:
    현재 트럭이 이동할 정류소 index를 선택한다.
    MaskablePPO가 action mask를 사용해 불가능한 정류소 선택을 제외한다.

Reward:
    원본 RebalanceEnv reward를 그대로 사용한다.

실행 예:
    PYTHONPATH=. python -m src.agents.ours.run_from_config \
        --config config/ours/ppo_topk12.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset

from src.agents.ours.common.bc_utils import collect_bc_data
from src.agents.ours.common.candidate_actions import maybe_wrap_candidate_actions
from src.agents.ours.common.data_overrides import apply_capacity_override, attach_forecast_override
from src.agents.ours.common.experiment_utils import (
    ENV_KW,
    EVAL_DATES,
    TRAIN_DATES,
    evaluate_most_imbalanced as evaluate_heuristic,
    load_rebalance_episodes as load_episodes,
    print_eval_table,
)
from src.agents.ours.common.future_demand import maybe_wrap_future_demand
from src.agents.ours.common.vae_latent import attach_vae_latent_override, maybe_wrap_vae_latent
from src.envs.rebalance_env import RebalanceEnv


def make_env(episodes, args: argparse.Namespace, seed: int | None = None, for_eval: bool = False):
    """공통 환경을 만들고, 필요하면 agent-local forecast wrapper를 적용한다."""
    env = RebalanceEnv(episodes, seed=seed, **ENV_KW)
    env = maybe_wrap_future_demand(env, args)
    env = maybe_wrap_vae_latent(env, args)
    return maybe_wrap_candidate_actions(env, args)


def evaluate(model: MaskablePPO, episodes: list, args: argparse.Namespace, seed: int) -> tuple[float, list[float]]:
    """고정 7일 평가셋에서 greedy PPO policy의 평균 reward를 계산한다."""
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


def pretrain_behavior_cloning(model: MaskablePPO, train_episodes: list, args: argparse.Namespace) -> dict[str, float]:
    """teacher action을 log-probability loss로 모방해 PPO policy를 먼저 초기화한다."""
    states, actions, masks = collect_bc_data(train_episodes, args, make_env)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(masks)),
        batch_size=args.bc_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.bc_lr)
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

            # PPO BC loss:
            #   MaskablePPO policy의 masked categorical distribution에서
            #   teacher action의 log probability를 최대화한다.
            dist = model.policy.get_distribution(x, action_masks=m)
            log_prob = dist.log_prob(y)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), args.max_bc_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            correct += int((dist.distribution.probs.argmax(dim=1) == y).sum().item())
            total += len(y)
        last_loss = total_loss / max(total, 1)
        last_acc = correct / max(total, 1)
        if epoch == 0 or (epoch + 1) % max(args.bc_log_every, 1) == 0:
            print(f"  BC epoch {epoch+1}/{args.bc_epochs}: loss={last_loss:.4f}, acc={last_acc:.3f}")
    return {"bc_samples": float(len(actions)), "bc_loss": last_loss, "bc_acc": last_acc}


def parse_args() -> argparse.Namespace:
    """MaskablePPO 비교 실험을 위한 CLI 옵션을 정의한다.

    PPO 자체 update는 sb3-contrib 구현을 사용하고, 여기서는 action mask,
    state 보강, 주기적 평가, conservative update 설정을 연결한다.
    보고서 기준 설정은 config/ours/*.yaml을 우선한다.
    """
    parser = argparse.ArgumentParser(description="MaskablePPO bike rebalancing agent")
    parser.add_argument("--district", default="마포구")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--episode-cache-dir", default="data/episode_cache")
    parser.add_argument("--no-episode-cache", action="store_true")
    parser.add_argument("--n-train-dates", type=int, default=200)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--tag", default="ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=256)
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-update-steps", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    """MaskablePPO 학습 루프와 주기적 7일 평가, best/final 저장을 실행한다."""
    args = parse_args()
    train_episodes = load_episodes(
        TRAIN_DATES[: args.n_train_dates],
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"PPO {args.district} load train" if args.progress else None,
    )
    eval_episodes = load_episodes(
        EVAL_DATES,
        args.district,
        args.processed_dir,
        None if args.no_episode_cache else args.episode_cache_dir,
        f"PPO {args.district} load eval" if args.progress else None,
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

    out_dir = Path("logs") / f"ppo_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    heuristic_mean, heuristic_rewards = evaluate_heuristic(eval_episodes, args.seed)
    print(f"=== MaskablePPO | tag={args.tag} ===")
    print(f"device={args.device}, obs_dim={obs_dim}, n_actions={n_actions}")
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

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl if args.target_kl > 0.0 else None,
        policy_kwargs={"net_arch": dict(pi=[args.hidden, args.hidden], vf=[args.hidden, args.hidden])},
        seed=args.seed,
        verbose=0,
        device=args.device,
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
        print_eval_table("ppo_bc_only", heuristic_rewards, final_rewards)
        return

    steps_done = 0
    progress_bar = None
    if args.progress:
        progress_bar = tqdm(
            total=args.total_timesteps,
            desc=f"PPO {args.district}",
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
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                use_masking=True,
                progress_bar=False,
            )
            steps_done += chunk
            if progress_bar is not None:
                progress_bar.update(chunk)
            if steps_done < next_eval_step and steps_done < args.total_timesteps:
                continue
            eval_reward, _ = evaluate(model, eval_episodes, args, args.seed)
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
    print_eval_table("ppo_best", heuristic_rewards, best_rewards)


if __name__ == "__main__":
    main()
