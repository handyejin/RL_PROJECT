"""DAgger — BC + on-policy state distribution aggregation.

Compounding error 문제 해결용:
  - Naive BC는 expert(휴리스틱) trajectory에만 학습 → BC가 실제로 만들어내는
    state 분포에서 OOD가 발생하면 connot recover.
  - DAgger는 매 iteration마다 **현재 BC가 만든 state**에 expert action을
    레이블링하여 dataset에 누적 → BC가 자신의 mistake 분포에서도 expert를
    모방하도록 학습.

iteration 흐름:
  1. 현재 BC로 train episode rollout. 각 state에서 expert(휴리스틱)의 action 기록.
     env는 BC action으로 진행 (BC가 만드는 state 분포 노출).
  2. 새 (state, expert_action) 쌍을 누적 dataset에 추가.
  3. 누적 dataset으로 BC 재학습 (기존 가중치에서 fine-tune).
  4. eval reward 측정 → best 갱신.

사용:
    python scripts/dagger.py --init-bc logs/bc_bc_ppo_heuristic_v2/bc_ppo_model.zip \
                              --init-dataset logs/bc_bc_ppo_heuristic/bc_dataset.npz \
                              --iterations 5 --tag dagger_v2_seed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.common.baselines import get_policy  # noqa: E402
from src.agents.models.ppo import MaskablePPO  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import TRAIN_DATES, EVAL_DATES, _load_yaml, _get  # noqa: E402


def rollout_with_expert(model: MaskablePPO, dates: list[str], district: str, n_trucks: int,
                        env_kwargs: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """현재 정책으로 trajectory 수집, 각 state에서 expert(휴리스틱) action을 레이블로 기록.

    env는 **현재 정책의 action**으로 진행 (BC가 실제로 만드는 state 분포 노출).
    """
    expert = get_policy("most_imbalanced")
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    for date in dates:
        ep = load_episode("data/processed", district=district,
                          episode_start=f"{date} 00:00")
        env = RebalanceEnv(ep, n_trucks=n_trucks, **env_kwargs)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            current_obs = obs.copy()
            expert_a = int(expert.act(env))  # expert가 본 답 — label로 기록
            mask = env.action_masks()
            policy_a, _ = model.predict(obs, deterministic=True, action_masks=mask)
            policy_a = int(policy_a)
            obs_list.append(current_obs)
            act_list.append(expert_a)
            obs, _, done, trunc, _ = env.step(policy_a)  # ← 정책 action으로 진행
            if trunc:
                break
    return (np.asarray(obs_list, dtype=np.float32),
            np.asarray(act_list, dtype=np.int64))


def evaluate_policy(model: MaskablePPO, dates: list[str], district: str, n_trucks: int,
                    env_kwargs: dict, seed: int) -> tuple[float, list[float]]:
    """7일 eval episodes에서 deterministic action으로 평균 reward 측정 (공정 metric)."""
    rewards = []
    for date in dates:
        ep = load_episode("data/processed", district=district,
                          episode_start=f"{date} 00:00")
        env = RebalanceEnv(ep, n_trucks=n_trucks, **env_kwargs)
        obs, _ = env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            mask = env.action_masks()
            a, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, r, done, trunc, _ = env.step(int(a))
            total += r
            if trunc:
                break
        rewards.append(total)
    return float(np.mean(rewards)), rewards


def fit_bc(model: MaskablePPO, obs: np.ndarray, actions: np.ndarray,
           lr: float, epochs: int, batch_size: int) -> tuple[float, float]:
    """현재 model의 policy를 (obs, actions)로 NLL fine-tune (warm start)."""
    policy = model.policy
    device = policy.device
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(actions).to(device)
    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy.set_training_mode(True)
    final_loss, final_acc = 0.0, 0.0
    for epoch in range(epochs):
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for x, y in loader:
            _, log_prob, _ = policy.evaluate_actions(x, y)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                dist = policy.get_distribution(x)
                pred = dist.distribution.logits.argmax(dim=-1)
                ep_correct += (pred == y).sum().item()
            ep_loss += loss.item() * len(y)
            ep_total += len(y)
        scheduler.step()
        final_loss = ep_loss / ep_total
        final_acc = ep_correct / ep_total
    return final_loss, final_acc


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(PROJECT_ROOT / "config" / "ppo_v4_base.yaml"))
    pre_args, _ = pre.parse_known_args()
    cfg = _load_yaml(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--init-bc", default=None,
                        help="시작 BC 모델 zip 경로. None이면 random init.")
    parser.add_argument("--init-dataset", default=None,
                        help="초기 expert dataset npz. 권장: 기존 BC dataset 재사용.")
    parser.add_argument("--district", default=_get(cfg, "district", default="마포구"))
    parser.add_argument("--n-trucks", type=int, default=_get(cfg, "truck", "n_trucks", default=3))
    parser.add_argument("--n-rollout-dates", type=int, default=30,
                        help="iteration별 BC rollout으로 trajectory 수집할 날짜 수")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--epochs-per-iter", type=int, default=60,
                        help="iteration별 BC retrain epoch")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="dagger")
    args = parser.parse_args()

    env_kwargs_train = dict(
        truck_capacity=_get(cfg, "truck", "capacity", default=20),
        target_fill_ratio=_get(cfg, "truck", "target_fill_ratio", default=0.5),
        urgent_low_ratio=_get(cfg, "env", "urgent_low", default=0.15),
        urgent_high_ratio=_get(cfg, "env", "urgent_high", default=0.85),
        strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
        w_travel_km=_get(cfg, "reward", "travel_km", default=-0.008),
        w_travel_step=_get(cfg, "reward", "travel_step", default=-0.002),
        w_work_per_bike=_get(cfg, "env", "w_work_per_bike", default=0.0),
        w_idle_visit=_get(cfg, "env", "w_idle_visit", default=0.0),
        future_demand_horizon=_get(cfg, "env", "future_demand_horizon", default=0),
        use_action_mask=True,
    )
    # eval은 공정 metric: bonus/shaping OFF
    env_kwargs_eval = dict(env_kwargs_train,
                           urgent_bonus=0.0, explore_bonus_scale=0.0, shaping_scale=0.0)

    out_dir = PROJECT_ROOT / "logs" / f"dagger_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    net_arch = list(_get(cfg, "ppo_v4", "net_arch",
                          default=_get(cfg, "ppo", "net_arch", default=[256, 256])))

    # ── 초기 BC 모델 로드 ──
    print("\n[init] loading initial BC model...")
    if args.init_bc:
        model = MaskablePPO.load(args.init_bc)
        print(f"  loaded init BC: {args.init_bc}")
    else:
        ep0 = load_episode("data/processed", district=args.district,
                           episode_start=f"{TRAIN_DATES[0]} 00:00")
        dummy_env = RebalanceEnv(ep0, n_trucks=args.n_trucks, **env_kwargs_train)
        model = MaskablePPO("MlpPolicy", dummy_env, learning_rate=args.lr,
                            policy_kwargs={"net_arch": net_arch}, verbose=0)
        print("  random init")

    # ── 초기 dataset (expert demonstrations) ──
    if args.init_dataset:
        z = np.load(args.init_dataset)
        obs_acc, act_acc = z["obs"], z["actions"]
        print(f"[init] loaded init dataset: {args.init_dataset} ({len(obs_acc)} pairs)")
    else:
        obs_acc = np.zeros((0, 0), dtype=np.float32)  # 첫 iteration에서 채워짐
        act_acc = np.zeros((0,), dtype=np.int64)
        print("[init] no initial dataset — iteration 1에서 첫 수집")

    # 초기 BC eval (baseline)
    init_reward, init_perday = evaluate_policy(model, EVAL_DATES, args.district, args.n_trucks,
                                                env_kwargs_eval, args.seed)
    print(f"\n[init] init BC eval reward: {init_reward:.2f} "
          f"(휴리스틱=-88.68, Δ={init_reward - (-88.68):+.2f})")
    print(f"  per-day: {[round(r, 1) for r in init_perday]}")

    history: list[dict] = [{
        "iter": 0, "loss": None, "acc": None, "eval_reward": init_reward,
        "per_day": init_perday, "dataset_size": len(obs_acc),
    }]
    best_reward = init_reward
    best_iter = 0
    model.save(out_dir / "best_model")  # init best

    for it in range(args.iterations):
        print(f"\n=== DAgger iteration {it+1}/{args.iterations} ===")
        # 1. BC로 rollout, expert action 기록
        dates = TRAIN_DATES[: args.n_rollout_dates]
        t0 = time.time()
        new_obs, new_act = rollout_with_expert(
            model, dates, args.district, args.n_trucks,
            env_kwargs_train, args.seed + it,
        )
        print(f"  rollout: {len(new_obs)} new (state, expert_action) pairs "
              f"({time.time() - t0:.1f}s)")
        # 2. dataset 누적
        if obs_acc.size == 0:
            obs_acc, act_acc = new_obs, new_act
        else:
            obs_acc = np.concatenate([obs_acc, new_obs])
            act_acc = np.concatenate([act_acc, new_act])
        print(f"  accumulated dataset: {len(obs_acc)} pairs")
        # 3. BC fine-tune (warm start)
        t0 = time.time()
        loss, acc = fit_bc(model, obs_acc, act_acc, args.lr,
                            args.epochs_per_iter, args.batch_size)
        print(f"  fine-tune {args.epochs_per_iter} epochs: "
              f"loss={loss:.4f}, acc={acc:.3f} ({time.time() - t0:.1f}s)")
        # 4. eval
        eval_reward, per_day = evaluate_policy(model, EVAL_DATES, args.district, args.n_trucks,
                                                 env_kwargs_eval, args.seed)
        delta = eval_reward - (-88.68)
        marker = "(best)" if eval_reward > best_reward else ""
        print(f"  eval reward (7일 평균): {eval_reward:.2f} "
              f"(휴리스틱=-88.68, Δ={delta:+.2f}) {marker}")
        print(f"    per-day: {[round(r, 1) for r in per_day]}")
        history.append({
            "iter": it + 1, "loss": loss, "acc": acc, "eval_reward": eval_reward,
            "per_day": per_day, "dataset_size": len(obs_acc),
        })
        if eval_reward > best_reward:
            best_reward = eval_reward
            best_iter = it + 1
            model.save(out_dir / "best_model")
        # iteration별 체크포인트
        model.save(out_dir / f"iter_{it+1}_model")

    print(f"\n=== DAgger 완료 ===")
    print(f"  init BC reward: {init_reward:.2f}")
    print(f"  best iter: {best_iter}, best eval reward: {best_reward:.2f}")
    print(f"  improvement: {best_reward - init_reward:+.2f}")
    print(f"  휴리스틱(-88.68) 대비 Δ = {best_reward - (-88.68):+.2f}")
    np.save(out_dir / "history.npy", history, allow_pickle=True)
    print(f"\n  best model → {out_dir}/best_model.zip")
    print(f"  history    → {out_dir}/history.npy")


if __name__ == "__main__":
    main()
