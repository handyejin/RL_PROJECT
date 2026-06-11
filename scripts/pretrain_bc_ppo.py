"""PPO-policy Behavior Cloning — 휴리스틱(most_imbalanced) 행동을 PPO 정책 형태로 모방.

PPO_V4(KL-to-BC)와 연동: 결과 zip은 PPO.load()로 불러올 수 있고
policy.evaluate_actions(obs, action)이 동작 → KL penalty 계산 가능.

흐름:
  1. 휴리스틱을 train episode에서 시뮬레이션 → (obs, action) 쌍 수집
     (기존 pretrain_bc.py와 동일 — bc_dataset.npz 재사용 가능)
  2. MaskablePPO 정책의 actor를 NLL(=cross-entropy) loss로 학습
  3. SB3 PPO zip으로 저장 → train.py --pretrain 으로 PPO_V4와 함께 사용

사용:
    python scripts/pretrain_bc_ppo.py --tag bc_ppo_pretrain
    # 기존 dataset 재사용:
    python scripts/pretrain_bc_ppo.py --dataset logs/bc_bc_pretrain/bc_dataset.npz --tag bc_ppo_reuse
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

from src.agents.models.ppo import MaskablePPO  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import TRAIN_DATES, _load_yaml, _get  # noqa: E402
from scripts.pretrain_bc import collect_data  # noqa: E402


def train_bc_ppo(obs: np.ndarray, actions: np.ndarray, sample_weights: np.ndarray | None,
                 args, env_kwargs: dict, net_arch: list) -> MaskablePPO:
    """MaskablePPO 정책의 actor를 NLL loss로 학습.

    PPO 정책은 Categorical 분포를 출력하므로 evaluate_actions(obs, target_action)의
    log_prob에 -1을 곱하면 cross-entropy loss와 동치. critic은 backward 신호가
    없어 자연스럽게 안 움직임.
    """
    print(f"\n[2/3] training BC (PPO actor) for {args.epochs} epochs...")
    ep0 = load_episode("data/processed", district=args.district,
                       episode_start=f"{TRAIN_DATES[0]} 00:00")
    dummy_env = RebalanceEnv(ep0, n_trucks=args.n_trucks, **env_kwargs)
    model = MaskablePPO(
        "MlpPolicy", dummy_env,
        learning_rate=args.lr,
        policy_kwargs={"net_arch": net_arch},
        verbose=0,
    )
    policy = model.policy
    device = policy.device

    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(actions).to(device)
    dataset = TensorDataset(obs_t, act_t)
    if sample_weights is not None:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset), replacement=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    if args.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    else:
        scheduler = None

    best_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        total_loss, correct, total = 0.0, 0, 0
        policy.set_training_mode(True)
        for x, y in loader:
            _, log_prob, _ = policy.evaluate_actions(x, y)
            loss = -log_prob.mean()  # NLL = cross-entropy on actions
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # acc: argmax of policy distribution == target
            with torch.no_grad():
                dist = policy.get_distribution(x)
                pred = dist.distribution.logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
            total_loss += loss.item() * len(y)
            total += len(y)
        if scheduler:
            scheduler.step()
        acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1}/{args.epochs}: loss={total_loss/total:.4f}, acc={acc:.3f}, lr={cur_lr:.2e}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"  best acc {best_acc:.3f} restored")

    return model


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    pre_args, _ = pre.parse_known_args()
    cfg = _load_yaml(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--district", default=_get(cfg, "district", default="마포구"))
    parser.add_argument("--n-trucks", type=int, default=_get(cfg, "truck", "n_trucks", default=3))
    parser.add_argument("--n-dates", type=int, default=_get(cfg, "training", "n_train_dates", default=60),
                        help="휴리스틱 데이터 수집할 날짜 수")
    parser.add_argument("--seed", type=int, default=_get(cfg, "training", "seed", default=42))
    parser.add_argument("--epochs", type=int, default=20, help="BC supervised epoch 수")
    parser.add_argument("--lr", type=float, default=3e-4, help="BC 학습률 (PPO actor는 DQN보다 작은 lr이 안정적)")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--peak-start", type=int, default=0)
    parser.add_argument("--peak-end", type=int, default=0)
    parser.add_argument("--peak-weight", type=float, default=1.0)
    parser.add_argument("--peak2-start", type=int, default=0)
    parser.add_argument("--peak2-end", type=int, default=0)
    parser.add_argument("--peak2-weight", type=float, default=1.0)
    parser.add_argument("--tag", default="bc_ppo_pretrain")
    parser.add_argument("--dataset", default=None,
                        help="이미 수집된 bc_dataset.npz 경로 (재수집 생략)")
    args = parser.parse_args()

    env_kwargs = dict(
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
    # PPO_V4의 default net_arch와 일치시킴
    net_arch = list(_get(cfg, "ppo_v4", "net_arch",
                          default=_get(cfg, "ppo", "net_arch", default=[256, 256])))

    out_dir = PROJECT_ROOT / "logs" / f"bc_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        print(f"\n[1/3] reusing dataset: {args.dataset}")
        z = np.load(args.dataset)
        obs, actions, times = z["obs"], z["actions"], z["times"]
        print(f"  loaded {len(obs)} (obs, action) pairs")
    else:
        obs, actions, times = collect_data(args, env_kwargs)
        np.savez(out_dir / "bc_dataset.npz", obs=obs, actions=actions, times=times)
        print(f"  saved dataset → {out_dir/'bc_dataset.npz'}")

    sample_weights = None
    windows = []
    if args.peak_weight != 1.0 and args.peak_end > args.peak_start:
        windows.append(("오전", args.peak_start, args.peak_end, args.peak_weight))
    if args.peak2_weight != 1.0 and args.peak2_end > args.peak2_start:
        windows.append(("저녁", args.peak2_start, args.peak2_end, args.peak2_weight))
    if windows:
        sample_weights = np.ones(len(times), dtype=np.float64)
        for label, s, e, w in windows:
            in_win = (times >= s) & (times <= e)
            sample_weights[in_win] = np.maximum(sample_weights[in_win], w)
            print(f"  {label} 피크 가중: env.t∈[{s},{e}] "
                  f"{int(in_win.sum())}/{len(times)} 쌍 ({in_win.mean()*100:.1f}%)을 ×{w}")

    model = train_bc_ppo(obs, actions, sample_weights, args, env_kwargs, net_arch)

    print(f"\n[3/3] saving SB3 PPO model...")
    model_path = out_dir / "bc_ppo_model"
    model.save(model_path)
    print(f"  model → {model_path}.zip")
    print(f"\n사용법: python scripts/train.py --algo ppo_v4 --pretrain {model_path}.zip --tag ppo_v4_bcppo")


if __name__ == "__main__":
    main()
