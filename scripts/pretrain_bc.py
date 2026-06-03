"""Behavior Cloning pretrain — 휴리스틱(most_imbalanced) 행동을 모방.

흐름:
  1. 휴리스틱을 60일 train episode에서 시뮬레이션 → (obs, action) 쌍 수집
  2. SB3 DQN의 q_net을 cross-entropy로 학습 (분류 문제)
  3. SB3 zip으로 저장 → train.py에서 --pretrain으로 로드

사용:
    python scripts/pretrain_bc.py --tag bc_pretrain
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baselines import get_policy  # noqa: E402
from src.agents.masked_dqn import MaskableDQN  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import TRAIN_DATES, _load_yaml, _get  # noqa: E402


def collect_data(args, env_kwargs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """휴리스틱을 train episode에서 시뮬레이션 → (obs, action, t) 쌍 수집.

    t는 결정 시점의 환경 시계(env.t, 0~143). 오전 피크 가중 샘플링에 사용.
    """
    policy = get_policy("most_imbalanced")
    dates = TRAIN_DATES[: args.n_dates]
    print(f"\n[1/3] collecting heuristic actions on {len(dates)} dates...")

    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    all_t: list[int] = []
    t0 = time.time()
    for i, date in enumerate(dates):
        ep = load_episode("data/processed", district=args.district, episode_start=f"{date} 00:00")
        env = RebalanceEnv(ep, n_trucks=args.n_trucks, **env_kwargs)
        obs, _ = env.reset(seed=args.seed)
        done = False
        while not done:
            current = obs.copy()
            action = policy.act(env)
            all_obs.append(current)
            all_actions.append(int(action))
            all_t.append(int(env.t))
            obs, _, done, _, _ = env.step(action)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(dates)} dates, pairs={len(all_obs)}")
    print(f"  total: {len(all_obs)} (obs, action) pairs ({time.time()-t0:.1f}s)")
    return (np.asarray(all_obs, dtype=np.float32),
            np.asarray(all_actions, dtype=np.int64),
            np.asarray(all_t, dtype=np.int64))


def train_bc(obs: np.ndarray, actions: np.ndarray, sample_weights: np.ndarray | None,
             args, env_kwargs: dict, net_arch: list) -> MaskableDQN:
    """SB3 MaskableDQN을 만들고, q_net을 cross-entropy로 학습.

    sample_weights가 주어지면 WeightedRandomSampler로 가중 샘플링(오전 피크 oversample).
    """
    print(f"\n[2/3] training BC for {args.epochs} epochs...")
    # SB3 DQN을 만들기 위한 dummy env (1 episode면 충분)
    ep0 = load_episode("data/processed", district=args.district,
                       episode_start=f"{TRAIN_DATES[0]} 00:00")
    dummy_env = RebalanceEnv(ep0, n_trucks=args.n_trucks, **env_kwargs)
    model = MaskableDQN(
        "MlpPolicy", dummy_env,
        learning_rate=1e-4,
        policy_kwargs={"net_arch": net_arch},
        verbose=0,
    )

    device = model.policy.device
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

    q_net = model.policy.q_net
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    if args.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    else:
        scheduler = None
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        total_loss, correct, total = 0.0, 0, 0
        q_net.train()
        for x, y in loader:
            logits = q_net(x)         # (batch, 146) — SB3에선 Q-values지만 BC에선 logits로 해석
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        if scheduler:
            scheduler.step()
        acc = correct / total
        # 매 20 epoch마다 출력 + best acc 저장
        if (epoch + 1) % 20 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1}/{args.epochs}: loss={total_loss/total:.4f}, acc={acc:.3f}, lr={cur_lr:.2e}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in q_net.state_dict().items()}

    # 최종 모델은 best acc 시점 가중치
    if best_state is not None:
        q_net.load_state_dict(best_state)
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
    parser.add_argument("--lr", type=float, default=1e-3, help="BC 학습률")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                        help="lr 스케줄: cosine은 학습 끝 10%까지 감소")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--peak-start", type=int, default=0,
                        help="오전 피크 가중 시작 env.t (포함). 0이면 가중 비활성")
    parser.add_argument("--peak-end", type=int, default=0,
                        help="오전 피크 가중 끝 env.t (포함). 0이면 가중 비활성")
    parser.add_argument("--peak-weight", type=float, default=1.0,
                        help="피크 구간 (obs,action) 쌍 oversample 배수 (1.0=비활성)")
    parser.add_argument("--peak2-start", type=int, default=0,
                        help="2차(저녁) 피크 가중 시작 env.t (포함). 0이면 비활성")
    parser.add_argument("--peak2-end", type=int, default=0,
                        help="2차(저녁) 피크 가중 끝 env.t (포함). 0이면 비활성")
    parser.add_argument("--peak2-weight", type=float, default=1.0,
                        help="2차(저녁) 피크 oversample 배수 (1.0=비활성)")
    parser.add_argument("--tag", default="bc_pretrain")
    parser.add_argument("--dataset", default=None,
                        help="이미 수집된 bc_dataset.npz 경로 (재수집 생략, obs/actions/times 로드)")
    args = parser.parse_args()

    # 학습 환경과 동일한 RebalanceEnv 설정
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
        # bonus·shaping은 휴리스틱 결정에 영향 없음 (정책은 reward 안 봄)
        use_action_mask=True,
    )
    net_arch = list(_get(cfg, "dqn", "net_arch", default=[256, 256]))

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

    # 피크 가중 샘플링: 각 윈도 [start,end]의 쌍을 해당 weight배 oversample.
    # 오전(peak)·저녁(peak2) 두 윈도 지원. 겹치지 않으면 단순 할당, 겹치면 큰 weight 우선.
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

    model = train_bc(obs, actions, sample_weights, args, env_kwargs, net_arch)

    print(f"\n[3/3] saving SB3 model...")
    model_path = out_dir / "bc_model"
    model.save(model_path)
    print(f"  model → {model_path}.zip")
    print(f"\n사용법: python scripts/train.py --pretrain {model_path}.zip --tag bc_finetune")


if __name__ == "__main__":
    main()
