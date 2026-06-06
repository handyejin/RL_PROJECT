"""score-regression warm-start — 예측형의 정류소별 점수를 회귀로 모방.

raw 146 정류소 공간에서 예측형을 신경망에 담는다. 분류(argmax) clone은 28% 정확도로
실패(-502)했으므로, 대신 예측형의 *정류소별 점수 벡터*(load별 predicted-imbalance)를
MSE 회귀로 학습 → argmax(Q)=예측형 선택 → prior ≈ 예측형(-459/-472).

obs엔 forecast(과거평균 미래수요) 포함 → RL/정책이 예측 정보를 입력으로 받음.

사용:
  python scripts/warmstart_scoreregress.py --tag ws_pred --n-dates 292 --horizon 3 --epochs 60
  → logs/ws_<tag>/bc_model.zip (DQfD --pretrain init로 사용 가능), 7일 eval 출력
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.masked_dqn import MaskableDQN  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402

SCORE_SCALE = 20.0  # 회귀 타깃 정규화 (capacity 규모) — argmax 불변, 학습 안정


def predictive_scores(env, gr, ge, H):
    """forecast 예측형의 정류소별 점수 (load별, raw). argmax=예측형 선택."""
    truck = env.trucks[env.current_truck]
    target = env.data.capacity.astype(np.float32) * env.target_fill_ratio
    t = env.t
    t_end = min(t + H, env.data.rentals.shape[0])
    fr = gr[t:t_end].sum(axis=0)
    fe = ge[t:t_end].sum(axis=0)
    predicted = env.bikes.astype(np.float32) + (fe - fr)
    if truck.load == 0:
        s = predicted - target
    elif truck.load >= env.truck_capacity:
        s = target - predicted
    else:
        s = np.abs(predicted - target)
    return s.astype(np.float32)


def main() -> None:
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ws_pred")
    ap.add_argument("--n-dates", type=int, default=292)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    district = _get(cfg, "district", default="마포구")
    n_trucks = _get(cfg, "truck", "n_trucks", default=3)
    net_arch = list(_get(cfg, "dqn", "net_arch", default=[256, 256]))
    H = args.horizon

    print(f"[1/4] train {args.n_dates}일 로드 + forecast 프로파일...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00")
          for d in TRAIN_DATES[: args.n_dates]]
    gr = np.stack([e.rentals for e in tr]).mean(0).astype(np.float32)
    ge = np.stack([e.returns for e in tr]).mean(0).astype(np.float32)

    ek = dict(truck_capacity=20, target_fill_ratio=0.5,
              urgent_low_ratio=_get(cfg, "env", "urgent_low", default=0.15),
              urgent_high_ratio=_get(cfg, "env", "urgent_high", default=0.85),
              strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
              w_travel_km=-0.008, w_travel_step=-0.002,
              future_demand_horizon=H, forecast_rent=gr, forecast_ret=ge,
              use_action_mask=True)

    print(f"[2/4] 예측형 (obs,score) 수집...")
    t0 = time.time()
    O, S = [], []
    for ep in tr:
        env = RebalanceEnv(ep, n_trucks=n_trucks, **ek)
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            sc = predictive_scores(env, gr, ge, H)
            mask = env.action_masks()
            masked = sc.copy(); masked[~mask] = -np.inf
            a = int(np.argmax(masked))
            O.append(np.asarray(obs, dtype=np.float32))
            S.append(sc / SCORE_SCALE)
            obs, _, done, trunc, _ = env.step(a)
            if trunc:
                break
    O = np.asarray(O, dtype=np.float32); S = np.asarray(S, dtype=np.float32)
    print(f"  {len(O):,} (obs, score146) 쌍 ({time.time()-t0:.1f}s)")

    print(f"[3/4] q_net MSE 회귀 학습 ({args.epochs} epochs)...")
    dummy = RebalanceEnv(tr[0], n_trucks=n_trucks, **ek)
    model = MaskableDQN("MlpPolicy", dummy, learning_rate=args.lr,
                        policy_kwargs={"net_arch": net_arch}, verbose=0)
    dev = model.policy.device
    q = model.policy.q_net
    opt = th.optim.Adam(q.parameters(), lr=args.lr)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(th.from_numpy(O).to(dev), th.from_numpy(S).to(dev)),
                        batch_size=args.batch_size, shuffle=True)
    for ep_i in range(args.epochs):
        tot = 0.0; n = 0
        q.train()
        for x, y in loader:
            pred = q(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        if (ep_i + 1) % 10 == 0 or ep_i == 0:
            print(f"  epoch {ep_i+1}/{args.epochs}: MSE={tot/n:.5f}")

    out_dir = PROJECT_ROOT / "logs" / f"ws_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "bc_model")
    print(f"  model → {out_dir/'bc_model.zip'}")

    print(f"[4/4] eval (7일, masked argmax)...")
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]
    model.policy.set_training_mode(False)
    rs = []
    for ep in ev:
        env = RebalanceEnv(ep, n_trucks=n_trucks, **ek)
        obs, _ = env.reset(seed=42)
        done = False; tot = 0.0
        while not done:
            mask = env.action_masks()
            a, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, r, done, _, _ = env.step(int(a))
            tot += r
        rs.append(tot)
    mean_r = float(np.mean(rs))
    print(f"\n  score-regression warm-start 7일 평균 = {mean_r:.2f}  (Δ휴리스틱 -500.02 = {mean_r+500.02:+.1f})")
    print(f"    (참고: forecast 예측형 직접 = ~-459(292일)/-472(60일), 분류 clone = -502)")


if __name__ == "__main__":
    main()
