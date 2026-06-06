"""MaskablePPO — forecast obs + 예측형 distillation warm-start + PPO 개선.

DQN+앵커는 "베끼거나 무너지거나"라 개선을 못 했다. PPO는 on-policy + trust region(clip)
이라 앵커 없이도 안정적으로 정책을 *조금씩 개선* → 예측형 규칙의 myopic함(트럭 독립·
이동시간 무시·시퀀싱 없음)을 넘을 여지를 탐색한다.

흐름:
  1. forecast obs 환경 (RL이 미래 예측을 입력으로 받음)
  2. warm-start: 예측형 점수로 정책을 distillation (soft CE) → -459 근처에서 출발
  3. PPO.learn() → 보상으로 정책 직접 개선 (clip이 붕괴 방지)

사용: python scripts/train_ppo.py --tag ppo1 --n-dates 292 --timesteps 500000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO  # noqa: E402
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # noqa: E402
from sb3_contrib.common.maskable.utils import get_action_masks  # noqa: E402
from sb3_contrib.common.wrappers import ActionMasker  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get, evaluate_heuristic  # noqa: E402


def _mask_fn(env):
    return env.action_masks()


def predictive_scores(env, gr, ge, H):
    truck = env.trucks[env.current_truck]
    target = env.data.capacity.astype(np.float32) * env.target_fill_ratio
    t = env.t; t_end = min(t + H, env.data.rentals.shape[0])
    fr = gr[t:t_end].sum(0); fe = ge[t:t_end].sum(0)
    predicted = env.bikes.astype(np.float32) + (fe - fr)
    if truck.load == 0:
        s = predicted - target
    elif truck.load >= env.truck_capacity:
        s = target - predicted
    else:
        s = np.abs(predicted - target)
    return s.astype(np.float32)


def eval_policy(model, eval_eps, ek, n_trucks):
    rs = []
    for ep in eval_eps:
        env = ActionMasker(RebalanceEnv(ep, n_trucks=n_trucks, **ek), _mask_fn)
        obs, _ = env.reset(seed=42)
        done, tot = False, 0.0
        while not done:
            m = get_action_masks(env)
            a, _ = model.predict(obs, deterministic=True, action_masks=m)
            obs, r, done, trunc, _ = env.step(int(a))
            tot += r
            if trunc:
                break
        rs.append(tot)
    return float(np.mean(rs))


class EvalCB(BaseCallback):
    def __init__(self, eval_eps, ek, n_trucks, heur, freq):
        super().__init__()
        self.eval_eps, self.ek, self.n_trucks, self.heur, self.freq = eval_eps, ek, n_trucks, heur, freq
        self.last = 0
        self.history = []

    def _on_step(self):
        if self.num_timesteps - self.last >= self.freq:
            self.last = self.num_timesteps
            r = eval_policy(self.model, self.eval_eps, self.ek, self.n_trucks)
            self.history.append((self.num_timesteps, r))
            mark = " ✅" if r > self.heur else ""
            print(f"[eval] step={self.num_timesteps:>8}  reward={r:.2f}  (휴 {self.heur:.2f}, Δ={r-self.heur:+.2f}){mark}", flush=True)
        return True


def warmstart_distill(model, train_eps, gr, ge, H, ek, n_trucks, steps, lr, temp):
    """예측형 점수로 정책 distillation: target=softmax(scores/T), soft CE."""
    print(f"  [warm-start] 예측형 distillation: {len(train_eps)}일 수집...")
    O, S = [], []
    for ep in train_eps:
        env = ActionMasker(RebalanceEnv(ep, n_trucks=n_trucks, **ek), _mask_fn)
        obs, _ = env.reset(seed=42)
        done = False
        base = env.env
        while not done:
            O.append(np.asarray(obs, np.float32))
            S.append(predictive_scores(base, gr, ge, H))
            m = env.action_masks()
            sc = S[-1].copy(); sc[~m] = -np.inf
            obs, _, done, trunc, _ = env.step(int(np.argmax(sc)))
            if trunc:
                break
    O = th.tensor(np.asarray(O), dtype=th.float32, device=model.device)
    S = th.tensor(np.asarray(S), dtype=th.float32, device=model.device)
    target = th.softmax(S / temp, dim=1)
    pol = model.policy
    opt = th.optim.Adam(pol.parameters(), lr=lr)
    n = len(O); bs = 256
    print(f"  [warm-start] distill {steps} steps on {n:,} samples (T={temp})...")
    for step in range(steps):
        idx = th.randint(0, n, (bs,), device=model.device)
        feat = pol.extract_features(O[idx])
        latent_pi, _ = pol.mlp_extractor(feat)
        logits = pol.action_net(latent_pi)
        loss = -(target[idx] * th.log_softmax(logits, dim=1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"    distill {step+1}/{steps}: soft-CE={loss.item():.4f}", flush=True)


def main():
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ppo1")
    ap.add_argument("--n-dates", type=int, default=292)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--timesteps", type=int, default=500000)
    ap.add_argument("--eval-freq", type=int, default=20000)
    ap.add_argument("--warmstart-steps", type=int, default=3000)
    ap.add_argument("--warmstart-lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--no-warmstart", action="store_true")
    args = ap.parse_args()

    district = _get(cfg, "district", default="마포구")
    n_trucks = _get(cfg, "truck", "n_trucks", default=3)
    net_arch = list(_get(cfg, "dqn", "net_arch", default=[256, 256]))
    H = args.horizon

    print(f"[1/4] {args.n_dates}일 로드 + forecast 프로파일...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00")
          for d in TRAIN_DATES[: args.n_dates]]
    gr = np.stack([e.rentals for e in tr]).mean(0).astype(np.float32)
    ge = np.stack([e.returns for e in tr]).mean(0).astype(np.float32)
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]

    ek = dict(truck_capacity=20, target_fill_ratio=0.5,
              urgent_low_ratio=_get(cfg, "env", "urgent_low", default=0.15),
              urgent_high_ratio=_get(cfg, "env", "urgent_high", default=0.85),
              strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
              w_travel_km=-0.008, w_travel_step=-0.002,
              future_demand_horizon=H, forecast_rent=gr, forecast_ret=ge, use_action_mask=True)

    heur = evaluate_heuristic(ev, n_trucks=n_trucks, truck_capacity=20, target_fill_ratio=0.5,
                              urgent_low_ratio=ek["urgent_low_ratio"], urgent_high_ratio=ek["urgent_high_ratio"],
                              w_travel_km=-0.008, w_travel_step=-0.002, future_demand_horizon=H)
    print(f"  휴리스틱(반응형) = {heur:.2f}")

    print(f"[2/4] MaskablePPO 생성...")
    train_env = ActionMasker(RebalanceEnv(tr, n_trucks=n_trucks, **ek), _mask_fn)
    model = MaskablePPO(MaskableActorCriticPolicy, train_env, learning_rate=args.lr,
                        n_steps=2048, batch_size=256, gamma=0.99, ent_coef=args.ent_coef,
                        policy_kwargs={"net_arch": net_arch}, verbose=0, seed=42)

    if not args.no_warmstart:
        warmstart_distill(model, tr, gr, ge, H, ek, n_trucks,
                          args.warmstart_steps, args.warmstart_lr, args.temp)
        r0 = eval_policy(model, ev, ek, n_trucks)
        print(f"  [warm-start] 직후 eval = {r0:.2f}  (Δ휴 {r0-heur:+.2f})")

    print(f"[3/4] PPO 학습 {args.timesteps:,} steps...")
    cb = EvalCB(ev, ek, n_trucks, heur, args.eval_freq)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    print(f"  완료 ({(time.time()-t0)/60:.1f}분)")

    out = PROJECT_ROOT / "logs" / f"ppo_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "ppo_model")
    np.save(out / "history.npy", cb.history, allow_pickle=True)
    print(f"[4/4] saved → {out}")
    if cb.history:
        best = max(cb.history, key=lambda x: x[1])
        print(f"\n=== 결과 === 휴리스틱 {heur:.2f} | best {best[1]:.2f} (step {best[0]:,}) "
              f"| {'✅ 추월' if best[1] > heur else '❌'}")


if __name__ == "__main__":
    main()
