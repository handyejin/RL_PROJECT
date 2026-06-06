"""다트럭 확장 — 정류소 수 vs 트럭 수에 따른 DQN 추월 경계 이동 확인.

단일트럭(dqn_small)에서 정류소 30+에선 DQN이 예측형에 졌다(고정 학습량). 트럭을 2~3대로
늘리면 한 대가 맡는 부담이 줄어 더 큰 문제도 커버 → 경계가 우측으로 밀리는지 확인.

비동기 다트럭: 매 step 비여행(free) 트럭이 순서대로 결정(재배치+다음 목적지). 트럭마다
이동시간 달라 비동기. 휴리스틱은 다른 트럭 목적지를 제외(exclude)해 중복 방지. DQN은
obs에 다른 트럭 위치(occupancy)를 넣어 협응을 학습.

동역학·리워드·정류소 선택은 SmallProblem / dqn_small.make_problem 재사용 (같은 잣대).
사용: python scripts/dqn_multitruck.py --n-stations 30 --n-trucks 2 --timesteps 400000
"""
from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

from scripts.dqn_small import make_problem, _future_net  # noqa: E402


# ───────────────────────── obs ─────────────────────────
def make_obs_mt(p, t, i, locs, loads, rems, dests, inv, no_forecast=False):
    K = p.K; nt = len(locs)
    loc_oh = np.zeros(K + 1, np.float32); loc_oh[locs[i]] = 1.0
    occ = np.zeros(K + 1, np.float32)                     # 다른 트럭들이 있는/향하는 곳
    for j in range(nt):
        if j == i:
            continue
        occ[dests[j] if rems[j] > 0 else locs[j]] += 1.0
    occ /= max(nt - 1, 1)
    parts = [
        inv / np.maximum(p.caps, 1),
        [loads[i] / max(p.truck_cap, 1)],
        loc_oh, occ,
        [(t - p.w0) / max(p.Twin, 1)],
    ]
    if not no_forecast:
        parts.append(_future_net(p, t) / np.maximum(p.caps, 1))
    return np.concatenate(parts).astype(np.float32)


def obs_dim_mt(K, no_forecast):
    return (3 * K + 3) if no_forecast else (4 * K + 4)


# ───────────────────────── 다트럭 휴리스틱 ─────────────────────────
class DoNothingMT:
    def __init__(self, p): self.p = p
    def act(self, t, i, locs, loads, rems, dests, inv, exclude):
        loc = locs[i]
        return (int(inv[loc]) if loc < self.p.K else 0), self.p.depot


class STRMT:
    def __init__(self, p): self.p = p
    def act(self, t, i, locs, loads, rems, dests, inv, exclude):
        p = self.p; loc = locs[i]
        if loc < p.K:
            dev = inv[loc] - p.target[loc]
            tv = int(p.target[loc] + p.band[loc]) if dev > p.band[loc] else \
                 int(p.target[loc] - p.band[loc]) if dev < -p.band[loc] else int(inv[loc])
        else:
            tv = int(p.target[0])
        oob = [n for n in range(p.K) if abs(inv[n] - p.target[n]) > p.band[n] and n not in exclude]
        dest = min(oob, key=lambda n: p.travel[loc, n]) if oob else (p.depot if loads[i] > 0 else loc)
        return tv, dest


class SLAMT:
    def __init__(self, p, horizon=6): self.p = p; self.h = horizon
    def act(self, t, i, locs, loads, rems, dests, inv, exclude):
        p = self.p; loc = locs[i]
        tv = int(p.target[loc]) if loc < p.K else int(p.target[0])
        k0 = t - p.w0; k1 = min(k0 + self.h, p.Twin)
        fut = (p.ge[k0:k1].sum(0) - p.gr[k0:k1].sum(0)) if k1 > k0 else np.zeros(p.K)
        score = np.abs(inv + fut - p.target)
        for n in exclude:
            if n < p.K:
                score[n] = -1
        best = int(np.argmax(score))
        return tv, (best if score[best] > 0 else p.depot)


class DQNPolicyMT:
    def __init__(self, model, p, no_forecast=False): self.model = model; self.p = p; self.no_forecast = no_forecast
    def act(self, t, i, locs, loads, rems, dests, inv, exclude):
        obs = make_obs_mt(self.p, t, i, locs, loads, rems, dests, inv, self.no_forecast)
        a, _ = self.model.predict(obs, deterministic=True)
        dest = int(a) // 3; lvl = int(a) % 3
        return self.p.level_to_target(locs[i], lvl), dest


# ───────────────────────── 다트럭 롤아웃(평가) ─────────────────────────
def _rollout_mt_impl(p, policy, dr, dg, nt):
    locs = [p.depot] * nt; loads = [0] * nt; rems = [0] * nt; dests = [p.depot] * nt
    inv = p.init_bikes.copy()
    reward = 0.0; unmet = 0
    for t in range(p.w0, p.w1):
        for i in range(nt):
            if rems[i] == 0:
                exclude = {locs[i]}
                for j in range(nt):
                    if j != i:
                        exclude.add(dests[j] if rems[j] > 0 else locs[j])
                tv, d = policy.act(t, i, locs, loads, rems, dests, inv, exclude)
                inv, loads[i], _ = p.rebalance_to(inv, loads[i], locs[i], tv)
                if d != locs[i]:
                    reward += p.w_travel_km * float(p.dist_km[locs[i], d])
                    rems[i] = max(int(p.travel[locs[i], d]), 1); dests[i] = d
                else:
                    dests[i] = locs[i]
        inv, so, fu = p._apply_step(inv, dr[t - p.w0], dg[t - p.w0])
        unmet += so + fu; reward += p.w_stockout * so + p.w_full * fu
        for i in range(nt):
            if rems[i] > 0:
                rems[i] -= 1; reward += p.w_travel_step
                if rems[i] == 0:
                    locs[i] = dests[i]
    return reward, unmet


def eval_stochastic_mt(p, policy, nt, n_sims, seed=123):
    rng = np.random.default_rng(seed); rs = []; us = []
    for _ in range(n_sims):
        dr = np.stack([rng.poisson(p.gr[t]) for t in range(p.Twin)]).astype(np.int64)
        dg = np.stack([rng.poisson(p.ge[t]) for t in range(p.Twin)]).astype(np.int64)
        r, u = _rollout_mt_impl(p, policy, dr, dg, nt); rs.append(r); us.append(u)
    return float(np.mean(rs)), float(np.std(rs)), float(np.mean(us))


def eval_actual_mt(p, policy, actual, nt):
    rs = []; us = []
    for dr, dg in actual:
        r, u = _rollout_mt_impl(p, policy, dr, dg, nt); rs.append(r); us.append(u)
    return float(np.mean(rs)), float(np.std(rs)), float(np.mean(us))


# ───────────────────────── 다트럭 Gym 환경 (DQN 학습) ─────────────────────────
class MultiTruckEnv(gym.Env):
    def __init__(self, p, nt, gamma=0.99, seed=0, no_forecast=False):
        super().__init__()
        self.p = p; self.nt = nt; self.gamma = gamma; self.no_forecast = no_forecast
        self.rng = np.random.default_rng(seed)
        K = p.K
        self.action_space = spaces.Discrete((K + 1) * 3)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(obs_dim_mt(K, no_forecast),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p = self.p
        self.t = p.w0
        self.locs = [p.depot] * self.nt; self.loads = [0] * self.nt
        self.rems = [0] * self.nt; self.dests = [p.depot] * self.nt
        self.inv = p.init_bikes.copy()
        self.queue = list(range(self.nt)); self.qi = 0          # 이번 step에 결정할 free 트럭들
        return self._obs(self.queue[0]), {}

    def _obs(self, i):
        return make_obs_mt(self.p, self.t, i, self.locs, self.loads, self.rems,
                           self.dests, self.inv, self.no_forecast)

    def _demand_step(self):
        p = self.p; t = min(self.t - p.w0, p.Twin - 1)
        rent = self.rng.poisson(p.gr[t]).astype(np.int64)
        ret = self.rng.poisson(p.ge[t]).astype(np.int64)
        self.inv, so, fu = p._apply_step(self.inv, rent, ret)
        return p.w_stockout * so + p.w_full * fu

    def step(self, a):
        p = self.p; i = self.queue[self.qi]
        dest = int(a) // 3; lvl = int(a) % 3
        tv = p.level_to_target(self.locs[i], lvl)
        self.inv, self.loads[i], _ = p.rebalance_to(self.inv, self.loads[i], self.locs[i], tv)
        rew = 0.0
        if dest != self.locs[i]:
            rew += p.w_travel_km * float(p.dist_km[self.locs[i], dest])
            self.rems[i] = max(int(p.travel[self.locs[i], dest]), 1); self.dests[i] = dest
        else:
            self.dests[i] = self.locs[i]
        self.qi += 1
        if self.qi < len(self.queue):                           # 같은 시각 다른 free 트럭 결정
            return self._obs(self.queue[self.qi]), float(rew), False, False, {}
        # 모든 free 트럭 결정 완료 → 다음 free 트럭 생길 때까지 시간 진행
        while self.t < p.w1:
            rew += self._demand_step()
            for j in range(self.nt):
                if self.rems[j] > 0:
                    self.rems[j] -= 1; rew += p.w_travel_step
                    if self.rems[j] == 0:
                        self.locs[j] = self.dests[j]
            self.t += 1
            free = [j for j in range(self.nt) if self.rems[j] == 0]
            if free and self.t < p.w1:
                self.queue = free; self.qi = 0
                return self._obs(free[0]), float(rew), False, False, {}
        return np.zeros(self.observation_space.shape, np.float32), float(rew), True, False, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stations", type=int, default=30)
    ap.add_argument("--n-trucks", type=int, default=2)
    ap.add_argument("--truck-cap", type=int, default=30)
    ap.add_argument("--target-ratio", type=float, default=0.5)
    ap.add_argument("--timesteps", type=int, default=400000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-sims", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--tag", default="mt")
    args = ap.parse_args()

    print(f"[1/4] 문제 구성 (N={args.n_stations}정류소·트럭{args.n_trucks}대·전일)...")
    pa = Namespace(n_stations=args.n_stations, truck_cap=args.truck_cap, target_ratio=args.target_ratio)
    prob, actual, sel = make_problem(pa)
    nt = args.n_trucks
    print(f"  K={prob.K}+depot, {prob.Twin} step, 트럭 {nt}대(cap{args.truck_cap})")

    print(f"[2/4] DQN 학습 ({args.timesteps:,} steps)...")
    env = MultiTruckEnv(prob, nt, gamma=args.gamma, seed=args.seed, no_forecast=args.no_forecast)
    model = DQN("MlpPolicy", env, learning_rate=args.lr, gamma=args.gamma,
                buffer_size=200000, learning_starts=5000, batch_size=128,
                train_freq=4, target_update_interval=2000,
                exploration_fraction=0.3, exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [256, 256]}, verbose=0, seed=args.seed)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    print(f"  완료 ({(time.time()-t0)/60:.1f}분)")
    out = PROJECT_ROOT / "logs" / f"dqn_mt_{args.tag}"; out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model")

    print(f"[3/4] 정책 비교 (리워드↑ 좋음, 원본 정의)...\n")
    policies = {
        "do-nothing": DoNothingMT(prob),
        "STR (반응형)": STRMT(prob),
        "SLA (예측형)": SLAMT(prob),
        "DQN (학습)": DQNPolicyMT(model, prob, no_forecast=args.no_forecast),
    }
    print(f"  {'정책':<20}{'리워드(확률30)':>16}{'미충족':>10}{'리워드(실제7)':>16}")
    print("  " + "-" * 64)
    res = {}
    for name, pol in policies.items():
        rs, ss, us = eval_stochastic_mt(prob, pol, nt, args.n_sims)
        ra, sa, ua = eval_actual_mt(prob, pol, actual, nt)
        res[name] = (rs, us, ra)
        print(f"  {name:<20}{rs:>11.2f}±{ss:<4.0f}{us:>9.1f}{ra:>13.2f}±{sa:<4.0f}")

    print(f"\n[4/4] 요약 (N={args.n_stations}, 트럭{nt})")
    sla = res["SLA (예측형)"]; dqn = res["DQN (학습)"]; str_ = res["STR (반응형)"]
    print(f"  SLA : {sla[0]:.2f}  /  STR : {str_[0]:.2f}  /  DQN : {dqn[0]:.2f}")
    print(f"  → DQN vs SLA {dqn[0]-sla[0]:+.2f}  {'✅ 예측형 추월' if dqn[0] > sla[0] else '❌ 미달'}")


if __name__ == "__main__":
    main()
