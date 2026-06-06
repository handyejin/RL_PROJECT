"""소규모 DQN — 10정류소·1트럭·전일(출퇴근 피크 포함)에서 휴리스틱 추월 시도.

추월 레시피(권장 조합):
  ① forecast를 state에 (미래 순수요) — 예측형과 동등한 정보
  ② 행동에 재배치량(레벨) 포함 — 목적지 + {비움/중간/채움}
  ③ potential-based reward shaping (Ng 1999, 정책불변) — Φ=−예측불균형 → 조밀 신용할당
  (선택) 예측형 warm-start(BC)로 floor=SLA 확보 후 fine-tune

환경은 rtdp_small.SmallProblem(동역학·Poisson 확률수요·비용) 재사용 → SLA/STR과 동일 잣대.
평가도 rtdp_small.rollout/eval_* 재사용(공정 비교).

사용: python scripts/dqn_small.py --timesteps 400000 --shaping 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

from src.envs.data_loader import load_episode  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402
from scripts.rtdp_small import (  # noqa: E402
    SmallProblem, STR, SLA, DoNothing, rollout, eval_stochastic, eval_actual,
)

OBS_H = 6  # forecast 관측 horizon (steps)


def make_problem(args):
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00")
          for d in TRAIN_DATES[:60]]
    gr_full = np.stack([e.rentals for e in tr]).mean(0)
    ge_full = np.stack([e.returns for e in tr]).mean(0)
    e0 = tr[0]
    # 출퇴근 불균형 압력 top-N 선택
    mw, ew = slice(42, 60), slice(102, 126)
    press = np.abs(ge_full[mw] - gr_full[mw]).sum(0) + np.abs(ge_full[ew] - gr_full[ew]).sum(0)
    sel = np.argsort(press)[::-1][: args.n_stations]
    print(f"  선택 정류소(idx): {sel.tolist()}  cap={e0.capacity[sel].tolist()}")

    w0, w1 = 0, e0.n_steps  # 전일 (출근·퇴근 피크 모두 포함)
    gr = gr_full[w0:w1][:, sel]; ge = ge_full[w0:w1][:, sel]
    caps = e0.capacity[sel]
    init = np.clip(np.round(caps * args.target_ratio), 0, caps).astype(np.int64)
    D = e0.distance_matrix; dist = D[np.ix_(sel, sel)]; coords = e0.station_coords[sel]
    cen = coords.mean(0, keepdims=True); dep_km = np.sqrt(((coords - cen) ** 2).sum(1)) * 111.0
    K = len(sel); full = np.zeros((K + 1, K + 1))
    full[:K, :K] = dist; full[:K, K] = dep_km; full[K, :K] = dep_km
    travel = np.ceil(full / (20.0 * 10.0 / 60.0)).astype(np.int64)
    np.fill_diagonal(travel, 1); travel[K, K] = 1
    prob = SmallProblem(gr, ge, caps, init, travel, w0, w1, args.truck_cap, args.target_ratio,
                        dist_km=full)  # 원본 RebalanceEnv reward 정의(stockout-1.0/full-0.8/km-0.008/step-0.002)
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]
    actual = [(e.rentals[w0:w1][:, sel].astype(np.int64), e.returns[w0:w1][:, sel].astype(np.int64)) for e in ev]
    return prob, actual, sel


def _future_net(p, k):
    t0 = k - p.w0; t1 = min(t0 + OBS_H, p.Twin)
    return (p.ge[t0:t1].sum(0) - p.gr[t0:t1].sum(0)) if t1 > t0 else np.zeros(p.K)


def make_obs(p, k, loc, load, inv, no_forecast=False):
    K = p.K
    loc_oh = np.zeros(K + 1, np.float32); loc_oh[loc] = 1.0
    parts = [
        inv / np.maximum(p.caps, 1),                 # 재고율 (K)
        [load / max(p.truck_cap, 1)],                # 적재율 (1)
        loc_oh,                                      # 위치 one-hot (K+1)
        [(k - p.w0) / max(p.Twin, 1)],               # 시각 (1)
    ]
    if not no_forecast:
        parts.append(_future_net(p, k) / np.maximum(p.caps, 1))  # forecast 미래 순수요율 (K)
    return np.concatenate(parts).astype(np.float32)


def obs_dim(K, no_forecast):
    return (2 * K + 3) if no_forecast else (3 * K + 3)


def potential(p, k, inv):
    """Φ(s) = −예측 불균형 (예측형이 줄이려는 양). shaping은 정책불변."""
    pred = inv + _future_net(p, k)
    return -float(np.abs(pred - p.target).sum()) / float(p.caps.sum())


# ───────────────────────── Gym 환경 ─────────────────────────
def decode_action(a, K, no_amount):
    """a → (dest, lvl). no_amount면 목적지만, 적재량은 중간(50%) 고정."""
    if no_amount:
        return int(a), 1
    return int(a) // 3, int(a) % 3


class DQNRebalanceEnv(gym.Env):
    def __init__(self, prob: SmallProblem, gamma=0.99, shaping_scale=1.0, seed=0,
                 no_amount=False, no_forecast=False):
        super().__init__()
        self.p = prob; self.gamma = gamma; self.shaping_scale = shaping_scale
        self.no_amount = no_amount; self.no_forecast = no_forecast
        self.rng = np.random.default_rng(seed)
        K = prob.K
        self.action_space = spaces.Discrete((K + 1) if no_amount else (K + 1) * 3)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(obs_dim(K, no_forecast),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.k = self.p.w0; self.loc = self.p.depot; self.load = 0
        self.inv = self.p.init_bikes.copy()
        return make_obs(self.p, self.k, self.loc, self.load, self.inv, self.no_forecast), {}

    def step(self, a):
        p = self.p; dest, lvl = decode_action(a, p.K, self.no_amount)
        tv = p.level_to_target(self.loc, lvl)
        self.inv, self.load, _ = p.rebalance_to(self.inv, self.load, self.loc, tv)
        phi0 = potential(p, self.k, self.inv)
        moving = (dest != self.loc)
        env_r = p.w_travel_km * float(p.dist_km[self.loc, dest]) if moving else 0.0  # 원본 reward 정의
        tau = max(int(p.travel[self.loc, dest]), 1)
        for s in range(tau):
            kk = self.k + s
            if kk >= p.w1:
                break
            t = min(kk - p.w0, p.Twin - 1)
            rent = self.rng.poisson(p.gr[t]).astype(np.int64)
            ret = self.rng.poisson(p.ge[t]).astype(np.int64)
            self.inv, so, fu = p._apply_step(self.inv, rent, ret)
            env_r += p.w_stockout * so + p.w_full * fu
            if moving:
                env_r += p.w_travel_step
        self.k = min(self.k + tau, p.w1); self.loc = dest
        phi1 = potential(p, self.k, self.inv)
        reward = env_r + self.shaping_scale * (self.gamma * phi1 - phi0)  # 원본 reward + (선택)shaping
        done = self.k >= p.w1
        return make_obs(p, self.k, self.loc, self.load, self.inv, self.no_forecast), float(reward), done, False, {}


# ───────────────────────── DQN 정책 래퍼 (평가용) ─────────────────────────
class DQNPolicy:
    def __init__(self, model, prob, no_amount=False, no_forecast=False):
        self.model = model; self.p = prob; self.no_amount = no_amount; self.no_forecast = no_forecast
    def act(self, k, loc, load, inv):
        obs = make_obs(self.p, k, loc, load, inv, self.no_forecast)
        a, _ = self.model.predict(obs, deterministic=True)
        dest, lvl = decode_action(a, self.p.K, self.no_amount)
        return self.p.level_to_target(loc, lvl), dest


def export_replay_json(p, sel, policy, dr, dg, path, district, label):
    """정책 1회 롤아웃을 step별 스냅샷 JSON으로 (docs/replay_viewer.html 포맷)."""
    e0 = load_episode("data/processed", district=district, episode_start=f"{TRAIN_DATES[0]} 00:00")
    station_ids = [str(e0.station_ids[s]) for s in sel]
    coords = e0.station_coords[sel].tolist()
    depot_coord = e0.station_coords[sel].mean(0).tolist()      # depot=정류소 무게중심
    # 뷰어가 트럭 loc/dest=depot(index K)로 station_coords/ids를 참조 → depot을 배열 끝에 추가
    coords_v = coords + [depot_coord]
    ids_v = station_ids + ["DEPOT"]
    loc = p.depot; load = 0; rem = 0; dest = p.depot; total = 0
    inv = p.init_bikes.copy()
    cum_so = cum_fu = 0; cum_km = 0.0; cum_r = 0.0
    snaps = []
    for t in range(p.w0, p.w1):
        r_prev = cum_r                                # 이 step에 귀속할 reward 계산용
        action_name = None; action_val = None
        if rem == 0:                                  # 정류소 도착·정차 상태 → 재배치 후 결정
            tv, d = policy.act(t, loc, load, inv)
            inv, load, _ = p.rebalance_to(inv, load, loc, tv)
            action_name = ids_v[d]; action_val = int(d)
            if d != loc:                              # 출발: loc는 origin 유지, rem=총 이동 step
                km = float(p.dist_km[loc, d]); cum_km += km; cum_r += p.w_travel_km * km
                total = max(int(p.travel[loc, d]), 1); rem = total; dest = d
            else:                                     # 정차
                dest = loc; total = 0
        inv, so, fu = p._apply_step(inv, dr[t - p.w0], dg[t - p.w0])
        cum_so += so; cum_fu += fu; cum_r += p.w_stockout * so + p.w_full * fu
        snap = {                                      # 이번 step의 트럭 상태(이동중이면 rem>0)
            "t": int(t), "bikes": inv.tolist(),
            "trucks": [{"loc": int(loc), "dest": int(dest), "load": int(load),
                        "remaining": int(rem), "total_steps": int(total)}],
            "action": action_val, "action_name": action_name, "actor_truck": 0,
            "current_truck": 0, "q_values": None,
            "cum_stockout": int(cum_so), "cum_full": int(cum_fu), "cum_km": round(cum_km, 2),
        }
        snaps.append(snap)
        if rem > 0:                                   # step 종료 후 이동 진행, 도착 시 loc 갱신
            rem -= 1; cum_r += p.w_travel_step
            if rem == 0:
                loc = dest; total = 0
        snap["reward"] = round(cum_r - r_prev, 3)     # 이 step 단위 reward (뷰어 step() 로그가 사용)
        snap["cum_reward"] = round(cum_r, 2)
    out = {
        "meta": {
            "district": district, "date": f"소규모 10정류소 ({label})", "algo": label,
            "model": label, "n_stations": int(p.K), "n_trucks": 1, "depot_index": int(p.depot),
            "station_ids": ids_v, "station_coords": coords_v, "depot_coord": depot_coord,
            "station_capacities": p.caps.astype(int).tolist(), "truck_capacity": int(p.truck_cap),
            "T_max": int(p.w1), "w0": int(p.w0), "step_minutes": 10, "rl_steps": len(snaps),
            "total_reward": round(cum_r, 2), "total_stockout": int(cum_so), "total_full": int(cum_fu),
            "total_km": round(cum_km, 2),
        },
        "snapshots": snaps,
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"  [export] {label}: {path}  (steps={len(snaps)}, reward={cum_r:.1f}, "
          f"stockout={cum_so}, full={cum_fu}, {path.stat().st_size/1024:.1f}KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stations", type=int, default=10)
    ap.add_argument("--truck-cap", type=int, default=30)
    ap.add_argument("--target-ratio", type=float, default=0.5)
    ap.add_argument("--timesteps", type=int, default=400000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--shaping", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-sims", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-amount", action="store_true", help="재배치량 lever 제거(목적지만, 50% 고정)")
    ap.add_argument("--no-forecast", action="store_true", help="obs에서 forecast 제거(현재상태만)")
    ap.add_argument("--tag", default="dqn1")
    ap.add_argument("--export-json", default=None, help="학습된 DQN(및 SLA) 시뮬을 JSON으로 저장할 경로 prefix (예: docs/replay_small)")
    ap.add_argument("--export-day", type=int, default=0, help="리플레이에 쓸 평가일 인덱스(0~6)")
    args = ap.parse_args()

    print(f"[1/4] 문제 구성 (10정류소·1트럭·전일, 출퇴근 포함)...")
    prob, actual, sel = make_problem(args)
    print(f"  K={prob.K}+depot, {prob.Twin} step(전일), 트럭cap={args.truck_cap}, shaping={args.shaping}")

    print(f"[2/4] DQN 학습 ({args.timesteps:,} steps, forecast obs + 재배치량 행동 + potential shaping)...")
    env = DQNRebalanceEnv(prob, gamma=args.gamma, shaping_scale=args.shaping, seed=args.seed,
                          no_amount=args.no_amount, no_forecast=args.no_forecast)
    model = DQN("MlpPolicy", env, learning_rate=args.lr, gamma=args.gamma,
                buffer_size=200000, learning_starts=5000, batch_size=128,
                train_freq=4, target_update_interval=2000,
                exploration_fraction=0.3, exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [256, 256]}, verbose=0, seed=args.seed)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    print(f"  완료 ({(time.time()-t0)/60:.1f}분)")
    out = PROJECT_ROOT / "logs" / f"dqn_small_{args.tag}"; out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model")

    print(f"[3/4] 정책 비교 (미충족수요↓)...\n")
    policies = {
        "do-nothing": DoNothing(prob),
        "STR (반응·최소재배치)": STR(prob),
        "SLA (예측형 lookahead)": SLA(prob),
        "DQN (학습)": DQNPolicy(model, prob, no_amount=args.no_amount, no_forecast=args.no_forecast),
    }
    print(f"  리워드 정의 = 원본 RebalanceEnv (stockout×-1.0 + full×-0.8 + km×-0.008 + step×-0.002), 높을수록↑ 좋음")
    print(f"  {'정책':<24}{'리워드(확률30)':>16}{'미충족(확률)':>14}{'리워드(실제7)':>16}")
    print("  " + "-" * 72)
    res = {}
    for name, pol in policies.items():
        rs, ss, us = eval_stochastic(prob, pol, args.n_sims)
        ra, sa, ua = eval_actual(prob, pol, actual)
        res[name] = (rs, us, ra)
        print(f"  {name:<24}{rs:>11.2f}±{ss:<4.0f}{us:>9.1f}{ra:>13.2f}±{sa:<4.0f}")

    print(f"\n[4/4] 요약 (리워드↑ 좋음)")
    sla = res["SLA (예측형 lookahead)"]; dqn = res["DQN (학습)"]; str_ = res["STR (반응·최소재배치)"]
    print(f"  SLA(예측형) : 리워드 {sla[0]:.2f} (미충족 {sla[1]:.1f})")
    print(f"  STR(반응형) : 리워드 {str_[0]:.2f} (미충족 {str_[1]:.1f})")
    print(f"  DQN        : 리워드 {dqn[0]:.2f} (미충족 {dqn[1]:.1f})")
    print(f"  → DQN vs SLA 리워드 {dqn[0]-sla[0]:+.2f} / vs STR {dqn[0]-str_[0]:+.2f}")
    print(f"  {'✅ 예측형 추월(리워드↑)' if dqn[0] > sla[0] else ('🔸 반응형만 추월' if dqn[0] > str_[0] else '❌ 추월 실패')}")

    if args.export_json:
        print(f"\n[+] 학습 시뮬 JSON 저장 (평가일 {args.export_day})...")
        dr, dg = actual[args.export_day]
        district = _get(_load_yaml(PROJECT_ROOT / "config" / "default.yaml"), "district", default="마포구")
        export_replay_json(prob, sel, DQNPolicy(model, prob, no_amount=args.no_amount, no_forecast=args.no_forecast),
                            dr, dg, f"{args.export_json}_dqn.json", district, "DQN")
        export_replay_json(prob, sel, SLA(prob), dr, dg, f"{args.export_json}_sla.json", district, "SLA(예측형)")


if __name__ == "__main__":
    main()
