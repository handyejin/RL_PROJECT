"""구역 분할 다트럭 — multi-agent 협응을 "공간 분할"로 우회.

발견: 단일트럭 DQN은 ~15정류소(sweet spot)에서 예측형을 가장 크게 추월(+52.6).
문제: 45정류소·3트럭 공유(multi-agent)는 협응 비정상성으로 DQN 학습 실패(≈do-nothing).
해법: 45정류소를 좌표 k-means로 **3구역×15**로 나누고, **구역당 트럭 1대 독립 학습**.
   → 협응 불필요(구역 분리) + 각 구역 = sweet spot → 각자 추월 → 합산하면 전체 추월.

각 구역: 단일트럭 SmallProblem(dqn_small 재사용), DQN 학습, SLA/STR/do-nothing과 비교.
전체 = 구역 합산(정류소 disjoint이라 미충족수요·reward 가산). 리워드=원본 정의.

사용: python scripts/dqn_zoned.py --n-stations 45 --n-zones 3 --timesteps 300000
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

from sklearn.cluster import KMeans  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402

from src.envs.data_loader import load_episode  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402
from scripts.rtdp_small import SmallProblem, STR, SLA, DoNothing, eval_stochastic, eval_actual  # noqa: E402
from scripts.dqn_small import DQNRebalanceEnv, DQNPolicy  # noqa: E402


def build_zone_problem(sel, gr_full, ge_full, e0, ev, truck_cap, target_ratio):
    """글로벌 정류소 인덱스 리스트(sel) → 단일트럭 SmallProblem + 평가일 수요."""
    w0, w1 = 0, e0.n_steps
    gr = gr_full[w0:w1][:, sel]; ge = ge_full[w0:w1][:, sel]
    caps = e0.capacity[sel]
    init = np.clip(np.round(caps * target_ratio), 0, caps).astype(np.int64)
    D = e0.distance_matrix; dist = D[np.ix_(sel, sel)]; coords = e0.station_coords[sel]
    cen = coords.mean(0, keepdims=True); dep_km = np.sqrt(((coords - cen) ** 2).sum(1)) * 111.0
    K = len(sel); full = np.zeros((K + 1, K + 1))
    full[:K, :K] = dist; full[:K, K] = dep_km; full[K, :K] = dep_km
    travel = np.ceil(full / (20.0 * 10.0 / 60.0)).astype(np.int64)
    np.fill_diagonal(travel, 1); travel[K, K] = 1
    prob = SmallProblem(gr, ge, caps, init, travel, w0, w1, truck_cap, target_ratio, dist_km=full)
    actual = [(e.rentals[w0:w1][:, sel].astype(np.int64), e.returns[w0:w1][:, sel].astype(np.int64)) for e in ev]
    return prob, actual


def train_zone_dqn(prob, timesteps, seed, lr=5e-4):
    env = DQNRebalanceEnv(prob, gamma=0.99, shaping_scale=1.0, seed=seed)
    model = DQN("MlpPolicy", env, learning_rate=lr, gamma=0.99,
                buffer_size=200000, learning_starts=5000, batch_size=128,
                train_freq=4, target_update_interval=2000,
                exploration_fraction=0.3, exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [256, 256]}, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stations", type=int, default=45)
    ap.add_argument("--n-zones", type=int, default=3)
    ap.add_argument("--truck-cap", type=int, default=30)
    ap.add_argument("--target-ratio", type=float, default=0.5)
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--n-sims", type=int, default=30)
    ap.add_argument("--zone-id", type=int, default=None, help="이 구역만 학습·평가(병렬용)")
    ap.add_argument("--dump", default=None, help="구역 결과 저장 경로(json)")
    ap.add_argument("--aggregate", default=None, help="구역 결과 json들(쉼표구분) 합산만 수행")
    args = ap.parse_args()

    if args.aggregate:                       # 병렬 구역 결과 합산 전용 모드
        agg = {k: {"r": 0.0, "u": 0.0, "ra": 0.0} for k in ["do-nothing", "STR", "SLA", "DQN"]}
        rows = []
        for p in args.aggregate.split(","):
            zr = json.load(open(p)); rows.append(zr)
            for k in agg:
                agg[k]["r"] += zr[k]["r"]; agg[k]["u"] += zr[k]["u"]; agg[k]["ra"] += zr[k]["ra"]
        print(f"[합산] {len(rows)}구역 (확률30 리워드↑ 좋음)")
        for r in sorted(rows, key=lambda x: x["zone"]):
            print(f"  구역{r['zone']}({r['n']}개): SLA {r['SLA']['r']:.1f} / DQN {r['DQN']['r']:.1f} "
                  f"{'✅' if r['DQN']['r'] > r['SLA']['r'] else '❌'}")
        print(f"\n전체 합산 (45정류소·{len(rows)}트럭):")
        for name in ["do-nothing", "STR", "SLA", "DQN"]:
            a = agg[name]
            print(f"  {name:<12} 확률 {a['r']:>9.1f} (미충족 {a['u']:.1f})  실제7일 {a['ra']:>9.1f}")
        d, s = agg["DQN"], agg["SLA"]
        print(f"\n  → 구역분할 DQN vs SLA: 확률 {d['r']-s['r']:+.1f} / 실제 {d['ra']-s['ra']:+.1f}  "
              f"{'✅ 추월' if d['r'] > s['r'] else '❌ 미달'}")
        return

    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")
    print(f"[1/4] {args.n_stations}정류소 선택 + {args.n_zones}구역 k-means 분할...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in TRAIN_DATES[:60]]
    gr_full = np.stack([e.rentals for e in tr]).mean(0); ge_full = np.stack([e.returns for e in tr]).mean(0)
    e0 = tr[0]
    mw, ew = slice(42, 60), slice(102, 126)
    press = np.abs(ge_full[mw] - gr_full[mw]).sum(0) + np.abs(ge_full[ew] - gr_full[ew]).sum(0)
    sel = np.argsort(press)[::-1][: args.n_stations]
    coords = e0.station_coords[sel]
    km = KMeans(n_clusters=args.n_zones, n_init=10, random_state=0).fit(coords)
    zones = [sel[km.labels_ == z].tolist() for z in range(args.n_zones)]
    for z, zs in enumerate(zones):
        print(f"  구역{z}: {len(zs)}개 정류소 {zs}")

    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]

    def run_zone(z):
        zs = zones[z]
        prob, actual = build_zone_problem(zs, gr_full, ge_full, e0, ev, args.truck_cap, args.target_ratio)
        model = train_zone_dqn(prob, args.timesteps, seed=1 + z)
        pols = {"do-nothing": DoNothing(prob), "STR": STR(prob), "SLA": SLA(prob),
                "DQN": DQNPolicy(model, prob)}
        row = {"zone": z, "n": len(zs)}
        for name, pol in pols.items():
            rs, _, us = eval_stochastic(prob, pol, args.n_sims)
            ra, _, _ = eval_actual(prob, pol, actual)
            row[name] = {"r": rs, "u": us, "ra": ra}
        return row

    # 단일 구역 모드(병렬용)
    if args.zone_id is not None:
        print(f"[단일구역 {args.zone_id}] 학습 {args.timesteps:,} steps...")
        t0 = time.time()
        row = run_zone(args.zone_id)
        print(f"  구역{args.zone_id}({row['n']}개) 완료 ({(time.time()-t0)/60:.1f}분): "
              f"SLA {row['SLA']['r']:.1f} / DQN {row['DQN']['r']:.1f} "
              f"{'✅' if row['DQN']['r'] > row['SLA']['r'] else '❌'}")
        if args.dump:
            json.dump(row, open(args.dump, "w"))
            print(f"  → dump {args.dump}")
        return

    print(f"[2/4] 구역별 단일트럭 DQN 독립 학습 (각 {args.timesteps:,} steps)...")
    agg = {k: {"r": 0.0, "u": 0.0, "ra": 0.0} for k in ["do-nothing", "STR", "SLA", "DQN"]}
    zone_rows = []
    t0 = time.time()
    for z in range(len(zones)):
        row = run_zone(z)
        for name in agg:
            agg[name]["r"] += row[name]["r"]; agg[name]["u"] += row[name]["u"]; agg[name]["ra"] += row[name]["ra"]
        zone_rows.append({"zone": z, "n": row["n"], "SLA": row["SLA"]["r"], "DQN": row["DQN"]["r"],
                          "STR": row["STR"]["r"], "do-nothing": row["do-nothing"]["r"]})
        print(f"  구역{z}({row['n']}개) 완료: SLA {row['SLA']['r']:.1f} / DQN {row['DQN']['r']:.1f} "
              f"{'✅' if row['DQN']['r'] > row['SLA']['r'] else '❌'}")
    print(f"  학습 완료 ({(time.time()-t0)/60:.1f}분)")

    print(f"\n[3/4] 구역별 (확률30 리워드↑ 좋음)")
    print(f"  {'구역':<6}{'n':>4}{'do-nothing':>12}{'STR':>10}{'SLA':>10}{'DQN':>10}")
    for r in zone_rows:
        print(f"  {r['zone']:<6}{r['n']:>4}{r['do-nothing']:>12.1f}{r['STR']:>10.1f}{r['SLA']:>10.1f}{r['DQN']:>10.1f}")

    print(f"\n[4/4] 전체 합산 (45정류소·3트럭 전체 시스템, 리워드↑ 좋음)")
    for name in ["do-nothing", "STR", "SLA", "DQN"]:
        a = agg[name]
        print(f"  {name:<12} 확률 {a['r']:>9.1f}  (미충족 {a['u']:.1f})   실제7일 {a['ra']:>9.1f}")
    d, s, st = agg["DQN"], agg["SLA"], agg["STR"]
    print(f"\n  → 구역분할 DQN(3트럭) vs 예측형 SLA: 확률 {d['r']-s['r']:+.1f} / 실제 {d['ra']-s['ra']:+.1f}")
    print(f"  {'✅ 구역분할 DQN이 예측형 추월' if d['r'] > s['r'] else '❌ 미달'}")
    print(f"  (cf. 같은 45·3트럭 multi-agent 공유 DQN은 ≈do-nothing으로 실패했음)")


if __name__ == "__main__":
    main()
