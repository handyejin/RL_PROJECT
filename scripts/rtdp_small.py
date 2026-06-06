"""소규모 RTDP — 서영현(2022) 논문 정공법 재현 (갈래 A: 추월 시연 셋업).

논문이 휴리스틱을 *실제로* 넘은 엔진은 model-free RL이 아니라 RTDP(테이블형 확률적
동적계획, Skellam/Poisson 전이로 명시적 lookahead). 테이블이라 상태 2^|N|이 폭발 →
정류소·트럭·시간을 줄인 소규모 문제에서만 가능.

갈래 A — RTDP가 빛날 조건을 갖춤(이전 셋업은 셋 다 빠져 추월 실패):
  ① 지리적으로 *퍼진* 정류소 선택 → 이동 2~4 step → 반응형은 너무 늦음(미리 배치해야)
  ② RTDP에 **적재량(목표 레벨) 선택권** {empty / mid / full} → 예측 over/under-fill
     (논문: RTDP는 STR보다 훨씬 많이 deliver 16~22 vs 2.7 → 버퍼 선제 확보)
  ③ 논문식 **STR=최소 재배치(밴드 가장자리까지만)**·반응형 vs RTDP=자유

축소: 마포구 07~10시, 퍼진 6정류소+depot, 트럭 1대, 18 step, 수요=Poisson(forecast).
상태 V키=(시각, 트럭위치, 정류소 3-레벨 밴드 인덱스). 동역학은 exact 정수재고.
RTDP: V(s)=min_a E[cost+γ^τ V(s')] (M-샘플), V0=0 admissible. 비용=미충족수요(↓).

사용: python scripts/rtdp_small.py --iters 5000 --n-stations 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.data_loader import load_episode  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402


# ───────────────────────── 소규모 문제 정의 ─────────────────────────
class SmallProblem:
    def __init__(self, gr, ge, caps, init_bikes, travel, w0, w1, truck_cap, target_ratio,
                 dist_km=None, w_stockout=-1.0, w_full=-0.8, w_travel_km=-0.008, w_travel_step=-0.002):
        self.gr = gr.astype(np.float64)        # (Twin, K) Poisson mean 대여
        self.ge = ge.astype(np.float64)        # (Twin, K) Poisson mean 반납
        self.caps = caps.astype(np.int64)
        self.init_bikes = init_bikes.astype(np.int64)
        self.travel = travel.astype(np.int64)  # (K+1,K+1) step, depot=K
        # 원본 RebalanceEnv와 동일한 reward 정의용
        self.dist_km = (np.zeros_like(travel, dtype=float) if dist_km is None else np.asarray(dist_km, float))
        self.w_stockout = w_stockout; self.w_full = w_full
        self.w_travel_km = w_travel_km; self.w_travel_step = w_travel_step
        self.w0, self.w1 = int(w0), int(w1)
        self.Twin = w1 - w0
        self.K = len(caps)
        self.depot = self.K
        self.truck_cap = int(truck_cap)
        self.target = np.clip(np.round(self.caps * target_ratio), 0, self.caps).astype(np.int64)
        self.band = np.maximum(1, np.round(0.2 * self.caps)).astype(np.int64)  # 안전버퍼 β=0.2
        # RTDP 적재 레벨 값 (정류소별): 0=거의 빔(반납 여유), 1=중간, 2=거의 참(대여 버퍼)
        self.level_val = np.stack([
            np.clip(np.round(0.1 * self.caps), 0, self.caps),
            self.target,
            np.clip(np.round(0.9 * self.caps), 0, self.caps),
        ]).astype(np.int64)  # (3, K)
        self._build_expectation_tables()

    @staticmethod
    def _poisson_pmf(lam, X):
        p = np.empty(X + 1); p[0] = np.exp(-lam)
        for x in range(1, X + 1):
            p[x] = p[x - 1] * lam / x
        return p

    def _build_expectation_tables(self):
        """기댓값 백업용 precompute: 정류소·스텝·재고별 E[미충족], E[처리량] 표(변동성 제거)."""
        maxc = int(self.caps.max())
        T, K = self.Twin, self.K
        self.maxc = maxc
        self.E_unmet_p = np.zeros((T, K, maxc + 1))   # E[(rent-b)+], b=재고
        self.E_unmet_r = np.zeros((T, K, maxc + 1))   # E[(ret-free)+], free=빈자리
        for t in range(T):
            for n in range(K):
                for lam, tbl in ((self.gr[t, n], self.E_unmet_p[t, n]),
                                 (self.ge[t, n], self.E_unmet_r[t, n])):
                    X = int(lam + 10 * np.sqrt(lam) + 10)
                    pmf = self._poisson_pmf(lam, X)
                    xs = np.arange(X + 1)
                    for b in range(maxc + 1):
                        excess = xs - b
                        tbl[b] = float((excess[excess > 0] * pmf[b + 1:]).sum())

    def expected_step(self, t_local, inv):
        """기댓값 1-step: (E[미충족], 다음 기대재고 float). inv float."""
        K = self.K
        b = np.clip(np.round(inv).astype(int), 0, self.maxc)
        idx = np.arange(K)
        lam_r = self.gr[t_local]; lam_g = self.ge[t_local]
        e_p = self.E_unmet_p[t_local, idx, b]                 # E[(rent-b)+]
        serv_p = lam_r - e_p                                  # E[min(rent,b)] = λ - E[(rent-b)+]
        inv_a = inv - serv_p
        free = np.clip(np.round(self.caps - inv_a).astype(int), 0, self.maxc)
        e_r = self.E_unmet_r[t_local, idx, free]
        serv_r = lam_g - e_r
        inv_next = np.clip(inv_a + serv_r, 0, self.caps)
        return float(e_p.sum() + e_r.sum()), inv_next

    def band_index(self, inv):
        dev = inv - self.target
        idx = np.ones(self.K, dtype=np.int8)
        idx[dev < -self.band] = 0
        idx[dev > self.band] = 2
        return tuple(int(x) for x in idx)

    def _apply_step(self, inv, rent, ret):
        served_r = np.minimum(rent, inv); unmet_p = rent - served_r; inv = inv - served_r
        free = self.caps - inv; served_g = np.minimum(ret, free); unmet_f = ret - served_g
        inv = inv + served_g
        return inv, int(unmet_p.sum()), int(unmet_f.sum())   # (inv, stockout, full)

    def rebalance_to(self, inv, load, loc, target_val):
        """loc 재고를 target_val로 (트럭 적재/용량 제약). depot이면 dump(load=0). → inv', load', moved."""
        inv = inv.copy()
        if loc == self.depot:
            return inv, 0, 0
        need = int(target_val - inv[loc])
        moved = 0
        if need > 0:
            give = min(need, load); inv[loc] += give; load -= give; moved = give
        elif need < 0:
            take = min(-need, self.truck_cap - load); inv[loc] -= take; load += take; moved = take
        return inv, load, moved

    def actions_rtdp(self, loc):
        """(lvl, dest). depot에선 lvl 무의미(1 고정)."""
        dests = list(range(self.K)) + [self.depot]
        lvls = [1] if loc == self.depot else [0, 1, 2]
        return [(lv, d) for lv in lvls for d in dests]

    def level_to_target(self, loc, lvl):
        return 0 if loc == self.depot else int(self.level_val[lvl, loc])  # depot은 무시(dump)


# ───────────────────────── RTDP ─────────────────────────
class RTDP:
    def __init__(self, prob: SmallProblem, gamma=0.95, m_backup=4, seed=1):
        self.p = prob; self.gamma = gamma; self.M = m_backup
        self.V = {}; self.rng = np.random.default_rng(seed)

    def key(self, k, loc, inv):
        return (k, loc, self.p.band_index(inv))   # 적재 제외·재고 밴드 축약(논문 fill-rate index)

    def getV(self, k, loc, inv):
        if k >= self.p.w1:
            return 0.0
        return self.V.get(self.key(k, loc, inv), 0.0)

    def _segment(self, k, loc, load, inv, lvl, dest, rng):
        tv = self.p.level_to_target(loc, lvl)
        inv1, load1, _ = self.p.rebalance_to(inv, load, loc, tv)
        tau = max(int(self.p.travel[loc, dest]), 1)
        cost = 0.0; cur = inv1
        for s in range(tau):
            kk = k + s
            if kk >= self.p.w1:
                break
            rent = rng.poisson(self.p.gr[min(kk - self.p.w0, self.p.Twin - 1)]).astype(np.int64)
            ret = rng.poisson(self.p.ge[min(kk - self.p.w0, self.p.Twin - 1)]).astype(np.int64)
            cur, so, fu = self.p._apply_step(cur, rent, ret); cost += so + fu
        return cost, min(k + tau, self.p.w1), dest, load1, cur

    def q_value(self, k, loc, load, inv, lvl, dest):
        tau = max(int(self.p.travel[loc, dest]), 1)
        g = self.gamma ** tau; tot = 0.0
        for _ in range(self.M):
            cost, k2, loc2, load2, inv2 = self._segment(k, loc, load, inv, lvl, dest, self.rng)
            tot += cost + g * self.getV(k2, loc2, inv2)
        return tot / self.M

    def _segment_expected(self, k, loc, load, inv, lvl, dest):
        """기댓값 세그먼트: 재배치 후 dest까지 τ step 기대 미충족수요 + 기대 후속재고(변동성 0)."""
        tv = self.p.level_to_target(loc, lvl)
        inv1, load1, _ = self.p.rebalance_to(inv, load, loc, tv)
        tau = max(int(self.p.travel[loc, dest]), 1)
        cost = 0.0; cur = inv1.astype(float)
        for s in range(tau):
            kk = k + s
            if kk >= self.p.w1:
                break
            um, cur = self.p.expected_step(min(kk - self.p.w0, self.p.Twin - 1), cur)
            cost += um
        return cost, min(k + tau, self.p.w1), dest, load1, cur

    def q_expected(self, k, loc, load, inv, lvl, dest):
        """분석적 기댓값 q — optimizer's curse 없는 백업/행동선택용."""
        tau = max(int(self.p.travel[loc, dest]), 1)
        cost, k2, loc2, load2, inv2 = self._segment_expected(k, loc, load, inv, lvl, dest)
        return cost + (self.gamma ** tau) * self.getV(k2, loc2, inv2)

    def train(self, iters, verbose_every=1000):
        p = self.p
        for it in range(iters):
            k = p.w0; loc = p.depot; load = 0; inv = p.init_bikes.copy()
            while k < p.w1:
                acts = p.actions_rtdp(loc)
                qs = [self.q_expected(k, loc, load, inv, lv, d) for (lv, d) in acts]
                best = int(np.argmin(qs))
                self.V[self.key(k, loc, inv)] = qs[best]
                lv, d = acts[best]
                _, k, loc, load, inv = self._segment(k, loc, load, inv, lv, d, self.rng)
            if verbose_every and (it + 1) % verbose_every == 0:
                print(f"    RTDP iter {it+1}/{iters}  |V|={len(self.V):,}", flush=True)

    def act(self, k, loc, load, inv):
        acts = self.p.actions_rtdp(loc)
        qs = [self.q_expected(k, loc, load, inv, lv, d) for (lv, d) in acts]  # 결정론 행동선택
        lv, d = acts[int(np.argmin(qs))]
        return self.p.level_to_target(loc, lv), d


# ───────────────────────── 휴리스틱 (act → (target_val, dest)) ─────────────────────────
class DoNothing:
    def __init__(self, p): self.p = p
    def act(self, k, loc, load, inv):
        return (int(inv[loc]) if loc < self.p.K else self.p.target[0]), self.p.depot


class STR:
    """반응형·최소 재배치: 밴드 밖이면 *밴드 가장자리까지만* 고치고, 가장 가까운 밴드밖으로."""
    def __init__(self, p): self.p = p
    def act(self, k, loc, load, inv):
        p = self.p
        # 현재 loc 최소 재배치 목표(밴드 가장자리)
        if loc < p.K:
            dev = inv[loc] - p.target[loc]
            tv = int(p.target[loc] + p.band[loc]) if dev > p.band[loc] else \
                 int(p.target[loc] - p.band[loc]) if dev < -p.band[loc] else int(inv[loc])
        else:
            tv = p.target[0]
        # 다음: 밴드 밖 정류소 중 가장 가까운 곳
        oob = [n for n in range(p.K) if abs(inv[n] - p.target[n]) > p.band[n] and n != loc]
        dest = min(oob, key=lambda n: p.travel[loc, n]) if oob else (p.depot if load > 0 else loc)
        return tv, dest


class SLA:
    """정적 lookahead: 목표(중간)로 재배치, 예측 미래 불균형 최대 정류소로."""
    def __init__(self, p, horizon=6): self.p = p; self.h = horizon
    def act(self, k, loc, load, inv):
        p = self.p
        tv = int(p.target[loc]) if loc < p.K else p.target[0]
        k0 = k - p.w0; k1 = min(k0 + self.h, p.Twin)
        fut = (p.ge[k0:k1].sum(0) - p.gr[k0:k1].sum(0)) if k1 > k0 else np.zeros(p.K)
        score = np.abs(inv + fut - p.target)
        if loc < p.K:
            score[loc] = -1
        best = int(np.argmax(score))
        return tv, (best if score[best] > 0 else p.depot)


# ───────────────────────── 평가(롤아웃) ─────────────────────────
def rollout(p, policy, dr, dg):
    """→ (reward, unmet, delivered). reward = 원본 RebalanceEnv 정의
       (stockout·w_so + full·w_full + 이동km·w_km + 이동step·w_step)."""
    k = p.w0; loc = p.depot; load = 0; inv = p.init_bikes.copy()
    reward = 0.0; unmet = 0; delivered = 0
    while k < p.w1:
        tv, dest = policy.act(k, loc, load, inv)
        inv, load, moved = p.rebalance_to(inv, load, loc, tv); delivered += moved
        moving = (dest != loc)
        if moving:
            reward += p.w_travel_km * float(p.dist_km[loc, dest])   # 이동거리 비용(1회)
        tau = max(int(p.travel[loc, dest]), 1)
        for s in range(tau):
            kk = k + s
            if kk >= p.w1:
                break
            inv, so, fu = p._apply_step(inv, dr[kk - p.w0], dg[kk - p.w0])
            unmet += so + fu
            reward += p.w_stockout * so + p.w_full * fu
            if moving:
                reward += p.w_travel_step                          # 이동 중 step 비용
        k = min(k + tau, p.w1); loc = dest
    return reward, unmet, delivered


def eval_stochastic(p, policy, n_sims, seed=123):
    """→ (mean_reward, std_reward, mean_unmet)."""
    rng = np.random.default_rng(seed); rs = []; us = []
    for _ in range(n_sims):
        dr = np.stack([rng.poisson(p.gr[t]) for t in range(p.Twin)]).astype(np.int64)
        dg = np.stack([rng.poisson(p.ge[t]) for t in range(p.Twin)]).astype(np.int64)
        r, u, _ = rollout(p, policy, dr, dg); rs.append(r); us.append(u)
    return float(np.mean(rs)), float(np.std(rs)), float(np.mean(us))


def eval_actual(p, policy, actual):
    """→ (mean_reward, std_reward, mean_unmet)."""
    rs = []; us = []
    for dr, dg in actual:
        r, u, _ = rollout(p, policy, dr, dg); rs.append(r); us.append(u)
    return float(np.mean(rs)), float(np.std(rs)), float(np.mean(us))


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stations", type=int, default=6)
    ap.add_argument("--w0", type=int, default=42)   # 07:00
    ap.add_argument("--w1", type=int, default=60)   # 10:00 (18 step)
    ap.add_argument("--truck-cap", type=int, default=30)
    ap.add_argument("--target-ratio", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--m-backup", type=int, default=4)
    ap.add_argument("--n-sims", type=int, default=30)
    ap.add_argument("--topk-pool", type=int, default=25)
    args = ap.parse_args()

    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")

    print(f"[1/5] 데이터 로드 + forecast 평균(Poisson mean)...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00")
          for d in TRAIN_DATES[:60]]
    gr_full = np.stack([e.rentals for e in tr]).mean(0)
    ge_full = np.stack([e.returns for e in tr]).mean(0)
    e0 = tr[0]; w0, w1 = args.w0, args.w1

    # ① 퍼진 정류소: top-pool 번잡 중 greedy farthest-point
    dem = (gr_full[w0:w1] + ge_full[w0:w1]).sum(0)
    pool = np.argsort(dem)[::-1][: args.topk_pool].tolist()
    D = e0.distance_matrix
    sel = [pool[0]]
    while len(sel) < args.n_stations:
        nxt = max((c for c in pool if c not in sel),
                  key=lambda c: min(D[c, s] for s in sel))
        sel.append(nxt)
    sel = np.array(sel)
    print(f"  선택 정류소(idx): {sel.tolist()}  cap={e0.capacity[sel].tolist()}")
    print(f"  정류소간 거리(km) 평균={D[np.ix_(sel,sel)][np.triu_indices(len(sel),1)].mean():.2f}")

    gr = gr_full[w0:w1][:, sel]; ge = ge_full[w0:w1][:, sel]
    caps = e0.capacity[sel]
    init = np.clip(np.round(caps * args.target_ratio), 0, caps).astype(np.int64)

    dist = D[np.ix_(sel, sel)]; coords = e0.station_coords[sel]
    cen = coords.mean(0, keepdims=True)
    dep_km = np.sqrt(((coords - cen) ** 2).sum(1)) * 111.0
    K = len(sel); full = np.zeros((K + 1, K + 1))
    full[:K, :K] = dist; full[:K, K] = dep_km; full[K, :K] = dep_km
    per_step_km = 20.0 * (10.0 / 60.0)
    travel = np.ceil(full / per_step_km).astype(np.int64)
    np.fill_diagonal(travel, 1); travel[K, K] = 1

    prob = SmallProblem(gr, ge, caps, init, travel, w0, w1, args.truck_cap, args.target_ratio,
                        dist_km=full)
    print(f"  K={K}+depot, 윈도우 {w0}~{w1}({prob.Twin} step), 트럭cap={args.truck_cap}")
    print(f"  travel(step):\n{travel}")

    print(f"[2/5] RTDP 학습 (iters={args.iters}, M={args.m_backup}, γ={args.gamma})...")
    t0 = time.time()
    agent = RTDP(prob, gamma=args.gamma, m_backup=args.m_backup, seed=1)
    agent.train(args.iters)
    print(f"  완료 ({time.time()-t0:.1f}s), |V|={len(agent.V):,}")

    print(f"[3/5] held-out 실제일 {len(EVAL_DATES)}일 수요 추출...")
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]
    actual = [(e.rentals[w0:w1][:, sel].astype(np.int64), e.returns[w0:w1][:, sel].astype(np.int64)) for e in ev]

    print(f"[4/5] 정책 비교...\n")
    policies = {
        "do-nothing": DoNothing(prob),
        "STR (반응·최소재배치)": STR(prob),
        "SLA (정적 lookahead)": SLA(prob),
        "RTDP (확률적 DP)": agent,
    }
    print(f"  {'정책':<24}{'리워드(확률30)':>16}{'미충족(확률)':>14}{'리워드(실제7)':>16}")
    print("  " + "-" * 70)
    res = {}
    for name, pol in policies.items():
        rs, ss, us = eval_stochastic(prob, pol, args.n_sims)
        ra, sa, ua = eval_actual(prob, pol, actual)
        res[name] = (rs, us, ra)
        print(f"  {name:<24}{rs:>10.2f}±{ss:<4.0f}{us:>10.2f}{ra:>12.2f}±{sa:<4.0f}")

    print(f"\n[5/5] 요약 (리워드↑·미충족↓)")
    b = res["STR (반응·최소재배치)"]; r = res["RTDP (확률적 DP)"]
    print(f"  STR  : 리워드 {b[0]:.2f} (미충족 {b[1]:.1f})")
    print(f"  RTDP : 리워드 {r[0]:.2f} (미충족 {r[1]:.1f})")
    print(f"  {'✅ RTDP가 STR 추월(리워드↑)' if r[0] > b[0] else '❌ 추월 실패'}")


if __name__ == "__main__":
    main()
