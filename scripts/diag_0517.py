"""Step 2 진단 — 05-17에서 RL(bc_v6)이 휴리스틱에 -61 지는 원인 분해.

동일한 fair eval 환경(shaping OFF)에서 휴리스틱과 bc_v6 best 모델을 각각 돌려
reward를 구성요소(stockout / full / travel)로 분해하고, 시간대별로 격차가
언제 벌어지는지 추적한다.

사용: python scripts/diag_0517.py [--date 2025-05-17]
"""
import argparse
import numpy as np

from src.envs.data_loader import load_episode
from src.envs.rebalance_env import RebalanceEnv
from src.agents.baselines import get_policy
from src.agents.masked_dqn import MaskableDQN

# fair eval 환경 설정 (config/default.yaml + 평가 시 shaping OFF)
ENV_KW = dict(
    n_trucks=3, truck_capacity=20, target_fill_ratio=0.5,
    urgent_low_ratio=0.15, urgent_high_ratio=0.85,
    urgent_bonus=0.0, strict_urgent_mask=True,
    w_travel_km=-0.008, w_travel_step=-0.002,
    explore_bonus_scale=0.0, shaping_scale=0.0,
    w_work_per_bike=0.0, w_idle_visit=0.0, future_demand_horizon=0,
)
W_STOCKOUT, W_FULL = -1.0, -0.8
MODEL = "logs/masked_dqn_bc_v6_finetune_100k/best/best_model.zip"


def make_env(date):
    ep = load_episode("data/processed", district="마포구",
                      episode_start=f"{date} 00:00")
    return ep, RebalanceEnv(ep, **ENV_KW)


def roll(env, act_fn, track_every=24):
    """episode 끝까지 굴리며 누적 지표 기록. (요약 dict, timeline) 반환."""
    env.reset(seed=42)
    total = 0.0
    timeline = []
    done = False
    step = 0
    while not done:
        a = act_fn(env)
        _, r, done, _, _ = env.step(a)
        total += float(r)
        step += 1
        if step % track_every == 0 or done:
            timeline.append(dict(
                step=step, t=int(env.t),
                stockout=int(env.cum_stockout),
                full=int(env.cum_full),
                km=float(env.cum_travel_km),
                reward=total,
            ))
    summary = dict(
        reward=total,
        stockout=int(env.cum_stockout),
        full=int(env.cum_full),
        km=float(env.cum_travel_km),
    )
    return summary, timeline


def decompose(s):
    so = W_STOCKOUT * s["stockout"]
    fu = W_FULL * s["full"]
    tr = s["reward"] - so - fu  # travel + 기타 (km*w_km + step*w_step)
    return so, fu, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-05-17")
    args = ap.parse_args()

    print(f"=== Step 2 진단: {args.date} (RL bc_v6 vs 휴리스틱) ===\n")

    # 휴리스틱
    ep, env = make_env(args.date)
    print(f"episode: stations={ep.n_stations}, steps={ep.n_steps}, "
          f"rentals={int(ep.rentals.sum())}, returns={int(ep.returns.sum())}")
    pol = get_policy("most_imbalanced")
    h_sum, h_tl = roll(env, lambda e: pol.act(e))

    # RL bc_v6
    _, env2 = make_env(args.date)
    model = MaskableDQN.load(MODEL, env=env2)

    def rl_act(e):
        obs = e._get_obs()
        mask = e.action_masks()
        a, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return int(a)

    r_sum, r_tl = roll(env2, rl_act)

    # 요약 비교
    print("\n--- 최종 결과 비교 ---")
    print(f"{'':14}{'휴리스틱':>12}{'RL(bc_v6)':>12}{'차이(RL-휴)':>14}")
    for k, label in [("reward", "총 reward"), ("stockout", "재고소진(건)"),
                     ("full", "만차(건)"), ("km", "이동거리(km)")]:
        h, r = h_sum[k], r_sum[k]
        print(f"{label:14}{h:>12.1f}{r:>12.1f}{r - h:>14.1f}")

    print("\n--- reward 구성요소 분해 ---")
    hso, hfu, htr = decompose(h_sum)
    rso, rfu, rtr = decompose(r_sum)
    print(f"{'':18}{'휴리스틱':>12}{'RL(bc_v6)':>12}{'기여 격차':>12}")
    print(f"{'stockout(-1.0)':18}{hso:>12.1f}{rso:>12.1f}{rso - hso:>12.1f}")
    print(f"{'full(-0.8)':18}{hfu:>12.1f}{rfu:>12.1f}{rfu - hfu:>12.1f}")
    print(f"{'travel(km+step)':18}{htr:>12.1f}{rtr:>12.1f}{rtr - htr:>12.1f}")
    print(f"{'합계':18}{h_sum['reward']:>12.1f}{r_sum['reward']:>12.1f}"
          f"{r_sum['reward'] - h_sum['reward']:>12.1f}")

    print("\n--- 시간대별 누적 격차 (RL reward - 휴 reward) ---")
    print(f"{'step':>6}{'t':>5}{'휴_so':>7}{'RL_so':>7}{'휴_full':>8}{'RL_full':>8}"
          f"{'휴_km':>8}{'RL_km':>8}{'Δreward':>10}")
    for h, r in zip(h_tl, r_tl):
        print(f"{h['step']:>6}{h['t']:>5}{h['stockout']:>7}{r['stockout']:>7}"
              f"{h['full']:>8}{r['full']:>8}{h['km']:>8.0f}{r['km']:>8.0f}"
              f"{r['reward'] - h['reward']:>10.1f}")


if __name__ == "__main__":
    main()
