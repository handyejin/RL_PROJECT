"""하루 시간대(타임스텝)별 누적 reward 추이 — 반응형 vs forecast vs oracle 예측형.

reward는 음수(서비스 실패)라 곡선이 우하향. 기울기 급한 곳 = 그 시간대 실패 많음.
예측형이 반응형 위로 벌어지는 지점 = 예측이 이득 보는 시간대(오전/저녁 러시).
7일 eval 평균. 출력: docs/intraday_reward.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baselines import MostImbalancedPolicy, PredictiveImbalancedPolicy  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402
from scripts.eval_forecast2 import ForecastPredictivePolicy  # noqa: E402


def main() -> None:
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")
    n_trucks = _get(cfg, "truck", "n_trucks", default=3)
    ek = dict(truck_capacity=20, target_fill_ratio=0.5, urgent_low_ratio=0.15, urgent_high_ratio=0.85,
              urgent_bonus=0.0, strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
              w_travel_km=-0.008, w_travel_step=-0.002, explore_bonus_scale=0.0, shaping_scale=0.0,
              w_work_per_bike=0.0, w_idle_visit=0.0, future_demand_horizon=0)

    print("프로파일(292일 평균) 구축...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in TRAIN_DATES]
    gr = np.stack([e.rentals for e in tr]).mean(0)
    ge = np.stack([e.returns for e in tr]).mean(0)
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]

    T = ev[0].rentals.shape[0]           # 144 steps
    grid = np.arange(T)

    def intraday_curve(make_pol):
        """7일 각각: step별 누적 reward를 t-grid에 보간 → 평균."""
        curves = []
        for ep in ev:
            env = RebalanceEnv(ep, n_trucks=n_trucks, **ek)
            env.reset(seed=42)
            ts, cums = [0], [0.0]
            cum, done = 0.0, False
            pol = make_pol(ep)
            while not done:
                _, r, done, _, _ = env.step(pol.act(env))
                cum += r
                ts.append(env.t); cums.append(cum)
            curves.append(np.interp(grid, ts, cums))
        return np.mean(curves, axis=0)

    print("정책별 시뮬레이션...")
    react = intraday_curve(lambda ep: MostImbalancedPolicy())
    fore = intraday_curve(lambda ep: ForecastPredictivePolicy(gr, ge, horizon=3))
    orac = intraday_curve(lambda ep: PredictiveImbalancedPolicy(horizon=3))

    hours = grid * 10 / 60.0   # step → 시각(시)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hours, react, color="#555555", lw=2, label=f"반응형 휴리스틱 (끝 {react[-1]:.0f})")
    ax.plot(hours, fore, color="#2ca02c", lw=2, label=f"forecast 예측형 ★ (끝 {fore[-1]:.0f})")
    ax.plot(hours, orac, color="#9467bd", lw=2, ls="--", label=f"oracle 예측형 (끝 {orac[-1]:.0f})")

    ax.set_xlabel("하루 시각 (시)")
    ax.set_ylabel("누적 reward (낮을수록 실패 누적)")
    ax.set_title("하루 시간대별 누적 reward — 7일 평균\n"
                 "곡선이 벌어지는 구간 = 예측형이 반응형보다 실패를 덜 내는 시간대")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.25)

    out = PROJECT_ROOT / "docs" / "intraday_reward.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved → {out}")
    # 콘솔에도 3시간 간격 표 출력
    print("\n  시각   반응형   forecast   oracle")
    for h in range(0, 25, 3):
        i = min(int(h * 6), T - 1)
        print(f"  {h:>2}시  {react[i]:>8.1f}{fore[i]:>10.1f}{orac[i]:>9.1f}")


if __name__ == "__main__":
    main()
