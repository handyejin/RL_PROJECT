"""forecast + 날씨 조건부 — 그날 수요 *수준*을 날씨로 보정해 oracle 회수율↑.

전체평균 프로파일(시간대 패턴)에, 그날 날씨로 예측한 *일 수준 배율*을 곱한다.
배율 모델: ratio = (그날 총수요 / 평균 총수요) ~ 선형(기온,강수,풍속,습도,주말).
날씨는 예보로 알 수 있으므로(수요 oracle과 달리) 배포에서 defensible.
사용: python scripts/eval_forecast3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baselines import BasePolicy, MostImbalancedPolicy, PredictiveImbalancedPolicy  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402
from scripts.train import EVAL_DATES, TRAIN_DATES, _load_yaml, _get  # noqa: E402
from scripts.eval_forecast2 import ForecastPredictivePolicy  # noqa: E402


def day_features(ep):
    """[1, mean_temp, total_precip, mean_wind, mean_humidity, is_weekend]."""
    w = ep.weather  # (T,4): temp, precip, wind, humidity
    if w is None:
        wf = [0.0, 0.0, 0.0, 0.0]
    else:
        wf = [float(w[:, 0].mean()), float(w[:, 1].sum()),
              float(w[:, 2].mean()), float(w[:, 3].mean())]
    is_wknd = 1.0 if ep.dayofweek >= 5 or ep.is_holiday else 0.0
    return np.array([1.0] + wf + [is_wknd], dtype=np.float64)


def main() -> None:
    cfg = _load_yaml(PROJECT_ROOT / "config" / "default.yaml")
    district = _get(cfg, "district", default="마포구")
    n_trucks = _get(cfg, "truck", "n_trucks", default=3)
    ek = dict(truck_capacity=20, target_fill_ratio=0.5, urgent_low_ratio=0.15, urgent_high_ratio=0.85,
              urgent_bonus=0.0, strict_urgent_mask=_get(cfg, "env", "strict_urgent_mask", default=False),
              w_travel_km=-0.008, w_travel_step=-0.002, explore_bonus_scale=0.0, shaping_scale=0.0,
              w_work_per_bike=0.0, w_idle_visit=0.0, future_demand_horizon=0)

    print(f"[1/3] {len(TRAIN_DATES)}일 로드 + 전체평균 프로파일 + 날씨 배율 회귀...")
    tr = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in TRAIN_DATES]
    gr = np.stack([e.rentals for e in tr]).mean(0)   # (T,N) 평균 대여
    ge = np.stack([e.returns for e in tr]).mean(0)
    glob_total = gr.sum()                            # 평균 일 총수요(대여 기준)

    # 일 수준 배율 회귀: ratio_d ~ 날씨/주말
    X = np.stack([day_features(e) for e in tr])             # (D, F)
    y = np.array([e.rentals.sum() / glob_total for e in tr])  # (D,)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred_ratio_tr = X @ coef
    ss_res = ((y - pred_ratio_tr) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    print(f"  배율 회귀 R²(train) = {1 - ss_res/ss_tot:.3f}  (날씨가 일 수요 수준을 얼마나 설명)")

    print(f"[2/3] eval {len(EVAL_DATES)}일 로드...")
    ev = [load_episode("data/processed", district=district, episode_start=f"{d} 00:00") for d in EVAL_DATES]

    def run(make_pol):
        rs = []
        for ep in ev:
            env = RebalanceEnv(ep, n_trucks=n_trucks, **ek); env.reset(seed=42)
            tot, done = 0.0, False; pol = make_pol(ep)
            while not done:
                _, r, done, _, _ = env.step(pol.act(env)); tot += r
            rs.append(tot)
        return float(np.mean(rs))

    base = run(lambda ep: MostImbalancedPolicy())
    orac = run(lambda ep: PredictiveImbalancedPolicy(horizon=3))
    print(f"\n[3/3] 비교 (7일 평균)\n")
    print(f"  {'반응형':<28}{base:>9.2f}")
    print(f"  {'oracle 예측형 H=3':<28}{orac:>9.2f}  Δ{orac-base:+.1f}  (상한)")

    def rec(label, val):
        print(f"  {label:<28}{val:>9.2f}  Δ{val-base:+.1f}  (oracle {100*(val-base)/(orac-base):.0f}%)")

    print()
    rec("전체평균 H=3", run(lambda ep: ForecastPredictivePolicy(gr, ge, horizon=3)))
    # 날씨 배율 적용: 그날 feature로 ratio 예측 → 프로파일 스케일
    def weather_scaled(ep):
        ratio = float(np.clip(day_features(ep) @ coef, 0.3, 2.0))
        return ForecastPredictivePolicy(gr * ratio, ge * ratio, horizon=3)
    rec("전체평균+날씨배율 H=3", run(weather_scaled))


if __name__ == "__main__":
    main()
