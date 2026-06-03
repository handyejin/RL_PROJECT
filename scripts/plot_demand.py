"""demand_10min.parquet 분포 시각화.

저장 위치: docs/figures/
- demand_hourly.png : 시간대별 평균 rentals/returns (평일 vs 주말)
- demand_dow_hour.png : 요일 × 시각 히트맵
- demand_daily.png : 일자별 총 수요
- demand_per_station.png : 정류소별 평균 일 수요 분포 (자치구 색상)
- demand_top_stations.png : top 20 정류소 막대
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def _pick_korean_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("AppleSDGothicNeo", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic"):
        for name in available:
            if cand in name:
                return name
    return "DejaVu Sans"


KR_FONT = _pick_korean_font()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "docs" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font=KR_FONT)
    mpl.rcParams["axes.unicode_minus"] = False
    print(f"font: {KR_FONT}")

    demand = pd.read_parquet(PROC / "demand_10min.parquet")
    stations = pd.read_parquet(PROC / "stations.parquet")[["station_id", "gu"]]
    demand = demand.merge(stations, on="station_id", how="left")
    demand["hour"] = demand["t"].dt.hour
    demand["dow"] = demand["t"].dt.dayofweek
    demand["date"] = demand["t"].dt.date

    # 1) 시간대별 평균 (평일 vs 주말)
    g = (
        demand.groupby(["hour", "is_weekend"])[["rentals", "returns"]]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for wkd, sub in g.groupby("is_weekend"):
        label = "주말/공휴일" if wkd else "평일"
        ax.plot(sub["hour"], sub["rentals"], marker="o", label=f"{label} rentals")
        ax.plot(sub["hour"], sub["returns"], marker="s", linestyle="--", label=f"{label} returns")
    ax.set(xlabel="hour of day", ylabel="avg per 10min · station", title="시간대별 평균 수요")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "demand_hourly.png", dpi=130)
    plt.close(fig)

    # 2) 요일 × 시각 히트맵 (총 rentals)
    pivot = demand.pivot_table(index="dow", columns="hour", values="rentals", aggfunc="sum")
    pivot.index = ["월", "화", "수", "목", "금", "토", "일"]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(pivot, cmap="rocket_r", cbar_kws={"label": "total rentals"}, ax=ax)
    ax.set(title="요일 × 시각 rentals 히트맵", xlabel="hour", ylabel="요일")
    fig.tight_layout()
    fig.savefig(OUT / "demand_dow_hour.png", dpi=130)
    plt.close(fig)

    # 3) 일자별 총 수요
    daily = demand.groupby("date")[["rentals", "returns"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(daily["date"], daily["rentals"], label="rentals", linewidth=1.2)
    ax.plot(daily["date"], daily["returns"], label="returns", linewidth=1.2, alpha=0.7)
    ax.set(title="일자별 총 수요 (2025)", xlabel="date", ylabel="trips/day")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "demand_daily.png", dpi=130)
    plt.close(fig)

    # 4) 정류소별 평균 일 수요 분포 (자치구별 색상)
    n_days = demand["date"].nunique()
    per_st = (
        demand.groupby(["station_id", "gu"])[["rentals", "returns"]]
        .sum()
        .div(n_days)
        .reset_index()
    )
    per_st["total"] = per_st["rentals"] + per_st["returns"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=per_st, x="total", hue="gu", multiple="stack", bins=40, ax=ax
    )
    ax.set(title="정류소별 평균 일 수요 분포", xlabel="trips/day (rentals+returns)", ylabel="정류소 수")
    fig.tight_layout()
    fig.savefig(OUT / "demand_per_station.png", dpi=130)
    plt.close(fig)

    # 5) Top 20 정류소
    top = per_st.nlargest(20, "total").merge(
        demand[["station_id"]].drop_duplicates(), on="station_id"
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, y="station_id", x="total", hue="gu", dodge=False, ax=ax)
    ax.set(title="평균 일 수요 Top 20 정류소", xlabel="trips/day", ylabel="station_id")
    fig.tight_layout()
    fig.savefig(OUT / "demand_top_stations.png", dpi=130)
    plt.close(fig)

    # 통계 요약
    print("=== 요약 ===")
    print(f"  기간: {demand['t'].min()} ~ {demand['t'].max()} ({n_days}일)")
    print(f"  정류소: {demand['station_id'].nunique()}")
    print(f"  총 rentals: {demand['rentals'].sum():,}")
    print(f"  총 returns: {demand['returns'].sum():,}")
    print(f"  평일 시간당 평균 rentals/station: {g[g.is_weekend == False]['rentals'].mean() * 6:.2f}")
    print(f"  주말 시간당 평균 rentals/station: {g[g.is_weekend == True]['rentals'].mean() * 6:.2f}")
    print(f"\n  자치구별 일평균 trips:")
    print(per_st.groupby("gu")["total"].sum().round(0).to_string())
    print(f"\n=== 저장: {OUT} ===")
    for p in sorted(OUT.glob("demand_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
