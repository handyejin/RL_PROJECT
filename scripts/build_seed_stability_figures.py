"""REINFORCE/A2C seed 안정성 표와 그래프를 생성한다.

최종 chronological split 기준 baseline에 맞춰 seed 반복 실험의 Delta를
다시 계산하고, 강화학습 보고서에 넣기 좋은 summary table과 figure를 만든다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG_DIR = DOCS / "figures"


def setup_plot_style() -> None:
    """보고서용 그래프 스타일을 통일한다."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.family": ["AppleGothic", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#E5E7EB",
        }
    )


def load_seed_detail() -> pd.DataFrame:
    """최신 seed detail을 읽고 chronological baseline 기준 Delta를 붙인다."""
    comparison = pd.read_csv(DOCS / "chronological_a2c_reinforce_comparison_current.csv")
    baseline = comparison.drop_duplicates("district").set_index("district")["baseline"]

    paths = sorted(DOCS.glob("rl_seed_sensitivity_a2c_reinforce_*.detail.csv"))
    paths = [p for p in paths if "chronological_corrected" not in p.name]
    if not paths:
        raise FileNotFoundError("seed detail csv not found")
    detail = pd.read_csv(paths[-1])
    detail["baseline_chronological"] = detail["district"].map(baseline)
    detail["best_delta_chronological"] = detail["best_reward"] - detail["baseline_chronological"]
    detail["final_delta_chronological"] = detail["final_reward"] - detail["baseline_chronological"]
    return detail


def summarize_by_district(detail: pd.DataFrame) -> pd.DataFrame:
    """구별 seed 평균/분산/승률을 요약한다."""
    rows = []
    for (algorithm, district), group in detail.groupby(["algorithm", "district"], sort=True):
        best = group["best_delta_chronological"].astype(float)
        final = group["final_delta_chronological"].astype(float)
        n = len(group)
        std = float(best.std(ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "algorithm": algorithm,
                "district": district,
                "seeds": n,
                "seed_list": ", ".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
                "mean_best_delta": float(best.mean()),
                "std_best_delta": std,
                "ci95_best_delta": ci95,
                "min_best_delta": float(best.min()),
                "max_best_delta": float(best.max()),
                "best_win_rate": float((best > 0).mean()),
                "mean_final_delta": float(final.mean()),
                "std_final_delta": float(final.std(ddof=1)) if n > 1 else 0.0,
                "final_win_rate": float((final > 0).mean()),
            }
        )
    return pd.DataFrame(rows).round(2)


def summarize_by_algorithm(district_summary: pd.DataFrame) -> pd.DataFrame:
    """알고리즘 단위 seed 안정성 지표를 요약한다."""
    rows = []
    for algorithm, group in district_summary.groupby("algorithm", sort=True):
        rows.append(
            {
                "algorithm": algorithm,
                "tested_districts": int(group["district"].nunique()),
                "avg_mean_best_delta": float(group["mean_best_delta"].mean()),
                "avg_std_best_delta": float(group["std_best_delta"].mean()),
                "avg_ci95_best_delta": float(group["ci95_best_delta"].mean()),
                "avg_best_win_rate": float(group["best_win_rate"].mean()),
                "avg_mean_final_delta": float(group["mean_final_delta"].mean()),
                "avg_final_win_rate": float(group["final_win_rate"].mean()),
            }
        )
    return pd.DataFrame(rows).round(2)


def plot_algorithm_stability(algo_summary: pd.DataFrame, path: Path) -> None:
    """알고리즘별 seed 표준편차와 평균 성능을 함께 보여준다."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    colors = {"A2C": "#059669", "REINFORCE": "#2563EB"}
    order = ["A2C", "REINFORCE"]
    data = algo_summary.set_index("algorithm").loc[order].reset_index()

    axes[0].bar(data["algorithm"], data["avg_mean_best_delta"], color=[colors[a] for a in data["algorithm"]], alpha=0.86)
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_title("Seed 평균 Best Delta", fontweight="bold")
    axes[0].set_ylabel("Mean Best Delta")
    axes[0].grid(axis="y", alpha=0.65)
    for idx, row in data.iterrows():
        axes[0].text(idx, row["avg_mean_best_delta"], f"{row['avg_mean_best_delta']:+.1f}", ha="center", va="bottom" if row["avg_mean_best_delta"] >= 0 else "top")

    axes[1].bar(data["algorithm"], data["avg_std_best_delta"], color=[colors[a] for a in data["algorithm"]], alpha=0.86)
    axes[1].set_title("Seed 표준편차 평균", fontweight="bold")
    axes[1].set_ylabel("Std of Best Delta")
    axes[1].grid(axis="y", alpha=0.65)
    for idx, row in data.iterrows():
        axes[1].text(idx, row["avg_std_best_delta"], f"{row['avg_std_best_delta']:.1f}", ha="center", va="bottom")

    fig.suptitle("Seed 안정성 요약: 평균 성능과 분산을 함께 확인", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_district_errorbars(district_summary: pd.DataFrame, path: Path) -> None:
    """구별 mean ± std seed 결과를 error bar로 그린다."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=False)
    colors = {"A2C": "#059669", "REINFORCE": "#2563EB"}
    for ax, algorithm in zip(axes, ["A2C", "REINFORCE"]):
        data = district_summary[district_summary["algorithm"] == algorithm].sort_values("mean_best_delta")
        y = np.arange(len(data))
        ax.errorbar(
            data["mean_best_delta"],
            y,
            xerr=data["std_best_delta"],
            fmt="o",
            color=colors[algorithm],
            ecolor="#94A3B8",
            elinewidth=2,
            capsize=4,
        )
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(data["district"])
        ax.set_xlabel("Best Delta mean ± 1 std")
        ax.set_title(f"{algorithm}: 구별 seed 안정성", fontweight="bold")
        ax.grid(axis="x", alpha=0.65)
        for yi, (_, row) in zip(y, data.iterrows()):
            ax.text(row["mean_best_delta"], yi + 0.18, f"{row['mean_best_delta']:+.1f}±{row['std_best_delta']:.1f}", fontsize=8, ha="center")
    fig.suptitle("Best/Worst 구 seed 반복 결과", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_seed_points(detail: pd.DataFrame, path: Path) -> None:
    """seed별 점을 직접 보여주는 strip plot을 만든다."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), sharex=False)
    colors = {42: "#111827", 123: "#2563EB", 777: "#F97316"}
    for ax, algorithm in zip(axes, ["A2C", "REINFORCE"]):
        data = detail[detail["algorithm"] == algorithm].copy()
        order = (
            data.groupby("district")["best_delta_chronological"].mean().sort_values().index.tolist()
        )
        y_map = {district: idx for idx, district in enumerate(order)}
        for _, row in data.iterrows():
            ax.scatter(
                row["best_delta_chronological"],
                y_map[row["district"]],
                s=62,
                color=colors.get(int(row["seed"]), "#64748B"),
                edgecolor="white",
                linewidth=0.7,
                label=str(int(row["seed"])),
            )
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel("Best Delta")
        ax.set_title(f"{algorithm}: seed별 Best Delta", fontweight="bold")
        ax.grid(axis="x", alpha=0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        seen[label] = handle
    fig.legend(seen.values(), [f"seed {label}" for label in seen], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Seed별 Best Delta 분포", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    """Markdown 표 문자열을 만든다."""
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                if "rate" in key:
                    values.append(f"{value * 100:.1f}%")
                else:
                    values.append(f"{value:.1f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(algo_summary: pd.DataFrame, district_summary: pd.DataFrame, paths: dict[str, Path], out_path: Path) -> None:
    """seed 안정성 결과를 별도 Markdown으로 저장한다."""
    content = f"""# REINFORCE/A2C Seed 안정성 분석

작성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 지표 설명

강화학습 실험에서는 같은 코드와 같은 hyperparameter를 사용해도 seed에 따라 결과가 달라질 수 있다. Seed는 neural network 초기값, action sampling, 학습 중 난수 선택을 결정한다.

본 분석은 Best/Worst 3개 구를 대상으로 seed `42`, `123`, `777`을 반복 실행한 결과를 정리한다. 표준적인 보고 방식에 맞춰 다음 지표를 사용했다.

| 지표 | 의미 |
|---|---|
| Mean Best Delta | seed별 Best Delta 평균 |
| Std Best Delta | seed별 Best Delta 표준편차. 작을수록 안정적 |
| 95% CI | `1.96 * std / sqrt(n)` 참고값. 단, n=3이라 엄밀한 신뢰구간보다는 변동성 참고용 |
| Win Rate | seed 반복 중 baseline을 넘은 비율 |

## 알고리즘별 요약

{markdown_table(algo_summary, [
    ('algorithm', 'Algorithm'),
    ('tested_districts', '구 수'),
    ('avg_mean_best_delta', '평균 Best Δ'),
    ('avg_std_best_delta', '표준편차 평균'),
    ('avg_ci95_best_delta', '95% CI 평균'),
    ('avg_best_win_rate', 'Best 승률'),
    ('avg_mean_final_delta', '평균 Final Δ'),
    ('avg_final_win_rate', 'Final 승률'),
])}

![algorithm seed stability]({paths['algorithm'].relative_to(DOCS)})

## 구별 Seed 안정성

{markdown_table(district_summary, [
    ('algorithm', 'Algorithm'),
    ('district', '구'),
    ('seed_list', 'Seeds'),
    ('mean_best_delta', 'Mean Best Δ'),
    ('std_best_delta', 'Std'),
    ('ci95_best_delta', '95% CI'),
    ('min_best_delta', 'Min'),
    ('max_best_delta', 'Max'),
    ('best_win_rate', 'Best 승률'),
    ('mean_final_delta', 'Mean Final Δ'),
])}

![district seed errorbars]({paths['errorbar'].relative_to(DOCS)})

![seed point plot]({paths['points'].relative_to(DOCS)})

## 해석

A2C는 seed가 바뀌어도 Best Delta 표준편차가 작아 안정적으로 재현된다. 반면 REINFORCE는 같은 구에서도 seed에 따라 Best Delta가 크게 바뀌며, 이는 Monte Carlo reward-to-go 기반 policy gradient가 높은 분산을 갖기 때문으로 해석된다.

따라서 최종 보고서에서는 A2C를 안정적인 주 모델로 제시하고, REINFORCE는 policy gradient baseline 및 일부 구에서의 가능성으로 설명하는 것이 적절하다.
"""
    out_path.write_text(content, encoding="utf-8")


def main() -> None:
    """seed 안정성 산출물을 생성한다."""
    setup_plot_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    detail = load_seed_detail()
    district_summary = summarize_by_district(detail)
    algo_summary = summarize_by_algorithm(district_summary)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    out_detail = DOCS / f"rl_seed_stability_chronological_detail_{timestamp}.csv"
    out_district = DOCS / f"rl_seed_stability_chronological_district_summary_{timestamp}.csv"
    out_algo = DOCS / f"rl_seed_stability_chronological_algorithm_summary_{timestamp}.csv"
    out_md = DOCS / f"rl_seed_stability_chronological_{timestamp}.md"
    paths = {
        "algorithm": FIG_DIR / f"seed_stability_algorithm_{timestamp}.png",
        "errorbar": FIG_DIR / f"seed_stability_district_errorbar_{timestamp}.png",
        "points": FIG_DIR / f"seed_stability_seed_points_{timestamp}.png",
    }

    detail.to_csv(out_detail, index=False, encoding="utf-8-sig")
    district_summary.to_csv(out_district, index=False, encoding="utf-8-sig")
    algo_summary.to_csv(out_algo, index=False, encoding="utf-8-sig")
    plot_algorithm_stability(algo_summary, paths["algorithm"])
    plot_district_errorbars(district_summary, paths["errorbar"])
    plot_seed_points(detail, paths["points"])
    build_markdown(algo_summary, district_summary, paths, out_md)

    print(f"wrote {out_detail}")
    print(f"wrote {out_district}")
    print(f"wrote {out_algo}")
    print(f"wrote {out_md}")
    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
