"""세션 요약 — 정책별 7일 평균 reward 비교 (휴리스틱 추월 정도).

출력: docs/summary_bars.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEUR = -500.02

# (label, reward, kind)  kind: heur / rl / deploy / oracle
ROWS = [
    ("oracle 예측형 H=3\n(완벽예지·상한)", -382.79, "oracle"),
    ("forecast 예측형 H=3\n(최종 채택·배포형)", -459.65, "deploy"),
    ("반응형 휴리스틱\n(기존 baseline)", -500.02, "heur"),
    ("강한BC+DQfD pretrain\n(RL 트랙 최고)", -499.14, "rl"),
    ("DQfD+anneal 1M", -515.03, "rl"),
    ("추상화 QRDQN 1M", -553.27, "rl"),
]

COLORS = {"oracle": "#9467bd", "deploy": "#2ca02c", "heur": "#555555", "rl": "#d62728"}


def main() -> None:
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    rows = sorted(ROWS, key=lambda r: r[1])   # 나쁜→좋은 (아래→위)
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [COLORS[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(range(len(rows)), vals, color=colors, alpha=0.85,
                   hatch=["///" if r[2] == "oracle" else "" for r in rows])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)

    # 휴리스틱 기준선
    ax.axvline(HEUR, color="black", ls="--", lw=1.2, alpha=0.7)
    ax.text(HEUR, len(rows) - 0.3, "  휴리스틱 -500", fontsize=9, color="black", va="top")

    # 값·Δ 라벨
    for i, (lab, v, kind) in enumerate(rows):
        d = v - HEUR
        txt = f"{v:.1f}  (Δ{d:+.1f})"
        ax.text(v - 2, i, txt, va="center", ha="right", fontsize=9, color="white", fontweight="bold")

    ax.set_xlabel("7일 평균 reward  (오른쪽=0에 가까울수록 좋음 = 서비스 실패 적음)")
    ax.set_title("정책별 성능 — '반응형→예측형' 설계가 천장을 올렸다\n"
                 "RL 트랙(빨강)은 휴리스틱 언저리, 예측형(초록/보라)이 크게 추월")
    ax.set_xlim(-575, -360)

    # 범례
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=COLORS["deploy"], label="예측형(배포형) ★ 최종"),
           Patch(facecolor=COLORS["oracle"], hatch="///", label="예측형(oracle 상한)"),
           Patch(facecolor=COLORS["heur"], label="기존 휴리스틱"),
           Patch(facecolor=COLORS["rl"], label="RL/BC 트랙")]
    ax.legend(handles=leg, loc="lower left", fontsize=8)
    ax.grid(True, axis="x", alpha=0.25)

    out = PROJECT_ROOT / "docs" / "summary_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
