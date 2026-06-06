"""학습 reward 추이 시각화 — 여러 run의 history.npy를 한 그래프에 비교.

사용: python scripts/plot_reward.py
출력: docs/reward_curves.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEURISTIC = -500.02
BC_LEVEL = -506.6   # 06-03 BC v6 best (참고선)

# (label, history.npy 경로, 색)
RUNS = [
    ("추상화 QRDQN 1M (Run4)", "logs/qrdqn_abstract_1M/history.npy", "#888888"),
    ("DQfD+CE 고정앵커 100k (Run5)", "logs/dqfd_dqfd_ce_100k/history.npy", "#1f77b4"),
    ("DQfD+CE+anneal 1M (Run6)", "logs/dqfd_dqfd_ce_anneal_1M/history.npy", "#d62728"),
]


def load(path: str):
    h = np.load(PROJECT_ROOT / path, allow_pickle=True)
    steps = np.array([d["timesteps"] for d in h])
    rew = np.array([d["eval_reward"] for d in h])
    return steps, rew


def main() -> None:
    # 한글 폰트 (macOS) — 플롯 생성 전에 설정해야 라벨에 적용됨
    try:
        plt.rcParams["font.family"] = "AppleGothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # 기준선
    ax.axhline(HEURISTIC, color="green", ls="-", lw=1.6, alpha=0.8,
               label=f"휴리스틱 baseline ({HEURISTIC})")
    ax.axhline(BC_LEVEL, color="orange", ls="--", lw=1.2, alpha=0.7,
               label=f"과거 BC best 참고선 ({BC_LEVEL})")
    ax.axhline(-499.14, color="purple", ls="-", lw=2.0, alpha=0.9,
               label="강한BC+DQfD pretrain = -499.1 [추월] (Run7)")

    for label, path, color in RUNS:
        p = PROJECT_ROOT / path
        if not p.exists():
            print(f"  skip (없음): {path}")
            continue
        steps, rew = load(path)
        ax.plot(steps, rew, color=color, lw=1.3, alpha=0.85, label=label)
        # best 지점 마커
        bi = int(np.argmax(rew))
        ax.scatter([steps[bi]], [rew[bi]], color=color, s=55, zorder=5,
                   edgecolor="white", linewidth=1)
        ax.annotate(f"best {rew[bi]:.1f}\n@{steps[bi]//1000}k",
                    (steps[bi], rew[bi]), textcoords="offset points",
                    xytext=(6, 8), fontsize=8, color=color)

    ax.set_xlabel("training step")
    ax.set_ylabel("eval reward (deterministic 7일 평균, raw)")
    ax.set_title("RL 트랙 reward 추이 — 휴리스틱(-500) 추월\n"
                 "위로 갈수록 좋음. 강한 BC+DQfD pretrain(보라선)이 처음으로 휴리스틱 추월(-499.1)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    # y축: 너무 낮은 초기 spike에 눌리지 않게 클램프
    ax.set_ylim(-720, -480)

    out = PROJECT_ROOT / "docs" / "reward_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
