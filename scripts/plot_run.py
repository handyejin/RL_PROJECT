"""단일 run의 step별 eval reward 곡선. history.npy 또는 로그 파싱 둘 다 지원.

사용:
  python scripts/plot_run.py logs/dqfd_<tag>/history.npy
  python scripts/plot_run.py logs/_dqfd/<run>.log        # history 없을 때(killed 등)
출력: docs/run_curve.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEURISTIC = -500.02

_EVAL_RE = re.compile(r"step=\s*([0-9]+)\s+reward=(-?[0-9.]+)")


def load_points(path: Path):
    """history.npy면 직접, .log면 eval 라인 파싱."""
    if path.suffix == ".npy":
        h = np.load(path, allow_pickle=True)
        return (np.array([d["timesteps"] for d in h]),
                np.array([d["eval_reward"] for d in h]))
    steps, rew = [], []
    for line in path.read_text().splitlines():
        m = _EVAL_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            rew.append(float(m.group(2)))
    return np.array(steps), np.array(rew)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/plot_run.py <history.npy | run.log>")
        sys.exit(1)
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    steps, rew = load_points(path)
    print(f"  {len(steps)} eval points, best={rew.max():.2f} @ {steps[int(np.argmax(rew))]}")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(HEURISTIC, color="green", lw=1.6, label=f"휴리스틱 ({HEURISTIC})")
    ax.plot(steps, rew, color="#d62728", lw=1.6, marker="o", ms=4, label="강한 BC + DQfD (online)")

    # 추월 구간(휴리스틱 위) 음영
    ax.fill_between(steps, HEURISTIC, rew, where=(rew >= HEURISTIC),
                    color="green", alpha=0.18, label="추월 구간")
    bi = int(np.argmax(rew))
    ax.scatter([steps[bi]], [rew[bi]], color="black", s=60, zorder=5)
    ax.annotate(f"best {rew[bi]:.1f} @ {steps[bi]//1000}k",
                (steps[bi], rew[bi]), textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("training step")
    ax.set_ylabel("eval reward (7일 평균, raw)")
    ax.set_title("강한 BC + DQfD pretrain → online RL: step별 reward\n"
                 "pretrain 직후 추월(-499) → online RL이 다시 끌어내림")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    out = PROJECT_ROOT / "docs" / "run_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
