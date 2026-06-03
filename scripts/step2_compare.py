"""Step 2 (potential-based shaping) vs Step 1 (baseline) 비교.

logs/masked_dqn_step{1,2}_{dqn,ddqn}_s{42,123,777}/history.npy를 모아서:
- best/final mean ± std (DQN/DDQN × Step1/Step2)
- shaping 효과 = Step2 − Step1
- DQN vs DDQN paired test (Step2에서)
- learning curve PNG (4-line: Step1 DQN/DDQN, Step2 DQN/DDQN)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS = PROJECT_ROOT / "logs"

SEEDS = [42, 123, 777]
STEPS = {"step1": "baseline", "step2": "+shaping"}
ALGOS = {"dqn": "DQN", "ddqn": "DDQN"}


def load_history(step: str, algo: str, seed: int):
    p = LOGS / f"masked_dqn_{step}_{algo}_s{seed}" / "history.npy"
    if not p.exists():
        return None
    return list(np.load(p, allow_pickle=True))


def summarize(runs):
    if not runs:
        return None
    bests = [max(r, key=lambda x: x["eval_reward"])["eval_reward"] for r in runs]
    finals = [r[-1]["eval_reward"] for r in runs]
    return {
        "n": len(runs),
        "best_mean": float(np.mean(bests)),
        "best_std": float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0,
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "bests": bests,
        "finals": finals,
    }


def curve_mean_std(runs):
    steps = [d["timesteps"] for d in runs[0]]
    matrix = np.array([[d["eval_reward"] for d in r] for r in runs])
    return steps, matrix.mean(0), matrix.std(0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(matrix.mean(0))


def main():
    # 4 그룹: (step1/step2) × (dqn/ddqn)
    groups = {}
    for step in STEPS:
        for algo in ALGOS:
            runs = [load_history(step, algo, s) for s in SEEDS]
            runs = [r for r in runs if r is not None]
            groups[f"{step}_{algo}"] = runs

    print("\n=== Step 1 vs Step 2 (potential-based shaping) ===\n")
    print(f"{'group':<20} {'n':>3}  {'best (mean±std)':<24}  {'final (mean±std)':<24}")
    summaries = {}
    for key, runs in groups.items():
        s = summarize(runs)
        if s is None:
            print(f"{key:<20}  (no data)")
            continue
        summaries[key] = s
        label = f"{STEPS[key.split('_')[0]]} {ALGOS[key.split('_')[1]]}"
        print(
            f"{label:<20} {s['n']:>3}  "
            f"{s['best_mean']:+7.2f} ± {s['best_std']:5.2f}        "
            f"{s['final_mean']:+7.2f} ± {s['final_std']:5.2f}"
        )

    # shaping 효과
    print("\n=== shaping 효과 (Step2 − Step1, best) ===")
    for algo in ALGOS:
        k1, k2 = f"step1_{algo}", f"step2_{algo}"
        if k1 in summaries and k2 in summaries:
            d = summaries[k2]["best_mean"] - summaries[k1]["best_mean"]
            print(f"  {ALGOS[algo]:<10} : Δ = {d:+.2f}")

    # DQN vs DDQN paired test in Step2
    print("\n=== Step2 안에서 DDQN vs DQN (paired, best) ===")
    if "step2_dqn" in summaries and "step2_ddqn" in summaries:
        d = np.array(summaries["step2_dqn"]["bests"])
        dd = np.array(summaries["step2_ddqn"]["bests"])
        if len(d) == len(dd) >= 2:
            diff = dd - d
            print(f"  DDQN − DQN: mean = {diff.mean():+.2f}, std = {diff.std(ddof=1):.2f}")
            try:
                from scipy import stats
                t, p = stats.ttest_rel(dd, d)
                print(f"  paired t : t = {t:.3f}, p = {p:.4f}")
            except Exception:
                pass

    # learning curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5.5))
        styles = {
            "step1_dqn":  ("tab:blue",   "-",  "Step1 DQN"),
            "step1_ddqn": ("tab:orange", "-",  "Step1 DDQN"),
            "step2_dqn":  ("tab:blue",   "--", "Step2 DQN (+shaping)"),
            "step2_ddqn": ("tab:orange", "--", "Step2 DDQN (+shaping)"),
        }
        for key, runs in groups.items():
            if not runs:
                continue
            color, ls, label = styles[key]
            steps, mean, std = curve_mean_std(runs)
            plt.plot(steps, mean, color=color, linestyle=ls, label=label, linewidth=1.5)
            plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12)
        plt.xlabel("timesteps")
        plt.ylabel("eval reward (7-day mean)")
        plt.title("Step 1 vs Step 2 — Potential-based Shaping")
        plt.grid(alpha=0.3)
        plt.legend(loc="lower right")
        out = PROJECT_ROOT / "docs" / "step2_curve.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\n  curve → {out}")
    except Exception as e:
        print(f"  (plot 생략: {e})")


if __name__ == "__main__":
    main()
