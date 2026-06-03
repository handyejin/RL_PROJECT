"""Step 1: DQN vs Double DQN baseline 비교.

logs/masked_dqn_step1_{dqn,ddqn}_s{42,123,777}/history.npy 모아서
- best / final eval reward
- learning curve (mean ± std)
- DQN vs DDQN paired test
출력.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS = PROJECT_ROOT / "logs"

SEEDS = [42, 123, 777]
ALGOS = {"dqn": "DQN", "ddqn": "Double DQN"}


def load_history(tag: str) -> list[dict] | None:
    p = LOGS / f"masked_dqn_{tag}" / "history.npy"
    if not p.exists():
        return None
    return list(np.load(p, allow_pickle=True))


def summarize(name: str, runs: list[list[dict]]) -> dict:
    bests = [max(r, key=lambda x: x["eval_reward"])["eval_reward"] for r in runs]
    finals = [r[-1]["eval_reward"] for r in runs]
    aucs = [float(np.mean([x["eval_reward"] for x in r])) for r in runs]
    return {
        "name": name,
        "n_seeds": len(runs),
        "best_mean": float(np.mean(bests)),
        "best_std": float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0,
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
        "auc_mean": float(np.mean(aucs)),
        "bests": bests,
        "finals": finals,
    }


def curve_mean_std(runs: list[list[dict]]):
    steps = [d["timesteps"] for d in runs[0]]
    matrix = np.array([[d["eval_reward"] for d in r] for r in runs])
    return steps, matrix.mean(0), matrix.std(0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(matrix.mean(0))


def main():
    by_algo: dict[str, list[list[dict]]] = {}
    missing = []
    for key in ALGOS:
        runs = []
        for s in SEEDS:
            tag = f"step1_{key}_s{s}"
            h = load_history(tag)
            if h is None:
                missing.append(tag)
            else:
                runs.append(h)
        by_algo[key] = runs

    if missing:
        print(f"⚠️  미완료: {missing}")
        print(f"   (있는 것만으로 집계)")

    summaries = {k: summarize(ALGOS[k], v) for k, v in by_algo.items() if v}

    print("\n=== Step 1 결과 ===")
    print(f"{'algo':<12} {'best (mean±std)':<22} {'final (mean±std)':<22} {'auc':<8}")
    for k, s in summaries.items():
        print(
            f"{s['name']:<12} "
            f"{s['best_mean']:+7.2f} ± {s['best_std']:5.2f}      "
            f"{s['final_mean']:+7.2f} ± {s['final_std']:5.2f}      "
            f"{s['auc_mean']:+7.2f}"
        )

    if "dqn" in summaries and "ddqn" in summaries and len(by_algo["dqn"]) == len(by_algo["ddqn"]) >= 2:
        d = np.array(summaries["dqn"]["bests"])
        dd = np.array(summaries["ddqn"]["bests"])
        diff = dd - d
        print(f"\nDDQN − DQN (best, paired): mean={diff.mean():+.2f}  std={diff.std(ddof=1):.2f}")
        try:
            from scipy import stats
            t, p = stats.ttest_rel(dd, d)
            w = stats.wilcoxon(dd, d) if len(d) >= 2 else None
            print(f"  paired t: t={t:.3f}, p={p:.4f}")
            if w is not None:
                print(f"  wilcoxon: stat={w.statistic:.3f}, p={w.pvalue:.4f}")
        except Exception as e:
            print(f"  (scipy 통계 생략: {e})")

    # learning curve 저장
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        colors = {"dqn": "tab:blue", "ddqn": "tab:orange"}
        for k, runs in by_algo.items():
            if not runs:
                continue
            steps, mean, std = curve_mean_std(runs)
            plt.plot(steps, mean, label=ALGOS[k], color=colors[k])
            plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color=colors[k])
        plt.xlabel("timesteps")
        plt.ylabel("eval reward (7-day mean)")
        plt.title("Step 1: DQN vs Double DQN (3 seeds)")
        plt.grid(alpha=0.3)
        plt.legend()
        out = PROJECT_ROOT / "docs" / "step1_curve.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\n  curve → {out}")
    except Exception as e:
        print(f"  (plot 생략: {e})")


if __name__ == "__main__":
    main()
