"""여러 모델의 history.npy를 비교하는 스크립트.

사용:
    python scripts/compare_models.py dqn_v1 mdqn_v1 ppo_v1
    python scripts/compare_models.py --show-plot dqn_v1 ppo_v1
    python scripts/compare_models.py --csv dqn_v1 mdqn_v1 ppo_v1 > comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_history(tag: str) -> dict[str, Any]:
    """logs/{algo}_{tag}/history.npy 로드."""
    log_dir = PROJECT_ROOT / "logs"
    
    # 매칭되는 디렉토리 찾기
    candidates = [
        d for d in log_dir.iterdir() if d.is_dir() and (
            d.name == tag or d.name.endswith(f"_{tag}")
        )
    ]
    
    if not candidates:
        raise FileNotFoundError(
            f"No logs found for tag '{tag}'. "
            f"Available: {[d.name for d in log_dir.iterdir() if d.is_dir()]}"
        )
    
    log_root = candidates[0]
    history_file = log_root / "history.npy"
    
    if not history_file.exists():
        raise FileNotFoundError(f"{history_file} not found")
    
    history = np.load(history_file, allow_pickle=True).tolist()
    algo, parsed_tag = log_root.name.rsplit("_", 1) if "_" in log_root.name else (log_root.name, log_root.name)

    return {
        "tag": parsed_tag,
        "algo": algo,
        "log_dir": str(log_root),
        "history": history,
    }


def print_table(results: list[dict]) -> None:
    """결과를 테이블 형식으로 출력."""
    if not results:
        return
    
    print("\n" + "="*120)
    print(f"{'Tag':<15} {'Algo':<12} {'N_Evals':<10} {'First':<12} {'Last':<12} {'Best':<12} {'Δ(Best-First)':<15}")
    print("="*120)
    
    for res in results:
        history = res["history"]
        if not history:
            continue
        
        rewards = [h["eval_reward"] for h in history]
        first, last, best = rewards[0], rewards[-1], max(rewards)
        delta = best - first
        
        print(
            f"{res['tag']:<15} {res['algo']:<12} {len(history):<10} "
            f"{first:>11.2f} {last:>11.2f} {best:>11.2f} {delta:>14.2f}"
        )
    
    print("="*120 + "\n")


def print_csv(results: list[dict]) -> None:
    """결과를 CSV 형식으로 출력."""
    print("Tag,Algo,Step,Reward")
    for res in results:
        for h in res["history"]:
            print(f"{res['tag']},{res['algo']},{h['timesteps']},{h['eval_reward']:.2f}")


def print_detailed(results: list[dict]) -> None:
    """상세 통계 출력."""
    for res in results:
        history = res["history"]
        if not history:
            continue
        
        rewards = np.array([h["eval_reward"] for h in history])
        steps = np.array([h["timesteps"] for h in history])
        
        print(f"\n{res['tag']} ({res['algo']})")
        print(f"  Log dir: {res['log_dir']}")
        print(f"  Evaluations: {len(rewards)}")
        print(f"  Reward min/mean/max: {rewards.min():.2f} / {rewards.mean():.2f} / {rewards.max():.2f}")
        print(f"  Reward std: {rewards.std():.2f}")
        print(f"  Total steps: {steps.max():,}")
        
        # 개선 추세
        diffs = np.diff(rewards)
        improving = (diffs > 0).sum()
        degrading = (diffs < 0).sum()
        print(f"  개선한 평가: {improving}, 악화한 평가: {degrading}")
        
        # 상위 3개
        top3_idx = np.argsort(rewards)[-3:][::-1]
        print("  Top 3 평가:")
        for rank, idx in enumerate(top3_idx, 1):
            print(f"    {rank}. step={steps[idx]:>7,} reward={rewards[idx]:>7.2f}")


def plot_comparison(results: list[dict]) -> None:
    """matplotlib으로 비교 그래프 그리기."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 미설치. 설치: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for res in results:
        history = res["history"]
        if not history:
            continue
        
        steps = [h["timesteps"] for h in history]
        rewards = [h["eval_reward"] for h in history]
        label = f"{res['tag']} ({res['algo']})"
        ax.plot(steps, rewards, marker='o', label=label, linewidth=2, markersize=4)
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Evaluation Reward", fontsize=12)
    ax.set_title("Model Comparison - Eval Reward Over Time", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "logs" / "model_comparison.png", dpi=100)
    print("📊 그래프 저장됨: logs/model_comparison.png")
    print("   (matplotlib 창이 나타나면 닫으세요)")
    try:
        plt.show()
    except:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="여러 모델의 학습 결과를 비교합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/compare_models.py dqn_v1 mdqn_v1 ppo_v1
  python scripts/compare_models.py --show-plot dqn_v1 ppo_v1
  python scripts/compare_models.py --csv dqn_v1 mdqn_v1 > comparison.csv
        """
    )
    parser.add_argument("tags", nargs="+", help="비교할 실험 태그들 (로그: logs/{algo}_{tag}/)")
    parser.add_argument("--csv", action="store_true", help="CSV 형식으로 출력")
    parser.add_argument("--show-plot", action="store_true", help="matplotlib 그래프 표시")
    parser.add_argument("--detailed", action="store_true", help="상세 통계 출력")
    
    args = parser.parse_args()
    
    # history 로드
    results = []
    for tag in args.tags:
        try:
            res = load_history(tag)
            results.append(res)
            print(f"✅ {tag} 로드됨 ({res['algo']}, {len(res['history'])} 평가)")
        except FileNotFoundError as e:
            print(f"❌ {tag}: {e}")
    
    if not results:
        print("❌ 로드된 결과가 없습니다.")
        sys.exit(1)
    
    # 출력
    if args.csv:
        print_csv(results)
    elif args.detailed:
        print_detailed(results)
    else:
        print_table(results)
    
    # 그래프
    if args.show_plot:
        plot_comparison(results)


if __name__ == "__main__":
    main()
