"""DQN 학습.

- 학습 환경: 1월 ~ 2월 중 N일을 랜덤 회전 → 일별 패턴에 robust한 정책 학습
- 평가 환경: 1/13~1/19 7일 고정 → run_baseline.py와 같은 셋이라 휴리스틱과 직접 비교 가능
- 매 eval_freq step마다 평가 → "점점 좋아지는 과정"을 stdout과 TensorBoard에 출력

사용:
    python scripts/train.py                          # 기본 10만 step
    python scripts/train.py --timesteps 500000       # 더 길게
    python scripts/train.py --tag run1               # 로그 디렉토리 구분
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from src.agents.baselines import get_policy  # noqa: E402
from src.agents.masked_dqn import MaskableDQN  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402
from src.envs.rebalance_env import RebalanceEnv  # noqa: E402

def _date_range(start: str, end: str) -> list[str]:
    """yyyy-mm-dd 문자열 리스트 (start~end 포함)."""
    import datetime
    d = datetime.date.fromisoformat(start)
    end_d = datetime.date.fromisoformat(end)
    out: list[str] = []
    while d <= end_d:
        out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


# 1년치(2025) → seed 42로 random 셔플 → 앞 80% train / 뒤 20% eval
# Random 분할 = 계절·공휴일·요일 분포가 균등 → 캘린더·날씨 feature 학습에 유리
import random as _r
_RNG = _r.Random(42)
_ALL_DATES = _date_range("2025-01-01", "2025-12-31")
_RNG.shuffle(_ALL_DATES)
_N_TRAIN = int(len(_ALL_DATES) * 0.8)
TRAIN_DATES = _ALL_DATES[:_N_TRAIN]                  # 292일 (셔플된 채 — 인덱스로 추출 시 random sample)
EVAL_DATES = sorted(_ALL_DATES[_N_TRAIN:_N_TRAIN + 7])  # 7일 (eval pool 73일 중 random 7 → sorted)


class HeuristicCompareCallback(BaseCallback):
    """평가 시점마다 휴리스틱 reward와 비교 출력."""

    def __init__(self, heuristic_reward: float, verbose: int = 1):
        super().__init__(verbose)
        self.heuristic_reward = heuristic_reward
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # EvalCallback이 last_mean_reward를 set하므로 여기선 pass
        pass


class EvalLoggerCallback(EvalCallback):
    """EvalCallback 확장 — 평가 종료 후 휴리스틱 대비 출력 + 히스토리 저장."""

    def __init__(self, *args, heuristic_reward: float = 0.0, history_store: list | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic_reward = heuristic_reward
        self.history_store = history_store if history_store is not None else []

    def _on_step(self) -> bool:
        ret = super()._on_step()
        if self.last_mean_reward is not None and len(self.evaluations_results) > 0:
            n_evals = len(self.evaluations_results)
            if len(self.history_store) < n_evals:
                ts = self.evaluations_timesteps[-1]
                rwd = float(np.mean(self.evaluations_results[-1]))
                self.history_store.append({"timesteps": int(ts), "eval_reward": rwd})
                diff = rwd - self.heuristic_reward
                marker = "✅" if diff > 0 else "  "
                print(
                    f"[eval {n_evals:2d}] step={ts:>7d}  reward={rwd:>7.2f}  "
                    f"(휴리스틱={self.heuristic_reward:.2f}, Δ={diff:+.2f}) {marker}"
                )
        return ret


class MaskedEvalCallback(BaseCallback):
    """MaskableDQN용 평가 콜백 — env.action_masks()를 predict에 넘겨 마스크 적용."""

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str | None,
        heuristic_reward: float,
        history_store: list,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.heuristic_reward = heuristic_reward
        self.history_store = history_store
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        rewards = []
        env = self.eval_env
        base = env.env if hasattr(env, "env") else env  # Monitor → underlying env
        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                mask = base.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                obs, r, done, trunc, _ = env.step(int(action))
                total += r
                if trunc:
                    break
            rewards.append(total)

        mean_r = float(np.mean(rewards))
        self.history_store.append({"timesteps": self.num_timesteps, "eval_reward": mean_r})

        diff = mean_r - self.heuristic_reward
        marker = "✅" if diff > 0 else "  "
        n_evals = len(self.history_store)
        print(
            f"[eval {n_evals:2d}] step={self.num_timesteps:>7d}  reward={mean_r:>7.2f}  "
            f"(휴리스틱={self.heuristic_reward:.2f}, Δ={diff:+.2f}) {marker}"
        )

        if mean_r > self.best_reward and self.best_model_save_path is not None:
            self.best_reward = mean_r
            Path(self.best_model_save_path).mkdir(parents=True, exist_ok=True)
            self.model.save(Path(self.best_model_save_path) / "best_model")
        return True


def evaluate_heuristic(
    eval_episodes: list,
    n_trucks: int = 3,
    seed: int = 42,
    urgent_low_ratio: float = 0.0,
    urgent_high_ratio: float = 1.0,
    urgent_bonus: float = 0.0,
    strict_urgent_mask: bool = False,
    w_travel_km: float = -0.01,
    w_travel_step: float = -0.005,
    explore_bonus_scale: float = 0.0,
) -> float:
    """eval set에 대해 most_imbalanced 휴리스틱의 평균 reward 측정 (DQN 비교 기준).

    DQN과 동일한 환경 설정(보상 가중치, 트리거, shaping 등)에서 평가해야 공정.
    """
    policy = get_policy("most_imbalanced")
    rewards = []
    for ep in eval_episodes:
        env = RebalanceEnv(
            ep, n_trucks=n_trucks,
            urgent_low_ratio=urgent_low_ratio,
            urgent_high_ratio=urgent_high_ratio,
            urgent_bonus=urgent_bonus,
            strict_urgent_mask=strict_urgent_mask,
            w_travel_km=w_travel_km,
            w_travel_step=w_travel_step,
            explore_bonus_scale=explore_bonus_scale,
        )
        env.reset(seed=seed)
        total = 0.0
        done = False
        while not done:
            _, r, done, _, _ = env.step(policy.act(env))
            total += r
        rewards.append(total)
    return float(np.mean(rewards))


def build_env(
    episodes,
    n_trucks: int,
    seed: int | None = None,
    monitor_dir: Path | None = None,
    use_action_mask: bool = True,
    urgent_low_ratio: float = 0.0,
    urgent_high_ratio: float = 1.0,
    urgent_bonus: float = 0.0,
    strict_urgent_mask: bool = False,
    w_travel_km: float = -0.01,
    w_travel_step: float = -0.005,
    explore_bonus_scale: float = 0.0,
):
    env = RebalanceEnv(
        episodes, n_trucks=n_trucks, seed=seed,
        use_action_mask=use_action_mask,
        urgent_low_ratio=urgent_low_ratio,
        urgent_high_ratio=urgent_high_ratio,
        urgent_bonus=urgent_bonus,
        strict_urgent_mask=strict_urgent_mask,
        w_travel_km=w_travel_km,
        w_travel_step=w_travel_step,
        explore_bonus_scale=explore_bonus_scale,
    )
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_dir / "monitor"))
    return env


def _load_yaml(path: str | Path) -> dict:
    """YAML config 로드. 파일 없거나 비어있으면 {} 반환."""
    p = Path(path)
    if not p.exists():
        return {}
    import yaml
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(cfg: dict, *keys, default=None):
    """nested dict 안전 접근. _get(cfg, 'truck', 'n_trucks', default=3)."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # ── Phase 1: --config만 먼저 파싱 → yaml 로드 ──
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"),
                     help="YAML config 파일 경로 (기본: config/default.yaml)")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_yaml(pre_args.config)
    if cfg:
        print(f"[config] loaded from {pre_args.config}")

    # ── Phase 2: 본 parser — default를 yaml에서 (CLI 인자로 override 가능) ──
    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--algo", choices=["dqn", "masked_dqn"],
                        default=_get(cfg, "algorithm", "name", default="dqn"),
                        help="dqn: vanilla, masked_dqn: invalid-action masking (action_masks)")
    parser.add_argument("--double-q", action="store_true",
                        default=_get(cfg, "algorithm", "double_q", default=False),
                        help="masked_dqn에서 Double DQN 타깃 사용")
    parser.add_argument("--no-action-mask", action="store_true",
                        help="env의 use_action_mask=False (마스킹 효과 비교용)")
    parser.add_argument("--district", default=_get(cfg, "district", default="마포구"))
    parser.add_argument("--n-trucks", type=int,
                        default=_get(cfg, "truck", "n_trucks", default=3))
    parser.add_argument("--timesteps", type=int,
                        default=_get(cfg, "training", "timesteps", default=100_000))
    parser.add_argument("--eval-freq", type=int,
                        default=_get(cfg, "training", "eval_freq", default=5_000))
    parser.add_argument("--lr", type=float,
                        default=_get(cfg, "dqn", "learning_rate", default=1e-4))
    parser.add_argument("--buffer-size", type=int,
                        default=_get(cfg, "dqn", "buffer_size", default=100_000))
    parser.add_argument("--batch-size", type=int,
                        default=_get(cfg, "dqn", "batch_size", default=64))
    parser.add_argument("--gamma", type=float,
                        default=_get(cfg, "dqn", "gamma", default=0.99))
    parser.add_argument("--exploration-fraction", type=float,
                        default=_get(cfg, "dqn", "exploration_fraction", default=0.3))
    parser.add_argument("--exploration-final-eps", type=float,
                        default=_get(cfg, "dqn", "exploration_final_eps", default=0.05))
    parser.add_argument("--seed", type=int,
                        default=_get(cfg, "training", "seed", default=42))
    parser.add_argument("--tag", default=_get(cfg, "training", "tag", default="run1"),
                        help="로그 디렉토리 구분")
    parser.add_argument("--n-train-dates", type=int,
                        default=_get(cfg, "training", "n_train_dates", default=20),
                        help="train pool 크기")
    parser.add_argument("--urgent-low", type=float,
                        default=_get(cfg, "env", "urgent_low", default=0.0),
                        help="bikes/capacity ≤ 이 값이면 빈 위급 (트리거)")
    parser.add_argument("--urgent-high", type=float,
                        default=_get(cfg, "env", "urgent_high", default=1.0),
                        help="bikes/capacity ≥ 이 값이면 가득 위급 (트리거)")
    parser.add_argument("--urgent-bonus", type=float,
                        default=_get(cfg, "env", "urgent_bonus", default=0.0),
                        help="위급 정류소 도착 시 보너스 reward (shaping)")
    parser.add_argument("--strict-mask", action="store_true",
                        default=_get(cfg, "env", "strict_urgent_mask", default=False),
                        help="위급 정류소만 선택 가능하게 마스킹")
    parser.add_argument("--w-travel-km", type=float,
                        default=_get(cfg, "reward", "travel_km", default=-0.01),
                        help="이동 거리 보상 가중치 (km당)")
    parser.add_argument("--w-travel-step", type=float,
                        default=_get(cfg, "reward", "travel_step", default=-0.005),
                        help="이동 시간 보상 가중치 (step당)")
    parser.add_argument("--explore-bonus", type=float,
                        default=_get(cfg, "env", "explore_bonus", default=0.0),
                        help="방문 빈도 기반 탐색 보너스 스케일")
    parser.add_argument("--eval-shaping", action="store_true",
                        default=_get(cfg, "evaluation", "shaping", default=False),
                        help="평가 환경에 shaping 적용. 기본 OFF (공정 metric)")
    args = parser.parse_args()

    log_root = PROJECT_ROOT / "logs" / f"{args.algo}_{args.tag}"
    tb_dir = log_root / "tb"
    eval_dir = log_root / "eval"
    log_root.mkdir(parents=True, exist_ok=True)

    algo_label = args.algo.upper() + (" (Double Q)" if args.double_q and args.algo == "masked_dqn" else "")
    print(f"=== {algo_label} training | tag={args.tag} | total_timesteps={args.timesteps:,} ===")
    print(f"logs → {log_root}")

    # episode 데이터 로드
    print(f"\n[1/4] loading episodes...")
    t0 = time.time()
    train_dates = TRAIN_DATES[: args.n_train_dates]
    train_episodes = [
        load_episode("data/processed", district=args.district, episode_start=f"{d} 00:00")
        for d in train_dates
    ]
    eval_episodes = [
        load_episode("data/processed", district=args.district, episode_start=f"{d} 00:00")
        for d in EVAL_DATES
    ]
    print(f"  train days: {len(train_episodes)}, eval days: {len(eval_episodes)} ({time.time()-t0:.1f}s)")

    # 휴리스틱 비교 기준
    print(f"\n[2/4] computing heuristic baseline on eval set...")
    t0 = time.time()
    # 평가 환경: 기본은 shaping 제거 (공정 metric). --eval-shaping이면 학습과 동일.
    eval_urgent_bonus = args.urgent_bonus if args.eval_shaping else 0.0
    eval_explore_bonus = args.explore_bonus if args.eval_shaping else 0.0
    heuristic_reward = evaluate_heuristic(
        eval_episodes, n_trucks=args.n_trucks, seed=args.seed,
        urgent_low_ratio=args.urgent_low, urgent_high_ratio=args.urgent_high,
        urgent_bonus=eval_urgent_bonus, strict_urgent_mask=args.strict_mask,
        w_travel_km=args.w_travel_km, w_travel_step=args.w_travel_step,
        explore_bonus_scale=eval_explore_bonus,
    )
    print(f"  most_imbalanced mean reward: {heuristic_reward:.2f}  ({time.time()-t0:.1f}s)")

    # 환경 구성
    use_mask = not args.no_action_mask
    print(f"  환경 설정: urgent_low={args.urgent_low}, urgent_high={args.urgent_high}, "
          f"urgent_bonus={args.urgent_bonus}, strict_mask={args.strict_mask}, "
          f"w_travel_km={args.w_travel_km}, w_travel_step={args.w_travel_step}")
    train_env_kwargs = dict(
        urgent_low_ratio=args.urgent_low,
        urgent_high_ratio=args.urgent_high,
        urgent_bonus=args.urgent_bonus,
        strict_urgent_mask=args.strict_mask,
        w_travel_km=args.w_travel_km,
        w_travel_step=args.w_travel_step,
        explore_bonus_scale=args.explore_bonus,
    )
    # 평가는 기본 shaping OFF (순수 metric) — --eval-shaping이면 학습과 동일
    eval_env_kwargs = {**train_env_kwargs,
                       "urgent_bonus": eval_urgent_bonus,
                       "explore_bonus_scale": eval_explore_bonus}
    print(f"  평가 환경: urgent_bonus={eval_env_kwargs['urgent_bonus']}, "
          f"explore_bonus={eval_env_kwargs['explore_bonus_scale']} "
          f"({'shaping OFF — 공정 metric' if not args.eval_shaping else 'shaping ON'})")

    train_env = build_env(
        train_episodes, args.n_trucks, seed=args.seed,
        monitor_dir=log_root / "monitor", use_action_mask=use_mask,
        **train_env_kwargs,
    )
    eval_env = build_env(
        eval_episodes, args.n_trucks, seed=args.seed + 1, use_action_mask=use_mask,
        **eval_env_kwargs,
    )

    # 모델
    print(f"\n[3/4] training {algo_label}...")
    common_kwargs = dict(
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        learning_starts=1000,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(tb_dir),
        verbose=0,
        seed=args.seed,
    )
    if args.algo == "masked_dqn":
        model = MaskableDQN("MlpPolicy", train_env, double_q=args.double_q, **common_kwargs)
    else:
        model = DQN("MlpPolicy", train_env, **common_kwargs)

    history: list[dict] = []
    if args.algo == "masked_dqn":
        eval_callback = MaskedEvalCallback(
            eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=len(EVAL_DATES),
            best_model_save_path=str(log_root / "best"),
            heuristic_reward=heuristic_reward,
            history_store=history,
        )
    else:
        eval_callback = EvalLoggerCallback(
            eval_env,
            best_model_save_path=str(log_root / "best"),
            log_path=str(eval_dir),
            eval_freq=args.eval_freq,
            n_eval_episodes=len(EVAL_DATES),
            deterministic=True,
            render=False,
            heuristic_reward=heuristic_reward,
            history_store=history,
        )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=False)
    print(f"  학습 완료 ({(time.time()-t0)/60:.1f}분)")

    # 최종 저장 & 결과 요약
    print(f"\n[4/4] saving model + history...")
    final_name = f"{args.algo}_final"
    model.save(log_root / final_name)
    np.save(log_root / "history.npy", history, allow_pickle=True)
    print(f"  model → {log_root / (final_name + '.zip')}")
    print(f"  history → {log_root / 'history.npy'} ({len(history)} eval points)")

    if history:
        best = max(history, key=lambda x: x["eval_reward"])
        last = history[-1]
        print(f"\n=== 결과 ===")
        print(f"  휴리스틱:        {heuristic_reward:.2f}")
        print(f"  {algo_label} 마지막:  {last['eval_reward']:.2f}  (step {last['timesteps']:,})")
        print(f"  {algo_label} 베스트:  {best['eval_reward']:.2f}  (step {best['timesteps']:,})")
        verdict = "✅ 휴리스틱 초과" if best["eval_reward"] > heuristic_reward else "❌ 휴리스틱 미달"
        print(f"  → {verdict}")


if __name__ == "__main__":
    main()
