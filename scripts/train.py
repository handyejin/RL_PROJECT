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

import gymnasium as gym  # noqa: E402
from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from src.agents.baselines import get_policy  # noqa: E402
from src.agents.masked_dqn import MaskableDQN  # noqa: E402
from src.agents.masked_qrdqn import MaskableQRDQN  # noqa: E402
from src.agents.dqfd import DQfDDQN, DemoBuffer, collect_demo_transitions  # noqa: E402
from src.envs.abstract_action import AbstractActionWrapper  # noqa: E402
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
        # _episodes는 RebalanceEnv 본체까지 내려가서 찾고,
        # action_masks/reset/step은 최외곽 env(추상화 wrapper 포함)에서 호출
        raw = env
        while not hasattr(raw, "_episodes") and hasattr(raw, "env"):
            raw = raw.env
        n_eps = min(self.n_eval_episodes, len(raw._episodes))
        for ep_idx in range(n_eps):
            # episode를 명시 선택 (deterministic 평가) + 고정 seed
            obs, _ = env.reset(seed=42, options={"episode_idx": ep_idx})
            done = False
            total = 0.0
            while not done:
                mask = env.action_masks()
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
    truck_capacity: int = 20,
    target_fill_ratio: float = 0.5,
    seed: int = 42,
    urgent_low_ratio: float = 0.0,
    urgent_high_ratio: float = 1.0,
    urgent_bonus: float = 0.0,
    strict_urgent_mask: bool = False,
    w_travel_km: float = -0.01,
    w_travel_step: float = -0.005,
    explore_bonus_scale: float = 0.0,
    shaping_scale: float = 0.0,
    w_work_per_bike: float = 0.0,
    w_idle_visit: float = 0.0,
    future_demand_horizon: int = 0,
) -> float:
    """eval set에 대해 most_imbalanced 휴리스틱의 평균 reward 측정 (DQN 비교 기준).

    DQN과 동일한 환경 설정(보상 가중치, 트리거, shaping 등)에서 평가해야 공정.
    """
    policy = get_policy("most_imbalanced")
    rewards = []
    for ep in eval_episodes:
        env = RebalanceEnv(
            ep, n_trucks=n_trucks, truck_capacity=truck_capacity,
            target_fill_ratio=target_fill_ratio,
            urgent_low_ratio=urgent_low_ratio,
            urgent_high_ratio=urgent_high_ratio,
            urgent_bonus=urgent_bonus,
            strict_urgent_mask=strict_urgent_mask,
            w_travel_km=w_travel_km,
            w_travel_step=w_travel_step,
            explore_bonus_scale=explore_bonus_scale,
            shaping_scale=shaping_scale,
            w_work_per_bike=w_work_per_bike,
            w_idle_visit=w_idle_visit,
            future_demand_horizon=future_demand_horizon,
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
    truck_capacity: int = 20,
    target_fill_ratio: float = 0.5,
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
    shaping_scale: float = 0.0,
    w_work_per_bike: float = 0.0,
    w_idle_visit: float = 0.0,
    future_demand_horizon: int = 0,
    reward_scale: float = 1.0,
    abstract_actions: bool = False,
    forecast_rent=None,
    forecast_ret=None,
    pred_horizon: int = 3,
):
    env = RebalanceEnv(
        episodes, n_trucks=n_trucks, truck_capacity=truck_capacity,
        target_fill_ratio=target_fill_ratio, seed=seed,
        use_action_mask=use_action_mask,
        w_work_per_bike=w_work_per_bike,
        w_idle_visit=w_idle_visit,
        future_demand_horizon=future_demand_horizon,
        urgent_low_ratio=urgent_low_ratio,
        urgent_high_ratio=urgent_high_ratio,
        urgent_bonus=urgent_bonus,
        strict_urgent_mask=strict_urgent_mask,
        w_travel_km=w_travel_km,
        w_travel_step=w_travel_step,
        explore_bonus_scale=explore_bonus_scale,
        shaping_scale=shaping_scale,
        forecast_rent=forecast_rent,
        forecast_ret=forecast_ret,
    )
    if abstract_actions:
        env = AbstractActionWrapper(env, pred_horizon=pred_horizon,
                                    forecast_rent=forecast_rent, forecast_ret=forecast_ret)
    if reward_scale != 1.0:
        env = RewardScale(env, reward_scale)
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_dir / "monitor"))
    return env


class RewardScale(gym.RewardWrapper):
    """보상에 상수 곱 (TD 타깃 크기 축소 → 발산 완화).

    선형 스케일이라 argmax 정책의 순서는 불변 — 학습 안정화 목적.
    평가 환경엔 적용하지 않아 휴리스틱과 raw reward로 공정 비교한다.
    action_masks()를 RebalanceEnv까지 위임해 마스킹 에이전트와 호환.
    """

    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, r):
        return r * self.scale

    def action_masks(self):
        return self.env.action_masks()


def _make_env_thunk(episodes, rank: int, *, n_trucks, truck_capacity,
                    target_fill_ratio, seed, use_action_mask, reward_scale,
                    abstract_actions, env_kwargs):
    """SubprocVecEnv용 env 생성 함수 (rank별 seed 오프셋으로 episode 회전 다양화)."""
    def _init():
        env = build_env(
            episodes, n_trucks, truck_capacity=truck_capacity,
            target_fill_ratio=target_fill_ratio,
            seed=(None if seed is None else seed + rank),
            monitor_dir=None, use_action_mask=use_action_mask,
            reward_scale=reward_scale, abstract_actions=abstract_actions, **env_kwargs,
        )
        return Monitor(env)
    return _init


def build_train_env(episodes, n_envs: int, *, n_trucks, truck_capacity,
                    target_fill_ratio, seed, monitor_dir, use_action_mask,
                    reward_scale, abstract_actions, env_kwargs):
    """학습용 환경. n_envs==1이면 단일 env, >1이면 SubprocVecEnv 병렬."""
    if n_envs <= 1:
        return build_env(
            episodes, n_trucks, truck_capacity=truck_capacity,
            target_fill_ratio=target_fill_ratio, seed=seed,
            monitor_dir=monitor_dir, use_action_mask=use_action_mask,
            reward_scale=reward_scale, abstract_actions=abstract_actions, **env_kwargs,
        )
    thunks = [
        _make_env_thunk(
            episodes, i, n_trucks=n_trucks, truck_capacity=truck_capacity,
            target_fill_ratio=target_fill_ratio, seed=seed,
            use_action_mask=use_action_mask, reward_scale=reward_scale,
            abstract_actions=abstract_actions, env_kwargs=env_kwargs,
        )
        for i in range(n_envs)
    ]
    return SubprocVecEnv(thunks)


def _load_yaml(path: str | Path) -> dict:
    """YAML config 로드. 파일 없거나 비어있으면 {} 반환."""
    p = Path(path)
    if not p.exists():
        return {}
    import yaml
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_device(requested: str) -> str:
    """요청한 device를 실제 사용 가능한 것으로 해석.

    - auto: mps > cuda > cpu 순으로 가능한 것 선택
    - mps/cuda 명시했는데 불가하면 이유 출력 후 cpu fallback
    """
    import torch
    mps_ok = torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()

    req = (requested or "auto").lower()
    if req == "auto":
        dev = "mps" if mps_ok else ("cuda" if cuda_ok else "cpu")
    elif req == "mps" and not mps_ok:
        reason = "빌드 안됨" if not torch.backends.mps.is_built() else "macOS 14.0+ 필요 또는 Apple Silicon 아님"
        print(f"  ⚠️  device=mps 요청했으나 사용 불가 ({reason}) → cpu fallback")
        dev = "cpu"
    elif req == "cuda" and not cuda_ok:
        print(f"  ⚠️  device=cuda 요청했으나 CUDA 없음 → cpu fallback")
        dev = "cpu"
    else:
        dev = req

    # MLP 정책 + CPU-bound env에서는 MPS가 오히려 느릴 수 있음 (SB3 권고)
    if dev == "mps":
        print("  ℹ️  MPS 사용 — 작은 MLP에선 CPU보다 느릴 수 있으니 step/s 비교 권장")
    return dev


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
    parser.add_argument("--algo", choices=["dqn", "masked_dqn", "qrdqn", "dqfd"],
                        default=_get(cfg, "algorithm", "name", default="dqn"),
                        help="dqn: vanilla, masked_dqn: invalid-action masking, "
                             "qrdqn: distributional(QR-DQN) + masking, "
                             "dqfd: demo(휴리스틱) anchor + large-margin (forgetting 방지)")
    parser.add_argument("--double-q", action="store_true",
                        default=_get(cfg, "algorithm", "double_q", default=False),
                        help="masked_dqn에서 Double DQN 타깃 사용")
    parser.add_argument("--n-steps", type=int,
                        default=_get(cfg, "dqn", "n_steps", default=1),
                        help="qrdqn n-step returns (장기 credit assignment). 1=일반 1-step TD")
    parser.add_argument("--n-quantiles", type=int,
                        default=_get(cfg, "dqn", "n_quantiles", default=200),
                        help="qrdqn 분위수 개수 (분포 해상도)")
    parser.add_argument("--no-action-mask", action="store_true",
                        help="env의 use_action_mask=False (마스킹 효과 비교용)")
    parser.add_argument("--district", default=_get(cfg, "district", default="마포구"))
    parser.add_argument("--n-trucks", type=int,
                        default=_get(cfg, "truck", "n_trucks", default=3))
    parser.add_argument("--truck-capacity", type=int,
                        default=_get(cfg, "truck", "capacity", default=20),
                        help="트럭 1대 적재 한도 (자전거 수)")
    parser.add_argument("--target-fill-ratio", type=float,
                        default=_get(cfg, "truck", "target_fill_ratio", default=0.5),
                        help="정류소를 capacity의 몇 비율로 채울지 (적재/하차 기준)")
    parser.add_argument("--lr-decay", action="store_true",
                        default=_get(cfg, "dqn", "lr_decay", default=False),
                        help="lr을 학습 진행에 따라 linear decay")
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
    parser.add_argument("--exploration-initial-eps", type=float,
                        default=_get(cfg, "dqn", "exploration_initial_eps", default=1.0),
                        help="ε 시작값 (기본 1.0). 강한 prior fine-tune 시 낮게(예 0.05) — 무작위 탐색이 prior 오염 방지")
    parser.add_argument("--learning-starts", type=int,
                        default=_get(cfg, "dqn", "learning_starts", default=1000),
                        help="첫 N step은 ε=1.0 random — replay buffer 다양화")
    parser.add_argument("--seed", type=int,
                        default=_get(cfg, "training", "seed", default=42))
    parser.add_argument("--device", default=_get(cfg, "training", "device", default="auto"),
                        help="학습 디바이스: auto(가능하면 mps>cuda>cpu) / cpu / mps / cuda")
    parser.add_argument("--n-envs", type=int,
                        default=_get(cfg, "training", "n_envs", default=1),
                        help="병렬 환경 수 (>1이면 SubprocVecEnv로 rollout 병렬 수집 → 속도↑)")
    parser.add_argument("--reward-scale", type=float,
                        default=_get(cfg, "dqn", "reward_scale", default=1.0),
                        help="학습 보상 상수 곱 (예 0.01). TD 타깃 축소 → 발산 완화. 평가엔 미적용")
    parser.add_argument("--max-grad-norm", type=float,
                        default=_get(cfg, "dqn", "max_grad_norm", default=10.0),
                        help="그래디언트 클리핑 norm (발산 방지). QRDQN 기본 None이라 명시 권장")
    parser.add_argument("--abstract-actions", action="store_true",
                        default=_get(cfg, "env", "abstract_actions", default=False),
                        help="146지선다(정류소) → 6개 의도(…/predictive)로 축소")
    parser.add_argument("--predictive-mode", choices=["oracle", "forecast"], default="forecast",
                        help="추상 predictive 의도의 미래 수요 출처: forecast(train 평균·배포형) / oracle(실제미래·상한)")
    parser.add_argument("--pred-horizon", type=int, default=3,
                        help="predictive 의도 선행 스텝 (3=30분)")
    # ── DQfD 전용 (--algo dqfd) ──
    parser.add_argument("--margin", type=float,
                        default=_get(cfg, "dqfd", "margin", default=0.8),
                        help="dqfd large-margin loss의 마진 ℓ (demo 행동 Q를 이만큼 1등으로 강제)")
    parser.add_argument("--lambda-margin", type=float,
                        default=_get(cfg, "dqfd", "lambda_margin", default=1.0),
                        help="dqfd margin loss 가중치 λ_E")
    parser.add_argument("--lambda-l2", type=float,
                        default=_get(cfg, "dqfd", "lambda_l2", default=1e-5),
                        help="dqfd L2 정규화 가중치")
    parser.add_argument("--lambda-bc", type=float,
                        default=_get(cfg, "dqfd", "lambda_bc", default=1.0),
                        help="dqfd BC(CrossEntropy) 모방 가중치 — margin보다 강한 모방 신호 (0이면 순수 DQfD)")
    parser.add_argument("--lambda-margin-final", type=float,
                        default=_get(cfg, "dqfd", "lambda_margin_final", default=None),
                        help="dqfd 앵커 annealing: λ_margin을 학습 끝까지 이 값으로 선형 감쇠 (미지정=상수)")
    parser.add_argument("--lambda-bc-final", type=float,
                        default=_get(cfg, "dqfd", "lambda_bc_final", default=None),
                        help="dqfd 앵커 annealing: λ_bc를 학습 끝까지 이 값으로 선형 감쇠 (미지정=상수). "
                             "전반 forgetting 차단 → 후반 RL이 휴리스틱 위로 개선")
    parser.add_argument("--dqfd-pretrain-steps", type=int,
                        default=_get(cfg, "dqfd", "pretrain_steps", default=20000),
                        help="dqfd: 환경 상호작용 전 demo만으로 Q 보정할 gradient step 수")
    parser.add_argument("--dqfd-pretrain-lr", type=float,
                        default=_get(cfg, "dqfd", "pretrain_lr", default=1e-3),
                        help="dqfd pretrain 전용 lr (supervised라 본 RL lr보다 높게 — BC와 동일 1e-3)")
    parser.add_argument("--demo-batch-size", type=int,
                        default=_get(cfg, "dqfd", "demo_batch_size", default=0),
                        help="dqfd 본학습 시 demo 미니배치 크기 (0이면 batch_size와 동일)")
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
    parser.add_argument("--shaping-scale", type=float,
                        default=_get(cfg, "env", "shaping_scale", default=0.0),
                        help="Potential-based shaping 스케일 (0=꺼짐). Φ(s)=-Σ|bikes-target|")
    parser.add_argument("--w-work", type=float,
                        default=_get(cfg, "env", "w_work_per_bike", default=0.0),
                        help="적재/하차 1대당 양수 reward")
    parser.add_argument("--w-idle", type=float,
                        default=_get(cfg, "env", "w_idle_visit", default=0.0),
                        help="적정 정류소 도착했는데 0대 옮긴 경우 페널티")
    parser.add_argument("--future-demand-horizon", type=int,
                        default=_get(cfg, "env", "future_demand_horizon", default=0),
                        help="0=비활성. >0이면 향후 N step net demand obs 포함")
    parser.add_argument("--eval-shaping", action="store_true",
                        default=_get(cfg, "evaluation", "shaping", default=False),
                        help="평가 환경에 shaping 적용. 기본 OFF (공정 metric)")
    parser.add_argument("--pretrain", default=None,
                        help="BC pretrain zip 경로. 지정 시 RL 시작 전 q_net 가중치 로드")
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
    # Potential-based shaping은 policy-invariant라 평가에 켜도 정책 우열 안 바뀜.
    # 다만 reward 절대값이 달라 휴리스틱과 직접 비교 위해 평가 시 OFF (공정 metric).
    eval_shaping_scale = args.shaping_scale if args.eval_shaping else 0.0
    heuristic_reward = evaluate_heuristic(
        eval_episodes, n_trucks=args.n_trucks, truck_capacity=args.truck_capacity,
        target_fill_ratio=args.target_fill_ratio, seed=args.seed,
        urgent_low_ratio=args.urgent_low, urgent_high_ratio=args.urgent_high,
        urgent_bonus=eval_urgent_bonus, strict_urgent_mask=args.strict_mask,
        w_travel_km=args.w_travel_km, w_travel_step=args.w_travel_step,
        explore_bonus_scale=eval_explore_bonus,
        shaping_scale=eval_shaping_scale,
        w_work_per_bike=args.w_work,
        w_idle_visit=args.w_idle,
        future_demand_horizon=args.future_demand_horizon,
    )
    print(f"  most_imbalanced mean reward: {heuristic_reward:.2f}  ({time.time()-t0:.1f}s)")

    # 환경 구성
    use_mask = not args.no_action_mask
    print(f"  환경 설정: urgent_low={args.urgent_low}, urgent_high={args.urgent_high}, "
          f"urgent_bonus={args.urgent_bonus}, strict_mask={args.strict_mask}, "
          f"w_travel_km={args.w_travel_km}, w_travel_step={args.w_travel_step}, "
          f"shaping_scale={args.shaping_scale}")
    train_env_kwargs = dict(
        urgent_low_ratio=args.urgent_low,
        urgent_high_ratio=args.urgent_high,
        urgent_bonus=args.urgent_bonus,
        strict_urgent_mask=args.strict_mask,
        w_travel_km=args.w_travel_km,
        w_travel_step=args.w_travel_step,
        explore_bonus_scale=args.explore_bonus,
        shaping_scale=args.shaping_scale,
        w_work_per_bike=args.w_work,
        w_idle_visit=args.w_idle,
        future_demand_horizon=args.future_demand_horizon,
    )
    # 평가는 기본 shaping OFF (순수 metric) — --eval-shaping이면 학습과 동일
    eval_env_kwargs = {**train_env_kwargs,
                       "urgent_bonus": eval_urgent_bonus,
                       "explore_bonus_scale": eval_explore_bonus,
                       "shaping_scale": eval_shaping_scale}
    print(f"  평가 환경: urgent_bonus={eval_env_kwargs['urgent_bonus']}, "
          f"explore_bonus={eval_env_kwargs['explore_bonus_scale']}, "
          f"shaping_scale={eval_env_kwargs['shaping_scale']} "
          f"({'shaping OFF — 공정 metric' if not args.eval_shaping else 'shaping ON'})")

    # 미래수요 출처 설정. forecast면 train 평균 프로파일을 obs(future_demand) + 추상 predictive에 주입.
    # forecast_gr/ge는 데모(forecast 예측형) 수집에도 재사용.
    forecast_gr = forecast_ge = None
    if args.abstract_actions:
        for d in (train_env_kwargs, eval_env_kwargs):
            d["pred_horizon"] = args.pred_horizon
    _need_forecast = args.predictive_mode == "forecast" and (
        args.abstract_actions or args.future_demand_horizon > 0)
    if _need_forecast:
        forecast_gr = np.stack([ep.rentals for ep in train_episodes]).mean(0).astype(np.float32)
        forecast_ge = np.stack([ep.returns for ep in train_episodes]).mean(0).astype(np.float32)
        for d in (train_env_kwargs, eval_env_kwargs):
            d["forecast_rent"] = forecast_gr
            d["forecast_ret"] = forecast_ge
        print(f"  forecast 주입: train {len(train_episodes)}일 평균 "
              f"(obs future_demand + predictive에 사용, H={args.pred_horizon})")
    elif args.abstract_actions:
        print(f"  predictive 의도 = oracle (실제 미래, H={args.pred_horizon}) — 상한")

    if args.reward_scale != 1.0:
        print(f"  reward_scale = {args.reward_scale} (학습만 적용, 평가는 raw)")
    if args.n_envs > 1:
        print(f"  n_envs = {args.n_envs} (SubprocVecEnv 병렬 rollout)")
    if args.abstract_actions:
        print(f"  abstract_actions = ON (146 → 5 의도)")
    train_env = build_train_env(
        train_episodes, args.n_envs, n_trucks=args.n_trucks,
        truck_capacity=args.truck_capacity, target_fill_ratio=args.target_fill_ratio,
        seed=args.seed, monitor_dir=(log_root / "monitor" if args.n_envs <= 1 else None),
        use_action_mask=use_mask, reward_scale=args.reward_scale,
        abstract_actions=args.abstract_actions, env_kwargs=train_env_kwargs,
    )
    # 평가 환경: reward_scale 미적용 (raw) → 휴리스틱과 공정 비교
    eval_env = build_env(
        eval_episodes, args.n_trucks, truck_capacity=args.truck_capacity,
        target_fill_ratio=args.target_fill_ratio,
        seed=args.seed + 1, use_action_mask=use_mask,
        abstract_actions=args.abstract_actions,
        **eval_env_kwargs,
    )

    # 모델
    print(f"\n[3/4] training {algo_label}...")
    # lr decay: progress 1.0→0.0이지만 0.1로 clamp (끝까지 lr=0.1×base 유지)
    if args.lr_decay:
        lr_base = args.lr
        lr_schedule = lambda progress: lr_base * max(0.1, progress)
        print(f"  lr_decay ON: {lr_base:.1e} → {lr_base*0.1:.1e} (linear, clamp 0.1)")
    else:
        lr_schedule = args.lr
    device = _resolve_device(args.device)
    print(f"  device = {device}")
    common_kwargs = dict(
        device=device,
        learning_rate=lr_schedule,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        exploration_initial_eps=args.exploration_initial_eps,
        learning_starts=args.learning_starts,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs={"net_arch": list(_get(cfg, "dqn", "net_arch", default=[256, 256]))},
        tensorboard_log=str(tb_dir),
        verbose=0,
        seed=args.seed,
    )
    if args.algo == "masked_dqn":
        model = MaskableDQN("MlpPolicy", train_env, double_q=args.double_q, **common_kwargs)
    elif args.algo == "qrdqn":
        # QRDQN: 분위수 분포 학습 → net_arch에 n_quantiles 추가, n-step 지원
        qr_kwargs = dict(common_kwargs)
        qr_kwargs["policy_kwargs"] = {
            **common_kwargs["policy_kwargs"],
            "n_quantiles": args.n_quantiles,
        }
        qr_kwargs["n_steps"] = args.n_steps
        print(f"  QR-DQN: n_quantiles={args.n_quantiles}, n_steps={args.n_steps}")
        model = MaskableQRDQN("MlpPolicy", train_env, **qr_kwargs)
    elif args.algo == "dqfd":
        if not use_mask:
            print("  ⚠️  dqfd는 마스킹 전제 — --no-action-mask 권장 안 함")
        demo_bs = args.demo_batch_size or args.batch_size
        anneal_msg = ""
        if args.lambda_margin_final is not None or args.lambda_bc_final is not None:
            anneal_msg = (f", anneal→(margin={args.lambda_margin_final}, bc={args.lambda_bc_final})")
        print(f"  DQfD: margin={args.margin}, λ_margin={args.lambda_margin}, "
              f"λ_bc={args.lambda_bc}, λ_l2={args.lambda_l2}, demo_batch={demo_bs}, "
              f"pretrain_steps={args.dqfd_pretrain_steps:,}{anneal_msg}")
        model = DQfDDQN(
            "MlpPolicy", train_env, double_q=True,
            margin=args.margin, lambda_margin=args.lambda_margin,
            lambda_l2=args.lambda_l2, lambda_bc=args.lambda_bc,
            lambda_margin_final=args.lambda_margin_final,
            lambda_bc_final=args.lambda_bc_final,
            demo_batch_size=demo_bs,
            **common_kwargs,
        )
    else:
        model = DQN("MlpPolicy", train_env, **common_kwargs)

    # BC pretrain 가중치 로드 — q_net & q_net_target 둘 다 (target도 같은 시작점)
    if args.pretrain:
        import torch
        from stable_baselines3.common.save_util import load_from_zip_file
        print(f"  [pretrain] loading {args.pretrain}")
        _, params, _ = load_from_zip_file(args.pretrain)
        state = params["policy"]
        # 'q_net.*' 키만 추출
        q_net_state = {k[len("q_net."):]: v for k, v in state.items() if k.startswith("q_net.")}
        model.policy.q_net.load_state_dict(q_net_state)
        model.policy.q_net_target.load_state_dict(q_net_state)
        print(f"  [pretrain] loaded q_net ({len(q_net_state)} tensors)")

    # DQfD: demo(teacher) full-transition 수집 → DemoBuffer 상주 → pre-training
    if args.algo == "dqfd":
        # teacher 선택:
        #  - 추상: "항상 predictive 의도"(warm-start)
        #  - raw + forecast obs: forecast 예측형(정류소 직접) → 예측 정책으로 앵커
        #  - 그 외: most_imbalanced
        if args.abstract_actions:
            from src.agents.baselines import ConstantIntentPolicy
            from src.envs.abstract_action import ACTION_NAMES
            pred_idx = ACTION_NAMES.index("predictive")
            demo_policy = ConstantIntentPolicy(idx=pred_idx)
            print(f"\n  [DQfD] 추상 action — demo = 항상 '{ACTION_NAMES[pred_idx]}' 의도(idx {pred_idx}) "
                  f"warm-start ({len(train_episodes)} days)...")
        elif forecast_gr is not None:
            from scripts.eval_forecast2 import ForecastPredictivePolicy
            demo_policy = ForecastPredictivePolicy(forecast_gr, forecast_ge, horizon=args.pred_horizon)
            print(f"\n  [DQfD] raw 정류소 — demo = forecast 예측형(H={args.pred_horizon}) "
                  f"({len(train_episodes)} days)...")
        else:
            demo_policy = None
            print(f"\n  [DQfD] demo = most_imbalanced ({len(train_episodes)} days)...")
        t_demo = time.time()
        # 학습 env와 동일 reward 설정 + 마스킹 (RewardScale 미적용, 추상은 학습과 동일하게)
        demo_env = build_env(
            train_episodes, args.n_trucks, truck_capacity=args.truck_capacity,
            target_fill_ratio=args.target_fill_ratio, seed=args.seed,
            monitor_dir=None, use_action_mask=use_mask,
            reward_scale=1.0, abstract_actions=args.abstract_actions, **train_env_kwargs,
        )
        d_obs, d_act, d_rew, d_next, d_done, d_mask = collect_demo_transitions(
            demo_env, policy=demo_policy, policy_name="most_imbalanced",
            reward_scale=args.reward_scale,
        )
        print(f"    collected {len(d_obs):,} transitions ({time.time()-t_demo:.1f}s), "
              f"reward_scale={args.reward_scale} 적용")
        demo_buffer = DemoBuffer(d_obs, d_act, d_rew, d_next, d_done, d_mask, device=model.device)
        model.set_demo_buffer(demo_buffer)
        if args.dqfd_pretrain_steps > 0:
            model.pretrain_on_demos(args.dqfd_pretrain_steps, lr=args.dqfd_pretrain_lr)

    history: list[dict] = []
    # VecEnv(n_envs>1)에선 콜백이 vec-step마다 호출되므로 eval_freq를 나눠 timestep 기준 유지
    cb_eval_freq = max(args.eval_freq // max(args.n_envs, 1), 1)
    if args.algo in ("masked_dqn", "qrdqn", "dqfd"):
        eval_callback = MaskedEvalCallback(
            eval_env,
            eval_freq=cb_eval_freq,
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
