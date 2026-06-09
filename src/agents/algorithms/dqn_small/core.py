"""DQN 소규모(쉬운) 환경 코어 — top-N 불균형 정류소 + 트럭 1대.

배경:
    구 전체(정류소 수십~수백) + 트럭 3대 환경에서는 model-free DQN이
    예측형 휴리스틱을 못 넘는다(상태·신용할당 폭발). 반대로 "배울 수 있는
    크기"로 줄이면 DQN이 휴리스틱을 추월한다.
    (참고 발견: N=15·1트럭에서 추월 격차 최대.)

이 파일이 하는 일:
    기존 공통 소스(`dqn_core` / `rebalance_env` / `data_loader`)는 일절
    수정하지 않고, 같은 RebalanceEnv 파이프라인 위에서 환경만 축소한다.

    1) 출퇴근 불균형 압력 top-N 정류소만 선택 (train 데이터 기준)
       press = |returns-rentals| 을 아침(07~10시)·저녁(17~21시) 구간에서 합산
    2) 선택한 정류소로 episode 전체를 subset (거리/이동/수요/거치대 일관 슬라이스)
    3) 트럭 수를 줄여(기본 1대) 신용할당을 단순화

    나머지(forecast state 보강, capacity override, candidate Top-K, BC,
    Double DQN, 7일 평가)는 dqn_core와 동일한 wrapper/헬퍼를 재사용한다.

실행 예:
    PYTHONPATH=. python -m src.agents.ours.common.dqn_small_core \\
        --district 영등포구 --processed-dir data/processed_seoul_all \\
        --forecast-path data/forecast_by_gu/demand_forecast_1h_영등포구.parquet \\
        --max-stations 15 --n-trucks 1 --total-timesteps 400000
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, TensorDataset

from src.agents.baselines import get_policy
from src.agents.ours.common.bc_utils import collect_bc_data
from src.agents.ours.common.candidate_actions import maybe_wrap_candidate_actions
from src.agents.ours.common.date_split import compute_split
from src.agents.ours.common.data_overrides import apply_capacity_override, attach_forecast_override
from src.agents.ours.common.future_demand import maybe_wrap_future_demand
from src.agents.ours.common.reward_shaping import maybe_wrap_agent_reward_shaping
from src.agents.masked_dqn import MaskableDQN
from src.agents.ours.common.stochastic_env import StochasticRebalanceEnv
from src.agents.ours.common.vae_latent import attach_vae_latent_override, maybe_wrap_vae_latent
from src.envs.data_loader import EpisodeData, load_episode
from src.envs.rebalance_env import RebalanceEnv


def date_range(start: str, end: str) -> list[str]:
    """시작일부터 종료일까지 날짜 문자열 목록을 만든다."""
    import datetime

    d = datetime.date.fromisoformat(start)
    end_d = datetime.date.fromisoformat(end)
    dates = []
    while d <= end_d:
        dates.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return dates


# dqn_core와 동일한 train/eval 날짜 분할(seed 42, 80/20, 평가 eval pool 전체 73일).
RNG = random.Random(42)
ALL_DATES = date_range("2025-01-01", "2025-12-31")
RNG.shuffle(ALL_DATES)
N_TRAIN = int(len(ALL_DATES) * 0.8)
TRAIN_DATES = ALL_DATES[:N_TRAIN]
EVAL_DATES = sorted(ALL_DATES[N_TRAIN:])  # eval pool 전체(73일)


# 트럭 수 외 나머지 env 설정은 dqn_core ENV_KW와 동일하게 유지한다.
BASE_ENV_KW = dict(
    truck_capacity=20,
    target_fill_ratio=0.5,
    urgent_low_ratio=0.15,
    urgent_high_ratio=0.85,
    urgent_bonus=0.0,
    strict_urgent_mask=True,
    w_travel_km=-0.008,
    w_travel_step=-0.002,
    explore_bonus_scale=0.0,
    shaping_scale=0.0,
    future_demand_horizon=0,
)

# 출퇴근 피크 구간 (10분 step 기준): 아침 07:00~10:00, 저녁 17:00~21:00.
MORNING_WINDOW = slice(42, 60)
EVENING_WINDOW = slice(102, 126)


def build_env_kw(args: argparse.Namespace) -> dict:
    """트럭 수만 args로 덮어쓴 env 설정을 만든다."""
    return dict(BASE_ENV_KW, n_trucks=args.n_trucks)


def build_raw_env(episodes, args: argparse.Namespace, env_kw: dict, seed: int | None):
    """demand-noise 설정에 따라 결정적/확률 RebalanceEnv를 만든다."""
    if getattr(args, "demand_noise", "none") == "poisson":
        return StochasticRebalanceEnv(
            episodes, seed=seed, demand_noise="poisson",
            demand_rate_scale=getattr(args, "demand_rate_scale", 1.0), **env_kw,
        )
    return RebalanceEnv(episodes, seed=seed, **env_kw)


def select_commute_imbalance_stations(episodes: list, n_stations: int) -> np.ndarray:
    """train episode 평균 수요에서 출퇴근 불균형 압력 top-N 정류소 index를 고른다.

    press[i] = |반납-대여| 합 (아침 피크) + |반납-대여| 합 (저녁 피크)
    압력이 큰 정류소일수록 재배치 가치가 크다 → sweet-spot 부분문제를 만든다.
    """
    rentals = np.stack([ep.rentals for ep in episodes]).mean(axis=0)  # (T, N)
    returns = np.stack([ep.returns for ep in episodes]).mean(axis=0)  # (T, N)
    T = rentals.shape[0]
    morning = MORNING_WINDOW if MORNING_WINDOW.stop <= T else slice(0, T)
    evening = EVENING_WINDOW if EVENING_WINDOW.stop <= T else slice(0, T)
    press = (
        np.abs(returns[morning] - rentals[morning]).sum(axis=0)
        + np.abs(returns[evening] - rentals[evening]).sum(axis=0)
    )
    n = min(n_stations, press.shape[0])
    top = np.argsort(press)[::-1][:n]
    return np.sort(top)  # 원래 정류소 순서를 보존해 index를 안정적으로 만든다.


def subset_episode(ep: EpisodeData, sel: np.ndarray) -> EpisodeData:
    """선택한 정류소 index로 EpisodeData의 정류소 축을 일관되게 슬라이스한다."""
    sel = np.asarray(sel, dtype=int)
    return EpisodeData(
        station_ids=[ep.station_ids[i] for i in sel],
        station_coords=ep.station_coords[sel],
        distance_matrix=ep.distance_matrix[np.ix_(sel, sel)],
        travel_steps=ep.travel_steps[np.ix_(sel, sel)],
        capacity=ep.capacity[sel],
        initial_bikes=ep.initial_bikes[sel],
        rentals=ep.rentals[:, sel],
        returns=ep.returns[:, sel],
        timestamps=ep.timestamps,
        dayofweek=ep.dayofweek,
        is_weekend=ep.is_weekend,
        is_holiday=ep.is_holiday,
        is_holiday_eve=ep.is_holiday_eve,
        weather=ep.weather,
    )


def load_episodes(dates: list[str], district: str, processed_dir: str) -> list:
    """날짜 목록을 RebalanceEnv episode 데이터로 변환한다."""
    return [
        load_episode(processed_dir, district=district, episode_start=f"{date} 00:00")
        for date in dates
    ]


def make_env(episodes, args: argparse.Namespace, env_kw: dict, seed: int | None = None, for_eval: bool = False):
    """축소된 환경을 만들고 forecast/shaping/candidate wrapper를 적용한다."""
    env = build_raw_env(episodes, args, env_kw, seed)
    env = maybe_wrap_future_demand(env, args)
    env = maybe_wrap_vae_latent(env, args)
    if not for_eval:
        env = maybe_wrap_agent_reward_shaping(env, args)
    return maybe_wrap_candidate_actions(env, args)


def _eval_seeds(args: argparse.Namespace, base_seed: int) -> list[int]:
    """확률 환경이면 여러 Poisson 실현으로 평균낼 seed 목록을 만든다.

    model/heuristic에 같은 seed를 쓰면 같은 수요 실현·트럭 시작을 공유해
    분산이 줄어든 공정한 paired 비교가 된다.
    """
    n = max(int(getattr(args, "eval_samples", 1)), 1)
    if getattr(args, "demand_noise", "none") != "poisson":
        n = 1  # 결정적 replay는 매번 동일 → 1회면 충분
    return [base_seed + i for i in range(n)]


def evaluate(model, episodes: list, args: argparse.Namespace, env_kw: dict, seed: int) -> tuple[float, list[float]]:
    """고정 평가셋에서 greedy DQN policy의 평균 reward를 계산한다(확률 환경은 다중 실현 평균)."""
    seeds = _eval_seeds(args, seed)
    rewards = []
    for ep in episodes:
        per_sample = []
        for s in seeds:
            env = make_env(ep, args, env_kw, seed=s, for_eval=True)
            obs, _ = env.reset(seed=s)
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=env.action_masks(),
                )
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total += float(reward)
                done = terminated or truncated
            per_sample.append(total)
        rewards.append(float(np.mean(per_sample)))
    return float(np.mean(rewards)), rewards


def evaluate_heuristic(episodes: list, args: argparse.Namespace, env_kw: dict, seed: int) -> tuple[float, list[float]]:
    """같은 축소 데이터에서 most_imbalanced 휴리스틱 reward를 계산한다(model과 같은 실현 공유)."""
    heuristic = get_policy("most_imbalanced")
    seeds = _eval_seeds(args, seed)
    rewards = []
    for ep in episodes:
        per_sample = []
        for s in seeds:
            env = build_raw_env(ep, args, env_kw, seed=s)
            env.reset(seed=s)
            done = False
            total = 0.0
            while not done:
                _, reward, terminated, truncated, _ = env.step(heuristic.act(env))
                total += float(reward)
                done = terminated or truncated
            per_sample.append(total)
        rewards.append(float(np.mean(per_sample)))
    return float(np.mean(rewards)), rewards


def pretrain_behavior_cloning(model, train_episodes: list, args: argparse.Namespace, env_kw: dict) -> dict[str, float]:
    """teacher action을 CrossEntropy로 모방해 DQN Q-network를 먼저 초기화한다."""
    def _make_env(eps, a, seed=None, for_eval=False):
        return make_env(eps, a, env_kw, seed=seed, for_eval=for_eval)

    states, actions, masks = collect_bc_data(train_episodes, args, _make_env)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(masks)),
        batch_size=args.bc_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.q_net.parameters(), lr=args.bc_lr)
    last_loss = 0.0
    last_acc = 0.0
    model.policy.set_training_mode(True)
    for epoch in range(args.bc_epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y, m in loader:
            x = x.to(model.device, dtype=torch.float32)
            y = y.to(model.device, dtype=torch.long)
            m = m.to(model.device, dtype=torch.bool)
            q_values = model.q_net(x)
            masked_q = q_values.masked_fill(~m, -1e9)
            loss = F.cross_entropy(masked_q, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.q_net.parameters(), args.max_bc_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            correct += int((masked_q.argmax(dim=1) == y).sum().item())
            total += len(y)
        last_loss = total_loss / max(total, 1)
        last_acc = correct / max(total, 1)
        if epoch == 0 or (epoch + 1) % max(args.bc_log_every, 1) == 0:
            print(f"  BC epoch {epoch+1}/{args.bc_epochs}: loss={last_loss:.4f}, acc={last_acc:.3f}")
    model.q_net_target.load_state_dict(model.q_net.state_dict())
    return {"bc_samples": float(len(actions)), "bc_loss": last_loss, "bc_acc": last_acc}


def print_eval_table(label: str, heuristic_rewards: list[float], model_rewards: list[float]) -> None:
    """평가 결과를 날짜별 표로 출력한다."""
    print(f"\n=== {label} vs 휴리스틱 ({len(EVAL_DATES)}일) ===")
    print(f"{'날짜':12}{'휴리스틱':>10}{'모델':>10}{'Δ(M-휴)':>9}")
    for date, h, r in zip(EVAL_DATES, heuristic_rewards, model_rewards):
        print(f"{date:12}{h:>10.1f}{r:>10.1f}{r - h:>9.1f}")
    print(
        f"{'평균':12}{np.mean(heuristic_rewards):>10.1f}{np.mean(model_rewards):>10.1f}"
        f"{np.mean(model_rewards) - np.mean(heuristic_rewards):>9.1f}"
    )


def parse_args() -> argparse.Namespace:
    """소규모 DQN 실험 CLI. dqn_core 옵션 + 환경 축소 옵션(--max-stations/--n-trucks)."""
    parser = argparse.ArgumentParser(description="Small-env DQN (top-N imbalance stations + few trucks)")
    parser.add_argument("--district", default="영등포구")
    parser.add_argument("--processed-dir", default="data/processed_seoul_all")
    parser.add_argument("--n-train-dates", type=int, default=200)
    parser.add_argument("--split-mode", choices=["random", "chronological"], default="random",
                        help="random: 1년 셔플 후 80/20. chronological: 1~10월 train / 10~12월 eval(OOD).")
    parser.add_argument("--total-timesteps", type=int, default=400_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--tag", default="dqn_small")
    parser.add_argument("--seed", type=int, default=42)
    # ── 환경 축소 옵션 (이 파일의 핵심) ───────────────────────────────
    parser.add_argument("--max-stations", type=int, default=15,
                        help="출퇴근 불균형 압력 top-N 정류소만 사용(0이면 전체).")
    parser.add_argument("--n-trucks", type=int, default=1, help="트럭 수(축소 환경 기본 1).")
    parser.add_argument("--demand-noise", choices=["none", "poisson"], default="none",
                        help="poisson이면 매 에피소드 수요를 Poisson(기록값)으로 재샘플(확률 환경).")
    parser.add_argument("--demand-rate-scale", type=float, default=1.0,
                        help="Poisson rate에 곱하는 배율(수요 강도 조절, 기본 1.0).")
    parser.add_argument("--eval-samples", type=int, default=1,
                        help="확률 환경 평가 시 날짜별 Poisson 실현 샘플 수(평균). 결정적이면 무시.")
    # ── DQN 하이퍼파라미터 (dqn_core와 동일 기본값) ───────────────────
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--double-q", action=argparse.BooleanOptionalAction, default=True)
    # ── BC 초기화 ─────────────────────────────────────────────────────
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-log-every", type=int, default=5)
    parser.add_argument("--max-bc-grad-norm", type=float, default=10.0)
    parser.add_argument("--bc-only", action="store_true")
    # ── state 보강 / capacity / forecast ─────────────────────────────
    parser.add_argument("--future-mode", default="forecast_projected_travel")
    parser.add_argument("--future-horizon", type=int, default=6)
    parser.add_argument("--capacity-path", default="data/processed/station_capacity.csv")
    parser.add_argument("--capacity-initial-fill-ratio", type=float, default=0.5)
    parser.add_argument("--forecast-path", default="")
    # ── VAE 수요 latent (선택, 기본 off) ──────────────────────────────
    parser.add_argument("--vae-mode", choices=["none", "demand_latent"], default="none",
                        help="demand_latent이면 VAE 수요 latent를 obs에 추가(forecast+VAE).")
    parser.add_argument("--vae-latent-path", default="",
                        help="구별 VAE latent parquet 경로(data/vae_latent_by_gu/...).")
    parser.add_argument("--vae-latent-dim", type=int, default=4)
    # ── candidate action (축소 환경에선 기본 off) ─────────────────────
    parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--candidate-mode", choices=["imbalance", "forecast_imbalance"], default="forecast_imbalance")
    parser.add_argument("--candidate-travel-coef", type=float, default=0.0)
    parser.add_argument("--candidate-zone-mode", choices=["none", "static3"], default="none")
    parser.add_argument("--candidate-zone-count", type=int, default=3)
    parser.add_argument("--candidate-zone-penalty", type=float, default=0.0)
    parser.add_argument("--candidate-feature-mode", choices=["none", "basic"], default="none")
    # ── reward shaping ───────────────────────────────────────────────
    parser.add_argument("--agent-shaping-mode", choices=["projected_imbalance"], default="projected_imbalance")
    parser.add_argument("--agent-shaping-scale", type=float, default=0.0)
    parser.add_argument("--agent-shaping-gamma", type=float, default=0.99)
    parser.add_argument("--rollback-to-best-on-eval", action="store_true")
    parser.add_argument("--finetune-patience", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    return parser.parse_args()


def main() -> None:
    """환경을 top-N 정류소·소수 트럭으로 축소한 뒤 MaskableDQN을 학습/평가한다."""
    args = parse_args()
    env_kw = build_env_kw(args)

    # split-mode에 따라 train/eval 날짜를 정한다. chronological은 1~10월 train,
    # 10~12월 eval(미래 기간 OOD). print_eval_table이 전역 EVAL_DATES를 참조하므로
    # 전역을 재할당해 표 출력 날짜도 함께 맞춘다.
    global TRAIN_DATES, EVAL_DATES
    TRAIN_DATES, EVAL_DATES = compute_split(args.split_mode, seed=args.seed)
    print(f"split-mode={args.split_mode}: train pool {len(TRAIN_DATES)}일, eval {len(EVAL_DATES)}일 "
          f"({EVAL_DATES[0]} ~ {EVAL_DATES[-1]})")

    train_episodes = load_episodes(TRAIN_DATES[: args.n_train_dates], args.district, args.processed_dir)
    eval_episodes = load_episodes(EVAL_DATES, args.district, args.processed_dir)

    full_n = train_episodes[0].n_stations
    if args.max_stations and args.max_stations > 0 and args.max_stations < full_n:
        sel = select_commute_imbalance_stations(train_episodes, args.max_stations)
        train_episodes = [subset_episode(ep, sel) for ep in train_episodes]
        eval_episodes = [subset_episode(ep, sel) for ep in eval_episodes]
        sel_ids = [train_episodes[0].station_ids[i] for i in range(len(sel))]
        print(f"환경 축소: {full_n} → {len(sel)} 정류소 (출퇴근 압력 top-N)")
        print(f"  선택 정류소 id: {sel_ids}")
    else:
        print(f"환경 축소 없음: 전체 {full_n} 정류소 사용")
    print(f"트럭 수: {args.n_trucks}")
    if args.demand_noise == "poisson":
        print(f"수요: Poisson 확률 (rate_scale={args.demand_rate_scale}, eval_samples={args.eval_samples})")
    else:
        print("수요: 결정적 replay (기록값)")

    all_episodes = train_episodes + eval_episodes
    capacity_stats = apply_capacity_override(
        all_episodes,
        args.capacity_path,
        args.capacity_initial_fill_ratio,
    )
    forecast_stats = attach_forecast_override(all_episodes, args.forecast_path)
    vae_stats = attach_vae_latent_override(all_episodes, args.vae_latent_path)

    train_env = DummyVecEnv([lambda: make_env(train_episodes, args, env_kw, seed=args.seed)])
    sample_env = make_env(eval_episodes[0], args, env_kw, seed=args.seed)
    obs_dim = int(sample_env.observation_space.shape[0])
    n_actions = int(sample_env.action_space.n)

    out_dir = Path("logs") / f"dqn_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    heuristic_mean, heuristic_rewards = evaluate_heuristic(eval_episodes, args, env_kw, args.seed)
    print(f"=== DQN(small) | tag={args.tag} | district={args.district} ===")
    print(f"device={args.device}, obs_dim={obs_dim}, n_actions={n_actions}")
    if capacity_stats:
        print(
            "capacity override: "
            f"matched={int(capacity_stats['capacity_matched'])}/{int(capacity_stats['capacity_total'])}, "
            f"mean_capacity={capacity_stats['capacity_mean']:.2f}"
        )
    if forecast_stats:
        print(
            "forecast override: "
            f"matched={int(forecast_stats['forecast_matched'])}/{int(forecast_stats['forecast_total'])}"
        )
    if vae_stats:
        print(
            "vae latent override: "
            f"matched={int(vae_stats['vae_matched'])}/{int(vae_stats['vae_total'])}, "
            f"latent_dim={int(vae_stats['vae_latent_dim'])}"
        )
    print(f"heuristic mean reward: {heuristic_mean:.2f}")

    model = MaskableDQN(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        n_steps=args.n_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs={"net_arch": [args.hidden, args.hidden]},
        seed=args.seed,
        verbose=0,
        device=args.device,
        double_q=args.double_q,
    )

    history = []
    best_reward = -np.inf
    best_step = 0
    best_policy_state = copy.deepcopy(model.policy.state_dict())
    patience_left = args.finetune_patience
    if args.bc_epochs > 0:
        bc_stats = pretrain_behavior_cloning(model, train_episodes, args, env_kw)
        print(
            f"BC done: samples={int(bc_stats['bc_samples'])}, "
            f"loss={bc_stats['bc_loss']:.4f}, acc={bc_stats['bc_acc']:.3f}"
        )
        eval_reward, _ = evaluate(model, eval_episodes, args, env_kw, args.seed)
        history.append({"timesteps": 0, "eval_reward": eval_reward, "stage": "bc"})
        best_reward = eval_reward
        best_policy_state = copy.deepcopy(model.policy.state_dict())
        model.save(out_dir / "best_model")
        print(f"timesteps={0:7d} eval={eval_reward:8.2f} stage=BC")

    if args.bc_only or args.total_timesteps <= 0:
        final_mean, final_rewards = evaluate(model, eval_episodes, args, env_kw, args.seed)
        model.save(out_dir / "final_model")
        np.save(out_dir / "history.npy", np.asarray(history or [{"timesteps": 0, "eval_reward": final_mean}], dtype=object))
        print_eval_table("dqn_small_bc_only", heuristic_rewards, final_rewards)
        return

    steps_done = 0
    while steps_done < args.total_timesteps:
        chunk = min(args.eval_every, args.total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        steps_done += chunk
        eval_reward, eval_rewards = evaluate(model, eval_episodes, args, env_kw, args.seed)
        history.append({"timesteps": steps_done, "eval_reward": eval_reward})
        if eval_reward > best_reward:
            best_reward = eval_reward
            best_step = steps_done
            best_policy_state = copy.deepcopy(model.policy.state_dict())
            patience_left = args.finetune_patience
            model.save(out_dir / "best_model")
        else:
            if args.rollback_to_best_on_eval:
                model.policy.load_state_dict(best_policy_state)
            if args.finetune_patience > 0:
                patience_left -= 1
        print(f"timesteps={steps_done:7d} eval={eval_reward:8.2f}")
        if args.finetune_patience > 0 and patience_left <= 0:
            print(f"fine-tuning early stop: best_step={best_step}, best_reward={best_reward:.2f}")
            break

    final_mean, final_rewards = evaluate(model, eval_episodes, args, env_kw, args.seed)
    model.save(out_dir / "final_model")
    if not history or abs(float(history[-1]["eval_reward"]) - final_mean) > 1e-9:
        history.append({"timesteps": steps_done, "eval_reward": final_mean, "stage": "final"})
    np.save(out_dir / "history.npy", np.asarray(history, dtype=object))

    print(f"best reward: {best_reward:.2f} at timesteps {best_step}")
    print(f"final reward: {final_mean:.2f}")
    print_eval_table("dqn_small_final", heuristic_rewards, final_rewards)


if __name__ == "__main__":
    main()
