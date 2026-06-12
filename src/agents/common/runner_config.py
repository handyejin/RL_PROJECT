####################
# 작성자 : 박제영
# 설명   : 우리 실험 runner가 공유하는 기본 경로, 하이퍼파라미터, 실행 명령 생성 유틸.
#          알고리즘 구현과 실행 옵션을 분리해 코드 중복과 설정 불일치를 줄인다.
####################

"""우리 실험 실행기에 공통으로 쓰는 경로, 기본값, 명령 생성 유틸.

이 파일은 알고리즘 자체를 구현하지 않는다. 목적은 `run_interactive.py`와
`run_from_config.py`가 같은 기본 설정과 같은 core module 경로를 사용하게 해서,
Top-K나 forecast/VAE 옵션이 실행기마다 다르게 흘러가는 실수를 줄이는 것이다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DISTRICTS = [
    "강남구",
    "강동구",
    "강북구",
    "강서구",
    "관악구",
    "광진구",
    "구로구",
    "금천구",
    "노원구",
    "도봉구",
    "동대문구",
    "동작구",
    "마포구",
    "서대문구",
    "서초구",
    "성동구",
    "성북구",
    "송파구",
    "양천구",
    "영등포구",
    "용산구",
    "은평구",
    "종로구",
    "중구",
    "중랑구",
]

# 최종 73일 chronological split의 A2C 결과 기준 Best/Worst 3개 구.
# Top-K, VAE, episode 수 ablation을 전체 25개 구 대신 빠르게 비교할 때 사용한다.
A2C_BEST3_DISTRICTS = ["노원구", "송파구", "영등포구"]
A2C_WORST3_DISTRICTS = ["성북구", "서대문구", "관악구"]

ALGORITHM_MODULES = {
    "reinforce": "src.agents.algorithms.reinforce.core",
    "a2c": "src.agents.algorithms.a2c.core",
    "dqn": "src.agents.algorithms.dqn.core",
    "qrdqn": "src.agents.algorithms.qrdqn.core",
    "ppo": "src.agents.algorithms.ppo.core",
    "bandit": "src.agents.algorithms.bandit.core",
}

# YAML에서 값이 빠졌을 때만 쓰는 fallback이다. 보고서 기준 설정은
# `config/ours/*.yaml`에 명시하고, interactive 실행은 이 값을 기본값으로 쓴다.
DEFAULT_RUNNER_VALUES: dict[str, Any] = {
    "algorithm": "a2c",
    "district": "강남구",
    "processed_dir": "data/processed_seoul_all",
    "forecast_dir": "data/forecast_by_gu",
    "vae_latent_dir": "data/vae_latent_by_gu",
    "capacity_path": "data/processed/station_capacity.csv",
    "episodes": 500,
    "total_timesteps": 170_000,
    "eval_every": 50,
    "eval_every_timesteps": 20_000,
    "n_train_dates": 200,
    "split_mode": "chronological",
    "seed": 42,
    "future_mode": "forecast_projected_travel",
    "future_horizon": 6,
    "vae_mode": "none",
    "vae_latent_dim": 4,
    "vae_epochs": 30,
    "vae_hidden": 32,
    "vae_batch_size": 1024,
    "vae_lr": 1e-3,
    "vae_beta": 0.01,
    "candidate_top_k": 12,
    "candidate_mode": "forecast_imbalance",
    "candidate_travel_coef": 0.20,
    "candidate_zone_mode": "static3",
    "candidate_zone_penalty": 1.0,
    "candidate_feature_mode": "basic",
    "ppo_learning_rate": 1e-4,
    "ppo_ent_coef": 0.003,
    "ppo_target_kl": 0.03,
    "ppo_clip_range": 0.1,
    "ppo_n_epochs": 5,
    "ppo_n_steps": 256,
    "ppo_batch_size": 128,
    "dqn_reward_scale": 0.01,
    "dqn_exploration_initial_eps": 0.3,
    "dqn_exploration_fraction": 0.2,
    "dqn_exploration_final_eps": 0.02,
    # QR-DQN 고유: distributional Q 분위수 개수와 quantile-Huber smoothing 폭.
    # core 기본값(200, 1.0)을 그대로 fallback으로 사용한다.
    "qrdqn_n_quantiles": 200,
    "qrdqn_kappa": 1.0,
    "bandit_alpha": 0.5,
    "bandit_l2": 1.0,
    "bandit_reward_scale": 0.01,
    "tag": "interactive",
    "device": "cpu",
    "progress": True,
    "dry_run": False,
}


def project_path(path: str | Path) -> Path:
    """상대경로를 프로젝트 루트 기준 절대경로로 변환한다."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def selected_districts(district: str) -> list[str]:
    """실험 대상 구 목록을 반환한다.

    `ALL`은 25개 구 전체를 의미한다. `A2C_BEST3`, `A2C_WORST3`,
    `A2C_BEST_WORST`는 최종 A2C 결과에서 뽑은 Best/Worst 구 묶음으로,
    ablation 실험을 빠르게 반복하기 위한 별칭이다.
    """
    key = str(district).upper()
    if key == "ALL":
        return DISTRICTS
    if key == "A2C_BEST3":
        return A2C_BEST3_DISTRICTS
    if key == "A2C_WORST3":
        return A2C_WORST3_DISTRICTS
    if key in {"A2C_BEST_WORST", "A2C_BESTWORST"}:
        return A2C_BEST3_DISTRICTS + A2C_WORST3_DISTRICTS
    if "," in str(district):
        return [part.strip() for part in str(district).split(",") if part.strip()]
    return [district]


def subprocess_env() -> dict[str, str]:
    """하위 학습 프로세스에서 프로젝트 import와 실시간 출력이 되도록 환경을 만든다."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


def build_vae_command(args: Any, district: str) -> list[str]:
    """선택한 구에 대한 VAE latent 생성 명령을 만든다."""
    return [
        sys.executable,
        "scripts/train_vae_demand_latent.py",
        "--district",
        district,
        "--processed-dir",
        str(project_path(args.processed_dir)),
        "--out-dir",
        str(project_path(args.vae_latent_dir)),
        "--latent-dim",
        str(args.vae_latent_dim),
        "--epochs",
        str(args.vae_epochs),
        "--hidden",
        str(args.vae_hidden),
        "--batch-size",
        str(args.vae_batch_size),
        "--lr",
        str(args.vae_lr),
        "--beta",
        str(args.vae_beta),
        "--device",
        "mps" if args.device == "mps" else "cpu",
    ] + (["--progress"] if args.progress else ["--no-progress"])


def build_training_command(args: Any, district: str) -> list[str]:
    """선택한 알고리즘/구에 맞는 core 실행 명령을 만든다."""
    module = ALGORITHM_MODULES[args.algorithm]
    forecast_path = project_path(args.forecast_dir) / f"demand_forecast_1h_{district}.parquet"
    vae_latent_path = project_path(args.vae_latent_dir) / f"vae_demand_latent_{district}.parquet"
    topk_label = f"topk{args.candidate_top_k}"
    tag_parts = [args.tag]
    split_label = getattr(args, "split_mode", "random")
    if split_label != "random" and split_label not in args.tag:
        tag_parts.append(split_label)
    if topk_label not in args.tag:
        tag_parts.append(topk_label)
    tag_parts.extend([args.algorithm, district])

    cmd = [
        sys.executable,
        "-m",
        module,
        "--processed-dir",
        str(project_path(args.processed_dir)),
        "--district",
        district,
        "--n-train-dates",
        str(args.n_train_dates),
        "--seed",
        str(args.seed),
        "--future-mode",
        args.future_mode,
        "--future-horizon",
        str(args.future_horizon),
        "--vae-mode",
        args.vae_mode,
        "--vae-latent-path",
        str(vae_latent_path if args.vae_mode != "none" else ""),
        "--vae-latent-dim",
        str(args.vae_latent_dim),
        "--capacity-path",
        str(project_path(args.capacity_path)),
        "--forecast-path",
        str(forecast_path),
        "--candidate-top-k",
        str(args.candidate_top_k),
        "--candidate-mode",
        args.candidate_mode,
        "--candidate-travel-coef",
        str(args.candidate_travel_coef),
        "--candidate-zone-mode",
        args.candidate_zone_mode,
        "--candidate-zone-penalty",
        str(args.candidate_zone_penalty),
        "--candidate-feature-mode",
        args.candidate_feature_mode,
        "--tag",
        "_".join(tag_parts),
        "--device",
        args.device,
    ]

    if args.algorithm in {"reinforce", "a2c"}:
        cmd += [
            "--episodes",
            str(args.episodes),
            "--eval-every",
            str(args.eval_every),
            "--split-mode",
            args.split_mode,
            "--normalize-advantages",
        ]
    else:
        cmd += ["--total-timesteps", str(args.total_timesteps), "--eval-every", str(args.eval_every_timesteps)]

    if args.algorithm == "dqn":
        # 학습 로그와 재현 가이드에서 Double/Dueling DQN 설정이 명확히 드러나도록 명시한다.
        cmd += [
            "--split-mode",
            args.split_mode,
            "--double-q",
            "--dueling-q",
            "--dqn-reward-scale",
            str(args.dqn_reward_scale),
            "--exploration-initial-eps",
            str(args.dqn_exploration_initial_eps),
            "--exploration-fraction",
            str(args.dqn_exploration_fraction),
            "--exploration-final-eps",
            str(args.dqn_exploration_final_eps),
        ]

    if args.algorithm == "qrdqn":
        # QR-DQN은 distributional Q (분위수 회귀)라 reward_scale/dueling 옵션이 없다.
        # exploration eps와 Double-Q는 DQN과 공통, n_quantiles/kappa만 추가로 노출한다.
        cmd += [
            "--split-mode",
            args.split_mode,
            "--double-q",
            "--exploration-initial-eps",
            str(args.dqn_exploration_initial_eps),
            "--exploration-fraction",
            str(args.dqn_exploration_fraction),
            "--exploration-final-eps",
            str(args.dqn_exploration_final_eps),
            "--n-quantiles",
            str(args.qrdqn_n_quantiles),
            "--kappa",
            str(args.qrdqn_kappa),
        ]

    if args.algorithm == "ppo":
        # Top-K rank action은 state마다 의미가 바뀌므로 PPO update를 보수적으로 제한한다.
        cmd += [
            "--split-mode",
            args.split_mode,
            "--learning-rate",
            str(args.ppo_learning_rate),
            "--ent-coef",
            str(args.ppo_ent_coef),
            "--target-kl",
            str(args.ppo_target_kl),
            "--clip-range",
            str(args.ppo_clip_range),
            "--n-epochs",
            str(args.ppo_n_epochs),
            "--n-steps",
            str(args.ppo_n_steps),
            "--batch-size",
            str(args.ppo_batch_size),
        ]

    if args.algorithm == "bandit":
        cmd += [
            "--split-mode",
            args.split_mode,
            "--bandit-alpha",
            str(args.bandit_alpha),
            "--bandit-l2",
            str(args.bandit_l2),
            "--bandit-reward-scale",
            str(args.bandit_reward_scale),
        ]

    if args.progress:
        cmd.append("--progress")
    return cmd


def ensure_vae_inputs(args: Any) -> bool:
    """VAE latent 생성에 필요한 전처리 파일이 있는지 확인한다."""
    processed_dir = project_path(args.processed_dir)
    missing = [
        str(processed_dir / name)
        for name in ["stations.parquet", "demand_10min.parquet"]
        if not (processed_dir / name).exists()
    ]
    if missing:
        print("\nVAE latent 생성에 필요한 파일이 없습니다.")
        for path in missing:
            print(f"  - {path}")
        print("먼저 서울 전체 전처리를 실행하세요.")
        return False
    return True


def ensure_training_inputs(args: Any, district: str) -> bool:
    """학습에 필요한 전처리/forecast/VAE 파일이 있는지 확인한다."""
    processed_dir = project_path(args.processed_dir)
    forecast_path = project_path(args.forecast_dir) / f"demand_forecast_1h_{district}.parquet"
    vae_latent_path = project_path(args.vae_latent_dir) / f"vae_demand_latent_{district}.parquet"
    required = [processed_dir / "stations.parquet", processed_dir / "trips.parquet", forecast_path]
    if args.vae_mode != "none":
        required.append(vae_latent_path)

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"\n[{district}] 필요한 파일이 없습니다.")
        for path in missing:
            print(f"  - {path}")
        print("전처리/수요예측 생성 후 다시 실행하세요.")
        return False
    return True
