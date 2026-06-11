"""우리 실험용 YAML 실행기.

팀원 공통 `config/default.yaml`과 `scripts/train.py`를 수정하지 않고,
`config/ours/*.yaml`에 적은 설정으로 REINFORCE/A2C/DQN/PPO/Bandit 실험을 실행한다.

YAML에서 관리하는 핵심 값:
    - algorithm, district
    - candidate_action.top_k
    - forecast/capacity 경로
    - DQN/PPO 하이퍼파라미터

주의:
    보고서 기준 설정은 config/ours/*.yaml에 둔다. 아래 DEFAULTS는 YAML에서
    값이 누락되었을 때 실행이 바로 깨지지 않게 하는 fallback일 뿐이다.

예:
    PYTHONPATH=. python -m src.agents.run_from_config \
      --config config/ours/dqn_topk3.yaml
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from src.agents.common.runner_config import (
    DEFAULT_RUNNER_VALUES,
    PROJECT_ROOT,
    build_training_command,
    ensure_training_inputs,
    selected_districts,
    subprocess_env,
)


# 보고서 기준 하이퍼파라미터는 YAML 파일에 명시한다.
# 이 dict는 YAML 누락값을 채우는 안전 fallback으로만 사용한다.
DEFAULTS: dict[str, Any] = {**DEFAULT_RUNNER_VALUES, "algorithm": "dqn", "tag": "yaml"}


def _load_yaml(path: Path) -> dict[str, Any]:
    """YAML 파일을 읽어 dict로 반환한다."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def _apply_section(values: dict[str, Any], section: dict[str, Any], mapping: dict[str, str]) -> None:
    """YAML section의 key를 runner option 이름으로 옮긴다."""
    for src_key, dest_key in mapping.items():
        if src_key in section:
            values[dest_key] = section[src_key]


def build_args(config: dict[str, Any], cli: argparse.Namespace) -> SimpleNamespace:
    """YAML 설정과 CLI override를 합쳐 run_interactive 호환 args를 만든다."""
    values = dict(DEFAULTS)

    if "algorithm" in config:
        values["algorithm"] = config["algorithm"]
    if "district" in config:
        values["district"] = config["district"]

    _apply_section(
        values,
        config.get("data", {}),
        {
            "processed_dir": "processed_dir",
            "forecast_dir": "forecast_dir",
            "vae_latent_dir": "vae_latent_dir",
            "capacity_path": "capacity_path",
        },
    )
    _apply_section(
        values,
        config.get("training", {}),
        {
            "episodes": "episodes",
            "total_timesteps": "total_timesteps",
            "eval_every": "eval_every",
            "eval_every_timesteps": "eval_every_timesteps",
            "n_train_dates": "n_train_dates",
            "seed": "seed",
            "split_mode": "split_mode",
            "tag": "tag",
            "device": "device",
            "progress": "progress",
        },
    )
    _apply_section(
        values,
        config.get("state", {}),
        {
            "future_mode": "future_mode",
            "future_horizon": "future_horizon",
            "vae_mode": "vae_mode",
            "vae_latent_dim": "vae_latent_dim",
        },
    )
    _apply_section(
        values,
        config.get("candidate_action", {}),
        {
            "top_k": "candidate_top_k",
            "mode": "candidate_mode",
            "travel_coef": "candidate_travel_coef",
            "zone_mode": "candidate_zone_mode",
            "zone_penalty": "candidate_zone_penalty",
            "feature_mode": "candidate_feature_mode",
        },
    )
    _apply_section(
        values,
        config.get("dqn", {}),
        {
            "reward_scale": "dqn_reward_scale",
            "exploration_initial_eps": "dqn_exploration_initial_eps",
            "exploration_fraction": "dqn_exploration_fraction",
            "exploration_final_eps": "dqn_exploration_final_eps",
        },
    )
    _apply_section(
        values,
        config.get("ppo", {}),
        {
            "learning_rate": "ppo_learning_rate",
            "ent_coef": "ppo_ent_coef",
            "target_kl": "ppo_target_kl",
            "clip_range": "ppo_clip_range",
            "n_epochs": "ppo_n_epochs",
            "n_steps": "ppo_n_steps",
            "batch_size": "ppo_batch_size",
        },
    )
    _apply_section(
        values,
        config.get("bandit", {}),
        {
            "alpha": "bandit_alpha",
            "l2": "bandit_l2",
            "reward_scale": "bandit_reward_scale",
        },
    )

    for key in ["algorithm", "district", "tag", "device"]:
        cli_value = getattr(cli, key, None)
        if cli_value:
            values[key] = cli_value
    if cli.seed is not None:
        values["seed"] = cli.seed
    if cli.candidate_top_k is not None:
        values["candidate_top_k"] = cli.candidate_top_k
    if cli.dry_run:
        values["dry_run"] = True

    return SimpleNamespace(**values)


def parse_args() -> argparse.Namespace:
    """YAML runner CLI 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="Run our RL experiment from YAML config.")
    parser.add_argument("--config", required=True, help="예: config/ours/dqn_topk3.yaml")
    parser.add_argument("--algorithm", choices=["reinforce", "a2c", "dqn", "ppo", "bandit"], default="")
    parser.add_argument("--district", default="", help="YAML district override. ALL 가능.")
    parser.add_argument("--candidate-top-k", type=int, default=None, help="YAML top_k override.")
    parser.add_argument("--tag", default="", help="YAML tag override.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="")
    parser.add_argument("--seed", type=int, default=None, help="YAML seed override.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """YAML 설정으로 선택한 구 또는 25개 구 전체를 순차 실행한다."""
    cli = parse_args()
    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    args = build_args(_load_yaml(config_path), cli)

    districts = selected_districts(args.district)
    print(f"[config] {config_path}")
    print(f"algorithm={args.algorithm}, districts={', '.join(districts)}")
    print(
        "candidate="
        f"top_k={args.candidate_top_k}, mode={args.candidate_mode}, "
        f"travel_coef={args.candidate_travel_coef}, zone={args.candidate_zone_mode}"
    )

    for index, district in enumerate(districts, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"[{index}/{len(districts)}] {district} 실행", flush=True)
        if not ensure_training_inputs(args, district):
            continue
        cmd = build_training_command(args, district)
        print("명령:", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        code = subprocess.run(cmd, cwd=PROJECT_ROOT, env=subprocess_env()).returncode
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
