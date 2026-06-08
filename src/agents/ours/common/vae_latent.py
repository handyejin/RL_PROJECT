"""VAE latent feature를 observation에 추가하는 agent 전용 wrapper.

목적:
    정류소별 과거 수요 패턴을 VAE로 압축한 latent vector를 RL state에 붙인다.
    원본 환경 파일은 수정하지 않고, agent가 받는 observation만 확장한다.

State 확장:
    s'_t = concat(s_t, z_t)

여기서 z_t는 현재 timestamp와 정류소에 해당하는 VAE latent vector이다.
예를 들어 latent_dim=4, 정류소 수 N이면 observation 뒤에 N*4개 feature가 추가된다.

주의:
    VAE는 action을 직접 고르는 알고리즘이 아니라 state representation을 보강하는
    보조 모델이다. 예측 parquet이 없거나 mode가 none이면 기존 state를 그대로 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class VaeLatentProvider:
    """timestamp/station_id별 VAE latent parquet을 episode 격자로 변환한다.

    입력 parquet 형식:
        station_id, dow, slot, vae_z_0, vae_z_1, ...

    예전 timestamp별 형식도 호환한다:
        t, station_id, vae_z_0, vae_z_1, ...

    학습 스크립트는 같은 요일/시간대의 과거 수요 패턴을 VAE로 압축해 이 파일을 만든다.
    RL agent는 현재 episode의 timestamp와 station_id 순서에 맞춰 latent를 읽는다.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"VAE latent file not found: {self.path}")
        frame = pd.read_parquet(self.path)
        required = {"station_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"VAE latent file missing columns: {sorted(missing)}")

        self.latent_columns = sorted([c for c in frame.columns if c.startswith("vae_z_")])
        if not self.latent_columns:
            raise ValueError("VAE latent file needs columns named vae_z_0, vae_z_1, ...")

        self.uses_timestamp = "t" in frame.columns
        if self.uses_timestamp:
            frame = frame[["t", "station_id", *self.latent_columns]].copy()
            frame["t"] = pd.to_datetime(frame["t"])
            frame = frame.drop_duplicates(["t", "station_id"], keep="last")
            self.frame = frame.set_index(["t", "station_id"]).sort_index()
        else:
            required_profile = {"dow", "slot"}
            missing_profile = required_profile - set(frame.columns)
            if missing_profile:
                raise ValueError(f"VAE latent profile missing columns: {sorted(missing_profile)}")
            frame = frame[["station_id", "dow", "slot", *self.latent_columns]].copy()
            frame["dow"] = frame["dow"].astype(int)
            frame["slot"] = frame["slot"].astype(int)
            frame = frame.drop_duplicates(["station_id", "dow", "slot"], keep="last")
            self.frame = frame.set_index(["station_id", "dow", "slot"]).sort_index()
        self.latent_dim = len(self.latent_columns)

    def grid_for_episode(self, episode) -> tuple[np.ndarray, int, int]:
        """episode의 timestamp/station 순서에 맞춘 (T, N, latent_dim) grid를 만든다."""
        if self.uses_timestamp:
            index = pd.MultiIndex.from_product(
                [episode.timestamps, episode.station_ids],
                names=["t", "station_id"],
            )
        else:
            timestamps = pd.Series(pd.DatetimeIndex(episode.timestamps))
            slots = (timestamps.dt.hour * 6 + (timestamps.dt.minute // 10)).astype(int).to_numpy()
            dows = timestamps.dt.dayofweek.astype(int).to_numpy()
            keys = [
                (station_id, int(dow), int(slot))
                for dow, slot in zip(dows, slots)
                for station_id in episode.station_ids
            ]
            index = pd.MultiIndex.from_tuples(keys, names=["station_id", "dow", "slot"])
        sub = self.frame.reindex(index)
        missing_rows = int(sub[self.latent_columns[0]].isna().sum())
        total_rows = int(len(sub))
        values = sub[self.latent_columns].fillna(0.0).to_numpy(dtype=np.float32)
        grid = values.reshape(len(episode.timestamps), len(episode.station_ids), self.latent_dim)
        return grid.astype(np.float32), total_rows - missing_rows, total_rows


class VaeLatentObservationWrapper(gym.Wrapper):
    """RebalanceEnv observation 뒤에 VAE latent feature를 붙인다."""

    VALID_MODES = {"none", "demand_latent"}

    def __init__(self, env, mode: str = "none", latent_dim: int = 0):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown VAE mode: {mode}")
        super().__init__(env)
        self.mode = mode
        self.latent_dim = int(latent_dim)

        base_space = env.observation_space
        extra_dim = 0 if mode == "none" else env.N * self.latent_dim
        low = np.concatenate(
            [
                np.asarray(base_space.low, dtype=np.float32),
                np.full(extra_dim, -5.0, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.asarray(base_space.high, dtype=np.float32),
                np.full(extra_dim, 5.0, dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.action_space

    def __getattr__(self, name):
        """wrapper에 없는 속성은 원본 env로 위임한다."""
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """원본 env reset 후 observation만 확장한다."""
        obs, info = self.env.reset(*args, **kwargs)
        return self._augment(obs), info

    def step(self, action):
        """원본 env step 후 next observation만 확장한다."""
        obs, reward, done, truncated, info = self.env.step(action)
        return self._augment(obs), reward, done, truncated, info

    def action_masks(self):
        """action mask는 원본 env의 규칙을 그대로 사용한다."""
        return self.env.action_masks()

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return obs
        return np.concatenate([obs.astype(np.float32, copy=False), self._latent_features()])

    def _latent_features(self) -> np.ndarray:
        """현재 step의 정류소별 latent vector를 펼쳐서 반환한다."""
        latent = getattr(self.env.data, "agent_vae_latent", None)
        if latent is None or len(latent) == 0:
            return np.zeros(self.env.N * self.latent_dim, dtype=np.float32)
        idx = min(int(self.env.t), len(latent) - 1)
        return latent[idx].astype(np.float32).reshape(-1)


def attach_vae_latent_override(episodes: list, path: str | Path) -> dict[str, float]:
    """episode data에 VAE latent grid를 붙인다.

    반환 통계:
        vae_matched: episode grid 중 parquet에서 찾은 row 수
        vae_total: 전체 row 수
        vae_latent_dim: latent dimension
    """
    if not path:
        return {}
    provider = VaeLatentProvider(path)
    matched = 0
    total = 0
    for ep in episodes:
        grid, m, t = provider.grid_for_episode(ep)
        ep.agent_vae_latent = grid
        matched += m
        total += t
    return {
        "vae_matched": float(matched),
        "vae_total": float(total),
        "vae_latent_dim": float(provider.latent_dim),
    }


def maybe_wrap_vae_latent(env, args):
    """CLI 옵션에 따라 env를 VAE latent wrapper로 감싼다."""
    mode = getattr(args, "vae_mode", "none")
    if mode == "none":
        return env
    latent = getattr(env.data, "agent_vae_latent", None)
    latent_dim = int(latent.shape[-1]) if latent is not None and len(latent) > 0 else int(getattr(args, "vae_latent_dim", 0))
    if latent_dim <= 0:
        raise ValueError("VAE latent_dim must be positive when --vae-mode is enabled")
    return VaeLatentObservationWrapper(env, mode=mode, latent_dim=latent_dim)
