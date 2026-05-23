import math
import os
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyYAML is required to load config/default.yaml. Install it with `pip install pyyaml`."
    ) from exc

from gymnasium import spaces

from . import loader, replay

# Global cache for rental dataframe to avoid loading multiple times
_rental_df_cache = {}


def _load_config():
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cfg_path = os.path.join(base, "config", "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_rental_df_cached(ddarengi_dir, cache_key="default", max_rows=None, sample_frac=None):
    """Load rental history data with caching to avoid reloading."""
    if cache_key in _rental_df_cache:
        print(f"Using cached rental data (cache_key={cache_key})")
        return _rental_df_cache[cache_key]
    
    print(f"Loading rental history from: {ddarengi_dir}")
    df = loader.load_rental_history_from_dir(
        ddarengi_dir,
        max_rows=max_rows,
        sample_frac=sample_frac,
    )
    _rental_df_cache[cache_key] = df
    return df


class RebalEnv(gym.Env):
    """A minimal Gymnasium environment for rebalancing using replay demand.

    - Observation: vector of normalized bike counts for a small set of stations + time features
    - Action: Discrete(K+1) where 0=NO-OP, i>0 = move one bike to station i-1
    - Reward: negative penalties from config for stockout/full, plus travel costs for moves

    This is intentionally simple and meant as a starting point for PPO training.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, station_ids: Optional[List[str]] = None, max_stations: int = 10):
        self.cfg = _load_config()
        self.step_min = int(self.cfg.get("simulation", {}).get("step_duration_min", 10))
        self.episode_min = int(self.cfg.get("simulation", {}).get("episode_duration_min", 1440))
        self.steps_per_episode = max(1, self.episode_min // self.step_min)

        # load rental history once (with caching)
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ddarengi_dir = os.path.join(base, "data", "ddarengi")
        
        data_cfg = self.cfg.get("data", {})
        max_rows = data_cfg.get("max_rows_to_load")
        sample_frac = data_cfg.get("sample_frac")
        if max_rows is not None:
            max_rows = int(max_rows)
        if sample_frac is not None:
            sample_frac = float(sample_frac)

        df = _load_rental_df_cached(
            ddarengi_dir,
            cache_key=f"default_{max_rows}_{sample_frac}",
            max_rows=max_rows,
            sample_frac=sample_frac,
        )
        print(f"Loaded {len(df)} rental records")
        
        self.rental_df = df

        # Verify that canonical columns exist after loading
        if "start_station_id" not in df.columns:
            raise ValueError(
                f"Column 'start_station_id' not found after loading CSV files. "
                f"Available columns: {list(df.columns)}"
            )

        # choose station list
        all_stations = []
        if "start_station_id" in df.columns:
            all_stations = list(df["start_station_id"].dropna().astype(str).unique())
        
        if station_ids:
            self.stations = [s for s in station_ids if s in all_stations]
        else:
            # pick top `max_stations` most frequent stations
            if not all_stations:
                raise ValueError("No valid stations found in rental history data")
            counts = df["start_station_id"].astype(str).value_counts()
            self.stations = counts.index[:max_stations].tolist()

        self.n_stations = len(self.stations)
        if self.n_stations == 0:
            raise ValueError("No stations available after filtering")

        print(f"Using {self.n_stations} stations: {self.stations}")

        # action: 0 = noop, 1..n_stations -> move one bike to station idx-1
        self.action_space = spaces.Discrete(self.n_stations + 1)

        # observation: bike counts normalized + time features (sin hour, cos hour)
        obs_len = self.n_stations + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # station capacity from config
        self.station_capacity = int(self.cfg.get("truck", {}).get("capacity", 20))
        print(f"Station capacity: {self.station_capacity}")

        # pre-aggregate events per step for the entire dataset timeline
        self._prepare_step_events()

        self.current_step = 0
        initial_fill_ratio = float(self.cfg.get("simulation", {}).get("initial_fill_ratio", 0.5))
        self.bikes = {s: int(self.station_capacity * initial_fill_ratio) for s in self.stations}
        print(f"Initial bike distribution (fill_ratio={initial_fill_ratio}): {self.bikes}")

        # reward weights from config
        self.r_stock = float(self.cfg.get("reward", {}).get("stockout", -1.0))
        self.r_full = float(self.cfg.get("reward", {}).get("full", -0.8))
        self.r_travel_km = float(self.cfg.get("reward", {}).get("travel_distance_km", -0.01))
        self.r_travel_step = float(self.cfg.get("reward", {}).get("travel_step", -0.005))
        
        print(f"Reward weights:")
        print(f"  stockout: {self.r_stock}")
        print(f"  full: {self.r_full}")
        print(f"  travel_km: {self.r_travel_km}")
        print(f"  travel_step: {self.r_travel_step}")

    def _prepare_step_events(self):
        """Prepare step events using vectorized operations for efficiency."""
        df = self.rental_df
        first = df["start_time"].min()
        if pd.isna(first):
            first = pd.Timestamp.now()
        
        # convert times to step index (vectorized)
        step_seconds = self.step_min * 60
        
        df = df.copy()
        df["rent_step"] = ((df["start_time"] - first).dt.total_seconds() // step_seconds).astype('Int64')
        if "end_time" in df.columns:
            df["return_step"] = ((df["end_time"] - first).dt.total_seconds() // step_seconds).astype('Int64')

        # Calculate max step
        max_rent_step = df["rent_step"].max() if not df["rent_step"].isna().all() else 0
        max_return_step = (df["return_step"].max() if "return_step" in df.columns and not df["return_step"].isna().all() else 0)
        self.max_data_step = int(max(max_rent_step, max_return_step))

        print(f"Preparing step events... (max_step={self.max_data_step})")
        
        # create mapping step -> list of (type, station) using groupby (more efficient)
        events = {}
        
        # Process rent events
        rent_events = df[["rent_step", "start_station_id"]].dropna().copy()
        rent_events = rent_events[rent_events["start_station_id"].astype(str).isin(self.stations)]
        for step, group in rent_events.groupby("rent_step"):
            step = int(step)
            events.setdefault(step, []).extend([("rent", str(sid)) for sid in group["start_station_id"].values])
        
        # Process return events
        if "return_step" in df.columns:
            return_events = df[["return_step", "end_station_id"]].dropna().copy()
            return_events = return_events[return_events["end_station_id"].astype(str).isin(self.stations)]
            for step, group in return_events.groupby("return_step"):
                step = int(step)
                events.setdefault(step, []).extend([("return", str(sid)) for sid in group["end_station_id"].values])

        self.step_events = events
        print(f"✓ Prepared events for {len(self.step_events)} steps")

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.bikes = {s: int(self.station_capacity * self.cfg.get("simulation", {}).get("initial_fill_ratio", 0.5)) for s in self.stations}
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        arr = np.array([self.bikes[s] / self.station_capacity for s in self.stations], dtype=np.float32)
        # time-of-day feature approximate
        hour = (self.current_step * self.step_min / 60.0) % 24
        tod_sin = math.sin(2 * math.pi * hour / 24)
        tod_cos = math.cos(2 * math.pi * hour / 24)
        obs = np.concatenate([arr, np.array([tod_sin, tod_cos], dtype=np.float32)])
        return obs

    def step(self, action):
        """Execute one step of the environment.
        
        Args:
            action: 0 = noop, i>0 = move one bike to station i-1
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        # apply action: 0 noop, i>0 move one bike to station i-1
        travel_cost = 0.0
        if action != 0:
            dest = self.stations[action - 1]
            # choose source as station with most bikes (>0)
            src = max(self.stations, key=lambda s: self.bikes.get(s, 0))
            if self.bikes.get(src, 0) > 0 and src != dest:
                self.bikes[src] -= 1
                # instant move (simplified)
                if self.bikes[dest] < self.station_capacity:
                    self.bikes[dest] += 1
                else:
                    # return failed (treated as full)
                    pass
                # cost for moving a bike: includes step cost + distance cost
                travel_cost = self.r_travel_km * 0.1 + self.r_travel_step  # 0.1 km per move
            else:
                # No valid bike to move, but still pay step cost
                travel_cost = self.r_travel_step
        else:
            # No-op action: only pay base step cost
            travel_cost = self.r_travel_step

        # process demand events for this step
        stockout = 0
        full = 0
        evs = self.step_events.get(self.current_step, [])
        for typ, station in evs:
            if typ == "rent":
                if self.bikes.get(station, 0) > 0:
                    self.bikes[station] -= 1
                else:
                    stockout += 1
            elif typ == "return":
                if self.bikes.get(station, 0) < self.station_capacity:
                    self.bikes[station] += 1
                else:
                    full += 1

        # reward is negative penalties + travel costs
        reward = stockout * self.r_stock + full * self.r_full + travel_cost

        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        obs = self._get_obs()
        info = {"stockout": stockout, "full": full, "travel_cost": travel_cost}
        return obs, float(reward), done, False, info

    def render(self, mode="human"):
        print(f"Step {self.current_step}/{self.steps_per_episode} bikes: {[self.bikes[s] for s in self.stations]}")

    def close(self):
        return None
