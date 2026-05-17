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


def _load_config():
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cfg_path = os.path.join(base, "config", "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

        # load rental history
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ddarengi_dir = os.path.join(base, "data", "ddarengi")
        df = loader.load_rental_history_from_dir(ddarengi_dir)
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


        # action: 0 = noop, 1..n_stations -> move one bike to station idx-1
        self.action_space = spaces.Discrete(self.n_stations + 1)

        # observation: bike counts normalized + time features (sin hour, cos hour)
        obs_len = self.n_stations + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # station capacity
        self.station_capacity = int(self.cfg.get("truck", {}).get("capacity", 20))

        # pre-aggregate events per step for the entire dataset timeline
        self._prepare_step_events()

        self.current_step = 0
        self.bikes = {s: int(self.station_capacity * self.cfg.get("simulation", {}).get("initial_fill_ratio", 0.5)) for s in self.stations}

        # reward weights
        self.r_stock = float(self.cfg.get("reward", {}).get("stockout", -1.0))
        self.r_full = float(self.cfg.get("reward", {}).get("full", -0.8))
        self.r_travel_km = float(self.cfg.get("reward", {}).get("travel_distance_km", -0.01))
        self.r_travel_step = float(self.cfg.get("reward", {}).get("travel_step", -0.005))

    def _prepare_step_events(self):
        # bin events by step index relative to the first timestamp
        df = self.rental_df
        first = df["start_time"].min()
        if pd.isna(first):
            first = pd.Timestamp.now()
        # convert times to step index
        step_seconds = self.step_min * 60

        def to_step(ts):
            if pd.isna(ts):
                return None
            return int((pd.to_datetime(ts) - first).total_seconds() // step_seconds)

        df = df.copy()
        df["rent_step"] = df["start_time"].apply(to_step)
        if "end_time" in df.columns:
            df["return_step"] = df["end_time"].apply(to_step)

        self.max_data_step = int(max(df["rent_step"].dropna().max() if not df["rent_step"].dropna().empty else 0,
                                     df["return_step"].dropna().max() if "return_step" in df.columns and not df["return_step"].dropna().empty else 0))

        # create mapping step -> list of (type, station)
        events = {}
        for _, row in df.iterrows():
            rs = row.get("rent_step")
            ss = row.get("start_station_id")
            if rs is not None and not pd.isna(rs) and ss in self.stations:
                events.setdefault(int(rs), []).append(("rent", str(ss)))
            if "return_step" in row and not pd.isna(row.get("return_step")):
                rts = row.get("return_step")
                es = row.get("end_station_id")
                if rts is not None and es in self.stations:
                    events.setdefault(int(rts), []).append(("return", str(es)))

        self.step_events = events

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
                # approximate travel cost per move
                travel_cost += self.r_travel_km * 0.1  # assume 0.1 km per move

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

        # reward is negative penalties
        reward = stockout * self.r_stock + full * self.r_full + travel_cost + self.r_travel_step

        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        obs = self._get_obs()
        info = {"stockout": stockout, "full": full}
        return obs, float(reward), done, False, info

    def render(self, mode="human"):
        print(f"Step {self.current_step}/{self.steps_per_episode} bikes: {[self.bikes[s] for s in self.stations]}")

    def close(self):
        return None
