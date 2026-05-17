from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class ReplayResult:
    stockout: int
    full: int
    total_events: int
    final_bikes: Dict[str, int]


class ReplaySimulator:
    """Simple event-based replay simulator for rental history.

    This simulator does NOT implement truck rebalancing yet. It replays
    rentals and counts stockout (rent attempt at empty station) and
    full (return attempt at full station) events given station capacities
    and an initial fill ratio.
    """

    def __init__(self, station_capacity: int = 20, initial_fill_ratio: float = 0.5):
        self.station_capacity = station_capacity
        self.initial_fill_ratio = initial_fill_ratio

    def _initialize_bikes(self, stations):
        bikes = {s: int(self.station_capacity * self.initial_fill_ratio) for s in stations}
        return bikes

    def run(self, rentals: pd.DataFrame, start_time=None, end_time=None) -> ReplayResult:
        df = rentals
        if start_time is not None:
            df = df[df["start_time"] >= pd.to_datetime(start_time)]
        if end_time is not None:
            df = df[df["start_time"] <= pd.to_datetime(end_time)]

        # build set of stations
        stations = set()
        if "start_station_id" in df.columns:
            stations.update(df["start_station_id"].dropna().astype(str).unique().tolist())
        if "end_station_id" in df.columns:
            stations.update(df["end_station_id"].dropna().astype(str).unique().tolist())

        bikes = self._initialize_bikes(stations)

        # build events
        events = []
        for _, row in df.iterrows():
            s = row.get("start_station_id")
            e = row.get("end_station_id")
            st = row.get("start_time")
            et = row.get("end_time")
            if pd.isna(st):
                continue
            events.append((st, "rent", str(s) if not pd.isna(s) else None))
            if not pd.isna(et) and e is not None:
                events.append((et, "return", str(e)))

        events.sort(key=lambda x: x[0])

        stockout = 0
        full = 0
        total = 0

        for ts, typ, station in events:
            if station is None:
                continue
            total += 1
            if typ == "rent":
                if bikes.get(station, 0) > 0:
                    bikes[station] -= 1
                else:
                    stockout += 1
            elif typ == "return":
                if bikes.get(station, 0) < self.station_capacity:
                    bikes[station] += 1
                else:
                    full += 1

        return ReplayResult(stockout=stockout, full=full, total_events=total, final_bikes=bikes)
