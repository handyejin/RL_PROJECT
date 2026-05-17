"""Run a simple rental-history replay using the pipeline modules.

Usage:
    python run_replay.py

Options can be set in `config/default.yaml`.
"""
import os
import json

import yaml

from src.ddarengi_pipeline import loader, replay


def main():
    base = os.path.dirname(__file__)
    cfg_path = os.path.join(base, "config", "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ddarengi_dir = os.path.join(base, "data", "ddarengi")
    print("Loading rental history from:", ddarengi_dir)
    df = loader.load_rental_history_from_dir(ddarengi_dir)

    sim = replay.ReplaySimulator(station_capacity=cfg.get("truck", {}).get("capacity", 20),
                                  initial_fill_ratio=cfg.get("simulation", {}).get("initial_fill_ratio", 0.5))

    split = cfg.get("data", {}).get("split", {})
    eval_start = split.get("eval_start")
    eval_end = split.get("eval_end")

    print("Running replay for evaluation period:", eval_start, "to", eval_end)
    result = sim.run(df, start_time=eval_start, end_time=eval_end)

    out = {
        "stockout": result.stockout,
        "full": result.full,
        "total_events": result.total_events,
    }

    processed_dir = cfg.get("data", {}).get("processed_dir", "data/processed/")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "replay_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Replay finished. Metrics:", out)
    print("Saved metrics to", out_path)


if __name__ == "__main__":
    main()
