"""Run evaluation using the trained PPO model.

Usage:
    python run_replay.py

This script:
1. Loads the trained PPO model
2. Evaluates it on the eval dataset
3. Saves metrics to replay_metrics.json
"""
import os
import json

import yaml
import numpy as np

from stable_baselines3 import PPO
from src.ddarengi_pipeline.env import RebalEnv


def main():
    base = os.path.dirname(__file__)
    cfg_path = os.path.join(base, "config", "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_path = os.path.join(base, "models", "ppo_rebal.zip")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train_ppo.py first to train the model.")
        return

    print(f"Loading trained model from: {model_path}")
    model = PPO.load(model_path)

    print("Creating evaluation environment...")
    env = RebalEnv(max_stations=8)

    print("Running evaluation with trained model...")
    obs, _ = env.reset()
    
    total_stockout = 0
    total_full = 0
    total_reward = 0
    total_steps = 0
    
    done = False
    while not done:
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_stockout += info.get("stockout", 0)
        total_full += info.get("full", 0)
        total_steps += 1
        
        done = terminated or truncated

    print(f"\nEvaluation Results:")
    print(f"  Total steps: {total_steps}")
    print(f"  Total stockout events: {total_stockout}")
    print(f"  Total full events: {total_full}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Avg reward per step: {total_reward / max(1, total_steps):.6f}")

    # Save metrics
    out = {
        "stockout": total_stockout,
        "full": total_full,
        "total_reward": float(total_reward),
        "total_steps": total_steps,
        "avg_reward_per_step": float(total_reward / max(1, total_steps)),
    }

    processed_dir = cfg.get("data", {}).get("processed_dir", "data/processed/")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "replay_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
