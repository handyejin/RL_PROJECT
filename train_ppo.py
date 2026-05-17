"""Train PPO on the RebalEnv environment.

Usage:
    python train_ppo.py

This script requires `stable-baselines3`, `gymnasium`, and `torch` installed.
"""
import os
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.ddarengi_pipeline.env import RebalEnv


def main():
    base = os.path.dirname(__file__)
    cfg_path = os.path.join(base, "config", "default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print('creating environment...')

    # create env
    env = RebalEnv(max_stations=8)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    ppo_cfg = cfg.get("ppo", {})
    learning_rate = float(ppo_cfg.get("learning_rate", 3e-4))
    total_timesteps = int(cfg.get("training", {}).get("total_timesteps", 10000))

    print('loading model and starting training...')

    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=learning_rate, policy_kwargs=ppo_cfg.get("policy_kwargs", {}))
    
    print('training model for', total_timesteps, 'timesteps...')
    model.learn(total_timesteps=total_timesteps)

    out_dir = os.path.join(base, "models")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "ppo_rebal.zip")
    model.save(model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    main()