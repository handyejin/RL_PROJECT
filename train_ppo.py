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

    print('Loading configuration from', cfg_path)
    print('Creating environment...')

    # create env
    env = RebalEnv(max_stations=8)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    # extract PPO hyperparameters from config
    ppo_cfg = cfg.get("ppo", {})
    learning_rate = float(ppo_cfg.get("learning_rate", 3e-4))
    n_steps = int(ppo_cfg.get("n_steps", 2048))
    batch_size = int(ppo_cfg.get("batch_size", 64))
    n_epochs = int(ppo_cfg.get("n_epochs", 10))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_range = float(ppo_cfg.get("clip_range", 0.2))
    ent_coef = float(ppo_cfg.get("ent_coef", 0.0))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
    policy_kwargs = ppo_cfg.get("policy_kwargs", {})
    
    total_timesteps = int(cfg.get("training", {}).get("total_timesteps", 500000))
    log_interval = int(cfg.get("training", {}).get("log_interval", 10))
    
    print(f'PPO Config:')
    print(f'  learning_rate: {learning_rate}')
    print(f'  n_steps: {n_steps}')
    print(f'  batch_size: {batch_size}')
    print(f'  n_epochs: {n_epochs}')
    print(f'  gamma: {gamma}')
    print(f'  gae_lambda: {gae_lambda}')
    print(f'  policy_kwargs: {policy_kwargs}')
    print(f'  total_timesteps: {total_timesteps}')

    print('Loading model and starting training...')

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs
    )
    
    print(f'Training model for {total_timesteps} timesteps...')
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

    out_dir = os.path.join(base, "models")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "ppo_rebal.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()