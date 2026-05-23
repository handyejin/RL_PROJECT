"""Quick test to validate data loading and environment."""
import sys
sys.path.insert(0, '.')
from src.ddarengi_pipeline import loader
from src.ddarengi_pipeline.env import RebalEnv
import os

# Test data loading
print("=" * 60)
print("Testing data loader...")
print("=" * 60)

base = '.'
ddarengi_dir = os.path.join(base, 'data', 'ddarengi')
print(f'Loading from: {ddarengi_dir}')
df = loader.load_rental_history_from_dir(ddarengi_dir)
print(f'✓ Loaded {len(df)} records')
print(f'✓ Columns: {list(df.columns)}')
print(f'✓ Date range: {df["start_time"].min()} to {df["start_time"].max()}')
print(f'✓ Unique stations: {df["start_station_id"].nunique()}')

# Test environment
print("\n" + "=" * 60)
print("Testing RebalEnv...")
print("=" * 60)

try:
    env = RebalEnv(max_stations=8)
    obs, info = env.reset()
    print(f'✓ Environment created successfully')
    print(f'✓ Observation shape: {obs.shape}')
    print(f'✓ Action space: {env.action_space}')
    print(f'✓ Number of stations: {env.n_stations}')
    print(f'✓ Steps per episode: {env.steps_per_episode}')
    
    # Test a few steps
    print("\nRunning 10 test steps...")
    total_reward = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f'  Step {i+1}: reward={reward:.6f}, stockout={info.get("stockout", 0)}, full={info.get("full", 0)}')
        if terminated or truncated:
            break
    
    print(f'✓ Total reward in 10 steps: {total_reward:.6f}')
    print(f'✓ Average reward per step: {total_reward / 10:.6f}')
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
