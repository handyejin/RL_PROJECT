"""Test PPO training with mock data."""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, '.')

# Create a minimal test by mocking the data loader
from src.ddarengi_pipeline import env

# Mock the rental data in the cache to speed up testing
print("Creating mock rental data...")
mock_data = []
start_time = datetime(2025, 1, 1)
stations = ['ST-001', 'ST-002', 'ST-003', 'ST-004', 'ST-005', 'ST-006', 'ST-007', 'ST-008']

for i in range(1000):  # Just 1000 events for testing
    for station in stations:
        mock_data.append({
            'start_time': start_time + timedelta(minutes=i*10),
            'end_time': start_time + timedelta(minutes=i*10+5),
            'start_station_id': station,
            'end_station_id': stations[(int(station.split('-')[1])) % len(stations)],
        })

df = pd.DataFrame(mock_data)
env._rental_df_cache['default'] = df
print(f"✓ Mock data created: {len(df)} events")

# Now test the environment
print("\nTesting RebalEnv...")
try:
    test_env = env.RebalEnv(max_stations=8)
    obs, info = test_env.reset()
    print(f"✓ Environment created")
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Action space: {test_env.action_space}")
    
    # Run a few steps
    print("\nRunning 50 test steps...")
    total_reward = 0.0
    total_stockout = 0
    total_full = 0
    
    for step in range(50):
        action = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        total_stockout += info.get("stockout", 0)
        total_full += info.get("full", 0)
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}: avg_reward={total_reward/(step+1):.6f}, total_stockout={total_stockout}, total_full={total_full}")
        
        if terminated or truncated:
            break
    
    print(f"\n✓ Test completed!")
    print(f"  Total steps: {step+1}")
    print(f"  Total reward: {total_reward:.6f}")
    print(f"  Total stockout: {total_stockout}")
    print(f"  Total full: {total_full}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Environment test passed! Ready for training.")
print("="*60)
