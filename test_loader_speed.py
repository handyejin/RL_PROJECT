"""Test data loading speed."""
import os
import glob
import pandas as pd
import time

ddarengi_dir = r"C:\RL_PROJECT\data\ddarengi"
pattern = os.path.join(ddarengi_dir, "*.csv")
files = sorted(glob.glob(pattern))

print(f"Found {len(files)} CSV files")

start = time.time()
for i, f in enumerate(files):
    t0 = time.time()
    try:
        df = pd.read_csv(f, encoding='cp949')
        elapsed = time.time() - t0
        print(f"  {i+1}. {os.path.basename(f):50} {len(df):>10} rows  {elapsed:>6.2f}s")
    except Exception as e:
        print(f"  {i+1}. {os.path.basename(f):50} ERROR: {e}")

elapsed = time.time() - start
print(f"\nTotal time: {elapsed:.2f}s")
