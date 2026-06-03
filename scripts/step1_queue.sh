#!/usr/bin/env bash
# Step 1: 6개 run 순차 학습 (DQN/DDQN × seed 42/123/777, 각 2M step).
# 사용: bash scripts/step1_queue.sh > logs/step1_queue.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

TS=2000000
EVAL_FREQ=40000

run_one () {
  local tag=$1
  local seed=$2
  local extra=$3
  echo ""
  echo "===================================="
  echo "  $tag (seed=$seed) $extra  timesteps=$TS"
  echo "  start: $(date +'%H:%M:%S')"
  echo "===================================="
  python scripts/train.py \
    --algo masked_dqn $extra \
    --tag "$tag" \
    --seed "$seed" \
    --timesteps "$TS" \
    --eval-freq "$EVAL_FREQ"
}

# DQN 3 seed
run_one step1_dqn_s42  42  ""
run_one step1_dqn_s123 123 ""
run_one step1_dqn_s777 777 ""

# DDQN 3 seed
run_one step1_ddqn_s42  42  "--double-q"
run_one step1_ddqn_s123 123 "--double-q"
run_one step1_ddqn_s777 777 "--double-q"

echo ""
echo "===================================="
echo "  ALL DONE: $(date +'%H:%M:%S')"
echo "===================================="
