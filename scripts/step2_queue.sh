#!/usr/bin/env bash
# Step 2: Potential-based shaping 적용 후 DQN/DDQN × 3 seed × 2M.
# 환경: Step 1과 동일 + shaping_scale 추가.
# 사용: bash scripts/step2_queue.sh > logs/step2_queue.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

TS=2000000
EVAL_FREQ=40000
SHAPING=0.1

run_one () {
  local tag=$1
  local seed=$2
  local extra=$3
  echo ""
  echo "===================================="
  echo "  $tag (seed=$seed) $extra  shaping=$SHAPING"
  echo "  start: $(date +'%H:%M:%S')"
  echo "===================================="
  python scripts/train.py \
    --algo masked_dqn $extra \
    --tag "$tag" \
    --seed "$seed" \
    --timesteps "$TS" \
    --eval-freq "$EVAL_FREQ" \
    --shaping-scale "$SHAPING"
}

# DQN 3 seed
run_one step2_dqn_s42  42  ""
run_one step2_dqn_s123 123 ""
run_one step2_dqn_s777 777 ""

# DDQN 3 seed
run_one step2_ddqn_s42  42  "--double-q"
run_one step2_ddqn_s123 123 "--double-q"
run_one step2_ddqn_s777 777 "--double-q"

echo ""
echo "===================================="
echo "  ALL DONE: $(date +'%H:%M:%S')"
echo "===================================="
