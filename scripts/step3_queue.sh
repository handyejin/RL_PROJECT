#!/usr/bin/env bash
# Step 3: 정류소별 실제 capacity 적용 (마포구 median 10, min 5, max 40).
# 환경 외 다른 설정은 Step 1 (baseline) 과 동일 — 단순 capacity 효과 측정.
# DQN 1 seed + DDQN 1 seed 만 (빠른 확인용).
# 사용: bash scripts/step3_queue.sh > logs/step3_queue.log 2>&1 &
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
  echo "  $tag (seed=$seed) $extra"
  echo "  start: $(date +'%H:%M:%S')"
  echo "===================================="
  python scripts/train.py \
    --algo masked_dqn $extra \
    --tag "$tag" \
    --seed "$seed" \
    --timesteps "$TS" \
    --eval-freq "$EVAL_FREQ"
}

run_one step3_dqn_s42  42  ""
run_one step3_ddqn_s42 42  "--double-q"

echo ""
echo "===================================="
echo "  ALL DONE: $(date +'%H:%M:%S')"
echo "===================================="
