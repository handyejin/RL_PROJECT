#!/usr/bin/env bash
# 단일 구 DQN-small 학습 (병렬 드라이버에서 xargs -P 로 호출).
# 사용법: run_seed_one.sh <topK> <seed> <gu>
set -u
cd "$(dirname "$0")/.."
TOPK="$1"; SEED="$2"; GU="$3"
OUTDIR="logs/runs/seedgrid"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/k${TOPK}_seed${SEED}_${GU}.log"
SUMMARY="$OUTDIR/summary_k${TOPK}_seed${SEED}.log"

echo "==== [k=${TOPK} seed=${SEED}] ${GU} 시작 $(date +%H:%M:%S) ====" >> "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.agents.ours.run_interactive \
  --algorithm dqn_small \
  --district "$GU" \
  --split-mode chronological \
  --timesteps 200000 \
  --eval-freq 20000 \
  --max-stations "$TOPK" \
  --n-trucks 1 \
  --seed "$SEED" \
  --tag "seed${SEED}_k${TOPK}" \
  > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[k=${TOPK} seed=${SEED}] ${GU} FAILED (rc=$RC) $(date +%H:%M:%S)" >> "$SUMMARY"
  exit 0
fi
AVG=$(grep "^평균" "$LOG" | tail -1)
echo "[k=${TOPK} seed=${SEED}] ${GU} done $(date +%H:%M:%S)  ${AVG}" >> "$SUMMARY"
