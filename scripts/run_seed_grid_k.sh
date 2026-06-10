#!/usr/bin/env bash
# DQN-small seed-grid 드라이버 (topK 파라미터화 버전)
# 사용법: run_seed_grid_k.sh <topK> <seed> <gu1> [gu2 ...]
#   - max-stations=<topK> 로 각 구를 chronological·200k·eval20000·n_trucks1 학습
#   - tag=seed<seed>_k<topK> 로 산출물/로그를 topK·seed별로 분리
#       산출물: logs/dqn_seed<seed>_k<topK>_dqn_small_<구>
#       구별 로그: logs/runs/seedgrid/k<topK>_seed<seed>_<구>.log
#       요약: logs/runs/seedgrid/summary_k<topK>_seed<seed>.log
#   - 한 구가 실패해도 다음 구로 계속 진행
set -u
cd "$(dirname "$0")/.."

TOPK="$1"; shift
SEED="$1"; shift
GUS=("$@")
OUTDIR="logs/runs/seedgrid"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary_k${TOPK}_seed${SEED}.log"
: > "$SUMMARY"
FAILED=()

for GU in "${GUS[@]}"; do
  LOG="$OUTDIR/k${TOPK}_seed${SEED}_${GU}.log"
  echo "==== [k=${TOPK} seed=${SEED}] ${GU} 시작 $(date +%H:%M:%S) ====" | tee -a "$SUMMARY"
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
    echo "[k=${TOPK} seed=${SEED}] ${GU} FAILED (rc=$RC)" | tee -a "$SUMMARY"
    FAILED+=("$GU")
    continue
  fi
  AVG=$(grep "^평균" "$LOG" | tail -1)
  echo "[k=${TOPK} seed=${SEED}] ${GU} done $(date +%H:%M:%S)  ${AVG}" | tee -a "$SUMMARY"
done

echo "==== k=${TOPK} seed=${SEED} 전체 완료 $(date +%H:%M:%S) ====" | tee -a "$SUMMARY"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "실패한 구: ${FAILED[*]}" | tee -a "$SUMMARY"
fi
