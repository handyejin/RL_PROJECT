#!/usr/bin/env bash
# DQN-small seed-grid 드라이버
# 사용법: run_seed_grid.sh <seed> <gu1> [gu2 ...]
#   - 각 구를 run_interactive 로 순차 학습 (chronological, 200k, topK15, n_trucks 1)
#   - tag=seed<seed> 로 산출물/로그를 seed별로 분리 → logs/dqn_seed<seed>_dqn_small_<구>
#   - 구별 stdout 로그: logs/runs/seedgrid/seed<seed>_<구>.log
#   - 한 구가 실패해도 다음 구로 계속 진행 (실패 구는 FAILED 목록에 기록)
set -u
cd "$(dirname "$0")/.."

SEED="$1"; shift
GUS=("$@")
OUTDIR="logs/runs/seedgrid"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary_seed${SEED}.log"
: > "$SUMMARY"
FAILED=()

for GU in "${GUS[@]}"; do
  LOG="$OUTDIR/seed${SEED}_${GU}.log"
  echo "==== [seed=${SEED}] ${GU} 시작 $(date +%H:%M:%S) ====" | tee -a "$SUMMARY"
  PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.agents.ours.run_interactive \
    --algorithm dqn_small \
    --district "$GU" \
    --split-mode chronological \
    --timesteps 200000 \
    --eval-freq 20000 \
    --max-stations 15 \
    --n-trucks 1 \
    --seed "$SEED" \
    --tag "seed${SEED}" \
    > "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[seed=${SEED}] ${GU} FAILED (rc=$RC)" | tee -a "$SUMMARY"
    FAILED+=("$GU")
    continue
  fi
  # 평가표의 '평균' 줄(휴리스틱/모델/Δ)을 요약에 기록
  AVG=$(grep "^평균" "$LOG" | tail -1)
  echo "[seed=${SEED}] ${GU} done $(date +%H:%M:%S)  ${AVG}" | tee -a "$SUMMARY"
done

echo "==== seed=${SEED} 전체 완료 $(date +%H:%M:%S) ====" | tee -a "$SUMMARY"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "실패한 구: ${FAILED[*]}" | tee -a "$SUMMARY"
fi
