#!/usr/bin/env bash
# BC 워밍업 수정 레시피 검증 (3구). v1 대비 변경점:
#   --exploration-initial-eps 0.1   (1.0 → 0.1: BC 정책을 랜덤이 지우지 않게)
#   --rollback-to-best-on-eval      (eval 악화 시 best 체크포인트로 복귀)
#   --learning-rate 5e-5            (1e-4 → 5e-5: 미세조정 안정화)
# 나머지는 v1/baseline과 동일: chronological·200k·eval20000·k12·n_trucks1·seed42,
#   teacher=future_heuristic, bc-epochs 30, bc-dates 60
#   산출물: logs/dqn_seed42bc2_k12_<구>/  · 로그: logs/runs/bctest2/<구>.log
set -u
cd "$(dirname "$0")/.."
GU="$1"
ROOT="$(pwd)"
OUTDIR="logs/runs/bctest2"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/${GU}.log"; SUMMARY="$OUTDIR/summary.log"

echo "==== [BC2] ${GU} 시작 $(date +%H:%M:%S) ====" >> "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.agents.algorithms.dqn_small.core \
  --processed-dir "$ROOT/data/processed_seoul_all" \
  --district "$GU" \
  --total-timesteps 200000 --eval-every 20000 --n-train-dates 200 --seed 42 \
  --bc-epochs 30 --bc-policy future_heuristic --bc-dates 60 --bc-lr 1e-3 \
  --exploration-initial-eps 0.1 --rollback-to-best-on-eval --learning-rate 5e-5 \
  --future-mode forecast_projected_travel --future-horizon 6 \
  --capacity-path "$ROOT/data/processed/station_capacity.csv" \
  --forecast-path "$ROOT/data/forecast_by_gu/demand_forecast_1h_${GU}.parquet" \
  --max-stations 12 --n-trucks 1 --split-mode chronological \
  --tag "seed42bc2_k12_${GU}" --device cpu \
  > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[BC2] ${GU} FAILED (rc=$RC) $(date +%H:%M:%S)" >> "$SUMMARY"
  tail -5 "$LOG" >> "$SUMMARY"
  exit 0
fi
HEUR=$(grep "heuristic mean reward" "$LOG" | tail -1)
BCROW=$(grep "stage=BC" "$LOG" | tail -1)
BEST=$(grep "best reward" "$LOG" | tail -1)
FINAL=$(grep "^평균" "$LOG" | tail -1)
echo "[BC2] ${GU} done $(date +%H:%M:%S)" >> "$SUMMARY"
echo "    ${HEUR}" >> "$SUMMARY"
echo "    BC-only: ${BCROW}" >> "$SUMMARY"
echo "    ${BEST}" >> "$SUMMARY"
echo "    최종(평균 휴리스틱/모델/Δ): ${FINAL}" >> "$SUMMARY"
