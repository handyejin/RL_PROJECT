#!/usr/bin/env bash
# SLA BC 워밍업 효과 검증 (3개 대표 구). no-BC baseline(k12 seed42)과 동일 설정에
# --bc-epochs/--bc-policy future_heuristic 만 추가해 BC 효과를 분리 측정한다.
#   teacher = future_heuristic (SLA류 forecast lookahead)
#   동일: chronological·200k·eval20000·max-stations12·n_trucks1·seed42
#   산출물: logs/dqn_seed42bc_k12_<구>/  · 로그: logs/runs/bctest/<구>.log
#   요약:   logs/runs/bctest/summary.log  (BC-only 평가 + 최종 Δ)
set -u
cd "$(dirname "$0")/.."
GU="$1"
ROOT="$(pwd)"
OUTDIR="logs/runs/bctest"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/${GU}.log"; SUMMARY="$OUTDIR/summary.log"

echo "==== [BC] ${GU} 시작 $(date +%H:%M:%S) ====" >> "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.agents.algorithms.dqn_small.core \
  --processed-dir "$ROOT/data/processed_seoul_all" \
  --district "$GU" \
  --total-timesteps 200000 --eval-every 20000 --n-train-dates 200 --seed 42 \
  --bc-epochs 30 --bc-policy future_heuristic --bc-dates 60 --bc-lr 1e-3 \
  --future-mode forecast_projected_travel --future-horizon 6 \
  --capacity-path "$ROOT/data/processed/station_capacity.csv" \
  --forecast-path "$ROOT/data/forecast_by_gu/demand_forecast_1h_${GU}.parquet" \
  --max-stations 12 --n-trucks 1 --split-mode chronological \
  --tag "seed42bc_k12_${GU}" --device cpu \
  > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[BC] ${GU} FAILED (rc=$RC) $(date +%H:%M:%S)" >> "$SUMMARY"
  tail -5 "$LOG" >> "$SUMMARY"
  exit 0
fi
HEUR=$(grep "heuristic mean reward" "$LOG" | tail -1)
BCROW=$(grep "stage=BC" "$LOG" | tail -1)
FINAL=$(grep "^평균" "$LOG" | tail -1)
echo "[BC] ${GU} done $(date +%H:%M:%S)" >> "$SUMMARY"
echo "    ${HEUR}" >> "$SUMMARY"
echo "    BC-only: ${BCROW}" >> "$SUMMARY"
echo "    최종(평균 휴리스틱/모델/Δ): ${FINAL}" >> "$SUMMARY"
