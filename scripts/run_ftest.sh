#!/usr/bin/env bash
# DQN-small 개선 실험: 학습연장(400k)+best롤백+forecast호라이즌12.
# baseline(k15 seed42 200k h6)과 비교. 휴리스틱은 forecast 미사용·raw env라 불변.
# 사용법: run_ftest.sh <gu>
#   변경점: --total-timesteps 400000  --rollback-to-best-on-eval  --future-horizon 12
#   고정:   k15·seed42·eval20000·n_trucks1·chronological·forecast_projected_travel
#   산출물: logs/dqn_ftest_h12_t400k_<gu>/  · 로그: logs/runs/ftest/<gu>.log
set -u
cd "$(dirname "$0")/.."
GU="$1"
ROOT="$(pwd)"
OUTDIR="logs/runs/ftest"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/${GU}.log"; SUMMARY="$OUTDIR/summary.log"

echo "==== [ftest h12 t400k] ${GU} 시작 $(date +%H:%M:%S) ====" >> "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.agents.algorithms.dqn_small.core \
  --processed-dir "$ROOT/data/processed_seoul_all" \
  --district "$GU" \
  --total-timesteps 400000 --eval-every 20000 --n-train-dates 200 --seed 42 \
  --bc-epochs 0 --rollback-to-best-on-eval \
  --future-mode forecast_projected_travel --future-horizon 12 \
  --capacity-path "$ROOT/data/processed/station_capacity.csv" \
  --forecast-path "$ROOT/data/forecast_by_gu/demand_forecast_1h_${GU}.parquet" \
  --max-stations 15 --n-trucks 1 --split-mode chronological \
  --tag "ftest_h12_t400k_${GU}" --device cpu \
  > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[ftest] ${GU} FAILED (rc=$RC) $(date +%H:%M:%S)" >> "$SUMMARY"
  tail -6 "$LOG" >> "$SUMMARY"; exit 0
fi
HEUR=$(grep "heuristic mean reward" "$LOG" | tail -1)
BEST=$(grep "best reward" "$LOG" | tail -1)
FINAL=$(grep "^평균" "$LOG" | tail -1)
echo "[ftest] ${GU} done $(date +%H:%M:%S)" >> "$SUMMARY"
echo "    ${HEUR} | ${BEST}" >> "$SUMMARY"
echo "    최종(평균 휴리스틱/모델/Δ): ${FINAL}" >> "$SUMMARY"
