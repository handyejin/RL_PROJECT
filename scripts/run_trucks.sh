#!/usr/bin/env bash
# 포화 가설 검증: 트럭 수를 늘리면 대형구 추월이 살아나나?
# --n-trucks 는 학습·평가·휴리스틱 동일 env에 적용 → 공정 비교 자동 보장.
# baseline(k15 seed42 200k h6 n_trucks1)에서 트럭수만 변경.
# 사용법: run_trucks.sh <n_trucks> <gu>
#   산출물: logs/dqn_trucks_n<n>_<gu>/  · 로그: logs/runs/trucks/n<n>_<gu>.log
set -u
cd "$(dirname "$0")/.."
NT="$1"; GU="$2"
ROOT="$(pwd)"
OUTDIR="logs/runs/trucks"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/n${NT}_${GU}.log"; SUMMARY="$OUTDIR/summary.log"

echo "==== [trucks n${NT}] ${GU} 시작 $(date +%H:%M:%S) ====" >> "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.agents.algorithms.dqn_small.core \
  --processed-dir "$ROOT/data/processed_seoul_all" \
  --district "$GU" \
  --total-timesteps 200000 --eval-every 20000 --n-train-dates 200 --seed 42 \
  --bc-epochs 0 \
  --future-mode forecast_projected_travel --future-horizon 6 \
  --capacity-path "$ROOT/data/processed/station_capacity.csv" \
  --forecast-path "$ROOT/data/forecast_by_gu/demand_forecast_1h_${GU}.parquet" \
  --max-stations 15 --n-trucks "$NT" --split-mode chronological \
  --tag "trucks_n${NT}_${GU}" --device cpu \
  > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[trucks n${NT}] ${GU} FAILED (rc=$RC) $(date +%H:%M:%S)" >> "$SUMMARY"
  tail -6 "$LOG" >> "$SUMMARY"; exit 0
fi
HEUR=$(grep "heuristic mean reward" "$LOG" | tail -1)
BEST=$(grep "best reward" "$LOG" | tail -1)
FINAL=$(grep "^평균" "$LOG" | tail -1)
echo "[trucks n${NT}] ${GU} done $(date +%H:%M:%S)" >> "$SUMMARY"
echo "    ${HEUR} | ${BEST}" >> "$SUMMARY"
echo "    최종(평균 휴리스틱/모델/Δ): ${FINAL}" >> "$SUMMARY"
