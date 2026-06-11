#!/usr/bin/env bash
# from-scratch DQN-small 단일 실행 (topK·timesteps 파라미터화). BC 없음, rollback 없음.
# baseline(k12 seed42 200k)과 동일 설정에서 topK·timesteps 만 바꿔 효과를 측정.
# 사용법: run_fs.sh <topK> <timesteps> <gu>
#   산출물: logs/dqn_fs_k<topK>_t<timesteps>_<gu>/
#   로그:   logs/runs/fs/k<topK>_t<timesteps>_<gu>.log
#   요약:   logs/runs/fs/summary.log
set -u
cd "$(dirname "$0")/.."
TOPK="$1"; STEPS="$2"; GU="$3"
ROOT="$(pwd)"
OUTDIR="logs/runs/fs"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/k${TOPK}_t${STEPS}_${GU}.log"; SUMMARY="$OUTDIR/summary.log"

echo "==== [k${TOPK} t${STEPS}] ${GU} 시작 $(date +%H:%M:%S) ====" >> "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.agents.algorithms.dqn_small.core \
  --processed-dir "$ROOT/data/processed_seoul_all" \
  --district "$GU" \
  --total-timesteps "$STEPS" --eval-every 20000 --n-train-dates 200 --seed 42 \
  --bc-epochs 0 \
  --future-mode forecast_projected_travel --future-horizon 6 \
  --capacity-path "$ROOT/data/processed/station_capacity.csv" \
  --forecast-path "$ROOT/data/forecast_by_gu/demand_forecast_1h_${GU}.parquet" \
  --max-stations "$TOPK" --n-trucks 1 --split-mode chronological \
  --tag "fs_k${TOPK}_t${STEPS}_${GU}" --device cpu \
  > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[k${TOPK} t${STEPS}] ${GU} FAILED (rc=$RC) $(date +%H:%M:%S)" >> "$SUMMARY"
  tail -5 "$LOG" >> "$SUMMARY"
  exit 0
fi
HEUR=$(grep "heuristic mean reward" "$LOG" | tail -1)
BEST=$(grep "best reward" "$LOG" | tail -1)
FINAL=$(grep "^평균" "$LOG" | tail -1)
echo "[k${TOPK} t${STEPS}] ${GU} done $(date +%H:%M:%S)" >> "$SUMMARY"
echo "    ${HEUR} | ${BEST}" >> "$SUMMARY"
echo "    최종(평균 휴리스틱/모델/Δ): ${FINAL}" >> "$SUMMARY"
