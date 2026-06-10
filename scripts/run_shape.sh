#!/usr/bin/env bash
# DQN-small 보상 셰이핑(PBRS) 효과 검증. baseline(k15 seed42 200k h6)에
# --agent-shaping-mode projected_imbalance --agent-shaping-scale <S> 만 추가.
# 셰이핑은 학습에만 적용(make_env for_eval=False), 평가는 clean → 휴리스틱 비교 공정.
# 사용법: run_shape.sh <scale> <gu>
#   산출물: logs/dqn_shape_s<scale>_<gu>/  · 로그: logs/runs/shape/s<scale>_<gu>.log
set -u
cd "$(dirname "$0")/.."
SCALE="$1"; GU="$2"
ROOT="$(pwd)"
OUTDIR="logs/runs/shape"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/s${SCALE}_${GU}.log"; SUMMARY="$OUTDIR/summary.log"

echo "==== [shape s${SCALE}] ${GU} 시작 $(date +%H:%M:%S) ====" >> "$SUMMARY"
PYTHONUNBUFFERED=1 PYTHONPATH=. python -m src.agents.algorithms.dqn_small.core \
  --processed-dir "$ROOT/data/processed_seoul_all" \
  --district "$GU" \
  --total-timesteps 200000 --eval-every 20000 --n-train-dates 200 --seed 42 \
  --bc-epochs 0 \
  --agent-shaping-mode projected_imbalance --agent-shaping-scale "$SCALE" --agent-shaping-gamma 0.99 \
  --future-mode forecast_projected_travel --future-horizon 6 \
  --capacity-path "$ROOT/data/processed/station_capacity.csv" \
  --forecast-path "$ROOT/data/forecast_by_gu/demand_forecast_1h_${GU}.parquet" \
  --max-stations 15 --n-trucks 1 --split-mode chronological \
  --tag "shape_s${SCALE}_${GU}" --device cpu \
  > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[shape s${SCALE}] ${GU} FAILED (rc=$RC) $(date +%H:%M:%S)" >> "$SUMMARY"
  tail -6 "$LOG" >> "$SUMMARY"; exit 0
fi
HEUR=$(grep "heuristic mean reward" "$LOG" | tail -1)
BEST=$(grep "best reward" "$LOG" | tail -1)
FINAL=$(grep "^평균" "$LOG" | tail -1)
echo "[shape s${SCALE}] ${GU} done $(date +%H:%M:%S)" >> "$SUMMARY"
echo "    ${HEUR} | ${BEST}" >> "$SUMMARY"
echo "    최종(평균 휴리스틱/모델/Δ): ${FINAL}" >> "$SUMMARY"
