#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$ROOT_DIR/lstm_autoencoder.py"
BASE_RESULTS_DIR="$ROOT_DIR/results"
MASTER_LOG="$BASE_RESULTS_DIR/lstm_nohup_master.log"

mkdir -p "$BASE_RESULTS_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inicio de ejecucion secuencial" | tee -a "$MASTER_LOG"

# Formato:
# run_id seq_length stride units middle_units batch epoch dropout l2 lr n_splits
RUNS=(
  "1 8 8 128 96 64 60 0.15 1e-5 1e-4 5"
)

for run in "${RUNS[@]}"; do
  read -r run_id seq_len stride units middle_units batch epoch dropout l2 lr n_splits <<< "$run"
  out_dir="$BASE_RESULTS_DIR/lstm_autoencoder${run_id}"

  mkdir -p "$out_dir"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${run_id} -> ${out_dir}" | tee -a "$MASTER_LOG"
  echo "  params: seq=${seq_len}, stride=${stride}, units=${units}, middle=${middle_units}, batch=${batch}, epoch=${epoch}, dropout=${dropout}, l2=${l2}, lr=${lr}, n_splits=${n_splits}" | tee -a "$MASTER_LOG"

  nohup python "$PY_SCRIPT" "$out_dir" "$seq_len" "$stride" "$units" "$middle_units" "$batch" "$epoch" \
    --dropout "$dropout" \
    --lambda_l2 "$l2" \
    --learning_rate "$lr" \
    --n_splits "$n_splits" \
    > "$out_dir/output.txt" 2>&1 &

  run_pid=$!
  wait "$run_pid"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run finished ${run_id}" | tee -a "$MASTER_LOG"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All runs finished" | tee -a "$MASTER_LOG"
