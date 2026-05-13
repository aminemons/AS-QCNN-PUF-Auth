#!/usr/bin/env bash
# =============================================================================
# run_training.sh
# Launch the full QCNN training pipeline in the background using nohup,
# so it keeps running even if your SSH connection drops.
#
# USAGE (on the remote workstation):
#   bash run_training.sh [config] [puf_types]
#
# EXAMPLES:
#   bash run_training.sh                                  # full config, all puf types
#   bash run_training.sh phase1_quantum/configs/small_config.yaml 5xor
#
# After training completes the script automatically commits and pushes all
# results to git so you can pull them from your local machine.
# =============================================================================

set -euo pipefail

# ── 1. Launcher Mode (Backgrounding) ──────────────────────────────────────────
# If the first argument isn't --internal, we relaunch ourselves via nohup.
if [[ "${1:-}" != "--internal" ]]; then
    mkdir -p results/logs
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    WRAPPER_LOG="results/logs/train_wrapper_${TIMESTAMP}.log"
    
    echo -e "\033[0;32m[INFO]\033[0m Launching training in the background using nohup..."
    nohup bash "$0" --internal "$@" > "${WRAPPER_LOG}" 2>&1 &
    
    PID=$!
    echo -e "\033[0;32m[INFO]\033[0m Training started successfully! (PID: ${PID})"
    echo -e "\033[0;32m[INFO]\033[0m You can now safely close this SSH session."
    echo -e "\033[0;32m[INFO]\033[0m To monitor progress live, run:"
    echo -e "       tail -f ${WRAPPER_LOG}"
    exit 0
fi

# ── 2. Internal Execution Mode ────────────────────────────────────────────────
shift # remove the --internal flag

CONFIG="${1:-phase1_quantum/configs/full_config.yaml}"
PUF_TYPES="${2:-1xor,3xor,5xor}"
LOG_FILE="results/logs/train_$(date +"%Y%m%d_%H%M%S").log"

echo "====================================================="
echo " QCNN Training — $(date)"
echo " Config    : ${CONFIG}"
echo " PUF types : ${PUF_TYPES}"
echo " Log file  : ${LOG_FILE}"
echo "====================================================="

# Activate conda env if available
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate qcnn 2>/dev/null || conda activate base
fi

# Run training
python phase1_quantum/02_train_qcnn.py \
    --config "${CONFIG}" \
    --puf_types "${PUF_TYPES}" \
    2>&1 | tee "${LOG_FILE}"

TRAIN_EXIT=${PIPESTATUS[0]}

if [ "${TRAIN_EXIT}" -ne 0 ]; then
    echo "[ERROR] Training exited with code ${TRAIN_EXIT}. Check ${LOG_FILE}"
    exit ${TRAIN_EXIT}
fi

echo ""
echo "====================================================="
echo " Training finished! Auto-pushing results to git..."
echo "====================================================="

cd "$(git rev-parse --show-toplevel)"

git add results/metrics/      2>/dev/null || true
git add results/matrices/     2>/dev/null || true
git add results/plots/        2>/dev/null || true
git add results/logs/         2>/dev/null || true
git add -f results/checkpoints/*/final_model.pt 2>/dev/null || true

TS="$(date +'%Y-%m-%d %H:%M')"
SUMMARY=""
if [ -f results/metrics/training_summary.json ]; then
    SUMMARY=$(python3 -c "
import json, sys
try:
    with open('results/metrics/training_summary.json') as f:
        d = json.load(f)
    parts = [f\"{r.get('puf_type','?')}: acc={r.get('test_acc',0):.4f}\" for r in d]
    print(' | '.join(parts))
except Exception as e:
    print('metrics available')
" 2>/dev/null || echo "metrics available")
fi

MSG="[auto] training complete ${TS}"
if [ -n "${SUMMARY}" ]; then
    MSG="${MSG} — ${SUMMARY}"
fi

git commit -m "${MSG}" || echo "Nothing new to commit."
git push origin master && echo "[OK] Results pushed to origin/master." \
                       || echo "[WARN] git push failed — check remote/auth."

echo "====================================================="
echo " ALL DONE. You can now pull on your local machine."
echo "====================================================="
