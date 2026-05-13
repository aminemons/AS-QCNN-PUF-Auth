#!/usr/bin/env bash
# =============================================================================
# run_training.sh
# Launch the full QCNN training pipeline inside a tmux session so it keeps
# running even if your SSH connection drops.
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

# ── Configuration ─────────────────────────────────────────────────────────────
SESSION="qcnn_train"
CONFIG="${1:-phase1_quantum/configs/full_config.yaml}"
PUF_TYPES="${2:-1xor,3xor,5xor}"
LOG_DIR="results/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if ! command -v tmux &>/dev/null; then
    error "tmux is not installed. Run: sudo apt-get install -y tmux"
    exit 1
fi

if ! command -v conda &>/dev/null && ! command -v python3 &>/dev/null; then
    error "Neither conda nor python3 found. Activate your environment first."
    exit 1
fi

mkdir -p "${LOG_DIR}"

# ── Kill any existing session with the same name ──────────────────────────────
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    warn "Session '${SESSION}' already exists."
    read -rp "Kill it and restart? [y/N] " ans
    if [[ "${ans}" =~ ^[Yy]$ ]]; then
        tmux kill-session -t "${SESSION}"
        info "Old session killed."
    else
        info "Attaching to existing session instead. Ctrl+B D to detach."
        tmux attach-session -t "${SESSION}"
        exit 0
    fi
fi

# ── The command that runs inside tmux ─────────────────────────────────────────
# Everything is wrapped in a single bash -c so it runs sequentially in one pane.
INNER_CMD=$(cat <<INNEREOF
set -euo pipefail
echo "====================================================="
echo " QCNN Training — \$(date)"
echo " Config    : ${CONFIG}"
echo " PUF types : ${PUF_TYPES}"
echo " Log file  : ${LOG_FILE}"
echo "====================================================="

# Activate conda env if available
if command -v conda &>/dev/null; then
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate qcnn 2>/dev/null || conda activate base
fi

# Run training — pipe to tee so you can both see output AND save to log
python phase1_quantum/02_train_qcnn.py \
    --config "${CONFIG}" \
    --puf_types "${PUF_TYPES}" \
    2>&1 | tee "${LOG_FILE}"

TRAIN_EXIT=\${PIPESTATUS[0]}

if [ "\${TRAIN_EXIT}" -ne 0 ]; then
    echo "[ERROR] Training exited with code \${TRAIN_EXIT}. Check ${LOG_FILE}"
    exit \${TRAIN_EXIT}
fi

echo ""
echo "====================================================="
echo " Training finished! Auto-pushing results to git..."
echo "====================================================="

# ── Auto git push ──────────────────────────────────────────────────────────
cd "\$(git rev-parse --show-toplevel)"

git add results/metrics/      2>/dev/null || true
git add results/matrices/     2>/dev/null || true
git add results/plots/        2>/dev/null || true
git add results/logs/         2>/dev/null || true
git add -f results/checkpoints/*/final_model.pt 2>/dev/null || true

TS="\$(date +'%Y-%m-%d %H:%M')"
SUMMARY=""
if [ -f results/metrics/training_summary.json ]; then
    SUMMARY=\$(python3 -c "
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

MSG="[auto] training complete \${TS}"
if [ -n "\${SUMMARY}" ]; then
    MSG="\${MSG} — \${SUMMARY}"
fi

git commit -m "\${MSG}" || echo "Nothing new to commit."
git push origin master && echo "[OK] Results pushed to origin/master." \
                       || echo "[WARN] git push failed — check remote/auth."

echo "====================================================="
echo " ALL DONE. You can now pull on your local machine:"
echo "   git pull origin main"
echo "====================================================="
INNEREOF
)

# ── Launch tmux session ───────────────────────────────────────────────────────
tmux new-session -d -s "${SESSION}" -x 220 -y 50
tmux send-keys -t "${SESSION}" "bash -c '${INNER_CMD//\'/\'\\\'\'}'" Enter

info "Session '${SESSION}' launched."
info "  Attach anytime : tmux attach -t ${SESSION}"
info "  Detach safely  : Ctrl+B  then  D"
info "  Kill session   : tmux kill-session -t ${SESSION}"
info "  Watch live log : tail -f ${LOG_FILE}"
echo ""
info "Attaching now (Ctrl+B D to detach and leave it running)..."
sleep 1
tmux attach-session -t "${SESSION}"
