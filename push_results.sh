#!/usr/bin/env bash
# push_results.sh
# Auto-commit and push all training results to GitHub.
# Run after training completes on the SSH workstation.

set -e

echo "Staging results..."
git add results/metrics/
git add results/matrices/
git add results/plots/
git add -f results/checkpoints/*/final_model.pt 2>/dev/null || true

TIMESTAMP=$(date +"%Y-%m-%d %H:%M")
SUMMARY=""

# Build summary from metrics if available
if [ -f results/metrics/eval_summary.json ]; then
    SUMMARY=$(python3 -c "
import json
with open('results/metrics/eval_summary.json') as f:
    d = json.load(f)
parts = [f\"{k}: {v.get('accuracy', 0):.4f}\" for k, v in d.items()]
print(' | '.join(parts))
" 2>/dev/null || echo "metrics available")
fi

MSG="results: $TIMESTAMP"
if [ -n "$SUMMARY" ]; then
    MSG="$MSG — $SUMMARY"
fi

git commit -m "$MSG" || echo "Nothing new to commit."
git push origin main
echo "Results pushed."
