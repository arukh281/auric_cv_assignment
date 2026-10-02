#!/usr/bin/env bash
# Phase 3 analysis of a finished run: analysis/errors.py (from the saved eval predictions, no inference) and
# analysis/gt_box_oracle.py (5.1, runs the model on the val GT boxes; GPU). Not part of _run.sh.
# Usage: bash scripts/run_analysis.sh <run name>     e.g.  bash scripts/run_analysis.sh b1_tile1024
# Log: $RUNS/<run name>/analysis.log. Outputs: $FIGS/<run name>/errors/ and $FIGS/<run name>/gt_oracle/.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh
NAME=${1:?usage: bash scripts/run_analysis.sh <run name>}
CFG=$(grep -l "^name: $NAME\$" configs/*.yaml | head -n 1 || true)
[ -n "$CFG" ] || { echo "no config in configs/ with name: $NAME"; exit 1; }
[ -f "$DATA_ROOT/_READY" ] || { echo "dataset not prepared: run  bash scripts/colab_setup.sh  first"; exit 1; }
PREDS="$RUNS/$NAME/eval/predictions.csv"
[ -f "$PREDS" ] || { echo "missing $PREDS: the run's eval has not finished"; exit 1; }
LOG="$RUNS/$NAME/analysis.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u +%FT%TZ) analysis $NAME ($CFG) | commit $(git rev-parse --short HEAD) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo no-gpu)"

echo "== errors.py"
$PY analysis/errors.py --preds "$PREDS" --name "$NAME" --data-root "$DATA_ROOT" --out-root "$FIGS"

echo "== gt_box_oracle.py"
$PY analysis/gt_box_oracle.py --config "$CFG" --runs-root "$RUNS" --data-root "$DATA_ROOT" --out-root "$FIGS"
echo "===== $(date -u +%FT%TZ) analysis $NAME DONE"
