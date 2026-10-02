#!/usr/bin/env bash
# Train one config (auto-resumes from last.pt on Drive; skips if already finished), then evaluate it.
# Called by run_b0.sh / run_b1.sh.  Usage: scripts/_run.sh <config> <run name> [extra eval.py args]
# Everything is appended to $RUNS/<run name>/run.log.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh
CFG=$1; NAME=$2; shift 2
[ -f "$DATA_ROOT/_READY" ] || { echo "dataset not prepared: run  bash scripts/colab_setup.sh  first"; exit 1; }
mkdir -p "$RUNS/$NAME"
LOG="$RUNS/$NAME/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -u +%FT%TZ) $NAME | commit $(git rev-parse --short HEAD) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo no-gpu)"

echo "== train"
$PY train.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" --work-dir "$WORK_DIR" --oom-fallback-batch 8

echo "== eval (last.pt)"
$PY eval.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" "$@"
$PY analysis/pred_review.py --preds "$RUNS/$NAME/eval/predictions.csv" --name "$NAME" --data-root "$DATA_ROOT" --out-root "$FIGS"
if [ -f "$RUNS/$NAME/eval/predictions_raw.csv" ] && grep -q 'mode: sliced' "$CFG"; then
  $PY analysis/merge_sensitivity.py --raw "$RUNS/$NAME/eval/predictions_raw.csv" --name "$NAME" --data-root "$DATA_ROOT" --out-root "$FIGS"
fi

echo "== checkpoint curve (real metric per kept checkpoint)"
$PY analysis/checkpoint_curve.py --config "$CFG" --runs-root "$RUNS" --data-root "$DATA_ROOT" --out-root "$FIGS"
echo "===== $(date -u +%FT%TZ) $NAME DONE"
