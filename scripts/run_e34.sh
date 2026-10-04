#!/usr/bin/env bash
# E3/E4 run (configs/e3_*.yaml, e4_*.yaml): train -> on last.pt only: sliced eval on val (eval/), holdout40
# (eval_holdout40/), train40 (eval_train40/, 40 sampled train images, seed 0) -> prediction review (val) ->
# class-agnostic on all three -> checkpoint curve on HOLDOUT40 (not val) -> errors.py on val.
# No val checkpoint curve, no best.pt. Usage: bash scripts/run_e34.sh <config>     Log: $RUNS/<name>/run.log
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh
CFG=${1:?usage: bash scripts/run_e34.sh <config>}
NAME=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CFG")
mkdir -p "$RUNS/$NAME"
exec > >(tee -a "$RUNS/$NAME/run.log") 2>&1
echo "===== $(date -u +%FT%TZ) $NAME | commit $(cat CODE_COMMIT 2>/dev/null || git rev-parse --short HEAD) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo no-gpu)"
echo "== train"
$PY train.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" --work-dir "$WORK_DIR" --oom-fallback-batch 8 ${TRAIN_EXTRA:-}
W="$RUNS/$NAME/train/weights/last.pt"
echo "== eval val / holdout40 / train40 (last.pt)"
$PY eval.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" --weights "$W"
$PY eval.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" --weights "$W" --split holdout --out "$RUNS/$NAME/eval_holdout40"
$PY eval.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" --weights "$W" --split train --max-images 40 --sample-seed 0 --out "$RUNS/$NAME/eval_train40"
$PY analysis/pred_review.py --preds "$RUNS/$NAME/eval/predictions.csv" --name "$NAME" --data-root "$DATA_ROOT" --out-root "$FIGS"
for D in eval eval_holdout40 eval_train40; do
  $PY analysis/class_agnostic.py --preds "$RUNS/$NAME/$D/predictions.csv" --data-root "$DATA_ROOT"
done
echo "== checkpoint curve on holdout40"
$PY analysis/checkpoint_curve.py --config "$CFG" --runs-root "$RUNS" --data-root "$DATA_ROOT" --out-root "$FIGS" --split holdout
echo "== errors.py (val, last.pt)"
$PY analysis/errors.py --preds "$RUNS/$NAME/eval/predictions.csv" --name "$NAME" --data-root "$DATA_ROOT" --out-root "$FIGS"
echo "===== $(date -u +%FT%TZ) $NAME DONE"
