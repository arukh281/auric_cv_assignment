#!/usr/bin/env bash
# B1h: B1 with the 40 images of splits/holdout40_seed0.txt left out of tiling and training. Re-run to resume.
#   1. scripts/_run.sh: train -> eval on val (eval/) -> prediction review -> merge sensitivity -> checkpoint curve
#   2. eval on the 40 held-out train images                        -> eval_holdout40/
#   3. eval on 40 seeded in-sample train images (holdout excluded)  -> eval_train40/
#   4. analysis/class_agnostic.py on all three
# All evals use the config's explicit eval.max_det (902). Log: $RUNS/b1h_tile1024_holdout40/run.log
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh
CFG=configs/b1h.yaml; NAME=b1h_tile1024_holdout40
bash scripts/_run.sh "$CFG" "$NAME" "$@"
exec > >(tee -a "$RUNS/$NAME/run.log") 2>&1
echo "== eval holdout40"
$PY eval.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" --split holdout --out "$RUNS/$NAME/eval_holdout40"
echo "== eval train40 (in-sample, seed 0, holdout excluded)"
$PY eval.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" --split train --max-images 40 --sample-seed 0 \
  --out "$RUNS/$NAME/eval_train40"
echo "== class-agnostic"
for D in eval eval_holdout40 eval_train40; do
  $PY analysis/class_agnostic.py --preds "$RUNS/$NAME/$D/predictions.csv" --data-root "$DATA_ROOT"
done
echo "===== $(date -u +%FT%TZ) $NAME B1h EXTRA EVALS DONE"
