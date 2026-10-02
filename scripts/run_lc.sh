#!/usr/bin/env bash
# Learning-curve / repeat run (configs/b1h_f25|f50|f75|seed1.yaml): train -> eval on val (eval/) -> prediction review ->
# merge sensitivity -> checkpoint curve (scripts/_run.sh), then eval on the 40 held-out train images (eval_holdout40/)
# and class_agnostic.py on both. All evals use the config's eval.max_det (902). Analysis: scripts/run_analysis.sh.
# Usage: bash scripts/run_lc.sh configs/b1h_f25.yaml        Log: $RUNS/<name>/run.log
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh
CFG=${1:?usage: bash scripts/run_lc.sh <config>}
NAME=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CFG")
bash scripts/_run.sh "$CFG" "$NAME" "${@:2}"
exec > >(tee -a "$RUNS/$NAME/run.log") 2>&1
echo "== eval holdout40"
$PY eval.py --config "$CFG" --data-root "$DATA_ROOT" --runs-root "$RUNS" --split holdout --out "$RUNS/$NAME/eval_holdout40"
echo "== class-agnostic"
for D in eval eval_holdout40; do
  $PY analysis/class_agnostic.py --preds "$RUNS/$NAME/$D/predictions.csv" --data-root "$DATA_ROOT"
done
echo "===== $(date -u +%FT%TZ) $NAME LC EVALS DONE"
