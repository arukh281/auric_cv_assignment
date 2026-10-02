#!/usr/bin/env bash
# Mac side, Kaggle via CLI: build kaggle_build/kernel_<mode>/ (gitignored) for MODE smoke|full and optionally push it.
# Usage: bash scripts/kaggle_cli_kernel.sh smoke|full [config, default configs/b1h.yaml] [--push]
# Kernel: <user>/auric-b1h-smoke or <user>/auric-b1h; private, GPU T4 x2 (machine_shape NvidiaTeslaT4), internet on;
# inputs: <user>/auric-cv-dataset and <user>/auric-cv-code. Entry point: scripts/kaggle_kernel_run.py.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?usage: smoke|full [config] [--push]}; KAGGLE=${KAGGLE:-.venv/bin/kaggle}; USER_=${KAGGLE_USER:-aradhya1211}
CFG=configs/b1h.yaml; PUSH=
for x in "${@:2}"; do case $x in --push) PUSH=1 ;; *) CFG=$x ;; esac; done
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }
RUN=$(.venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CFG")
if [ "$CFG" = configs/b1h.yaml ]; then BASE=auric-b1h; SCRIPT=scripts/run_b1h.sh
else BASE=auric-$(echo "$RUN" | tr '_' '-'); SCRIPT="scripts/run_lc.sh $CFG"; fi
case $MODE in smoke) SLUG=$BASE-smoke ;; full) SLUG=$BASE ;; *) echo "MODE must be smoke or full"; exit 1 ;; esac
D=kaggle_build/kernel_${SLUG}; rm -rf "$D"; mkdir -p "$D"
sed -e "s|__MODE__|$MODE|" -e "s|__CONFIG__|$CFG|" -e "s|__RUN__|$RUN|" -e "s|__SCRIPT__|$SCRIPT|" \
  scripts/kaggle_kernel_run.py > "$D/run.py"
cat > "$D/kernel-metadata.json" <<JSON
{
  "id": "$USER_/$SLUG",
  "title": "$SLUG",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["$USER_/auric-cv-dataset", "$USER_/auric-cv-code"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON
echo "built $D for $USER_/$SLUG ($CFG, run $RUN, script: $SCRIPT)"
if [ -n "$PUSH" ]; then $KAGGLE kernels push -p "$D" --accelerator NvidiaTeslaT4; fi
