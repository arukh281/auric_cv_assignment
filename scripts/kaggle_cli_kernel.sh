#!/usr/bin/env bash
# Mac side, Kaggle via CLI: build kaggle_build/kernel_<mode>/ (gitignored) for MODE smoke|full and optionally push it.
# Usage: bash scripts/kaggle_cli_kernel.sh smoke|full [--push]
# Kernel: <user>/auric-b1h-smoke or <user>/auric-b1h; private, GPU T4 x2 (machine_shape NvidiaTeslaT4), internet on;
# inputs: <user>/auric-cv-dataset and <user>/auric-cv-code. Entry point: scripts/kaggle_kernel_run.py.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?usage: smoke|full [--push]}; KAGGLE=${KAGGLE:-.venv/bin/kaggle}; USER_=${KAGGLE_USER:-aradhya1211}
case $MODE in smoke) SLUG=auric-b1h-smoke ;; full) SLUG=auric-b1h ;; *) echo "MODE must be smoke or full"; exit 1 ;; esac
D=kaggle_build/kernel_$MODE; rm -rf "$D"; mkdir -p "$D"
sed "s/__MODE__/$MODE/" scripts/kaggle_kernel_run.py > "$D/run.py"
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
echo "built $D for $USER_/$SLUG"
if [ "${2:-}" = "--push" ]; then $KAGGLE kernels push -p "$D" --accelerator NvidiaTeslaT4; fi
