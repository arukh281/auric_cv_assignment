#!/usr/bin/env bash
# Mac side, Kaggle via CLI: build (and optionally push) a kernel that runs setup and then ONE shell command in the repo.
# Usage: bash scripts/kaggle_cli_script_kernel.sh <slug> '<command>' [--source <user>/<kernel>]... [--push]
# Same machine settings as scripts/kaggle_cli_kernel.sh (private, T4 x2, internet, data + code datasets); --source adds
# a kernel_sources entry (its output is mounted read-only under /kaggle/input). Folder: kaggle_build/kernel_<slug>/.
set -euo pipefail
cd "$(dirname "$0")/.."
SLUG=${1:?usage: <slug> <command> [--source k]... [--push]}; CMD=${2:?command}; shift 2
KAGGLE=${KAGGLE:-.venv/bin/kaggle}; USER_=${KAGGLE_USER:-aradhya1211}; PUSH=; SOURCES=()
while [ $# -gt 0 ]; do case $1 in --push) PUSH=1 ;; --source) SOURCES+=("$2"); shift ;; *) echo "unknown $1"; exit 1 ;; esac; shift; done
B64=$(printf '%s' "$CMD" | base64 | tr -d '\n')
D=kaggle_build/kernel_$SLUG; rm -rf "$D"; mkdir -p "$D"
sed -e "s|__MODE__|script|" -e "s|__CONFIG__||" -e "s|__RUN__||" -e "s|__SCRIPT__|b64:$B64|" scripts/kaggle_kernel_run.py > "$D/run.py"
SRC_JSON=$(printf '"%s",' "${SOURCES[@]+"${SOURCES[@]}"}" | sed 's/,$//; s/^""$//')
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
  "kernel_sources": [$SRC_JSON],
  "model_sources": []
}
JSON
echo "built $D for $USER_/$SLUG (sources: ${SOURCES[*]:-none})"
if [ -n "$PUSH" ]; then $KAGGLE kernels push -p "$D" --accelerator NvidiaTeslaT4; fi
