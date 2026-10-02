#!/usr/bin/env bash
# Idempotent Colab setup, run from the Colab Terminal in the repo root:  bash scripts/colab_setup.sh
#  1. installs requirements.txt WITHOUT replacing Colab's torch / torchvision / numpy / opencv
#  2. copies + unzips the dataset from Drive to /content/data (skipped if already done)
#  3. saves the environment lock to Drive
#  4. runs the known-answer tests
# Prerequisite: Drive mounted at /content/drive (mount once from a notebook cell: drive.mount("/content/drive")).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh

[ -d /content/drive/MyDrive ] || { echo "Drive is not mounted. In a notebook cell run:  from google.colab import drive; drive.mount('/content/drive')"; exit 1; }
[ -f "$DRIVE_DATA" ] || { echo "dataset not found: $DRIVE_DATA"; exit 1; }
mkdir -p "$RUNS"

echo "== python / GPU"
$PY -c 'import sys; print(sys.version)'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: no GPU visible"

echo "== install (Colab torch/torchvision/numpy/opencv pinned as constraints)"
$PY -m pip freeze | grep -iE '^(torch|torchvision|numpy|opencv-python|opencv-python-headless)==' > /content/colab_constraints.txt || true
cat /content/colab_constraints.txt
$PY -m pip install -q -r requirements.txt -c /content/colab_constraints.txt
$PY -c 'import torch, ultralytics; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "ultralytics", ultralytics.__version__)'

echo "== dataset"
if [ -f "$DATA_ROOT/_READY" ]; then
  echo "already prepared: $DATA_ROOT"
else
  rm -rf "$DATA_ROOT" /content/_data_tmp
  echo "copying $DRIVE_DATA ($(du -h "$DRIVE_DATA" | cut -f1)) to /content ..."
  cp "$DRIVE_DATA" /content/dataset.zip
  unzip -q /content/dataset.zip -d "$DATA_ROOT"
  rm -f /content/dataset.zip   # local copy only; the Drive zip is never touched
  # if the zip had a wrapper folder, move the folder holding classmap.txt + train/ up to DATA_ROOT
  root=$(dirname "$(find "$DATA_ROOT" -name classmap.txt -path '*' | head -1)")
  if [ "$root" != "$DATA_ROOT" ]; then mv "$root" /content/_data_tmp && rm -rf "$DATA_ROOT" && mv /content/_data_tmp "$DATA_ROOT"; fi
  touch "$DATA_ROOT/_READY"
fi
$PY -c "import sys; sys.path.insert(0, '.'); from detlib.data import ensure_data_yaml; print(ensure_data_yaml('$DATA_ROOT'))"
echo "train images: $(ls "$DATA_ROOT/train/images" | wc -l)   val images: $(ls "$DATA_ROOT/val/images" | wc -l)"

echo "== environment lock -> $RUNS/colab_env_lock.txt"
$PY -m pip freeze > "$RUNS/colab_env_lock.txt"

echo "== tests"
DATA_ROOT="$DATA_ROOT" $PY tests/test_pipeline.py 2>&1 | grep -E '^(PASS|FAIL)|tests passed|noisy case|Error|assert' || true
DATA_ROOT="$DATA_ROOT" $PY tests/test_pipeline.py > /dev/null 2>&1 && echo "SETUP OK" || { echo "TESTS FAILED: run  DATA_ROOT=$DATA_ROOT $PY tests/test_pipeline.py  to see why"; exit 1; }
