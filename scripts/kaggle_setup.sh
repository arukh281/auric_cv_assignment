#!/usr/bin/env bash
# Idempotent Kaggle setup, run in the repo root (notebook cell:  !bash scripts/kaggle_setup.sh). Mirrors colab_setup.sh:
#  1. git pull (token from Kaggle Secrets "GH_TOKEN" via kaggle_secrets; never printed; skipped if unavailable
#     or SKIP_PULL=1)
#  2. installs requirements.txt WITHOUT replacing Kaggle's torch / torchvision / numpy / opencv (pinned constraints)
#  3. unzips /kaggle/input/<dataset>/cv_dataset.zip (or copies an already-extracted dataset folder) to
#     /kaggle/tmp/data (skipped if already done)
#  4. saves the environment lock to $RUNS/kaggle_env_lock.txt
#  5. runs all test files in tests/
# Paths come from scripts/_env.sh (RUNS=/kaggle/working/runs, WORK_DIR=/kaggle/tmp/work).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh
[ "$PLATFORM" = kaggle ] || { echo "not a Kaggle environment (no $KAGGLE_ROOT/input); use scripts/colab_setup.sh"; exit 1; }
mkdir -p "$RUNS" "$KAGGLE_ROOT/tmp"

echo "== git pull"
if [ "${SKIP_PULL:-0}" = 1 ]; then
  echo "skipped (SKIP_PULL=1)"
else
  set +x
  T=$($PY -c 'from kaggle_secrets import UserSecretsClient as U; print(U().get_secret("GH_TOKEN"))' 2>/dev/null || true)
  if [ -n "$T" ]; then
    git pull -q "https://$T@github.com/arukh281/auric_cv_assignment.git" "$(git rev-parse --abbrev-ref HEAD)" \
      && echo "pulled: $(git log -1 --oneline)" || echo "WARNING: git pull failed; continuing at $(git log -1 --oneline)"
  else
    echo "no GH_TOKEN secret available; continuing at $(git log -1 --oneline)"
  fi
  unset T
fi

echo "== python / GPU"
$PY -c 'import sys; print(sys.version)'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARNING: no GPU visible"

echo "== install (Kaggle torch/torchvision/numpy/opencv pinned as constraints)"
CONSTRAINTS="$KAGGLE_ROOT/tmp/kaggle_constraints.txt"
$PY -m pip freeze | grep -iE '^(torch|torchvision|numpy|opencv-python|opencv-python-headless)==' > "$CONSTRAINTS" || true
cat "$CONSTRAINTS"
if [ "${SKIP_INSTALL:-0}" = 1 ]; then echo "install skipped (SKIP_INSTALL=1)"; else
  $PY -m pip install -q -r requirements.txt -c "$CONSTRAINTS"
fi
$PY -c 'import torch, ultralytics; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "ultralytics", ultralytics.__version__)'

echo "== dataset"
if [ -f "$DATA_ROOT/_READY" ]; then
  echo "already prepared: $DATA_ROOT"
else
  rm -rf "$DATA_ROOT" "$KAGGLE_ROOT/tmp/_data_tmp"
  mkdir -p "$DATA_ROOT"
  if [ -n "$DRIVE_DATA" ] && [ -f "$DRIVE_DATA" ]; then
    echo "unzipping $DRIVE_DATA ($(du -h "$DRIVE_DATA" | cut -f1)) ..."
    unzip -q "$DRIVE_DATA" -d "$DATA_ROOT"          # /kaggle/input is read-only; nothing there is modified
  else
    SRC=$(find "$KAGGLE_ROOT/input" -name classmap.txt 2>/dev/null | head -n 1 || true)
    [ -n "$SRC" ] || { echo "dataset not found: no cv_dataset.zip or classmap.txt under $KAGGLE_ROOT/input"; exit 1; }
    echo "copying extracted dataset $(dirname "$SRC") ..."
    cp -r "$(dirname "$SRC")/." "$DATA_ROOT/"      # copy: labels must be writable (Ultralytics label cache)
  fi
  # if the zip had a wrapper folder, move the folder holding classmap.txt + train/ up to DATA_ROOT
  root=$(dirname "$(find "$DATA_ROOT" -name classmap.txt | head -n 1)")
  if [ "$root" != "$DATA_ROOT" ]; then mv "$root" "$KAGGLE_ROOT/tmp/_data_tmp" && rm -rf "$DATA_ROOT" && mv "$KAGGLE_ROOT/tmp/_data_tmp" "$DATA_ROOT"; fi
  touch "$DATA_ROOT/_READY"
fi
$PY -c "import sys; sys.path.insert(0, '.'); from detlib.data import ensure_data_yaml; print(ensure_data_yaml('$DATA_ROOT'))"
echo "train images: $(ls "$DATA_ROOT/train/images" | wc -l)   val images: $(ls "$DATA_ROOT/val/images" | wc -l)"

echo "== environment lock -> $RUNS/kaggle_env_lock.txt"
$PY -m pip freeze > "$RUNS/kaggle_env_lock.txt"

echo "== tests (all files in tests/)"
FAILED=0
for t in tests/test_*.py; do
  if DATA_ROOT="$DATA_ROOT" $PY "$t" > "$KAGGLE_ROOT/tmp/$(basename "$t" .py).log" 2>&1; then
    echo "PASS $t: $(grep -E 'tests passed' "$KAGGLE_ROOT/tmp/$(basename "$t" .py).log" | tail -n 1)"
  else
    echo "FAIL $t (log: $KAGGLE_ROOT/tmp/$(basename "$t" .py).log)"; tail -n 15 "$KAGGLE_ROOT/tmp/$(basename "$t" .py).log"; FAILED=1
  fi
done
[ "$FAILED" = 0 ] && echo "SETUP OK" || { echo "TESTS FAILED"; exit 1; }
