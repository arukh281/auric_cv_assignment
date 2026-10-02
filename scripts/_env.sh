# Shared paths for the Colab / Kaggle scripts (sourced, not run). Override any of these by exporting it first.
# Platform: Kaggle if KAGGLE_KERNEL_RUN_TYPE is set or /kaggle/input exists (KAGGLE_ROOT overrides /kaggle for
# testing); otherwise the Colab defaults below, unchanged.
KAGGLE_ROOT=${KAGGLE_ROOT:-/kaggle}
if [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ] || [ -d "$KAGGLE_ROOT/input" ]; then
  PLATFORM=kaggle
  # dataset: /kaggle/input/<dataset>/cv_dataset.zip, or a folder Kaggle already extracted (holds classmap.txt)
  DRIVE_DATA=${DRIVE_DATA:-$(ls -d "$KAGGLE_ROOT"/input/*/cv_dataset.zip 2>/dev/null | head -n 1 || true)}
  RUNS=${RUNS:-$KAGGLE_ROOT/working/runs}
  DATA_ROOT=${DATA_ROOT:-$KAGGLE_ROOT/tmp/data}
  WORK_DIR=${WORK_DIR:-$KAGGLE_ROOT/tmp/work}
else
  PLATFORM=colab
  AURIC=${AURIC:-/content/drive/MyDrive/auric}
  DRIVE_DATA=${DRIVE_DATA:-$AURIC/cv_dataset.zip}
  RUNS=${RUNS:-$AURIC/runs}
  DATA_ROOT=${DATA_ROOT:-/content/data}
  WORK_DIR=${WORK_DIR:-/content/work}
fi
FIGS=${FIGS:-$RUNS/figures}
PY=${PY:-python3}
export PYTHONUNBUFFERED=1
