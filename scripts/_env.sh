# Shared paths for the Colab scripts (sourced, not run). Override any of these by exporting it first.
AURIC=${AURIC:-/content/drive/MyDrive/auric}
DRIVE_DATA=${DRIVE_DATA:-$AURIC/cv_dataset.zip}
RUNS=${RUNS:-$AURIC/runs}
DATA_ROOT=${DATA_ROOT:-/content/data}
WORK_DIR=${WORK_DIR:-/content/work}
FIGS=${FIGS:-$RUNS/figures}
PY=${PY:-python3}
export PYTHONUNBUFFERED=1
