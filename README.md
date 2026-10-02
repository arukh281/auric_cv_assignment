# Auric CV assignment: 5-class truck detector

YOLO11s (COCO-pretrained) on a 5-class overhead truck dataset. Target: mAP50 ≥ 0.75 on the provided val set.

## Layout
| Path | What |
|---|---|
| `train.py` | Train one config. Resumable: re-running the same command continues from `last.pt` |
| `eval.py` | Val images → predictions → metrics (full-image or sliced). Same scorer for every model |
| `tools/make_tiles.py` | Tiles the TRAIN split at native resolution |
| `analysis/eda.py` | Phase 1 dataset analysis. **Needs the full dataset; do not re-run on a subset** (it overwrites `figures/eda/`) |
| `analysis/pred_review.py` | TP/FP/FN grids, confusion matrix with background |
| `detlib/` | Shared tiling, scoring, merging, environment logging |
| `configs/` | `b0.yaml`, `b1.yaml`, `data.yaml` |
| `notebooks/colab_train.ipynb` | Colab driver for B0 and B1 |
| `tests/test_pipeline.py` | Known-answer tests: scoring, tiling, merging |
| `runs/<run>/` | Config, command, env, metrics, training curves per run. Not in git (`.gitignore`); kept on Drive with the weights |
| `analysis/merge_sensitivity.py` | Re-scores saved raw tile predictions under other merge settings (no re-inference) |
| `EXPERIMENTS.md`, `REPORT.md` | Experiment log and report |

## Dataset
Expected layout (the `cv_dataset.zip` root):
```
classmap.txt  data.yaml  train/images  train/labels  val/images  val/labels
```
Labels are YOLO txt (`cls xc yc w h`, normalized). Locally, `data/` is a symlink to the dataset folder.

## Run on Colab (training)
**Environment:** a Colab **T4 GPU (16 GB)** runtime on **Python 3.13**.
- Colab's preinstalled CUDA torch/torchvision are used as-is and **never reinstalled**. The install cell runs
  `pip install -r requirements.txt -c <constraints pinned to Colab's own torch, torchvision, numpy, opencv>`.
- The exact environment is saved twice:
  - `DRIVE_RUNS/colab_env_lock.txt` (full `pip freeze`)
  - each run's `env/*_pip_freeze.txt` and `env/*_hardware.json` (Python, torch, CUDA, cuDNN, GPU name e.g.
    "Tesla T4", GPU memory, Ultralytics version, git commit)
- To reproduce: use the same Colab runtime, install with that lock file, and keep torch as shipped.

**Training:**
- AMP (fp16) is on.
- `last.pt` is written to Drive every epoch, so a disconnect loses at most one epoch.
- If batch 16 does not fit on the T4, `train.py` retries at batch 8 and records the batch actually used.

**Steps:**
1. Upload `cv_dataset.zip` to Google Drive (the notebook default is `MyDrive/auric/cv_dataset.zip`). Check that it
   shows the full size (~6.1 GB).
2. Push this repo to GitHub. For a private repo, add a GitHub token as a Colab secret named `GH_TOKEN`
   (key icon in the left bar), and turn on notebook access for this notebook.
3. Open `notebooks/colab_train.ipynb` in Colab and choose Runtime → Change runtime type → **T4 GPU**.
4. Edit the first cell: `DRIVE_DATA`, `DRIVE_RUNS`, `REPO_URL`, `BRANCH`.
5. Run the setup cells, then the tests. The tests must print `tests passed`.
6. Run the **1-epoch timing probe** and its estimate cell. It reports seconds per epoch, the estimated hours for
   100 epochs of B0 and of B1, and the number of tiles B1 trains on. **If B1 is over ~8 h, stop and choose a
   schedule first.**
7. Run B0, then B1 (each: train cell, then eval cell). Expected outputs under `DRIVE_RUNS`:
   - `b0_full640/` and `b1_tile1024/`: `config.yaml`, `command.txt`, `env/`, `train/weights/last.pt`,
     `training_summary.json`, `training_curves.png`, `eval/metrics.json`, `eval/per_class.csv`,
     `eval/scorer_comparison.csv`, `eval/bootstrap_ci.csv`, `eval/pr_curves.png`
   - `figures/<run>/`: TP/FP/FN grids, confusion matrix, and `merge_sensitivity.csv` (B1)
   - `../runs_export.zip`: everything except weights
8. **After a disconnect:** reconnect, re-run the setup cells, then re-run the same training cell. It resumes from
   the last saved epoch on Drive. B1 tiles are rebuilt identically from the seed.

## Run locally
```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt torch torchvision
.venv/bin/python tests/test_pipeline.py                     # known-answer tests
.venv/bin/python train.py --config configs/b0.yaml          # data/ = dataset root, outputs to runs/b0_full640
.venv/bin/python eval.py  --config configs/b0.yaml          # evaluates runs/b0_full640/train/weights/last.pt
.venv/bin/python eval.py  --config configs/b1.yaml --weights path/to/last.pt   # any checkpoint, sliced
.venv/bin/python analysis/pred_review.py --preds runs/b0_full640/eval/predictions.csv --name b0_full640
```
Re-score saved predictions without inference: `eval.py --config ... --from-preds <predictions.csv>`.
