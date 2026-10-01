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
| `runs/<run>/` | Config, command, env, metrics per run (weights live on Drive, not in git) |
| `EXPERIMENTS.md`, `REPORT.md` | Experiment log and report |

## Dataset
Expected layout (the `cv_dataset.zip` root):
```
classmap.txt  data.yaml  train/images  train/labels  val/images  val/labels
```
Labels are YOLO txt (`cls xc yc w h`, normalized). Locally, `data/` is a symlink to the dataset folder.

## Run on Colab (training)
1. Upload `cv_dataset.zip` to Google Drive, e.g. `MyDrive/cv_dataset.zip`.
2. Push this repo to GitHub. For a private repo, add a GitHub token as a Colab secret named `GH_TOKEN`
   (key icon in the left bar; enable notebook access).
3. Open `notebooks/colab_train.ipynb` in Colab and pick a GPU runtime (Runtime → Change runtime type → L4 or A100).
4. Edit the first cell: `DRIVE_DATA`, `DRIVE_RUNS`, `REPO_URL`, `BRANCH`.
5. Run all cells. Expected outputs, under `DRIVE_RUNS` on Drive:
   - `b0_full640/` and `b1_tile1024/`: `config.yaml`, `command.txt`, `env/`, `train/weights/last.pt`,
     `eval/metrics.json`, `eval/per_class.csv`, `eval/bootstrap_ci.csv`, `eval/pr_curves.png`
   - `figures/<run>/`: TP/FP/FN grids and the confusion matrix
   - `../auric_runs_export.zip`: everything except weights, to copy into this repo's `runs/` and `figures/`
6. **After a disconnect:** reconnect, re-run the setup cells, then re-run the same training cell. It resumes from
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
