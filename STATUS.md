# Status: Phase 2 code (2026-10-02)

## What works, and how it was tested (locally, on the 42-image `data_small`, CPU)
- **Scoring:** `tests/test_pipeline.py`, 9 known-answer tests, all pass.
  - Predictions identical to GT → mAP50 = 1.0, with every bootstrap sample = 1.0.
  - Every other GT box only → COCO AP = 51/101 exactly; Ultralytics-style = 0.75.
  - Wrong class, or boxes moved off-image → 0.
  - An FP ranked above the TP → 0.5.
  - A duplicate prediction counts as FP.
- **Tiling:**
  - Tiles cover the whole image with ≥ 256 px overlap, including the 7204 x 5932 and 1369 x 1334 px extremes.
  - Every real val box lies whole inside some tile.
  - Label round-trip is exact (to 1e-4 px).
  - The 50% visibility rule behaves as specified.
- **Tile merge:** cross-tile duplicates and edge fragments are removed. The merge is class-wise. WBF works.
- **`tools/make_tiles.py`** (3 images, as required):
  - Prints the max box side (161 px) and refuses an overlap at or below it.
  - Output is byte-identical with 1 or 3 workers.
  - Labels align visually.
  - The empty-tile keep rate came out at 15/47 against p = 0.2. I checked the RNG separately: it is unbiased
    (0.198 over 20k seeds), so this is small-sample chance (binomial p = 0.037).
- **`train.py` B0 and B1:** 2-epoch CPU smoke runs at imgsz 320 completed.
  - The run folder gets `config.yaml`, `data.yaml`, `command.txt`, `env/` (pip freeze, hardware, git commit),
    `init_weights.json` (sha256), `tiling_params.json` (B1) and Ultralytics `train/`.
- **Resume:** I killed a run after epoch 1 and re-ran the same command. It continued with epochs 2-4, and
  `command.txt` logs both invocations.
- **`eval.py`:**
  - Full (B0) and sliced (B1) modes both write `predictions.csv`, `metrics.json`, `per_class.csv`,
    `bootstrap_ci.csv`, `pr_curves.png` and `eval_args.json`.
  - max_det = 334, from the max of 167 GT boxes per val image (EDA table).
  - `--ultra-crosscheck` runs end to end.
- **`analysis/pred_review.py`:** writes TP/FP-kind/FN grids, a confusion matrix with background, and overview images.
- **Notebook data cell:** run locally on a mini dataset as a flat zip, a zip with a wrapper folder, and a plain
  folder. `data.yaml` is generated when missing.

## Not tested / uncertain
- **The scorer has not yet been compared against Ultralytics on a model that detects anything.** Smoke models
  score 0, so 0 = 0 proves little. The local check was stopped to spare the Mac. B0's Colab eval cell runs
  `--ultra-crosscheck`: compare `metrics.json` → `mAP50_ultralytics_interp` with `ultralytics_val.mAP50`.
  Expect them to be close, not identical: Ultralytics uses rect batching and its own matching.
- **Colab-only parts never ran here:** Drive mount, git clone with a token, CUDA, determinism on GPU, and the time
  and memory of batch 16 at 1024. If B1 runs out of memory on L4, lower `batch` in `configs/b1.yaml`.
  That is a recorded config change.
- **Choices you should confirm or change:**
  1. B1 runs 100 epochs (the original Phase 2 spec) against 50 for B0, which confounds the comparison.
  2. Empty-tile keep fraction is 0.2.
  3. The tile merge uses NMS with intersection-over-smaller at 0.6.
  4. `last.pt` is reported, with no best-epoch selection on val. Ultralytics still validates once at the final epoch
     and saves a `best.pt`; ignore both.
  5. The headline mAP50 is COCO-interpolated. The Ultralytics-interpolated value is reported next to it, and is
     what Ultralytics prints.
- `analysis/eda.py` must not be re-run on the subset: it would overwrite the full-dataset tables.

## First commands
On the Mac (once): publish the repo. It has no remote yet, and Phase 2 is on branch `worktree-phase2-pipeline`.
```bash
cd ~/Desktop/Work/auric_cv_assignment
git merge --ff-only worktree-phase2-pipeline          # bring Phase 2 onto main
gh repo create auric_cv_assignment --private --source . --push
```
On Colab: open `notebooks/colab_train.ipynb` (File → Open notebook → GitHub, or upload it), set the first cell:
```python
DRIVE_DATA = "/content/drive/MyDrive/cv_dataset.zip"
DRIVE_RUNS = "/content/drive/MyDrive/auric_runs"
REPO_URL   = "https://github.com/arukh281/auric_cv_assignment.git"
BRANCH     = "main"
```
Add Colab secret `GH_TOKEN` (private repo), choose an L4/A100 runtime, then Runtime → Run all.
