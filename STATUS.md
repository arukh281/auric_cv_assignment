# Status: Phase 2 code (2026-10-02)

## Round 4 (after the T4 timing probe): current plan
- Both runs: 50 epochs, SGD lr0 0.01 momentum 0.937 (explicit), AMP, `cache: disk` on /content, seed 0.
  Weights are kept at epochs 10/20/30/40/50 by a `train.py` hook; Ultralytics' `save_period` would keep 1/11/21/...
- Run from the Colab **Terminal**: `scripts/colab_setup.sh`, then `scripts/run_b1.sh`, then `scripts/run_b0.sh`;
  monitor with `scripts/status.sh`. See README → "Run on Colab from the Terminal".
- `analysis/checkpoint_curve.py` gives real-metric mAP50 per kept checkpoint.
- Tested locally: 12 tests pass, shell scripts pass `bash -n`, `status.sh` dry-run on a fake run.
  Not run anywhere yet: training with the new config, `checkpoint_curve.py`, and the setup and run scripts on Colab.

## Decisions applied (round 2)
1. **B0 now runs 100 epochs**, matching B1. Both runs set `val: true`, only to log per-epoch train/val losses and
   Ultralytics val mAP. `last.pt` is still what gets evaluated.
   - `train.py` writes `training_summary.json` (batch used, iterations per epoch, total iterations) and
     `training_curves.png/.csv` into each run folder.
   - EXPERIMENTS.md notes that B1 does more iterations per epoch because it trains on tiles.
2. **Empty-tile fraction** stays at 0.2.
3. **Tile merge default** stays at NMS on IoS 0.6.
   - `eval.py` now saves the pre-merge `predictions_raw.csv`.
   - `analysis/merge_sensitivity.py` re-scores it at IoS 0.5/0.6/0.7, IoU 0.5 and no-merge, without re-inference,
     and writes `figures/<run>/merge_sensitivity.csv`.
4. **Headline metric** is COCO 101-point AP50.
   - `eval.py` exports COCO JSON and adds pycocotools AP50 (maxDets = max_det) next to our scorer and the Ultralytics
     number, in `scorer_comparison.csv` and `metrics.json`.
   - It warns if ours and pycocotools differ by more than 0.005.
- **Out-of-memory fallback:** `train.py --oom-fallback-batch 8`. If the configured batch runs out of GPU memory before
  any epoch is saved, it retries at 8 and records `batch_requested`, `batch` and `batch_note` in `config.yaml`, plus
  `batch_used` in `training_summary.json`.
- `.gitignore` excludes `data`, `data/`, `data_small/`, `runs/`, `*.zip` and weights (`*.pt`, `*.pth`, `*.onnx`,
  `*.engine`, `*.safetensors`).

## IMPORTANT: the round-2 changes have NOT been run
You asked for nothing to run on the Mac, so every round-2 change was written but never executed. That covers:
- the pycocotools cross-check
- `finalize` / raw predictions and `merge_sensitivity.py`
- the out-of-memory fallback and `training_summary` / curves
- 3 new tests: `test_pycocotools_known_answers`, `test_finalize_merges_tile_duplicates`, `test_oom_fallback`
- the hand-edited notebook cells

**The notebook now runs `tests/test_pipeline.py` on Colab before any training. If any test fails, stop there.**

## Tested earlier (round 1, on the Mac, before the no-run instruction)
- 9 known-answer tests:
  - perfect predictions → mAP50 = 1.0
  - half recall → exactly 51/101 (COCO) and 0.75 (Ultralytics-style)
  - wrong class or off-image boxes → 0
  - ranking and duplicates handled
  - tiles cover every image, and every box fits whole in some tile
  - tile merge behaves as specified
- Tiler output is deterministic across worker counts.
- 2-epoch smoke runs of B0 and B1 trained and evaluated end to end.
- Kill-and-resume works.
- The notebook's data cell works for a flat zip, a wrapped zip and a plain folder.
- `pred_review.py` runs.
- `eval.py` and `train.py` have changed since round 1, so treat the round-1 smoke tests as covering the old versions only.

## Still uncertain
- **Our scorer has not been checked against Ultralytics val on a model that actually detects trucks.** B0's Colab
  eval does this (`--ultra-crosscheck`); expect close, not identical.
- **Per-epoch val for B1 runs on un-sliced 1024 px full images.** It shows training dynamics only and is not
  comparable with the sliced metric.
- **Ultralytics also saves `best.pt` when `val=true`. Ignore it:** `last.pt` is the reported checkpoint.
- `analysis/eda.py` must not be re-run on a subset.

## First commands
1. Create the GitHub repo yourself and push `main`.
2. On Colab, open `notebooks/colab_train.ipynb` and set the first cell: `DRIVE_DATA`, `DRIVE_RUNS`, `REPO_URL`,
   `BRANCH = "main"`.
3. If the repo is private, add a `GH_TOKEN` secret.
4. Choose a **T4** runtime (Python 3.13). Run the setup cells, the tests and the **1-epoch timing probe**, then
   stop. Send me the `timing_estimate.csv` numbers (or the printed table) so the schedule can be decided before the
   full B0 and B1 runs.

## T4 / Python 3.13 notes (round 3, also not run)
- Colab torch is kept as shipped. The other requirements are installed against a constraints file built from
  Colab's own torch, torchvision, numpy and opencv versions.
- The full lock is saved to `DRIVE_RUNS/colab_env_lock.txt`. Each run's `env/` records the GPU (e.g. "Tesla T4"),
  GPU memory, CUDA, torch and Ultralytics versions.
- The requirement pins are the versions I tested locally on Python 3.11. Whether every pin has a Python 3.13 wheel
  has not been checked here: the install cell will fail loudly if one doesn't. Update `requirements.txt` with the
  version that does install.
- Both configs set `amp: true`. `last.pt` is written to Drive every epoch.
- B1 at batch 16 / 1024 px may not fit in 16 GB. The fallback to batch 8 is automatic and recorded.
