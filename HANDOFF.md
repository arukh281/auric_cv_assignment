# Handoff (2026-10-02)

Read this first, then `STATUS.md`, `EXPERIMENTS.md`, `README.md`. Repo: `arukh281/auric_cv_assignment` (private),
local checkout `~/Desktop/Work/auric_cv_assignment`.

## The task
Auric AI computer-vision take-home (brief: `Computer_Vision_Assignment_Auric_AI_Technologies__1_.pdf`, not in git).
- Build a 5-class detector (Cargo Truck, Truck w/Box, Truck w/Flatbed, Truck Tractor, Truck w/Liquid) with
  **≥ 0.75 mAP50 on the provided val set**.
- Diagnosis, experimental reasoning and reproducibility are graded as much as the score.
- Deliverables: report, final weights, eval code (val images → predictions → metrics), training code and configs,
  requirements, README.
- Research questions to come: 5.1 GT-box oracle, 5.2 hard examples, 5.3 value of 500 more labels, 5.4 smallest subset
  reaching 90% of full-data mAP.

**Working rules (from the user):**
- The user writes Hypothesis and Conclusion in EXPERIMENTS.md. Leave them as "TODO (me)".
- Never invent numbers. Every number must come from a file the code produced.
- Analysis lives in `analysis/` scripts that write figures and CSVs.
- STOP at the end of each phase.
- Never delete anything on Drive. Never print the GitHub token.
- No heavy compute on the Mac (M4, disk nearly full). All training happens on Colab.
- Do not start a new training run without the user's go-ahead.

## Dataset facts (Phase 1, `figures/eda/`, from `analysis/eda.py` on the FULL dataset)
- Train: 443 images, 7618 boxes. Val: **22 images**, 1552 boxes. Overhead satellite imagery, ~3000 px images
  (val up to 7204 × 5932).
- Tiny objects: median sqrt(box area) is 22.05 px (train) and 25.50 px (val). 77.2% of train boxes are COCO "small".
  The largest box side is 161 px.
- Imbalance: Cargo Truck 50%, Box 31%. Truck w/Liquid has only **20 val instances** in 8 images, so its AP is noisy.
- No train/val near-duplicates (pHash). Labels are clean apart from 36 boxes under 4 px and 1 empty train image.
- **Never re-run `analysis/eda.py` locally**: `data/` now points at a 42-image subset (`data_small/`), and a re-run
  would overwrite the full-data tables.

## Decisions and why
| Decision | Why |
|---|---|
| YOLO11s, **COCO-pretrained**; **no xView weights** | Our classes are xView classes. xView checkpoints may have seen our val scenes, which would inflate val mAP with no way to measure it (REPORT.md). |
| **B0** = whole image letterboxed to 640 (reference) | Shows the cost of downscaling: ~3200 px → 640 shrinks a median truck to ~4–5 px. |
| **B1** = train on **1024 tiles at native resolution, overlap 256**, 20% empty tiles kept, clipped box kept if ≥ 50% visible | No downscaling. Overlap 256 > max box side 161, so every box is whole in at least one tile (tested). Val is not tiled for training: eval slices full-res val and scores against the original full-image labels. |
| **Tile merge**: class-wise NMS on intersection-over-smaller (IoS) 0.6 | IoS also removes tile-edge fragments that sit inside a full box. `analysis/merge_sensitivity.py` re-scores the saved raw tile predictions at IoS 0.5/0.6/0.7, IoU 0.5 and no-merge, without re-inference. |
| **Metric**: headline = **COCO 101-point AP50**, our own scorer (`detlib/scoring.py`), same code for B0 and B1 | Standard definition. pycocotools AP50 and the Ultralytics-interpolated value are reported alongside. A warning fires if ours and pycocotools differ by > 0.005; on a noisy test case they differ by 0.0009. Ultralytics' interpolation reads higher and caps a perfect AP at 0.995. |
| **Evaluate `last.pt`**, never a checkpoint picked on val | Val is the only test set. Per-epoch Ultralytics val (`val: true`) is logged for curves only. For B1 it is un-sliced, so it is not comparable with the real metric. |
| **50 epochs for both**, equal schedules | The timing probe showed B1 ≈ 4.6 min/epoch on a T4. 50 epochs fits the free-tier ~4h50m session. B1 still does more iterations per epoch (3785 tiles vs 443 images), recorded in `training_summary.json`. |
| **Explicit SGD, lr0 0.01, momentum 0.937** | `optimizer=auto` picks AdamW under 10k iterations and SGD above, so B0 (~2.8k iterations) and B1 (~23.7k) would have used different optimizers. |
| **`cache: disk`** on /content | The probe showed B0 was CPU-bound decoding 3000 px PNGs. Ultralytics warns that RAM caching is non-deterministic. The cache sits next to the images on /content, not on Drive. |
| AMP, seed 0, `deterministic: true`, batch 16 with automatic fallback to 8 on out-of-memory (recorded) | T4 16 GB; reproducibility. |
| Checkpoints at completed epochs 10/20/30/40/50 (`train.py` hook), `last.pt` every epoch on Drive | Ultralytics' `save_period` counts from epoch 0 (it would keep 1/11/21…). `analysis/checkpoint_curve.py` scores each kept checkpoint with the real metric. |
| Colab's preinstalled CUDA torch kept; other requirements installed against constraints | Colab runs Python 3.13. Reinstalling torch breaks the CUDA stack. The env lock is saved to Drive per session and per run. |

## What has been run, and results so far
- **Phase 1 EDA** (full data, on the Mac): done. Results above and in `figures/eda/`.
- **Local smoke tests** (Mac, 42-image subset, CPU):
  - 12 known-answer tests pass: scoring vs pycocotools, tiling, merge, OOM fallback.
  - 2-epoch B0/B1 pipeline runs, kill-and-resume, and notebook data-cell tests all passed with older code.
- **T4 timing probe** (1 epoch each, old config: `optimizer=auto`), from `runs/_timing/` on Drive as reported by
  the user:
  - B0: 28 it/epoch at 5.9 s/it, ~3.4 min/epoch including val.
  - B1: 3785 tiles (2772 with boxes, 1013 empty), 499 partial boxes dropped, 237 it/epoch at 1.1 s/it, 10.5 GB GPU
    memory at batch 16, ~4.6 min/epoch.
  - cls_loss after epoch 1: B0 146.7, B1 7.9.
- **No B0 or B1 mAP results exist yet.**

## Running now
- **B1** (`configs/b1.yaml`, run `b1_tile1024`) has been training on Colab T4 since **~11:43 (2026-10-02, user's
  local time)**.
  - Started from the Colab Terminal with `scripts/run_b1.sh` at commit `bebdd2b`.
  - Expected ~3.8 h for 50 epochs, based on the probe.
  - When training finishes, the script runs eval (sliced), the prediction review, merge sensitivity and the
    checkpoint curve by itself.
  - Log: `/content/drive/MyDrive/auric/runs/b1_tile1024/run.log`. Monitor with `bash scripts/status.sh`.

## Known issues
- **Local `.venv` is broken**: `~/Desktop/Work/auric_cv_assignment/.venv` is a symlink to itself, created at 10:52
  today, so the real environment is gone. Recreate it only if local runs are needed:
  `uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt torch torchvision`. That is ~1.2 GB and the
  disk is tight.
- **The local root checkout may lag GitHub `main`.** Run `git pull` in `~/Desktop/Work/auric_cv_assignment`. Work was
  done in a worktree, `.claude/worktrees/phase2-pipeline` (branch `worktree-phase2-pipeline`), which can be deleted
  once merged.
- **Not yet exercised on Colab with the new config:** `checkpoint_curve.py`, the exact-epoch checkpoint hook and the
  finished-run skip. The first B1 run will exercise them; check `run.log` for tracebacks.
- **Colab Terminal cannot mount Drive or read secrets.** Mount from a notebook cell. Paste the token at the silent
  prompt when cloning (README).
- **Scorer vs Ultralytics val:** not yet compared on a real model. B0's eval runs `--ultra-crosscheck`.

## Exact next steps
1. Watch B1 with `bash scripts/status.sh`. If the session dies, start a new one: mount Drive from a cell, re-clone
   (README step 2), `bash scripts/colab_setup.sh`, then `nohup bash scripts/run_b1.sh > /dev/null 2>&1 &`. It
   resumes from `last.pt`.
2. When `run.log` shows `b1_tile1024 DONE`, report from these files (no other numbers):
   - `runs/b1_tile1024/eval/per_class.csv`, `metrics.json` and `scorer_comparison.csv`
   - `training_summary.json`
   - `figures/b1_tile1024/checkpoint_curve.csv` and `merge_sensitivity.csv`
3. Start B0: `nohup bash scripts/run_b0.sh > /dev/null 2>&1 &`, then report the same files for `b0_full640`,
   including the Ultralytics cross-check.
4. Copy the run folders (without weights) and figures from Drive into the repo docs. Fill the Results sections in
   EXPERIMENTS.md. The user fills Hypothesis and Conclusion. **STOP (end of Phase 2).**
5. Phase 3, only on the user's go-ahead. `analysis/errors.py` and `analysis/gt_box_oracle.py` (5.1) are written and
   tested on synthetic predictions and a random-weight model (`tests/test_analysis.py`); not yet run on a trained model.
   They are not wired into `scripts/_run.sh`. `analysis/errors.py`:
   - TIDE-style error bins: classification, localization, both, duplicate, background FP, missed GT, each with the
     mAP50 gain from oracle-fixing it.
   - FN/FP rates sliced by class, size bin, objects per image and brightness, plus size × class tables.
   - Confusion matrix with a background row/column.
   - Example crops per error type and class.
