# Experiment log

Each entry: Observation, Hypothesis, Changes vs. previous run, Results, Conclusion, Next step.
Every number cites the file it came from. Hypothesis and Conclusion are written by the author.

---

## Pre-run timing probe (1 epoch each, Colab T4 free tier; old config: optimizer=auto, 100 epochs)
Source: `runs/_timing/` on Drive (`timing_estimate.csv`, `train/results.csv`, `training_summary.json`,
`tiling_params.json`), as reported from the Colab run. The probe folders are not in git.
- B0: 28 it/epoch at 5.9 s/it, ~3.4 min/epoch including val. Throughput looked CPU-bound (decoding ~3000 px PNGs
  with 2 dataloader workers).
- B1: 3785 tiles, 237 it/epoch at 1.1 s/it, 10.5 GB GPU memory at batch 16, ~4.6 min/epoch including val.
- Tiling: 3785 tiles written, 2772 with boxes, 1013 empty kept, 499 partially visible boxes dropped; overlap 256 >
  max box side 161.
- `optimizer=auto` chose AdamW (lr 0.001111) for the probe. It would switch to SGD above 10k iterations, so the
  planned B0 (~2.8k iterations) and B1 (~23.7k) would have used different optimizers.
- cls_loss after epoch 1: B0 146.7, B1 7.9.
- Free-tier sessions cap at ~4h50m.

**Changes made because of the probe (both runs):** optimizer set explicitly (SGD, lr0 0.01, momentum 0.937),
50 epochs each, `cache: disk` on /content, checkpoints kept at completed epochs 10/20/30/40/50, and
`analysis/checkpoint_curve.py` scores every kept checkpoint with the real metric.

---

## B0: naive full-image baseline (`configs/b0.yaml`, run `b0_full640`)

**Observation**
- Images are large: median 3228 x 2891 px in train and 3114 x 2911.5 px in val (`figures/eda/tables/images.csv`).
- Objects are tiny: median sqrt(box area) is 22.05 px in train and 25.50 px in val (`figures/eda/summary.json`).
  77.2% of train boxes (5881 of 7618) are COCO "small", under 32² px (`figures/eda/tables/size_bins.csv`).
- Letterboxing a ~3200 px image to 640 px scales it by ~0.2, so a median truck would cover roughly 4-5 px.
  This is derived from the numbers above, not measured.

**Hypothesis**
TODO (me)

**Changes vs. previous run**
First run. Starts from COCO-pretrained YOLO11s (`yolo11s.pt`).
- Whole image at imgsz 640, 50 epochs (same as B1), batch 16 (falls back to 8 on out-of-memory; the batch actually
  used is recorded), seed 0, deterministic=True.
- SGD lr0 0.01, momentum 0.937 (explicit; same as B1). AMP on. `cache: disk`. Ultralytics default augmentations and LR
  schedule. Weights kept at epochs 10/20/30/40/50.
- Run order: B1 first, then B0.
- `val=true` only to log per-epoch val losses and Ultralytics val mAP (`training_curves.png/.csv`). `last.pt` is
  evaluated; no checkpoint is selected on val.
- Eval: `eval.py`, full image at 640, conf 0.001, NMS IoU 0.7, max_det 334.

**Results**
Pending Colab. Fill from:
- `runs/b0_full640/eval/per_class.csv` and `metrics.json` (headline = COCO 101-pt AP50; the scorer comparison against
  pycocotools and Ultralytics val is in `scorer_comparison.csv`)
- `training_summary.json` (total iterations, batch used)
- `figures/b0_full640/checkpoint_curve.csv/.png` (real-metric mAP50 per kept checkpoint)
- `figures/b0_full640/` (TP/FP/FN grids, confusion matrix)

**Conclusion**
TODO (me)

**Next step**
Pending results.

---

## B1: native-resolution tiled baseline (`configs/b1.yaml`, run `b1_tile1024`)

**Observation**
- Same object-size evidence as B0.
- The largest train box side is 161 px (`figures/eda/tables/boxes.csv`, also printed by `tools/make_tiles.py`).
  A 256 px tile overlap therefore puts every box entirely inside at least one tile.
  `tests/test_pipeline.py` verifies this on the local val labels.

**Hypothesis**
TODO (me)

**Changes vs. previous run (B0)**
- Train on 1024 x 1024 tiles cut from full-resolution train images, at imgsz 1024 (no downscaling).
  - Overlap 256.
  - 20% of box-free tiles kept (seed 0).
  - A clipped box is kept if at least 50% of its area is visible in the tile.
- Val is not tiled for training. The evaluated model is `last.pt`.
- Epochs: 50 and SGD lr0 0.01 momentum 0.937, the same as B0. **Note:** B1 still does more iterations per epoch than B0. One epoch passes over every
  written tile (many per source image), not over 443 whole images. So equal epochs are not equal gradient steps.
  The actual counts are in each run's `training_summary.json` (`iterations_per_epoch`, `total_iterations`), and any
  B0-vs-B1 difference has to be read with that in mind.
- Eval: sliced inference on full-res val with the same tile geometry.
  - Class-wise NMS across tiles, using intersection-over-smaller at 0.6.
  - conf 0.001, max_det 334.
  - Scored against the original full-image labels with the same code as B0.

**Results**
Pending Colab. Fill from:
- `runs/b1_tile1024/eval/per_class.csv`, `metrics.json` and `scorer_comparison.csv`
- `training_summary.json` (iterations, batch used) and `tiling_params.json` (tile counts)
- `figures/b1_tile1024/checkpoint_curve.csv/.png` (sliced mAP50 per kept checkpoint)
- `figures/b1_tile1024/merge_sensitivity.csv` (mAP50 under IoS 0.5/0.6/0.7, IoU 0.5 and no merge, re-scored from
  the saved raw tile predictions without re-inference)

Per-epoch val curves for B1 come from Ultralytics' val on un-sliced full images at 1024. They show training
dynamics only and are not comparable with the sliced metric.

**Conclusion**
TODO (me)

**Next step**
Pending results.
