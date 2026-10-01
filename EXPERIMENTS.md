# Experiment log

Each entry: Observation, Hypothesis, Changes vs. previous run, Results, Conclusion, Next step.
Every number cites the file it came from. Hypothesis and Conclusion are written by the author.

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
First run. Starts from COCO-pretrained YOLO11s (`yolo11s.pt`). Whole image at imgsz 640, 50 epochs, batch 16,
seed 0, deterministic=True, Ultralytics default augmentations / optimizer (`optimizer=auto`) / LR schedule.
`val=false`: no per-epoch validation and no best-epoch selection on val; `last.pt` is evaluated.
Eval: `eval.py`, full image at 640, conf 0.001, NMS IoU 0.7, max_det 334.

**Results**
Pending Colab. Fill from `runs/b0_full640/eval/per_class.csv` and `metrics.json`. Qualitative results are in
`figures/b0_full640/` (TP/FP/FN grids, confusion matrix).

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
- Epochs: 100, against 50 for B0. **Note:** this confounds the comparison with B0. An epoch is also a different
  number of gradient steps in the two setups (tiles vs whole images).
- Eval: sliced inference on full-res val with the same tile geometry.
  - Class-wise NMS across tiles, using intersection-over-smaller at 0.6.
  - conf 0.001, max_det 334.
  - Scored against the original full-image labels with the same code as B0.

**Results**
Pending Colab. Fill from `runs/b1_tile1024/eval/per_class.csv` and `metrics.json`, plus `tiling_params.json` for
tile counts.

**Conclusion**
TODO (me)

**Next step**
Pending results.
