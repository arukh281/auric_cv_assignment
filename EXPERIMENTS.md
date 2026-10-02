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
- Compute: 443 images x 50 epochs, batch=16 (`configs/b0.yaml`) -> ceil(443/16) x 50 = 28 x 50 = 1400 total
  iterations. B1 sees ~8.5x more iterations (11850 / 1400 = 8.46), so the comparison is "equal epochs, not equal
  compute"; this is a known confound.
- Eval: `eval.py`, full image at 640, conf 0.001, NMS IoU 0.7, max_det 334.

**Results**
Pending Colab. Fill from:
- `runs/b0_full640/eval/per_class.csv` and `metrics.json` (headline = COCO 101-pt AP50; the scorer comparison against
  pycocotools and Ultralytics val is in `scorer_comparison.csv`)
- `training_summary.json` (total iterations, batch used)
- `figures/b0_full640/checkpoint_curve.csv/.png` (real-metric mAP50 per kept checkpoint)
- `figures/b0_full640/` (TP/FP/FN grids, confusion matrix)

**Conclusion**
**Result:** mAP50 = 0.002, indicating almost no useful detection.

At 640 px, only 26% of trucks aligned sufficiently with a prediction slot, compared with 93% using 1024 px tiles. This confirms that tiling was necessary, although it did not solve every problem.

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

*Pre-registration (written by the author before any sliced B1 metric was seen):*
- Written at: 2 Oct 2026, 1:35 pm IST, before seeing any sliced B1 result
- Hypothesis: Trucks are tiny objects in huge 3000 px images. Shrinking to 640 px makes a truck about 5 px, too small to learn: B0's training error started extremely high (cls_loss 146.7). Tiling at full resolution keeps trucks at about 22 px, so B1 should find far more trucks than B0.
- Predicted overall mAP50 range: 0.40–0.60, below the 0.75 target. 22 px is still small for YOLO's finest level (stride 8 means about 3 grid cells per truck), and val images are 10× denser than train.
- Predicted best class / worst class and why:
  - Best: Cargo Truck: about half of all training data and the most val instances (800).
  - Worst: Truck w/Liquid: only 2.5% of training data and 20 val instances, so its score will be low and very uncertain (wide CI).
- Predicted dominant error type (Loc/Cls/Dupe/Bkg/Both/Miss): Classification error (Cls): the model will find most trucks but often name the wrong type, because from above at 22 px the five types look similar.
- What result would make me reject the hypothesis: (a) B1 is not clearly better than B0, with confidence intervals that don't overlap, or (b) small trucks are missed about as often as large ones.

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
- Compute: 3785 tiles x 50 epochs, batch=16 (`configs/b1.yaml`) -> ceil(3785/16) x 50 = 237 x 50 = 11850 total
  iterations. B1 sees ~8.5x more iterations than B0 (11850 / 1400 = 8.46), so the comparison is "equal epochs, not
  equal compute"; this is a known confound. (If the out-of-memory fallback to batch 8 triggers, the batch actually
  used is in `training_summary.json`.)
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
**Result:** mAP50 = 0.071 on validation (95% CI: 0.042–0.124), well below my predicted 0.40–0.60.

I was right about classification being a major problem: the model often confuses Cargo and Box trucks. However, overall performance was much worse than expected. Tiling fixed the image-resolution issue, but trucks between 8 and 48 px are still missed about 71–79% of the time (at per-class F1-optimal thresholds). Only the largest trucks (48–96 px) perform somewhat better, with a miss rate of about 52%.

Rejection condition (b) was met: on validation, truck size is no longer the main limitation.

**Diagnosis log**

* **Scoring bug?** Unlikely. The pipeline scored 0.40 on 40 training images the model had already seen.
* **Tile merging?** Helps rather than hurts. Disabling it dropped the score from 0.071 to 0.048.
* **Detection limit?** Minor effect; increasing it improved the score by just 0.006.
* **Domain shift?** No clear evidence. AUCs were 0.34–0.45, with all 95% confidence intervals including 0.5.
* **Poor image quality?** Not the main cause; normal and blurry/hazy images performed similarly.

*What we found:* The model misses about 42% of trucks and correctly classifies their type only 55% of the time, barely above always predicting Cargo (51.5%). Average per-class accuracy is 39%, compared with 20% for random guessing, suggesting it has learned something but remains weak.

**Open question:** Is the validation score unusually low, or does the model perform similarly on any unseen image? B1h tests this using 40 held-out training images.

**Next step**
Pending results.
