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

Rejection condition (b) was largely met: trucks of 8–48 px (92% of val trucks) are missed at a flat 71–79%, so size isn't what limits most detections. Only the largest 5% (48–96 px, n = 73) are found more often (52% missed).

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

---

## B1h: B1 with 40 train images held out (`configs/b1h.yaml`, run `b1h_tile1024_holdout40`)

### Results

| Images                   |        mAP50 (95% CI) | Trucks found |
| ------------------------ | --------------------: | -----------: |
| Training images (seen)   | **0.378 (0.30–0.46)** |          92% |
| Held-out training images | **0.151 (0.06–0.19)** |          85% |
| Validation images        | **0.107 (0.06–0.17)** |          66% |

*Trucks found* = share of trucks matched by any prediction of any class, at any confidence (IoU 0.5, down to the eval
floor conf 0.001): 735/796, 629/738 and 1032/1552 GT. Sources:
`results/b1h_tile1024_holdout40/{eval_train40,eval_holdout40,eval}/metrics.json` and `class_agnostic.json`
(Kaggle kernel `aradhya1211/auric-b1h` v1, code `4a17eb1`, 2 × Tesla T4, `last.pt`, max_det 902 for all three).

### Conclusion

B1h confirms that the model struggles with unseen images, not just the validation set. Its mAP50 drops from 0.378 on training images to 0.151 on held-out images, indicating poor generalization with the current dataset of around 400 images.

The held-out and validation confidence intervals overlap, suggesting that poor generalization is the main issue rather than a unique problem with the validation set. However, the model finds fewer trucks on validation images (66% vs. 85%). The cause is not yet known. Density is a candidate, but within the validation set miss rates did not vary with density, so this remains untested.

Classification remains the biggest bottleneck, consistent with B1. More training data, stronger augmentation, and a separate classifier trained on cropped trucks are potential next steps.

**Pre-registration:** No prediction was written before this run.

**Important caveat:** B1h scored 0.107 on validation, compared with B1's 0.071. Differences of this size can arise from run-to-run randomness and platform differences; the small change in the detection limit explains only about 0.006. A single run is therefore not enough to establish a meaningful improvement.

### Next steps

* Test whether adding 500 labelled images improves generalization using learning curves on the held-out set.
* Experiment with stronger data augmentation.
* Evaluate a separate classifier on cropped truck images to improve Cargo vs. Box classification.

---

## Plan from here (2026-10-02, after B1h)

| # | What | Answers | Cost |
|---|---|---|---|
| 1 | Learning curves: train on 25/50/75% of the data, plus a repeat of the 100% run with a different seed | §5.3 "would 500 more labels help?", plus how big run-to-run noise really is | about 7 h of Kaggle GPU, in the background |
| 2 | Smallest useful subset | §5.4 | reuses the curves, plus 1–2 runs |
| 3 | Which training examples resist learning | §5.2 | about 1 run |
| 4 | Final analysis + report + README | §6 | your writing |

---

## LC: learning curves (§5.3), runs b1h_f25 / b1h_f50 / b1h_f75 + b1h_seed1

**Pre-registration (written by the author before any learning-curve result was seen)**
- Written at: 2 Oct 2026, 7:06 PM IST, before any learning-curve run
- Question: would 500 more labelled images meaningfully raise mAP50?
- Prediction: the curve is still rising at 100%. The big gap between seen (0.38) and unseen (0.15) images is a classic
  sign the model is still data-hungry.
- Predicted held-out mAP50 at 25% (about 100 images): 0.07–0.12. Learning curves usually bend: the first images teach
  the most, so a quarter of the data typically gives more than a quarter of the score (well above 0.038, pure
  proportion) but still clearly below 0.151.
- Decision rule: if the extrapolated gain at +500 images is smaller than the seed-to-seed noise, more labels alone
  won't help.

### Results

All runs used the same number of training iterations (~10,750), with warmup and close-mosaic scaled to match, so data size is the only variable.

| Training images | Held-out mAP50 | Held-out class-agnostic AP50 |
| --------------: | -------------: | ---------------------------: |
|       101 (25%) |          0.061 |                        0.160 |
|       202 (50%) |          0.095 |                        0.232 |
|       302 (75%) |          0.105 |                        0.293 |
|      403 (100%) |  0.151 / 0.133 |                0.362 / 0.362 |

403 row: seed 0 (B1h) / seed 1. Sources: `figures/learning_curve/learning_curve.csv` and `power_law_fit.csv`
(Kaggle kernels `aradhya1211/auric-b1h-{f25,f50,f75,seed1}`, code `ab32af2`; B1h as the 100% seed-0 point).

### Conclusion

The learning curve is still rising, with no clear sign of flattening. As training data increases, both overall performance and truck detection improve. My prediction that the curve would keep rising was correct.

At 25% of the data, mAP50 reached 0.061, slightly below my predicted range of 0.07–0.12. However, the 95% confidence interval (0.012–0.090) overlaps that range, so the result is reasonably close to the prediction.

A comparison between two runs revealed that the random seed affects classification much more than detection. The validation mAP50 differed by 0.043, while class-agnostic AP50 differed by only 0.004. This suggests that finding trucks is relatively stable, but predicting their types remains unreliable. Class-agnostic metrics are therefore a more consistent signal of detection performance than per-class mAP from a single run.

### Would 500 additional labels help?

The learning curve suggests that more labelled data would improve performance, particularly truck detection. However, extrapolating beyond the current 403 images is uncertain, and even an optimistic projection falls well short of the target mAP50 of 0.75.

By my pre-registered rule: the projected held-out gain at +500 images (+0.080 mAP50) is larger than the seed-to-seed noise (0.017), so more labels should help. But the projection (0.215, interval 0.076–0.276) extrapolates 2.24× beyond the data and was flagged unreliable, so only the direction is trustworthy, not the size.

**Conclusion:** More data should help, but additional labels alone are unlikely to achieve the target. Classification needs a separate improvement strategy.

### Smallest useful training subset

No random subset tested retained 90% of full-data performance. Even 75% of the images achieved 0.105, about 74% of the full-data held-out mAP50 (0.142, the mean of both seeds). For detection alone (class-agnostic) it retained 81%. The remaining question is whether a carefully selected subset can retain more performance than a random sample of the same size.

---

## §5.1 If locations were perfect

Method: for every true truck box in val, I took the model's own class scores at that
location and checked whether its top class was right. This removes detection from the
picture and tests classification alone.

Results (val, 1,552 trucks):
- B1: 55% correct (39% averaged per class). B1h: 60% (44%).
- Always answering "Cargo" would give 51.5%; random guessing gives 20% per class.

Conclusion: even with perfect locations, the model names the truck type correctly only
slightly more often than always saying "Cargo". It has learned something (44% vs 20%
per class), but not much. The error breakdown (B1h) agrees: fixing wrong-class errors would
add +0.147 mAP50, far more than fixing box positions (+0.022) or missed trucks (+0.044).
So classification, not localization, is the main ceiling on performance.
Main confusion: Cargo Truck vs Truck w/Box.

Sources: `figures/{b1_tile1024,b1h_tile1024_holdout40}/gt_oracle/comparison.csv` and `confusion_pool.csv` (pooled-anchor
mode, all GT), `figures/b1h_tile1024_holdout40/errors/tide_dAP.csv`. For B1 the same fixes give +0.190 / +0.014 / +0.041
(`figures/b1_tile1024/errors/tide_dAP.csv`).
