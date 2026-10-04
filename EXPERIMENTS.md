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

---

## S54: smart vs random subsets (§5.4), runs b1h_smart50 / b1h_smart75 + b1h_f50_seed1 / b1h_f75_seed1

**Pre-registration (written before any S54 run was launched)**
- Written at: 3 Oct 2026, 2:14 AM IST
- Pool: the same 403 non-holdout images. Sizes: 202 (50%) and 302 (75%), to compare with the existing random f50 and f75.
- Selection "smart": (1) class coverage first: add images in order of rarest class contained (Liquid, then Tractor,
  then Flatbed, then Box) until each class's GT box count is at least its proportional share for that subset size;
  (2) fill the rest by greedy k-center (farthest-point) on image embeddings, seed 0. Embedding = mean of DINOv2-small
  CLS embeddings over the image's 1024 px tiles (fallback: ImageNet ResNet18 if DINOv2 can't be loaded; the one used
  is recorded). Never include 1938.png. Lists: splits/train_smart50_seed0.txt and splits/train_smart75_seed0.txt.
- Runs (identical to the LC runs: ~10,750 iterations with scaled warmup and close_mosaic, max_det 902 evals on val and
  holdout40, class-agnostic on both, run_analysis): b1h_smart50 (seed 0), b1h_smart75 (seed 0), b1h_f50_seed1 and
  b1h_f75_seed1 (the existing random subsets, training seed 1, to measure noise at those sizes).
- Decision rule: "Smart selection beats random at a given size only if its held-out class-agnostic AP50 exceeds BOTH
  random runs at that size by more than their seed spread. Same check reported for held-out mAP50 as secondary. A
  subset 'recovers 90%' if its held-out mAP50 >= 0.90 x 0.1420 = 0.128."
- Prediction: none provided. The overnight instructions contained the unfilled placeholder
  "<<< WRITE YOUR GUESS HERE, e.g. "smart75 beats random75 on class-agnostic AP50 but stays below 0.128 mAP50" >>>",
  so no prediction was written before the runs.

---

## E1 / E2: is B1h undertrained, and does scale 0.5 hurt small trucks? (`configs/e1_b1h_150ep.yaml`, `configs/e2_b1h_150ep_scale02.yaml`)

**Pre-registration (written 3 Oct 2026, 11:09 PM IST, before any E1/E2 result exists; spec from the author's planning)**

**Observation**
- B1h at epoch 50 still had train cls_loss 1.60 (`train/results.csv` of kernel aradhya1211/auric-b1h; plotted in
  `figures/sanity/b1h_loss_curves.png`), and the losses were still falling over epochs 42–50 (cls −0.030/epoch).
- B1h val mAP50 was still rising over the last 20 epochs: 0.067 → 0.097 → 0.106 at epochs 30 / 40 / 50
  (`figures/b1h_tile1024_holdout40/checkpoint_curve.csv`).
- B1h reached only 0.378 mAP50 on its own training images (train40; `results/b1h_tile1024_holdout40/eval_train40/metrics.json`).
- Together these look like underfitting.

**Hypotheses**
- E1 tests undertraining.
- E2 tests whether scale 0.5 (which can shrink a ~22 px truck to ~11 px) hurts small trucks.

**Changes vs. B1h**
- E1 changes only the epochs: 50 → 150. Everything else is unchanged: seed 0, holdout list, eval max_det 902,
  checkpoints every 10 epochs, and warmup_epochs 3 / close_mosaic 10.
- E2 changes epochs (50 → 150) and scale (0.5 → 0.2), so E2 − E1 isolates scale.
- Both are trained at the same time in one Kaggle session: E1 on GPU 0, E2 on GPU 1, workers 2 each, each with its
  own tile/cache folder (`scripts/run_pair.py`). Each then runs the normal eval on val and holdout40,
  class-agnostic scoring, and run_analysis.
- Note: 3× the iterations of B1h (32,250 vs 10,750), so the comparison with B1h is not at equal compute, by design.

**Predictions**
- E1 supports undertraining if its val and holdout mAP50 beat B1h by more than the seed spread (0.043 val / 0.017
  holdout) and train40 rises well above 0.38.
- E2 supports the scale hypothesis if it beats E1 by more than that spread, mainly on boxes under 32 px.
- Smaller differences count as noise.

**Run-time rule (agreed before launch)**
- After 2 epochs, each run's training finish is projected inside the kernel (`scripts/run_pair.py`).
- If both are projected to finish within 10.5 h of kernel start (about 4.2 min per epoch or less), both continue.
- Otherwise E2 is stopped and relaunched as its own kernel after E1 finishes.
- Decision: not applied. E1 and E2 ran as two separate GPU kernels (see the launch record below), so the in-kernel rule was not used.

---

## SANITY: why does B1h reach only ~0.38 mAP50 on its own training images? (kernel `aradhya1211/auric-sanity`, code `f2388f1`)

**Observation**
- B1h scores 0.378 mAP50 on 40 of its own training images (`results/b1h_tile1024_holdout40/eval_train40/metrics.json`).
- Its train losses at epoch 50 are box 1.588 / cls 1.596 / dfl 1.003, and they are still falling. Slope over epochs
  42–50 (after the close_mosaic jump at 41): box −0.0099, cls −0.030, dfl −0.0018 per epoch
  (`results/b1h_tile1024_holdout40/train/args.yaml` run; curves in `figures/sanity/b1h_loss_curves.png`).

**Hypothesis**
The low in-sample score could come from (a) wrong training settings, (b) broken tile labels, or (c) a model or
pipeline that cannot fit this data at all.

**Changes**
No new detector training on the full data. Three checks:
1. **Effective settings:** read from the B1h run's `train/args.yaml`, copied to
   `results/b1h_tile1024_holdout40/train/args.yaml`.
2. **Label check:** B1h's tiles were regenerated on Kaggle with the same code, holdout list and indices. Every written
   tile's labels were compared with labels re-derived from the source image via `detlib.tiling.clip_boxes`. 12 tiles
   were rendered.
3. **Overfit test:** YOLO11s on 16 tiles covering all 5 classes (100 GT boxes), imgsz 1024, every augmentation off,
   300 epochs, scored with our scorer on the same tiles (`analysis/sanity_check.py`).

**Results**
- **Effective B1h settings** (`args.yaml`):
  - imgsz 1024, mosaic 1.0, scale 0.5, close_mosaic 10, rect false.
  - translate 0.1, fliplr 0.5, hsv 0.015 / 0.7 / 0.4, erasing 0.4.
  - SGD lr0 0.01, lrf 0.01, warmup 3, epochs 50, batch 16.
- **Tiling rules** (`tools/make_tiles.py`):
  - `e0.2` keeps 20% of box-free tiles (seed 0).
  - `v0.5` keeps a box cut by a tile edge only if ≥ 50% of its area is inside the tile.
- **Tiling counts** (`results/sanity/label_check.json`, identical to B1h's `tiling_params.json`):
  - 7192 tile windows; 3439 written (2504 with boxes, 935 empty kept); 3753 empty tiles dropped.
  - Box instances in written tiles: 12,455, of which 11,967 whole and 488 clipped but kept.
  - 451 box-in-tile instances dropped as < 50% visible (each such box is whole in another tile).
- **Label check** (`results/sanity/label_check.json`):
  - 3439 tiles / 12,455 boxes checked; **85 tiles mismatched**; 0 tiles with a class id outside 0–4 or a coordinate
    outside [0, 1].
  - Mismatches cluster in a few source images (first examples: 1058, 1095, 1127, 1175). Only tile names were saved,
    not the per-box differences.
  - One mismatched tile, `1127__x2304_y768`, is among the 12 rendered (`figures/sanity/label_tiles/`). Its boxes look
    aligned with vehicles when viewed. **Not yet diagnosed:** it is unknown whether these are real label differences
    or an artefact of the checker's box-pairing, e.g. coincident boxes with different classes.
- **Overfit test** (`results/sanity/overfit/`; training took 457 s for 300 epochs on a T4):

| epoch | mAP50 | Cargo | Box | Flatbed | Tractor | Liquid |
|---|---|---|---|---|---|---|
| 50 | 0.355 | 0.221 | 0.454 | 0.161 | 0.329 | 0.611 |
| 100 | 0.987 | 0.966 | 0.996 | 1.000 | 0.971 | 1.000 |
| 300 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

  - Final train losses at epoch 300: box 0.060 / **cls 0.097** / dfl 0.739, vs B1h's cls_loss 1.60 at epoch 50.
  - 0 GT boxes unmatched at conf ≥ 0.25 (`failures.csv`). Renders are in `figures/sanity/overfit/`.

**Conclusion**
- **Verdict: NOT PASS** under the pre-agreed rule, which requires zero label mismatches and overfit AP50 ≥ 0.9.
  - The overfit criterion passed (AP50 1.000 at epoch 300).
  - The label criterion did not (85 mismatched tiles).
- **E1/E2 were not launched.**
- Facts only, no interpretation yet: the model and pipeline can fit tiles of this data perfectly when augmentation is
  off. The 85 mismatched tiles are unexplained.

**Next**
- Diagnose the 85 mismatches: re-run the label check with per-box differences saved (CPU-only Kaggle kernel), to tell
  label problems from checker artefacts.
- Then re-apply the PASS rule before launching E1/E2.

**Kaggle GPU hours** (budget 29.8 h until the quota refresh on 2026-10-10 05:30 local)

| kernel | purpose | start → first seen complete (IST) | GPU h (upper bound from polls) |
|---|---|---|---|
| auric-sanity v1 | label check + 16-tile overfit | 22:53 → 23:11 | ≤ 0.30 |

- `kaggle quota` at ~23:15 reported 0.24 h used, 29.76 h remaining. It may not yet include the end of auric-sanity.

### SANITY → E1/E2: override of NOT PASS (author's decision, 3 Oct 2026, 11:22 PM IST)

The author overrode the NOT PASS and launched E1/E2 the same night. Reasons, as given:
- The overfit test passed (AP50 1.000 on all classes, cls_loss 0.097), so the pipeline can fit the labels.
- B1h's losses were still falling at epoch 50, consistent with undertraining.
- The 85 flagged tiles (2.5%) are unexplained and possibly a checker artefact. E1 and E2 use exactly B1h's tiles, so
  the E1/E2 vs B1h comparison is unaffected either way.

Follow-up launched at the same time: CPU-only kernel `aradhya1211/auric-label-mismatch` (no GPU quota), which re-checks
the flagged tiles with per-box differences (`analysis/label_mismatch.py`).

### E1 / E2: launch record

- E1 `aradhya1211/auric-e1-b1h-150ep` and E2 `aradhya1211/auric-e2-b1h-150ep-scale02` were pushed as **two separate
  GPU kernels** at 23:20 IST on 2 Oct, code `e72d3b8`.
  - Each trains on one T4 (device 0) with the committed configs and B1h's workers.
  - Each then runs `scripts/run_lc.sh` (eval on val + holdout40, class-agnostic) and `scripts/run_analysis.sh`.
- Both showed RUNNING at 23:21 IST. Kaggle accepted both, so the paired kernel and its 2-epoch rule were not used.
- E1/E2 run-time decision: n/a (separate sessions; Kaggle's 12 h session cap applies to each).

**Kaggle GPU budget** (29.8 h until the quota refresh on 2026-10-10 05:30 local)

| kernel | purpose | start (IST) | GPU h |
|---|---|---|---|
| auric-sanity v1 | label check + 16-tile overfit | 22:53 → 23:11 | ≤ 0.30 (measured from polls) |
| auric-e1-b1h-150ep v1 | E1, 150 epochs | 23:20 | estimate ~5–7 (3× B1h's ~1.8 h kernel, incl. 15-checkpoint curve) |
| auric-e2-b1h-150ep-scale02 v1 | E2, 150 epochs, scale 0.2 | 23:20 | estimate ~5–7 |
| auric-label-mismatch v1 | label re-check | 23:2x | 0 (CPU-only) |

- Total committed tonight: about 14 h at most, of 29.8 h.
- Actual hours to be filled from `kaggle quota` and the kernel logs after the runs.

### E1 / E2: Results (4 Oct 2026)

Kernels `aradhya1211/auric-e1-b1h-150ep` and `aradhya1211/auric-e2-b1h-150ep-scale02`, code `e72d3b8`, 1 × T4 each.
train40: CPU-only kernel `aradhya1211/auric-e1e2-train40` (code `1f3c5bf`), same sliced settings and scorer as B1h's
train40, on the same 40 images (verified: identical `sampled_images`).

**Training actually done**
- **E1 stopped at epoch 145 of 150.** Ultralytics' EarlyStopping (default `patience=100`, never overridden in any of
  our configs) watches Ultralytics' own un-sliced val fitness. Its best was at epoch 45, so it stopped 100 epochs later.
  - 31,175 iterations, 4.43 h of training (`results/e1_b1h_150ep/train/results.csv`, `run.log`).
  - `last.pt` = epoch 145, which had 4 of the 10 planned no-mosaic epochs.
- E2 completed 150 epochs (32,250 iterations, 4.13 h).

| | B1h (50 ep) | E1 (150 ep, stopped at 145) | E2 (150 ep, scale 0.2) |
|---|---|---|---|
| val mAP50 (95% CI) | 0.1065 (0.0563–0.1653) | 0.0680 (0.0355–0.1243) | 0.0620 (0.0326–0.1104) |
| val AP50 Cargo / Box / Flatbed / Tractor / Liquid | 0.131 / 0.180 / 0.071 / 0.006 / 0.145 | 0.085 / 0.177 / 0.024 / 0.001 / 0.053 | 0.062 / 0.115 / 0.037 / 0.018 / 0.078 |
| holdout40 mAP50 (95% CI) | 0.1507 (0.0558–0.1874) | 0.0923 (0.0302–0.1384) | 0.1112 (0.0344–0.1391) |
| train40 mAP50 | 0.378 | **0.774** | **0.906** |
| class-agnostic AP50 val / holdout / train40 | 0.255 / 0.362 / – | 0.210 / 0.259 / 0.900 | 0.178 / 0.278 / 0.936 |
| final train loss box / cls / dfl | 1.588 / 1.596 / 1.003 (ep 50) | 1.179 / 0.795 / 0.893 (ep 145) | 0.864 / 0.543 / 0.837 (ep 150) |

Sources: `results/<run>/eval/per_class.csv`, `eval_holdout40/per_class.csv`, `eval_train40/{metrics,class_agnostic}.json`,
`train/results.csv` (B1h: `results/b1h_tile1024_holdout40/...`).

**Checkpoint curves** (val mAP50, sliced, our scorer; `results/<run>/checkpoint_curve/checkpoint_curve.csv`).
Holdout per checkpoint was never computed (the checkpoint curve scores val only).

| epoch | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 | 130 | 140 | last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 0.030 | 0.069 | 0.058 | 0.062 | 0.066 | 0.067 | 0.072 | 0.087 | 0.100 | 0.084 | 0.069 | 0.063 | 0.071 | 0.067 | 0.068 (145) |
| E2 | 0.049 | 0.062 | 0.070 | 0.105 | 0.107 | 0.088 | 0.079 | 0.068 | 0.068 | 0.062 | 0.062 | 0.060 | 0.064 | 0.066 | 0.062 (150) |

Val mAP50 is not rising at the end. It peaks (E1 0.100 at epoch 90; E2 0.107 at epochs 40–50) and then falls back.

**Scale test: val recall by GT box size** (conf ≥ 0.25, IoU ≥ 0.5, right class; computed from each run's
`eval/predictions.csv` with `detlib.scoring.match_image`, the matcher `errors.py` uses; B1h's values match its
`errors/op_gt.csv`).

| sqrt(area) | n GT | B1h | E1 | E2 |
|---|---|---|---|---|
| < 16 px | 263 | 0.057 | 0.061 | 0.061 |
| 16–32 px | 818 | 0.119 | 0.127 | 0.103 |
| ≥ 32 px | 471 | 0.274 | 0.255 | 0.208 |

AP50 by size was not computed. E1's `errors/` and `figures/` folders are missing from its Kaggle output (cause unknown).

### E1 / E2: Conclusion (verdicts against the pre-registered predictions; noise = seed spread 0.043 val / 0.017 holdout)

- **E1 (undertraining): not supported.**
  - The pre-registered condition was val and holdout above B1h by more than the spread, and train40 well above 0.38.
  - train40 rose a lot (0.378 → 0.774). But held-out fell: val −0.039 (within noise, wrong direction) and holdout
    −0.058 (beyond noise).
  - Training loss fell (cls 1.60 → 0.80) while held-out performance got worse, with val peaking mid-training. That is
    the pattern of overfitting, not undertraining.
  - Caveat: E1 stopped at epoch 145 (EarlyStopping), not 150.
- **E2 vs E1 (scale 0.5 hurts small trucks): inconclusive, leaning not supported.**
  - E2 − E1: val −0.006 (noise); holdout +0.019 (just above the 0.017 spread).
  - Recall on boxes under 16 px is identical (0.061 vs 0.061). 16–32 px is lower for E2 (0.103 vs 0.127).
  - The predicted gain "mainly on boxes under 32 px" did not appear.
- **What the two runs show together:** with 3× B1h's iterations, both runs fit their training images far better
  (train40 0.77 / 0.91) while held-out mAP50 stayed at or below B1h's. The gap between seen and unseen images widened.

### E1 / E2: Next
- Ultralytics' `patience` should be set explicitly (0 or above the epoch count) in every config so runs end where
  configured. It was 100 by default for all runs so far; only E1 was long enough to be affected.
- The evidence now points at generalisation and overfitting, not training length. Candidates already noted in this
  log: stronger augmentation or regularisation, more data (§5.3), and a separate crop classifier for Cargo vs Box.

### SANITY: label-mismatch re-check (4 Oct 2026; CPU-only kernel `aradhya1211/auric-label-mismatch`, code `1f3c5bf`)

- **All 85 flagged tiles** (from 32 source images) were re-paired one-to-one by IoU:
  - **3,278 of 3,278 boxes "same"**: identical class, every coordinate within 0.5 px.
  - 0 shifted, 0 re-classed, 0 missing, 0 extra (`results/sanity/label_mismatch/summary.json`, `box_diffs.csv`).
- **Source images of the flagged tiles:** 2,856 GT boxes, 0 pairs at IoU ≥ 0.9, 0 exact duplicates, 19 pairs at
  IoU ≥ 0.5 (`source_overlaps.csv`).
- The flagged tiles are the densest ones (up to 263 boxes per tile). 8 of the 85 renders are in
  `figures/sanity/label_mismatch/`.
- **Conclusion: a checker artefact, not a label problem.** The original check (`sanity_check.check_labels`) paired
  boxes by sorting coordinates. In dense tiles, near-equal coordinates (float round-trip of the YOLO text) reorder and
  mis-pair; IoU pairing finds every label identical.
- With this, both criteria of the sanity PASS rule hold: 0 real label mismatches, and overfit AP50 1.000.

### Kaggle GPU hours (update 4 Oct 2026, 10:xx IST)

| kernel | GPU h |
|---|---|
| auric-sanity v1 | ≤ 0.30 (22:53–23:11 IST, 2 Oct) |
| auric-e1-b1h-150ep v1 | ~4.7 (run.log 17:55–22:36 UTC, 3 Oct, incl. evals; plus setup) |
| auric-e2-b1h-150ep-scale02 v1 | ~4.4 (run.log 18:08–22:31 UTC, 3 Oct) |
| auric-label-mismatch v1, auric-e1e2-train40 v1 | 0 (CPU-only) |

`kaggle quota`: **9.56 h used, 20.44 h remaining** of 30 h (refresh 2026-10-10).
