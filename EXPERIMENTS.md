# Experiment log

Each entry: Observation, Hypothesis, Changes vs. previous run, Results, Conclusion, Next step.
Every number cites the file it came from. Hypothesis and Conclusion are written by the author.

## Index of runs (added 2026-10-04 audit)

mAP50 = COCO 101-point AP50 averaged over the 5 classes, on `last.pt`. val = 22-image official val; holdout40 = 40
train images never trained on by B1h-family runs (`splits/holdout40_seed0.txt`); train40 = 40 sampled seen train
images. Sources: `results/<run>/eval/metrics.json`, `results/<run>/eval_holdout40/metrics.json`,
`results/<run>/eval_train40/metrics.json` (`mAP50`). "—" = not evaluated; UNKNOWN = should exist but cannot be sourced
from a repo file. GPU-h: Kaggle kernel hours where recorded (MORNING.md table for the 3 Oct runs; the GPU-hours tables
in this file for SANITY/E1/E2), else UNKNOWN with the training time from `results/<run>/train/results.csv` (`time`
column) in brackets, which is a lower bound.

| run | purpose | status | val mAP50 | holdout40 mAP50 | train40 mAP50 | GPU-h | entry |
|---|---|---|---|---|---|---|---|
| B0 `b0_full640` | naive whole image at 640 | done | 0.0020 | — | — | UNKNOWN (Colab; train 0.64 h) | [B0](#b0-naive-full-image-baseline-configsb0yaml-run-b0_full640) |
| B1 `b1_tile1024` | native-resolution 1024 tiles | done | 0.0715 | — (trained on these images) | 0.4027 | UNKNOWN (Colab; train 2.34 h) | [B1](#b1-native-resolution-tiled-baseline-configsb1yaml-run-b1_tile1024) |
| B1h `b1h_tile1024_holdout40` | B1 with holdout40 removed; generalisation test; current best | done | 0.1065 | 0.1507 | 0.3778 | UNKNOWN (train 1.52 h) | [B1h](#b1h-b1-with-40-train-images-held-out-configsb1hyaml-run-b1h_tile1024_holdout40) |
| `b1h_seed1` | B1h, training seed 1 (seed noise) | done | 0.0634 | 0.1333 | — | UNKNOWN (train 1.45 h) | [LC](#lc-learning-curves-53-runs-b1h_f25--b1h_f50--b1h_f75--b1h_seed1) |
| `b1h_f25` | learning curve, 101 images | done | 0.0169 | 0.0606 | — | UNKNOWN (train 1.49 h) | [LC](#lc-learning-curves-53-runs-b1h_f25--b1h_f50--b1h_f75--b1h_seed1) |
| `b1h_f50` | learning curve, 202 images | done | 0.0492 | 0.0948 | — | UNKNOWN (train 1.41 h) | [LC](#lc-learning-curves-53-runs-b1h_f25--b1h_f50--b1h_f75--b1h_seed1) |
| `b1h_f75` | learning curve, 302 images | done | 0.0846 | 0.1045 | — | UNKNOWN (train 1.35 h) | [LC](#lc-learning-curves-53-runs-b1h_f25--b1h_f50--b1h_f75--b1h_seed1) |
| `b1h_f50_seed1` | random 202, seed 1 (§5.4 noise) | done | 0.0573 | 0.0816 | — | ≤ 1.6 (MORNING.md) | [S54](#s54-smart-vs-random-subsets-54-runs-b1h_smart50--b1h_smart75--b1h_f50_seed1--b1h_f75_seed1) |
| `b1h_f75_seed1` | random 302, seed 1 (§5.4 noise) | done | 0.0736 | 0.1230 | — | ≤ 1.75 (MORNING.md) | [S54](#s54-smart-vs-random-subsets-54-runs-b1h_smart50--b1h_smart75--b1h_f50_seed1--b1h_f75_seed1) |
| `b1h_smart50` | smart-selected 202 images (§5.4) | done; no author conclusion | 0.0562 | 0.1052 | — | ≤ 1.7 (MORNING.md) | [S54](#s54-smart-vs-random-subsets-54-runs-b1h_smart50--b1h_smart75--b1h_f50_seed1--b1h_f75_seed1) |
| `b1h_smart75` | smart-selected 302 images (§5.4) | done; no author conclusion | 0.0832 | 0.1040 | — | ≤ 1.9 (MORNING.md) | [S54](#s54-smart-vs-random-subsets-54-runs-b1h_smart50--b1h_smart75--b1h_f50_seed1--b1h_f75_seed1) |
| E1 `e1_b1h_150ep` | undertraining test: 150 epochs | done (stopped at 145 by val-driven early stopping) | 0.0680 | 0.0923 | 0.7737 | ~4.7 (estimate) | [E1/E2](#e1--e2-is-b1h-undertrained-and-does-scale-05-hurt-small-trucks-configse1_b1h_150epyaml-configse2_b1h_150ep_scale02yaml), [results](#e1--e2-results-4-oct-2026) |
| E2 `e2_b1h_150ep_scale02` | scale 0.2 vs 0.5, 150 epochs | done | 0.0620 | 0.1112 | 0.9056 | ~4.4 (estimate) | [E1/E2](#e1--e2-is-b1h-undertrained-and-does-scale-05-hurt-small-trucks-configse1_b1h_150epyaml-configse2_b1h_150ep_scale02yaml), [results](#e1--e2-results-4-oct-2026) |
| SANITY (`auric-sanity`) | settings, tile-label check, 16-tile overfit test | done; NOT PASS, overridden; PASS after re-check | — (overfit AP50 1.000 on its 16 training tiles, `results/sanity/overfit/eval_ep300/metrics.json`) | — | — | ≤ 0.30 | [SANITY](#sanity-why-does-b1h-reach-only-038-map50-on-its-own-training-images-kernel-aradhya1211auric-sanity-code-f2388f1), [override](#sanity--e1e2-override-of-not-pass-authors-decision-3-oct-2026-1122-pm-ist) |
| label-mismatch (`auric-label-mismatch`) | IoU re-check of the 85 flagged tiles | done: checker artefact | — | — | — | 0 (CPU) | [re-check](#sanity-label-mismatch-re-check-4-oct-2026-cpu-only-kernel-aradhya1211auric-label-mismatch-code-1f3c5bf) |
| CHECK 0 | which weights produced the scores | done | — | — | — | 0 (no compute) | [CHECK 0](#check-0-4-oct-2026-which-weights-produced-the-reported-scores) |
| E3 `e3_b1h_dota` | DOTA-pretrained init | launched 4 Oct, no results | — | — | — | UNKNOWN (expected ~2) | [E3/E4](#e3--e4-two-remedies-for-overfitting-at-b1hs-length-pre-registered-4-oct-2026-before-any-result) |
| E4 `e4_b1h_flipud_mixup` | flipud 0.5 + mixup 0.1 | launched 4 Oct, no results | — | — | — | UNKNOWN (expected ~2) | [E3/E4](#e3--e4-two-remedies-for-overfitting-at-b1hs-length-pre-registered-4-oct-2026-before-any-result) |
| CPU kernels fp-crop / e1-figs | FP audit + crop classifier; E1 figures/errors | launched, no results | — | — | — | 0 (CPU) | [E3/E4](#e3--e4-two-remedies-for-overfitting-at-b1hs-length-pre-registered-4-oct-2026-before-any-result) |

Seed noise for comparisons: B1h vs b1h_seed1 differ by 0.043 val / 0.017 holdout40 mAP50 (rows above).

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
*Not recorded before the run; reconstructed afterwards (2026-10-04 audit; the field said "TODO (me)" from `f8d9117`
until after B0's results were committed in `99fbf95`):* a whole image letterboxed to 640 px shrinks a median truck
to ~4–5 px, so a naive full-image YOLO11s should detect very few trucks and serve as a floor for the tiled B1.
No numeric prediction was made.

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
(Filled 2026-10-04 audit from the result files; the field previously said "Pending Colab".)
- Val mAP50 **0.0020** (95% CI 0.0009–0.0045) (`results/b0_full640/eval/per_class.csv`, `metrics.json`).
  Per class AP50: Cargo 0.0022, Box 0.0077, Flatbed 0.0000, Tractor 0.0000, Liquid 0.0000 (same file). No predictions
  at all for Flatbed, Tractor or Liquid (`n_pred` 0).
- Scorer cross-check: ours 0.001986 = pycocotools 0.001986; Ultralytics val 0.0024
  (`results/b0_full640/eval/scorer_comparison.csv`, `ultralytics_crosscheck.json`).
- Training: 50/50 epochs, batch 16 used, 28 it/epoch, 1400 iterations (`results/b0_full640/training_summary.json`);
  0.64 h of training (`results/b0_full640/train/results.csv`, `time` column).
- Checkpoint curve (val mAP50, epochs 10/20/30/40/50): 0.0001 / 0.0008 / 0.0006 / 0.0010 / 0.0020
  (`figures/b0_full640/checkpoint_curve.csv`).
- holdout40 and train40: not evaluated (—).
- GT-box oracle: 396 of 1552 GT boxes have an anchor at IoU ≥ 0.5 ("unflagged"); pooled accuracy 0.481, mean
  per-class 0.215 (`figures/b0_full640/gt_oracle/comparison.csv`).

**Conclusion**
**Result:** mAP50 = 0.002, indicating almost no useful detection.

At 640 px, only 26% of trucks aligned sufficiently with a prediction slot, compared with 93% using 1024 px tiles. This confirms that tiling was necessary, although it did not solve every problem.

**Next step**
(Filled 2026-10-04 audit; was "Pending results.") B1, the native-resolution tiled baseline below. B1 was in fact run
before B0 (see Changes).

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
(Filled 2026-10-04 audit from the result files; the field previously said "Pending Colab".)
- Val mAP50 **0.0715** (95% CI 0.0419–0.1242) (`results/b1_tile1024/eval/per_class.csv`, `metrics.json`).
  Per class AP50: Cargo 0.0889, Box 0.1299, Flatbed 0.0530, Tractor 0.0064, Liquid 0.0792 (same file).
- Scorer cross-check: ours = pycocotools (0.07148, abs diff 0.0) (`results/b1_tile1024/eval/scorer_comparison.csv`).
- Training: 50/50 epochs, batch 16 used, 3785 tiles, 237 it/epoch, 11850 iterations
  (`results/b1_tile1024/training_summary.json`); 2.34 h of training (`results/b1_tile1024/train/results.csv`).
- Checkpoint curve (val mAP50, epochs 10/20/30/40/50): 0.036 / 0.058 / 0.055 / 0.056 / 0.071
  (`figures/b1_tile1024/checkpoint_curve.csv`).
- Merge sensitivity (max_det 334): IoS 0.5 0.0707, IoS 0.6 0.0715 (default), IoS 0.7 0.0722, IoU 0.5 0.0718, no merge
  0.0484 (`figures/b1_tile1024/merge_sensitivity.csv`). IoS 0.6 at max_det 3000: 0.0776
  (`figures/b1_tile1024_maxdet3000/merge_sensitivity.csv`).
- train40 (seen images): mAP50 **0.4027** (0.3175–0.4985) (`results/b1_tile1024/eval_train40/per_class.csv`).
  holdout40: not evaluated (—; B1 trained on those images).
- Class-agnostic val AP50 0.2374, recall at conf 0.001 0.5838 (906/1552) (`results/b1_tile1024/eval/class_agnostic.json`).
- Error bins (dAP50 if fixed): cls +0.190, bkg +0.053, missed +0.041, loc +0.014
  (`figures/b1_tile1024/errors/tide_dAP.csv`).

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
(Filled 2026-10-04 audit; was "Pending results.") B1h: B1 retrained with 40 train images held out, to test whether
the low val score reflects poor generalisation to any unseen image (Open question above).

---

## B1h: B1 with 40 train images held out (`configs/b1h.yaml`, run `b1h_tile1024_holdout40`)

### Observation
(Added 2026-10-04 audit from the B1 entry.) B1 scored 0.0715 val mAP50 but 0.4027 on 40 seen train images
(`results/b1_tile1024/eval/metrics.json`, `results/b1_tile1024/eval_train40/metrics.json`). It was open whether the
low score is specific to val or applies to any unseen image (B1 "Open question").

### Hypothesis
*Not recorded before the run; reconstructed afterwards:* if the model scores about as badly on held-out train images
as on val, the problem is generalisation from ~400 images, not something peculiar to val. No prediction was written
(see "Pre-registration" below).

### Changes vs. previous run (B1)
(Added 2026-10-04 audit from `configs/b1h.yaml` and the result files.)
- 40 train images held out (`splits/holdout40_seed0.txt`, seed 0, stratified by rarest class; 1938.png excluded):
  403 images → 3439 tiles, 215 it/epoch, 10750 iterations (`results/b1h_tile1024_holdout40/training_summary.json`).
- Eval max_det 902 instead of 334 (`results/b1h_tile1024_holdout40/eval/metrics.json`, `max_det`).
- Platform: Kaggle 2 × T4, torch 2.10.0+cu128, instead of Colab T4 torch 2.11.0+cu130
  (`results/<run>/env/train_hardware.json`). Code `4a17eb1`.
- Evaluated on val, holdout40 and train40, all on `last.pt`.

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

**Observation** (field label added 2026-10-04 audit)
B1h: train40 0.378 vs holdout40 0.151 vs val 0.107 mAP50 (`results/b1h_tile1024_holdout40/{eval_train40,eval_holdout40,eval}/metrics.json`).

**Hypothesis** (field label added 2026-10-04 audit): the pre-registration below, committed in `4716578` (2 Oct 19:06)
before the LC results commit `089e471`.

**Changes vs. B1h** (field label added 2026-10-04 audit; from `configs/b1h_f*.yaml`, `results/<run>/training_summary.json`)
- Nested class-stratified subsets of the 403 non-holdout images (`splits/train_f{25,50,75}_seed0.txt`): 101 / 202 /
  302 images, trained 199 / 101 / 67 epochs so total iterations stay ~10,750 (10746 / 10706 / 10720); warmup and
  close_mosaic scaled to match.
- b1h_seed1: B1h with training seed 1 (10750 iterations), to measure seed noise.

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

**Corrected 2026-10-04:** the brief asks about 500 labelled *instances*, not images. 500 instances is about 29 images at
17.1 boxes per image (6880 boxes / 403 images). The same fit gives about +0.006 holdout mAP50 (403 → 432 images), below the
0.017 seed spread. The 903-image extrapolation below answers "500 more images". See REPORT.md §5.3.


The learning curve suggests that more labelled data would improve performance, particularly truck detection. However, extrapolating beyond the current 403 images is uncertain, and even an optimistic projection falls well short of the target mAP50 of 0.75.

By my pre-registered rule: the projected held-out gain at +500 images (+0.080 mAP50) is larger than the seed-to-seed noise (0.017), so more labels should help. But the projection (0.215, interval 0.076–0.276) extrapolates 2.24× beyond the data and was flagged unreliable, so only the direction is trustworthy, not the size.

**Conclusion:** More data should help, but additional labels alone are unlikely to achieve the target. Classification needs a separate improvement strategy.

### Smallest useful training subset

No random subset tested retained 90% of full-data performance. Even 75% of the images achieved 0.105, about 74% of the full-data held-out mAP50 (0.142, the mean of both seeds). For detection alone (class-agnostic) it retained 81%. The remaining question is whether a carefully selected subset can retain more performance than a random sample of the same size.

### Next
(Field added 2026-10-04 audit, from the text above and the Plan table.) §5.4 smart-vs-random subsets (S54 below);
classification needs its own remedy (see §5.1).

---

## §5.1 If locations were perfect

**Observation** (field added 2026-10-04 audit): B1's error bins put wrong-class errors first (cls +0.190 dAP50,
`figures/b1_tile1024/errors/tide_dAP.csv`).

**Hypothesis** — *Not recorded before the run; reconstructed afterwards:* classification, not localisation, limits
mAP50. (The entry was first committed with its conclusion in `3e4b9c8`; no earlier hypothesis exists.)

**Changes**: no training. Analysis only (`analysis/gt_box_oracle.py`, pooled-anchor mode) on the existing B0, B1 and
B1h weights.

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
Check (2026-10-04 audit): B1 pooled accuracy 0.553 / mean per-class 0.387; B1h 0.601 / 0.436
(`figures/{b1_tile1024,b1h_tile1024_holdout40}/gt_oracle/comparison.csv`, mode pool, subset all). B1h dAP50 cls
+0.147, loc +0.022, missed +0.044 (`figures/b1h_tile1024_holdout40/errors/tide_dAP.csv`). All match.

**Next** (field added 2026-10-04 audit, from B1h "Next steps"): a separate classifier on cropped trucks for Cargo vs
Box (now planned as the E3/E4 side kernel (b)).

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

**Six-field summary** (added 2026-10-04 audit)
- **Observation:** no random 50% or 75% subset kept 90% of full-data held-out mAP50 (LC "Smallest useful training subset").
- **Hypothesis:** the author's prediction for §5.4 was **never provided** (see "Prediction" above). Any hypothesis
  stated for S54 is *not recorded before the run; reconstructed afterwards:* a coverage + diversity selected subset
  keeps more held-out performance than a random subset of the same size.
- **Changes:** as in the pre-registration above (committed `b02413c`, 3 Oct 02:14, before the first S54 result
  commit `40067b8`). Runs: b1h_smart50 202 images / 104 epochs / 10712 it; b1h_smart75 302 / 66 / 10758;
  b1h_f50_seed1 202 / 101 / 10706; b1h_f75_seed1 302 / 67 / 10720 (`results/<run>/training_summary.json`).
  Note: these four ran on Kaggle torch 2.11.0 / Python 3.13.15, the seed-0 random runs on torch 2.10.0 / Python
  3.12.13 (HANDOFF §8 item 4; `results/<run>/env/train_hardware.json`).
- **Results** (held-out, `figures/subset_compare/decision_rule.csv`):

| size | metric | smart | random s0 | random s1 | seed spread | smart − best random | exceeds both by > spread |
|---|---|---|---|---|---|---|---|
| 202 | class-agnostic AP50 (primary) | 0.2841 | 0.2324 | 0.2213 | 0.0111 | +0.0517 | True |
| 202 | mAP50 (secondary) | 0.1052 | 0.0948 | 0.0816 | 0.0132 | +0.0103 | False |
| 302 | class-agnostic AP50 (primary) | 0.3343 | 0.2934 | 0.3062 | 0.0128 | +0.0281 | True |
| 302 | mAP50 (secondary) | 0.1040 | 0.1045 | 0.1230 | 0.0185 | −0.0189 | False |

  Val mAP50: smart50 0.0562, smart75 0.0832, f50_seed1 0.0573, f75_seed1 0.0736 (`figures/subset_compare/subset_compare.csv`).
  No 50% or 75% run reached the 0.128 "recovers 90%" threshold; the highest was f75_seed1 at 0.1230 (same file).
- **Corrected 2026-10-04:** Conclusion and Next were "not yet written" / UNKNOWN; added below.

**Conclusion (written 2026-10-04, after the results; no prediction was recorded before the runs).**
- **No tested subset reached the threshold.** "Recovers 90%" means holdout40 mAP50 ≥ 0.128; the best 50% or 75% run
  was random seed 1 at 302 images, 0.1230 (`figures/subset_compare/subset_compare.csv`). The smallest qualifying
  subset is therefore larger than 302 images and **was not found**.
- **Smart vs random by size** (holdout40; `figures/subset_compare/decision_rule.csv`):

  | size | metric | smart | random s0 / s1 | margin over best random | seed spread | beats both by > spread |
  |---|---|---|---|---|---|---|
  | 202 | class-agnostic AP50 (primary) | 0.284 | 0.232 / 0.221 | +0.052 | 0.011 | yes |
  | 202 | mAP50 | 0.105 | 0.095 / 0.082 | +0.010 | 0.013 | no |
  | 302 | class-agnostic AP50 (primary) | 0.334 | 0.293 / 0.306 | +0.028 | 0.013 | yes |
  | 302 | mAP50 | 0.104 | 0.105 / 0.123 | −0.019 | 0.018 | no |

  Smart selection finds more trucks (class-agnostic) but does not improve class-aware mAP50, the metric the threshold
  uses. The primary metric was chosen before the runs, so the "smart wins" result is real by the pre-registered rule;
  it does not answer the 90% question.
- **Per class** (holdout40 AP50, `results/<run>/eval_holdout40/per_class.csv`; smart / random s0 / random s1):
  - 202 images: Cargo 0.057 / 0.051 / 0.032; Box 0.412 / 0.371 / 0.345; Flatbed 0.036 / 0.014 / 0.021;
    Tractor 0.001 / 0.000 / 0.002; Liquid 0.019 / 0.037 / 0.008.
  - 302 images: Cargo 0.079 / 0.049 / 0.062; Box 0.421 / 0.422 / 0.439; Flatbed 0.013 / 0.038 / 0.033;
    Tractor 0.006 / 0.001 / 0.035; Liquid 0.000 / 0.012 / 0.047.
  - Smart is highest for Cargo at both sizes and for Box and Flatbed at 202. Otherwise it is not consistently ahead.
    Tractor (35 holdout instances) and Liquid (29) are too few for any per-class conclusion; there is no per-class
    seed spread.
- **Disclosures:**
  - The implemented coverage step adds the image with the **most boxes of the class still needed**
    (`tools/make_smart_subsets.py`). The pre-registration says "in order of rarest class contained". The runs used
    the implemented rule.
  - No author prediction was recorded before the runs.
  - The smart and seed-1 runs ran on a different Kaggle torch build (2.11.0) than the seed-0 random runs (2.10.0).

**Next.** Fractions above 75% (for example 85% and 95%, about 343 and 383 images) were not tested.
- The answer lies there, since 75% falls short (best 0.123) and 100% passes (0.151 / 0.133).
- Each size needs three runs (smart plus two random seeds).
- They were not run because E1/E2 showed the bottleneck is generalisation and classification, not the choice of
  training images (§3.4). The remaining GPU budget (20.44 h until 10 Oct, before E3/E4) went to experiments that
  could move mAP50 rather than refine a subset size.

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
  Audit 2026-10-04: the epoch-50 losses 1.58793 / 1.59568 / 1.00336 are in
  `results/b1h_tile1024_holdout40/train/results.csv` (args.yaml holds settings, not losses).

**Hypothesis**
*Not recorded before the run; reconstructed afterwards:* this hypothesis and the "pre-agreed" PASS rule (zero label
mismatches and overfit AP50 ≥ 0.9) were first committed together with the results in `e72d3b8`; no earlier commit
records them (checked `f2388f1`, `d8b5d47`, `c436c3a`). The low in-sample score could come from (a) wrong training settings, (b) broken tile labels, or (c) a model or
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
- **Corrected 2026-10-04 (label check):** the 85 flags are a checker artefact, not label errors.
  `analysis/sanity_check.check_labels` paired boxes by a coordinate `np.lexsort` (`analysis/sanity_check.py` line 75),
  which mis-pairs near-equal coordinates in dense tiles. The CPU re-check with one-to-one IoU pairing
  (`analysis/label_mismatch.py`, kernel `aradhya1211/auric-label-mismatch`) found all 3,278 boxes in the 85 tiles
  identical ("same": 3278; `results/sanity/label_mismatch/summary.json`). See "SANITY: label-mismatch re-check" below.
  The checker code is **not yet fixed** (`check_labels` still uses lexsort pairing).
- **Corrected 2026-10-04 ("E1/E2 were not launched"):** true when written (`e72d3b8`, 23:17 IST), but the author then
  overrode the NOT PASS before the re-check and launched E1/E2 (see "SANITY → E1/E2: override of NOT PASS", committed
  `1f3c5bf`, 23:22 IST). The re-check (committed later in `727d7c7`) showed both PASS criteria hold: 0 real label
  mismatches, overfit AP50 1.000.

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
  GPU kernels** at 23:20 IST on 3 Oct, code `e72d3b8`. **Corrected 2026-10-04: was "2 Oct", source says 3 Oct** (E1/E2 pre-registration dated 3 Oct 23:09 IST, commit `d8b5d47` 2026-10-03 23:09 +0530; code `e72d3b8` committed 2026-10-03 23:17 +0530).
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
    Audit 2026-10-04: `run.log` is not in the repo; 145 epochs × 215 it = 31,175 and 4.43 h (`time` 15947 s at epoch
    145) are from `results.csv`.
  - `last.pt` = epoch 145, which had 5 of the 10 planned no-mosaic epochs.
    **Corrected 2026-10-04: was "4 of the 10", source says 5** (epochs 141–145): `close_mosaic: 10` with
    `epochs: 150` (`results/e1_b1h_150ep/train/args.yaml`) turns mosaic off from epoch 141, where dfl_loss jumps
    0.885 → 0.903 (`results/e1_b1h_150ep/train/results.csv`), matching B1h's jump at epoch 41.
- **Corrected 2026-10-04 (early stop was val-driven):** E1's stopping point was chosen by val. Ultralytics' default
  `patience: 100` (`results/e1_b1h_150ep/train/args.yaml` line 7) watched per-epoch fitness on the official val set
  (0.1·mAP50 + 0.9·mAP50-95 from Ultralytics' un-sliced val). Its best was epoch 45 (fitness 0.00514, from
  `results/e1_b1h_150ep/train/results.csv`), so training stopped at 145. E1's `last.pt` therefore depends on val; see
  CHECK 0. E2's best fitness was at epoch 61 (same computation on `results/e2_b1h_150ep_scale02/train/results.csv`),
  so 150 came first and E2 was not affected.
- E2 completed 150 epochs (32,250 iterations, 4.13 h) (`results/e2_b1h_150ep_scale02/training_summary.json`,
  `train/results.csv`).

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
**Corrected 2026-10-04: B1h class-agnostic train40 was "–", source says 0.649**
(`results/b1h_tile1024_holdout40/eval_train40/class_agnostic.json`, `AP50_class_agnostic` 0.6490). All other cells
checked against the files above and match.

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

Audit 2026-10-04: the B1h and E2 columns match `figures/{b1h_tile1024_holdout40,e2_b1h_150ep_scale02}/errors/op_gt.csv`
(rows `conf0.25`, `matched`, binned by `size`). The E1 column is not saved in any repo file (E1 has no `errors/`), so
it cannot be re-sourced: E1 values UNVERIFIED.
**Corrected 2026-10-04 (later):** E1's `errors/` and figures were regenerated from its saved val predictions on CPU-only kernel `aradhya1211/auric-e1-figs` (code `734299d`; `pred_review.py`, `merge_sensitivity.py`, `errors.py`). The E1 column is now verified: `figures/e1_b1h_150ep/errors/op_gt.csv` (conf0.25 rows) gives 0.061 / 0.127 / 0.255 for n = 263 / 818 / 471, the same as the table.

AP50 by size was not computed. E1's `errors/` and `figures/` folders were missing from its Kaggle output (cause unknown); regenerated later (see above).

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

- **Observation:** the sanity label check flagged 85 of 3439 tiles (`results/sanity/label_check.json`), undiagnosed.
- **Hypothesis:** stated in the SANITY "Not yet diagnosed" note (`e72d3b8`, before this re-check): the flags are either
  real label differences or an artefact of the checker's box pairing. No prediction of which.
- **Changes:** no training. Same tiles re-compared with one-to-one IoU pairing and per-box differences saved
  (`analysis/label_mismatch.py`), CPU only.
- **Results:**
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
- **Next** (field added 2026-10-04 audit, from HANDOFF.md "Open items"): fix `analysis/sanity_check.check_labels` to
  pair by IoU as `analysis/label_mismatch.py` does. Not done yet (the lexsort pairing is still in
  `analysis/sanity_check.py`).

### Kaggle GPU hours (update 4 Oct 2026, 10:xx IST)

| kernel | GPU h |
|---|---|
| auric-sanity v1 | ≤ 0.30 (22:53–23:11 IST, 3 Oct; **Corrected 2026-10-04: was "2 Oct", source says 3 Oct**: its code `f2388f1` is dated 2026-10-03 22:54 +0530) |
| auric-e1-b1h-150ep v1 | ~4.7 (run.log, not in the repo; 17:55–22:36 UTC, 3 Oct, incl. evals; plus setup) |
| auric-e2-b1h-150ep-scale02 v1 | ~4.4 (run.log 18:08–22:31 UTC, 3 Oct) |
| auric-label-mismatch v1, auric-e1e2-train40 v1 | 0 (CPU-only) |

`kaggle quota`: **9.56 h used, 20.44 h remaining** of 30 h (refresh 2026-10-10).

## CHECK 0 (4 Oct 2026): which weights produced the reported scores?

*Six-field labels (added 2026-10-04 audit; content below unchanged):* **Observation** — E1 stopped at 145, not 150.
**Hypothesis** — *not recorded before the check; reconstructed afterwards:* some reported score may rest on weights
or stopping points chosen by val. **Changes** — none (audit of `metrics.json` `weights`, `scripts/_run.sh`,
`data.yaml`). **Results** — the bullets below up to "From now on". **Conclusion** — all scores use `last.pt`; one
val leak (E1's early stop). **Next** — the "From now on" bullets.

- Every reported score used `last.pt`. Each `metrics.json` records its `weights`; all of these end in
  `train/weights/last.pt`:
  - B1h: `results/b1h_tile1024_holdout40/{eval,eval_holdout40,eval_train40}`
  - E1: `results/e1_b1h_150ep/{eval,eval_holdout40,eval_train40}`
  - E2: `results/e2_b1h_150ep_scale02/{eval,eval_holdout40,eval_train40}`
  - every other run in `results/`
- No reported score used `best.pt`. `scripts/_run.sh` calls `eval.py` without `--weights`, and its default is
  `last.pt`.
- The training `data.yaml` has `val: /kaggle/tmp/data/val/images`, the official val set
  (`results/e1_b1h_150ep/data.yaml`). With `val: true`, Ultralytics computed its own un-sliced val mAP every epoch.
  That was used only for curves.
- **One leak of val into training.** Ultralytics EarlyStopping (default `patience=100`) watches that per-epoch val
  fitness. It stopped E1 at epoch 145, because its best fitness was at epoch 45.
  - So E1's `last.pt` is epoch 145, a stopping point chosen by val.
  - Effect: 5 fewer epochs (145 instead of 150). E2 ran all 150 epochs without early stopping and shows the same
    pattern (train40 up, holdout40 down), so the E1/E2 verdict does not rest on this.
  - B1h (50 epochs) and E2 (150 epochs) never triggered it.
- The val checkpoint curves (E1/E2 "peak at epoch 90" and so on) were reported as curves, never used to pick weights.
- From now on:
  - every config sets `patience: 0`
  - only `last.pt` is evaluated
  - per-checkpoint curves are scored on holdout40, not val (`analysis/checkpoint_curve.py --split holdout`)

## E3 / E4: two remedies for overfitting at B1h's length (pre-registered 4 Oct 2026, before any result)

*Six-field labels (added 2026-10-04 audit; content below unchanged):* **Observation** — the "Context" line.
**Hypothesis** — "Working hypothesis" and the per-run author's hypotheses and predictions (committed in `734299d`
before any E3/E4 result). **Changes** — the settings list and per-run bullets. **Results** — none yet (launched 4 Oct;
no result files in `results/` at the time of this audit). **Conclusion** — none yet. **Next** — not recorded.

Context: E1/E2 rejected undertraining. From B1h to E1/E2, train40 rose 0.378 → 0.774 / 0.906 while holdout40 fell
0.151 → 0.092 / 0.111.

Working hypothesis: the model overfits and fails to generalise from limited, fine-grained data.

Both runs use B1h's recipe (`configs/b1h.yaml`) with these settings:
- 50 epochs, `patience: 0`
- checkpoints every 10 epochs, each scored on **holdout40**
- full val / holdout40 / train40 eval on `last.pt` only (`scripts/run_e34.sh`)

Each runs as its own single-GPU kernel.

- **E3, aerial pretraining** (`configs/e3_b1h_dota.yaml`):
  - Architecture `yolo11s.yaml`. Weights come from Ultralytics' DOTA-trained `yolo11s-obb.pt`: every tensor whose
    name and shape match is copied (`train.py transfer_weights`). The count is recorded in `init_weights.json`.
  - Hypothesis (author's): aerial vehicle features generalise better from our small dataset.
  - Prediction (author's): holdout40 mAP50 beats B1h (0.1507) by more than 0.017 (B1h's holdout seed spread), and
    the train40-minus-holdout40 gap shrinks (B1h: 0.378 − 0.151 = 0.227).
- **E4, overhead augmentation** (`configs/e4_b1h_flipud_mixup.yaml`):
  - B1h plus `flipud: 0.5` and `mixup: 0.1`.
  - Hypothesis (author's): overfitting. Aerial images have no "up", so vertical flips add free variety.
  - Prediction (author's): same as E3.
  - **Two settings change at once, so E4 tests the pair together, not each one separately.**
- Expected cost: about 2 GPU-h each.
- Seed noise for the verdicts: 0.017 on holdout40 (B1h seed 0 vs seed 1).

CPU-only kernels run alongside (no GPU quota):
- (a) Background false-positive audit (`analysis/fp_audit.py`): the 60 highest-confidence B1h holdout40 predictions
  with IoU < 0.1 to every GT box, as a contact sheet, counted by eye.
- (b) Crop classifier (`analysis/crop_classifier.py`):
  - ImageNet ResNet18 trained on GT crops (2× context, 96 px) from the train images minus holdout40.
  - Scored as per-class accuracy on holdout40 GT crops.
  - Then B1h's holdout40 predictions are re-labelled with it (score = detector conf × p(class)), scored with
    `eval.py --from-preds`, and compared with 0.1507.
  - Val is not used.
- (c) E1's missing `figures/` and `errors/`, regenerated from its saved val predictions.

## §5.2 test-half confirmation (pre-registered 4 Oct 2026, before the test half is read)

- **Halves:** `splits/s52_inspect_seed0.txt` (202 images) and `splits/s52_test_seed0.txt` (201 images), committed in
  `b02413c` before any §5.2 computation.
- **Test-half data:** `results/s52/per_box_test.csv` has never been opened. It will be read only inside the CPU
  kernel by `analysis/s52_confirm.py`, after this entry is committed.
- **Where the claims come from:** Claude derived them from the inspect half
  (`figures/s52/inspect/counts_by_{class,size}.csv`). They are not author predictions; no author prediction for §5.2
  was recorded.
- **Pass rule:** a claim passes on the test half if its point estimate meets the criterion and so does the 2.5th
  percentile of a 2000-sample bootstrap over images (seed 0).
- **Control run:** the same script also runs on the inspect half and must reproduce the inspect values below.

| claim | inspect value (source) | criterion on test half |
|---|---|---|
| C1 Small boxes are the ones never detected: never-detected share for boxes < 16 px minus the share for boxes ≥ 32 px | 608/903 − 72/744 = 0.673 − 0.097 = 0.576 (`counts_by_size.csv`) | > 0 |
| C2a Truck w/Box is learned early more than any other class: its learned-early share minus the highest other class's | 167/976 = 0.171 vs Flatbed 1/333 = 0.003 → 0.168 (`counts_by_class.csv`) | > 0 |
| C2b Truck w/Box is forgotten more than any other class: its forgotten share minus the highest other class's | 183/976 = 0.188 vs Tractor 26/349 = 0.074 → 0.113 | > 0 |
| C3 Truck w/Liquid is never learned more than any other class: its never-learned share minus the highest other class's | 29/64 = 0.453 vs Flatbed 60/333 = 0.180 → 0.273 | > 0 |
| C4 Most never-detected boxes are below threshold, not invisible: share with an epoch-50 prediction of any class at IoU ≥ 0.5 and conf ≥ 0.001 | 0.793 (in-session analysis 3 Oct, not saved to a file; this run saves it) | > 0.5 |
| C5 Cargo Truck boxes are often never detected: never-detected share among Cargo Truck boxes | 846/1789 = 0.473 | > 0.3 |

- **What a failure means:** the inspect-half pattern does not generalise and is not reported as a finding.

### §5.2 test-half confirmation: Results (4 Oct 2026; CPU-only kernel `aradhya1211/auric-s52-confirm` v2, code `95953f3`)

- **Control:** on the inspect half the script reproduced every inspect value the claims came from (C1 0.576, C2a 0.168,
  C2b 0.113, C3 0.273, C4 0.793, C5 0.473; `results/s52_confirm/inspect/claims.csv`). C4's confidence-floor number is
  now saved to a file for the first time.
- **Test half:** 201 images, 3369 GT boxes. Categories: learned-late 1356, never-detected 1270, forgotten 319,
  never-learned 280, learned-early 144 (`results/s52_confirm/test/summary.json`).

| claim | test estimate (95% CI, image bootstrap) | criterion | result |
|---|---|---|---|
| C1 never-detected share, < 16 px minus ≥ 32 px | 0.631 (0.559–0.699) | > 0 | **pass** |
| C2a Box learned-early share minus highest other class | 0.118 (0.061–0.157) | > 0 | **pass** |
| C2b Box forgotten share minus highest other class | 0.157 (0.090–0.209) | > 0 | **pass** |
| C3 Liquid never-learned share minus highest other class | 0.281 (0.164–0.375) | > 0 | **pass** |
| C4 never-detected boxes with an IoU ≥ 0.5 prediction at conf ≥ 0.001 | 0.778 (0.716–0.848) | > 0.5 | **pass** |
| C5 Cargo never-detected share | 0.474 (0.427–0.532) | > 0.3 | **pass** |

Source: `results/s52_confirm/test/claims.csv`.
- **Conclusion:** all five pre-registered claims (C1–C5, commit `95953f3`; six rows because C2 has parts a and b) hold on the unseen test half, by the pre-registered
  rule.
- **Interpretation** (written after the result; no author prediction was recorded):
  - Never-detected boxes are mostly small.
  - About 78% of them do get a box from the model, but below the 0.25 operating threshold. So "never detected" is
    largely "never confident", not "invisible".
  - Truck w/Box is both the class learned earliest and the one most often forgotten later.
  - Truck w/Liquid is the class most often found but never classified correctly.

### Val vs holdout40 recall gap (4 Oct 2026; CPU-only kernel `aradhya1211/auric-recall-gap`, code `cc152a5`)

- **Observation:**
  - B1h finds 1032/1552 = 0.665 of val trucks but 629/738 = 0.852 of holdout40 trucks (any class, IoU ≥ 0.5,
    conf ≥ 0.001). `analysis/recall_gap.py` reproduces both totals exactly (`results/recall_gap/summary.json`).
  - The difference is 0.187, with an image-bootstrap CI of 0.043–0.307 (`results/recall_gap/standardised.csv`).
- **Hypotheses** (written with the analysis, before its result): the gap comes from
  1. different box sizes (scale or resolution shift),
  2. denser scenes in val (median 97 boxes in a box's image vs 58), or
  3. images of a different size, or
  4. none of these (plain scene difference).
- **Changes:** none (analysis of saved predictions). Holdout recall is re-weighted to val's distribution of each
  factor.
- **Results** (conf 0.001):

| factor | holdout recall re-weighted to val's mix | share of the gap it explains | val bins covered by holdout data |
|---|---|---|---|
| none | 0.852 | – | – |
| box size | 0.847 | 3% | 99.9% |
| boxes per image | 0.869 | −9% | 60% (holdout has no image with 60–100 boxes) |
| image megapixels | 0.839 | 7% | 89% |

  Recall at matched box size, holdout vs val (`results/recall_gap/recall_by_factor.csv`):

  | box size | holdout | val |
  |---|---|---|
  | 8–16 px | 0.639 | 0.548 |
  | 16–24 px | 0.890 | 0.648 |
  | 24–32 px | 0.936 | 0.732 |
  | 32–48 px | 0.895 | 0.725 |

  At ≥ 100 boxes per image: holdout 0.910 (2 images) vs val 0.709 (5 images).
- **Conclusion:**
  - **Size and image size explain almost none of the gap** (3% and 7%). Val recall is lower in every size bin from
    8 to 48 px.
  - **Density does not explain it where it can be tested:** in the densest bin val is still 0.20 lower. The
    60–100 bin cannot be compared because holdout has no images in it.
  - What remains is a difference between the scenes themselves, at the same size, density and resolution. This
    analysis cannot say what that difference is. Candidates from `analysis/notes/visual_inspection.md` (uncertain):
    haze, blur, ports and dense truck yards.
  - Caveat: val has only 22 images, so within-bin comparisons rest on few images.
  - The earlier domain classifier (§3.3) did not separate train from val (AUCs include 0.5). That makes a strong
    global appearance shift less likely, but it was run on a 42-image local subset only.
- **Next:** none planned. This is diagnosis for §3.

### SANITY: label check re-run with the fixed checker (4 Oct 2026; CPU-only kernel `aradhya1211/auric-sanity-labels`, code `cc152a5`)

- `analysis/sanity_check.py` now pairs boxes one-to-one by IoU (commit `cc152a5`).
- Re-run with `--labels-only`: **0 of 3439 tiles mismatched**, 12455 box labels checked, 0 out-of-range values
  (`results/sanity/label_check_iou/label_check.json`).
- Same tiles and box count as the original check, which flagged 85 (`results/sanity/label_check.json`).
- The label criterion of the sanity PASS rule now holds with the checker itself, not only with the separate re-check.

## Deliverables: predict.py reproducibility check (4 Oct 2026; CPU-only kernel `aradhya1211/auric-predict-test`, code `bb7c14b`)

- **Command:** `predict.py --weights <B1h last.pt> --images val/images --labels val/labels --max-images 2 --device cpu`.
  The images are the first two val images by name, 1181.png and 1206.png. The weights' SHA-256 `3fa24066…7ffb`
  matches `results/b1h_tile1024_holdout40/eval/metrics.json`.
- **Compared with B1h's saved val predictions** (GPU, 2 Oct) by `tools/compare_preds.py`, pairing same class and
  IoU ≥ 0.9:
  - **Merged predictions:** 243/243 (1181.png) and 902/902 (1206.png) paired.
    - Max |conf| difference 2.7e-6.
    - Max coordinate difference 0.00018 px.
  - **Raw tile predictions:** 499/499 and 3660/3660 paired, with differences of the same size.
  - Sources: `results/predict_test/compare.json`, `compare_raw.json`.
- **2-image mAP50:** 0.0577 (`results/predict_test/metrics.json`). This is only a smoke value, not comparable with the
  22-image score.
- **Conclusion:** `predict.py` reproduces the saved predictions on CPU, up to floating-point noise.
