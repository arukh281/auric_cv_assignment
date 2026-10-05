# Experiment log

Each entry: Observation, Hypothesis, Changes vs. previous run, Results, Conclusion, Next step.
Every number cites the file it came from. This log was written with AI assistance (Claude Code for code, analysis and drafting; a chat assistant for experiment planning), as the brief permits. Hypotheses, predictions and decision rules marked "author's" were chosen and approved by the author before the run, some drafted with AI assistance. Other hypotheses, interpretations and conclusions were drafted by Claude Code and reviewed by the author. Pre-registration status is stated per entry, from git history.

*Corrected 2026-10-04: this line previously said "Hypothesis and Conclusion are written by the author", which was inaccurate.*

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
| E3 `e3_b1h_dota` | DOTA-pretrained init | done; prediction not supported (reversed) — see "E3: Results" | 0.0881 | 0.0817 | 0.7291 | 1.64 (kernel time) | [E3/E4](#e3--e4-two-remedies-for-overfitting-at-b1hs-length-pre-registered-4-oct-2026-before-any-result) |
| E4 `e4_b1h_flipud_mixup` | flipud 0.5 + mixup 0.1 | done; partly supported — see "E4: Results" | 0.0761 | 0.1304 | 0.2657 | 1.84 (kernel time) | [E3/E4](#e3--e4-two-remedies-for-overfitting-at-b1hs-length-pre-registered-4-oct-2026-before-any-result) |
| E6 `e6_b1h_rfs` | repeat-factor sampling, B1h steps | running (launched 13:01, 4 Oct) | — | — | — | pending | "E6" entry |
| E7 `e7_b1h_dota_frozen` | DOTA init + frozen backbone | running (launched 13:14, 4 Oct) | — | — | — | pending | "E7" entry |
| E8 `e8_b1h_flipud_mixup_100ep` | E4 recipe, 100 epochs | pre-registered; waits for E6 and E7 | — | — | — | (~3.7 expected) | "E8" entry |
| TTA (`auric-tta-b1h`) | flips + 1.5× on B1h tiles | running | — | pending | — | 0 (CPU) | "TTA on B1h" entry |
| Two-stage on E4 boxes | crop classifier relabels E4 | pre-registered; waits for fp-crop | — | — | — | 0 (CPU) | "Two-stage" entry |
| fp-crop (`auric-fp-crop`) | background-FP audit + crop classifier on B1h | running | — | — | — | 0 (CPU) | "E3 / E4" pre-registration, CPU (a)/(b) |
| CPU analyses, done | e1-figs, s52-confirm, s52-floor, recall-gap, sanity-labels, predict-test, s54-char-merge | done | — | — | — | 0 (CPU) | their entries below |
| CPU analyses, running | lc-per-class; B1h / E1 / E2 holdout checkpoint curves | running | — | — | — | 0 (CPU) | — |

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

At 640 px, only 26% of trucks aligned sufficiently with a prediction slot, compared with 93% using 1024 px tiles.
*Source added 2026-10-04:* 26% = 396/1552 (`figures/b0_full640/gt_oracle/comparison.csv`, unflagged); 93% = 1449/1552
(`figures/b1_tile1024/gt_oracle/comparison.csv`, unflagged). This confirms that tiling was necessary, although it did not solve every problem.

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

*Note 2026-10-04 (pre-registration text left unchanged):* val is about 4.1× denser than train by mean boxes per
image, 70.5 vs 17.2 (`figures/eda/tables/boxes_per_image_stats.csv`), not 10×. By median it is 76.5 vs 7.
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

*Forward pointer 2026-10-04:*
- This was B1 at F1-optimal thresholds.
- B1h at conf 0.25 shows a clear size gradient in right-class recall: 0.057 / 0.119 / 0.274 for < 16 / 16–32 / ≥ 32 px
  (E1/E2 results table; `figures/b1h_tile1024_holdout40/errors/op_gt.csv`).
- §5.2 C1 confirmed on unseen training boxes that never-detected boxes are mostly small.
- What still holds: across 8–48 px, most trucks are missed at every size, so size alone does not explain the bulk of
  the misses.
- What is weakened: "size isn't what limits" is too strong. Size matters, and most strongly below 16 px.

**Diagnosis log**

* **Scoring bug?** Unlikely. The pipeline scored 0.40 on 40 training images the model had already seen.
* **Tile merging?** Helps rather than hurts. Disabling it dropped the score from 0.071 to 0.048.
* **Detection limit?** Minor effect; increasing it improved the score by just 0.006.
* **Domain shift?** No clear evidence. AUCs were 0.34–0.45, with all 95% confidence intervals including 0.5.
  *Forward pointer 2026-10-04:* this was a weak test (42 local images). The later recall-gap analysis found val recall
  0.187 below holdout40 (CI 0.043–0.307), not explained by box size, image size or density. That points to a
  scene-level shift. See "Val vs holdout40 recall gap".
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

The held-out and validation confidence intervals overlap, suggesting that poor generalization is the main issue rather than a unique problem with the validation set.
*Forward pointer 2026-10-04:* poor generalisation remains the main issue, but val is also specifically harder: recall
0.665 vs 0.852, gap 0.187 (CI 0.043–0.307), not explained by size, image size or density. See "Val vs holdout40
recall gap". However, the model finds fewer trucks on validation images (66% vs. 85%). The cause is not yet known. Density is a candidate, but within the validation set miss rates did not vary with density, so this remains untested.

Classification remains the biggest bottleneck, consistent with B1. More training data, stronger augmentation, and a separate classifier trained on cropped trucks are potential next steps.

**Pre-registration:** No prediction was written before this run.

**Important caveat:** B1h scored 0.107 on validation, compared with B1's 0.071. Differences of this size can arise from run-to-run randomness and platform differences; the small change in the detection limit explains only about 0.006. A single run is therefore not enough to establish a meaningful improvement.

### Next steps

* Test whether adding 500 labelled images improves generalization using learning curves on the held-out set.
* Experiment with stronger data augmentation.
* Evaluate a separate classifier on cropped truck images to improve Cargo vs. Box classification.

---

## Plan from here (2026-10-02, after B1h)

*Historical (marked 2026-10-04): the plan as written on 2 Oct. Later work is in the entries below.*

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

**Two verdicts, side by side (added 2026-10-04):**

| question | verdict | basis |
|---|---|---|
| (a) Pre-registered: 500 more **images** (403 → 903) | helps by the pre-registered rule (+0.080 > 0.017 noise); direction only, because the fit extrapolates 2.24× and is flagged unreliable | `figures/learning_curve/power_law_fit.csv` |
| (b) The brief: 500 more **instances** (~29 images, current class mix) | **no**: +0.006, below the 0.017 noise | same fit at 403 → 432 images; REPORT.md §5.3 |
| (b′) 500 targeted instances of one class | pending (`auric-lc-per-class`) | — |

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

**Subset characterisation (added 2026-10-04):** smart subsets hold +32% (202) / +24% (302) more boxes than
random ones of the same image count, 74% / 95% of the pool's Tractor boxes vs 35% / 49%, and more dense images; box sizes
are unchanged. This confounds smart vs random: no box-count-matched random subset was run. Details: REPORT.md §5.4,
`figures/subset_compare/subset_characterisation.csv`, `subset_class_coverage.csv`.

**Validation-mAP50 version (brief's definition, added 2026-10-04):** threshold 0.9 × 0.1065 = 0.0959; no subset reaches
it (best 0.085, random 302); full-data seed 1 itself scores 0.063. Weak evidence: val seed spread 0.043.

**Stated explicitly (added 2026-10-04):** The smallest successful subset we tested is the full training set (403 images); no tested smaller subset (202 or 302 images, smart or random) reached 90% on holdout40 or on val.

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

*Update 2026-10-04:* the label checker used here had a pairing bug. It was fixed in `cc152a5`, and the re-run with
the fixed checker gives 0 of 3439 tiles mismatched (`results/sanity/label_check_iou/label_check.json`).

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
- *Update 2026-10-04:* done. The checker was fixed in `cc152a5`, and its re-run gives 0 of 3439 tiles mismatched
  (see "SANITY: label check re-run with the fixed checker").

### Kaggle GPU hours (update 4 Oct 2026, 10:xx IST)

| kernel | GPU h |
|---|---|
| auric-sanity v1 | ≤ 0.30 (22:53–23:11 IST, 3 Oct; **Corrected 2026-10-04: was "2 Oct", source says 3 Oct**: its code `f2388f1` is dated 2026-10-03 22:54 +0530) |
| auric-e1-b1h-150ep v1 | ~4.7 (run.log, not in the repo; 17:55–22:36 UTC, 3 Oct, incl. evals; plus setup) |
| auric-e2-b1h-150ep-scale02 v1 | ~4.4 (run.log 18:08–22:31 UTC, 3 Oct) |
| auric-label-mismatch v1, auric-e1e2-train40 v1 | 0 (CPU-only) |

`kaggle quota`: **9.56 h used, 20.44 h remaining** of 30 h (refresh 2026-10-10).

*Update 2026-10-04, ~14:15 IST:*

| kernel | GPU h |
|---|---|
| auric-e3-dota v1 | 1.64 (kernel time, log `time` 5890 s) |
| auric-e4-flipud-mixup v1 | 1.84 (kernel time, log `time` 6613 s) |
| auric-e6-rfs, auric-e7-dota-frozen | running |
| every other 4 Oct kernel | 0 (CPU-only) |

`kaggle quota` at ~14:15 IST: **15.48 h used, 14.52 h remaining**. The web UI at 14:23 showed 10 h 53 min reserved by
the running sessions (E6, E7).

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
- **Extended 2026-10-04: early-stopping check for every run.** Epochs completed (rows in `results/<run>/train/results.csv`)
  vs configured epochs:

  | run | configured | completed | stopped early? | best Ultralytics val fitness at epoch | epochs without improvement at end |
  |---|---|---|---|---|---|
  | b0_full640, b1_tile1024, b1h_tile1024_holdout40, b1h_seed1 | 50 | 50 | no | — | — |
  | b1h_f25 | 199 | 199 | no | 112 | 87 |
  | b1h_f50 / b1h_f50_seed1 | 101 | 101 / 101 | no | 54 / 68 | 47 / 33 |
  | b1h_f75 / b1h_f75_seed1 | 67 | 67 / 67 | no | 45 / 44 | 22 / 23 |
  | b1h_smart50 / b1h_smart75 | 104 / 66 | 104 / 66 | no | 64 / 52 | 40 / 14 |
  | e1_b1h_150ep | 150 | **145** | **yes, at 145** | 45 | 100 |
  | e2_b1h_150ep_scale02 | 150 | 150 | no | 61 | 89 |
  | e3_b1h_dota, e4_b1h_flipud_mixup | 50 | 50 | no (`patience: 0`) | — | — |

  - Fitness = 0.1 × mAP50 + 0.9 × mAP50-95, Ultralytics' default weights. It is recomputed from the per-epoch
    `metrics/mAP50(B)` and `metrics/mAP50-95(B)` columns, which are Ultralytics' un-sliced val metrics.
  - Patience was 100 (default) in every run before E3; `patience` is in `results/<run>/train/args.yaml` where that
    file was copied.
  - **Only E1 stopped early.** All learning-curve and §5.4 runs completed their configured epochs, so the
    "equal iterations" design of §5.3 and §5.4 holds, and early stopping did not affect them.
  - b1h_f25 came closest, ending 87 epochs after its best fitness. A 13-epoch longer schedule would have triggered it.
- From now on:
  - every config sets `patience: 0`
  - only `last.pt` is evaluated
  - per-checkpoint curves are scored on holdout40, not val (`analysis/checkpoint_curve.py --split holdout`)

## E3 / E4: two remedies for overfitting at B1h's length (pre-registered 4 Oct 2026, before any result)

*Six-field labels (added 2026-10-04 audit; content below unchanged):* **Observation** — the "Context" line.
**Hypothesis** — "Working hypothesis" and the per-run author's hypotheses and predictions (committed in `734299d`
before any E3/E4 result). **Changes** — the settings list and per-run bullets. **Results** and **Conclusion** — see
"E3: Results" and "E4: Results" below (*updated 2026-10-04*). **Next** — E7 (from E3) and E8 / two-stage (from E4).

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

## E6: rare-class repeat-factor sampling (pre-registered 4 Oct 2026, before any run; `configs/e6_b1h_rfs.yaml`)

- **Observation:**
  - Truck Tractor and Truck w/Liquid have the lowest holdout40 AP (0.007, 0.073;
    *Corrected 2026-10-04: Flatbed's holdout40 AP, 0.069, is below Liquid's 0.073, so the two lowest are Tractor and
    Flatbed. Liquid is third. The rest of the observation, rarity and C3, stands.*
    `results/b1h_tile1024_holdout40/eval_holdout40/per_class.csv`).
  - They are the rarest training classes (625 and 192 train instances; `figures/eda/tables/class_counts.csv`).
  - §5.2 C3: Liquid is the class most often found but never classified correctly.
- **Hypothesis (author's):** rare classes are under-sampled. Showing their tiles more often raises their AP.
- **Changes vs B1h** (only these):
  - LVIS repeat-factor sampling (`detlib/rfs.py`, `train.py write_rfs_list`). For each class c, f_c is the fraction of
    training **tiles** containing c. Tiles are the images Ultralytics trains on; the author's instruction's "training images" are
    (*corrected 2026-10-04: was "the brief's"*)
    read as tiles.
  - r_c = max(1, √(0.1 / f_c)), and each tile's r is the maximum r_c over its classes. The list
    `<run>/train_rfs.txt` holds floor(r) copies of each tile plus one more with probability r − floor(r) (seed 0).
    The per-class f_c and r_c are written to `<run>/rfs.json` and will be reported here.
  - Epochs are set at run time so that epochs × ceil(list length / 16) is closest to B1h's 10,750 iterations. Only
    which tiles are seen changes, not how many steps are taken.
  - Caveat: warmup (3) and close_mosaic (10) are counted in epochs, so they cover a different number of iterations
    if the epoch count differs from 50.
  - `patience: 0`; `last.pt` only; holdout40 scored at every 10-epoch checkpoint (`scripts/run_e34.sh`).
- **Prediction (author's):** Truck Tractor and Truck w/Liquid holdout40 AP rise; overall holdout40 mAP50 changes by
  less than 0.017 (noise).
- **Decision rules:**
  - The standing E3/E4 rules apply.
  - E6 alone cannot make a new final model unless it beats B1h on holdout40 by more than 0.017.
  - Per the standing rule for any winner, it would then also need a seed-1 repeat that beats B1h.
- **Launch:** when a GPU slot is free (E3 and E4 currently hold both).
- **Amendment (4 Oct 2026, before launch, at the author's request):**
  - Warmup and close_mosaic are counted in epochs, so with a longer epoch they would differ from B1h in steps and
    become a second change.
  - `train.py` now sets both at run time from the repeated list's iterations per epoch (config key
    `match_schedule_to`):
    - warmup_epochs = 3 × 215 / it_ep (a float), giving the same 645 warmup steps as B1h. Ultralytics' floor of 100
      warmup iterations does not bind.
    - close_mosaic = round(10 × 215 / it_ep), aiming for B1h's 2150 no-mosaic steps.
  - The residual difference in no-mosaic steps (close_mosaic × it_ep − 2150) is written to `<run>/rfs.json` and will be
    reported here with the results. It is at most half an epoch of steps.

## TTA on B1h (pre-registered 4 Oct 2026, before any run; `analysis/tta_eval.py`, CPU-only kernel)

- **Observation:**
  - §5.2 C4: about 78% of never-detected boxes do get a prediction, but at low confidence.
  - Small trucks dominate the misses.
- **Hypothesis (author's):** averaging predictions over flips and an upscale stabilises confidence on small or
  ambiguous trucks.
- **Changes:** none to the model (B1h `last.pt`). Each 1024 tile is predicted four ways, and every prediction is
  mapped back to original coordinates:
  1. original
  2. horizontal flip
  3. vertical flip
  4. 1.5× bilinear upscale, predicted at imgsz 1536
- **Merging and scoring:** the union of predictions goes through the usual merge (class-wise NMS on IoS 0.6,
  max_det 902) and scorer, via `eval.py --from-preds`. Ultralytics `augment=True` is not used, because it also
  downscales.
- **Scored subsets:** {orig}, {orig, hflip}, {orig, vflip}, {orig, up1.5} and all four. {orig} must reproduce
  B1h's holdout40 score of 0.1507.
- **Prediction (author's):** holdout40 mAP50 with all four variants improves on B1h by more than 0.017.
- **Rule:** score on holdout40 first. Only if all four beat 0.1507 + 0.017, score once on val and apply TTA to
  whichever model ends up final. The single-variant subsets are descriptive; the pre-registered test is the
  all-four set.

### §5.2 confidence floor, descriptive (4 Oct 2026; CPU-only kernel `aradhya1211/auric-s52-floor`, code `a4d9307`, `analysis/s52_floor.py`)

**Computed after the test half was opened; not pre-registered.** The three thresholds in the author's 4 Oct message
(≥ 70%, chance < 1%, 30–50% at ≥ 0.10) were not committed before the test half was read, so they are not judged here.

| conf floor | inspect: never-detected with IoU ≥ 0.5 prediction | test | chance (inspect / test) |
|---|---|---|---|
| ≥ 0.001 | 0.793 (1022/1288) | 0.778 (988/1270) | 0.0020 / 0.0009 |
| ≥ 0.05 | 0.538 | 0.543 | 0.0003 / 0.0009 |
| ≥ 0.10 | 0.396 | 0.400 | 0.0003 / 0.0006 |

- Source: `results/s52_floor/{inspect,test}/floor.csv`. No chance boxes were dropped.
- The inspect values reproduce the unsaved 3 Oct session analysis (0.793 / 0.538 / 0.396 / 0.002).

### E3: Results (4 Oct 2026; GPU kernel `aradhya1211/auric-e3-dota`, code `734299d`, ~1.64 GPU-h kernel time)

- **Weight transfer:** 493 of 499 tensors (166 of 169 modules) were copied from `yolo11s-obb.pt`
  (sha256 `43fa6310…a45a`).
  - Not transferred: the three final class-score convolutions `model.23.cv3.{0,1,2}.2` (DOTA has 15 classes, we
    have 5).
  - Source: `results/e3_b1h_dota/init_weights.json`; Ultralytics log "Transferred 493/499 items".

| | B1h | E3 |
|---|---|---|
| val mAP50 (95% CI) | 0.1065 (0.056–0.165) | 0.0881 (0.050–0.152) |
| holdout40 mAP50 (95% CI) | 0.1507 (0.056–0.187) | **0.0817 (0.030–0.104)** |
| train40 mAP50 | 0.378 | 0.729 |
| train40 − holdout40 | 0.227 | **0.647** |
| class-agnostic AP50, val / holdout40 / train40 | 0.255 / 0.362 / 0.649 | 0.227 / 0.251 / 0.877 |
| holdout40 AP50: Cargo / Box / Flatbed / Tractor / Liquid | 0.096 / 0.509 / 0.069 / 0.007 / 0.073 | 0.057 / 0.326 / 0.025 / 0.001 / 0.000 |

Sources: `results/e3_b1h_dota/{eval,eval_holdout40,eval_train40}/{per_class.csv,class_agnostic.json}`. Every eval
used `last.pt` (field `weights` in `metrics.json`).

Holdout40 checkpoint curve (`results/e3_b1h_dota/checkpoint_curve_holdout/checkpoint_curve_holdout.csv`):

| epoch | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| holdout40 mAP50 | 0.132 | 0.131 | 0.114 | 0.088 | 0.082 |

- **Verdict against the pre-registered prediction:** **not supported, reversed.**
  - The prediction was holdout40 above 0.1507 + 0.017 and a smaller train40 − holdout40 gap. Holdout40 fell by 0.069
    and the gap nearly tripled.
  - The DOTA initialisation makes the model fit the training images faster and further, and generalise worse.
- **Interpretation** (written after the result):
  - The most plausible reading is that the backbone transfers aerial features that speed up memorisation of these
    403 images, not generalisation.
  - The holdout40 curve peaks at epoch 10, so more aggressive regularisation or a much shorter schedule might help.
    That is untested.
  - *Corrected 2026-10-04 (author):* picking an epoch on holdout40 would not be test-set selection. Holdout40 is the
    validation split and val is the test set, so it would be legitimate if declared in advance, with val reported
    once afterwards. This project keeps `last.pt` as pre-registered and reports the per-checkpoint holdout curves
    only descriptively. E3's best checkpoint (0.132, epoch 10) is below B1h anyway.
  - Single seed.
- **Decision rule:** E3 does not beat B1h on holdout40, so it does not enter E5.

## E7: DOTA initialisation with a frozen backbone (pre-registered 4 Oct 2026, before any run; `configs/e7_b1h_dota_frozen.yaml`)

- **Observation (E3):** DOTA initialisation raised train40 to 0.729 while holdout40 fell to 0.082. The gap went from
  0.227 to 0.647, and the holdout40 curve peaked at epoch 10 (0.132).
- **Hypothesis (author's):** E3 overwrote the transferred aerial features while memorising the training images.
  Freezing them reduces overfitting.
- **Changes vs E3** (only this): `freeze: 11` in the Ultralytics train args. That freezes `model.0`–`model.10`, the
  whole yolo11s backbone: Conv and C3k2 blocks 0–8, SPPF at 9 and C2PSA at 10 (Ultralytics `yolo11.yaml` backbone
  section). The neck (11–22) and the Detect head (23) are trained.
  - The kernel writes the frozen layer indices and the total and trainable parameter counts to
    `<run>/trainable_params.json` (new `train.py` callback); they will be reported here.
  - Ultralytics freezing sets `requires_grad = False`; BatchNorm running statistics in frozen layers still update in
    train mode.
- **Unchanged from E3:** 50 epochs, `patience: 0`, `last.pt` only, holdout40 scored at every 10-epoch checkpoint
  (`scripts/run_e34.sh`).
- **Prediction (author's):**
  - **vs E3:** a smaller train40 − holdout40 gap than E3's 0.647, and holdout40 mAP50 above E3's 0.0817 by more than
    0.017 (that is, > 0.099).
  - **vs B1h:** we do **not** predict a win over B1h (0.1507).
- **Decision rules:** the standing rules. E7 enters E5 only if it beats B1h on holdout40 by more than 0.017.
- **Launch:** in the next free GPU slot after E4 finishes (E6 holds the other).

**Holdout-curve note (author, 4 Oct):** holdout40 is the validation split; val is the test set. Epoch selection on
holdout40 would be legitimate if declared in advance. This project keeps `last.pt` as pre-registered. Per-checkpoint
holdout40 curves of E1–E4 and E6 are reported descriptively.

### E4: Results (4 Oct 2026; GPU kernel `aradhya1211/auric-e4-flipud-mixup`, code `734299d`, ~1.84 GPU-h kernel time)

| | B1h | E4 (flipud 0.5 + mixup 0.1) |
|---|---|---|
| val mAP50 (95% CI) | 0.1065 (0.056–0.165) | 0.0761 (0.054–0.107) |
| holdout40 mAP50 (95% CI) | 0.1507 (0.056–0.187) | **0.1304 (0.047–0.156)** |
| train40 mAP50 | 0.378 | 0.266 |
| train40 − holdout40 | 0.227 | **0.135** |
| class-agnostic AP50, val / holdout40 / train40 | 0.255 / 0.362 / 0.649 | **0.290 / 0.400** / 0.575 |
| holdout40 AP50: Cargo / Box / Flatbed / Tractor / Liquid | 0.096 / 0.509 / 0.069 / 0.007 / 0.073 | 0.091 / 0.531 / 0.024 / 0.006 / 0.000 |
| val AP50: Cargo / Box / Flatbed / Tractor / Liquid | 0.131 / 0.180 / 0.071 / 0.006 / 0.145 | 0.112 / 0.110 / 0.120 / 0.039 / 0.000 |
| final train losses box / cls / dfl | 1.588 / 1.596 / 1.003 | 1.694 / 1.863 / 1.031 |

Sources: `results/e4_b1h_flipud_mixup/{eval,eval_holdout40,eval_train40}/{per_class.csv,class_agnostic.json}`,
`train/results.csv`. All evals used `last.pt`.

Holdout40 checkpoint curve (`results/e4_b1h_flipud_mixup/checkpoint_curve_holdout/checkpoint_curve_holdout.csv`):

| epoch | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| holdout40 mAP50 | 0.102 | 0.116 | 0.127 | 0.129 | 0.130 |

- **Verdict against the pre-registered prediction:** **partly supported.**
  - The gap shrank: 0.135 vs 0.227.
  - Holdout40 mAP50 did **not** beat B1h by more than 0.017; it is 0.020 *lower*, about one seed spread, so within
    noise in either direction.
- **The direction is opposite to E1–E3.**
  - E4 fits the training images less (train40 0.266), and holdout40 is still rising at epoch 50.
  - Class-agnostic AP *improved* on both held-out sets: holdout40 0.400 vs 0.362, val 0.290 vs 0.255. Class-aware
    mAP50 fell, mostly because Liquid fell to 0 and Flatbed dropped on holdout40.
  - Read together: the augmentation helps finding trucks but not naming them, and it under-fits in 50 epochs.
  - Single seed. The two settings (flipud and mixup) are tested together, not separately.
- **Decision rule:**
  - Neither E3 nor E4 beats B1h on holdout40 by more than 0.017, so there is **no E5** from E3/E4.
  - Per the standing rule, B1h stays the final model unless E6 or E7 beats it. Both were explicitly requested and
    pre-registered after this rule; E7 launched in E4's slot at 13:14.

## Two-stage: crop classifier on E4's boxes (pre-registered 4 Oct 2026, before any run)

- **Observation:** E4 finds more held-out trucks than B1h (class-agnostic holdout40 AP50 0.400 vs 0.362) but names
  them worse (mAP50 0.130 vs 0.151; `results/e4_b1h_flipud_mixup/eval_holdout40/`).
- **Hypothesis (author's):** the crop classifier names E4's detections better than E4's own head.
- **Changes:** none to E4.
  - The crop classifier trained in `auric-fp-crop` (`model.pt`, trained on train-minus-holdout40 GT crops) is
    loaded with `analysis/crop_classifier.py --load-model`.
  - It is applied to E4's holdout40 predictions exactly as to B1h's, with both scoring variants: argmax
    (score = conf × p_max) and allclass (conf × p_c per class).
  - Scored with `eval.py --from-preds --split holdout`.
- **Prediction (author's):** E4 + classifier beats B1h's 0.1507 on holdout40 by more than 0.017, that is > 0.168.
  This is pre-registered for the argmax variant, as for B1h; allclass is reported too.
- **Rule:** only if it passes, score once on val.

## E8: E4's recipe for 100 epochs (pre-registered 4 Oct 2026, before any run; `configs/e8_b1h_flipud_mixup_100ep.yaml`)

- **Amendment to the decision rules (author, 4 Oct).**
  - The standing rule said that if neither E3 nor E4 beats B1h, there is no more GPU training.
  - E4 showed a new pattern: less overfitting (gap 0.135) and holdout40 still rising at epoch 50, though slowly:
    +0.002 and +0.001 over the last 20 epochs (0.127 → 0.129 → 0.130).
  - So longer training is retested with the extra augmentation.
- **Hypothesis (author's):** with augmentation limiting overfitting, longer training helps instead of hurting
  (contrast with E1, plain B1h recipe at 150 epochs: holdout40 0.092).
- **Changes vs E4** (only this): `epochs: 100`. close_mosaic stays 10 epochs as in E4 (so not scaled to steps,
  unlike E6). Also `patience: 0`, `last.pt` only, holdout40 scored at every 10-epoch checkpoint (`scripts/run_e34.sh`).
- **Prediction (author's):** holdout40 mAP50 > 0.1507 + 0.017 = 0.168, with train40 − holdout40 staying below E1's
  0.682. Stated honestly: E4's flattening curve makes a big gain unlikely.
- **Cost and launch:** about 3.7 GPU-h. It launches in the next free GPU slot after E6 or E7 finishes, provided
  `kaggle quota` shows more than 5 h remaining after subtracting 3.7 h and the running kernels' remaining time;
  otherwise ask the author.
- **Launch-rule addition (author, 4 Oct, before launch).** At launch, the quota available minus what running sessions
  reserve must be at least E8's 3.7 h plus a 1 h margin.
  - `kaggle quota` (CLI 2.2.4) does not report the reserved amount; the web UI showed 10 h 53 min reserved by E6 and
    E7 at 14:23 IST.
  - So the watcher launches E8 only after **both** E6 and E7 have finished, when nothing is reserved. It then requires
    at least 4.7 h available and more than 5 h left after E8. Otherwise it does not launch, and the author is asked.

## Background false-positive audit (4 Oct 2026; CPU-only kernel `aradhya1211/auric-fp-crop`, code `734299d`)

- **Observation:** B1h's largest error bin on val is background false positives (TIDE Bkg: 9380 predictions,
  +0.077 if fixed; `figures/b1h_tile1024_holdout40/errors/tide_dAP.csv`). The visual notes mention unlabelled trucks.
- **Hypothesis** (from the E3/E4 pre-registration, CPU item a): missing labels, or vehicles outside our five classes,
  cap what any model can score.
- **Changes:** none. This is an audit of saved predictions.
  - Selection: from B1h's 12897 holdout40 predictions, the 11684 with IoU < 0.1 to every GT box.
  - The 60 most confident of those (conf 0.527–0.804) are cropped with context into one sheet: red = prediction,
    green = GT (`figures/b1h_tile1024_holdout40/fp_audit_sheet.png`).
- **Results** (counted by eye by Claude Code from the sheet; per-crop verdicts in the `verdict` column of
  `results/fp_audit/fp_audit.csv`):

  | verdict | count |
  |---|---|
  | truck-like vehicle with no GT box | 46 |
  | other vehicle (bus, van) | 3 (#38, #44, #53) |
  | background (rail track, shadow, ground) | 4 (#1, #45, #49, #52) |
  | unclear at this resolution | 7 (#8, #10, #16, #17, #33, #47, #57) |

  - In several crops the prediction sits in a row of labelled trucks where one neighbour is unlabelled (#7, #30, #31,
    #36, #41, #43).
- **Conclusion:**
  - Most of the model's confident "false positives" on holdout40 look like real, unlabelled trucks. Missing labels are
    real and frequent among high-confidence predictions, so they cap measured precision. Background errors are a
    minority at the top of the confidence ranking.
  - Limits:
    - One viewer, not blind to the hypothesis.
    - Only 60 of 11684 unmatched predictions, all high-confidence.
    - Low-resolution crops.
    - Whether a truck-like vehicle belongs to one of the five classes cannot always be judged.
  - The size of the effect on mAP50 is not estimated here.
- **Framing (added 2026-10-04, author's direction):**
  - The 46 truck-like unlabelled detections could be either:
    (a) vehicles of our five classes that are missing from the labels, or
    (b) truck types this dataset does not include. xView, whose classes ours match, also has Pickup Truck, Utility
    Truck, Trailer and a generic Truck class.
  - The crops cannot tell (a) from (b).
  - Either way the metric penalises detections of real vehicles. Measured mAP50 is therefore a lower bound on how
    well the model finds trucks.
  - Caveats kept: one viewer, not blind to the hypothesis, low-resolution crops, 60 of 11684 unmatched predictions.
- **Next:** a re-labelling audit of a val sample, to estimate how much measured mAP50 the missing labels cost (a
  candidate in REPORT §6.4).

## Crop classifier on B1h's holdout40 detections (4 Oct 2026; CPU-only kernel `aradhya1211/auric-fp-crop`, code `734299d`)

- **Observation:** classification, mainly Cargo vs Box, is the main loss (§3.2, §5.1).
- **Hypothesis** (E3/E4 pre-registration, CPU item b): a dedicated classifier on crops with context names trucks
  better than the detector head.
- **Changes:**
  - ImageNet ResNet18, trained on 6880 GT crops (2× context, 96 px) from the 403 train images, excluding holdout40.
  - 12 epochs, class-balanced sampling, last epoch used.
  - Then B1h's 12897 holdout40 predictions are re-labelled: argmax (score = conf × p_max) and allclass
    (conf × p_c for each class).
- **Results:**
  - Holdout40 GT-crop accuracy 0.610 overall, mean per-class 0.370
    (`results/crop_classifier/summary.json`, `holdout_gt_accuracy.csv`):

    | class | n | accuracy |
    |---|---|---|
    | Cargo | 321 | 0.701 |
    | Box | 297 | 0.690 |
    | Flatbed | 56 | 0.196 |
    | Tractor | 35 | 0.229 |
    | Liquid | 29 | 0.034 |

  - Re-labelled holdout40 mAP50 (`results/crop_classifier/eval_holdout40_{argmax,allclass}/per_class.csv`):
    - argmax **0.0999** (0.031–0.126)
    - allclass **0.1167** (0.040–0.147)
    - B1h's own head scores **0.1507**
- **Conclusion:**
  - The classifier is worse than the detector's own head. It lowers holdout40 mAP50 by 0.034 (allclass) to 0.051
    (argmax).
  - Its holdout accuracy (0.610) is similar to B1h's GT-box oracle accuracy on val (0.601, a different set), so this
    simple crop classifier does not find extra separability.
  - It changed the class of 38% of the predictions (`detector_class_kept_by_argmax` 0.618).
  - The decision rule for applying it as a second stage (> +0.017) is **not met**.
  - Single seed and a single setting (96 px, 2× context).
- **Next:** the pre-registered two-stage test on E4's boxes (running) uses this same model.

## TTA on B1h: Results (4 Oct 2026; CPU-only kernel `aradhya1211/auric-tta-b1h`, code `095cf9b`)

| variants | holdout40 mAP50 (95% CI) |
|---|---|
| orig (reproduces B1h's 0.1507) | 0.1507 (0.056–0.187) |
| orig + hflip | 0.1447 (0.057–0.177) |
| orig + vflip | 0.1481 (0.061–0.183) |
| orig + up1.5 | 0.1391 (0.060–0.165) |
| **all four (pre-registered test)** | **0.1381 (0.062–0.165)** |

Source: `results/tta_b1h/tta_summary.csv`.
- **Verdict:** **not supported.** The prediction was above 0.1507 + 0.017; TTA lowered holdout40 mAP50 by 0.013.
- Every added variant lowers the score; the 1.5× upscale lowers it most. Val is not scored (the rule requires a
  pass first).
- Interpretation (after the result): adding variants adds low-confidence duplicates and wrong-class boxes that the
  class-wise merge keeps, which hurts precision more than the extra recall helps.

## B1h holdout40 checkpoint curve, descriptive (4 Oct 2026; CPU-only kernel `aradhya1211/auric-b1h-holdout-curve`, code `e5a115e`)

| epoch | 10 | 20 | 30 | 40 | 50 (= `last.pt`) |
|---|---|---|---|---|---|
| holdout40 mAP50 | 0.097 | 0.107 | 0.124 | 0.154 | 0.151 |

- Source: `results/b1h_tile1024_holdout40/checkpoint_curve_holdout/checkpoint_curve_holdout.csv`. Epoch 50
  reproduces the reported 0.1507.
- **Reading:** B1h rises through epoch 40 and is flat from 40 to 50: −0.003, well within the 0.017 seed noise. There
  is no clear peak before epoch 50, so B1h's own recipe shows no sign of overfitting within 50 epochs. Overfitting
  appears only when training goes longer (E1) or fits faster (E2, E3).
- B1h's reported numbers are unchanged (`last.pt`).

## §5.3 per class: 500 targeted instances (4 Oct 2026; CPU-only kernel `aradhya1211/auric-lc-per-class`, code `095cf9b`)

- **Method:** per class, holdout40 AP is fitted against that class's training-instance count, AP = a·n^b on the
  positive points.
  - Seven points: f25, f50 (+ seed 1), f75 (+ seed 1), full (+ seed 1). Smart subsets are excluded.
  - The projection adds 500 instances of that class only; Δ mAP50 = ΔAP / 5.
- Source: `figures/learning_curve/lc_per_class.csv`, `lc_per_class_points.csv`.

| class | train instances (full) | holdout boxes | points used | measured n range | projected ΔAP at +500 | Δ mAP50 | flag |
|---|---|---|---|---|---|---|---|
| Liquid | 163 | 29 | 6 of 7 | 36–163 | +0.081 | +0.016 | **unreliable extrapolation** (+500 is 4.1× the largest measured n) |
| Box | 2069 | 297 | 7 | 475–2069 | +0.055 | +0.011 | 1.24× |
| Flatbed | 606 | 56 | 7 | 159–606 | +0.037 | +0.007 | 1.83× |
| Cargo | 3452 | 321 | 7 | 893–3452 | +0.015 | +0.003 | 1.14× |
| Tractor | 590 | 35 | 5 of 7 | 87–590 | −0.000 | −0.000 | AP not rising with n (b < 0) |

- **Answer to the brief's per-class question:**
  - No single-class allocation of 500 instances projects a gain above the 0.017 noise.
  - Liquid projects the most (+0.016), but that rests on a 4.1× extrapolation from 29 holdout boxes, so it is
    unreliable.
  - Among reliable fits, Box benefits most (+0.011). Tractor shows no benefit.
- **Independent §5.4 evidence on Tractor** (no curve fit):
  - At 302 images the smart subset held 95% of the pool's Tractor boxes vs 49% for random (559 vs 290), yet Tractor
    holdout40 AP was 0.006 vs 0.001 / 0.035 for the random seeds. At 202 images: 438 vs 205 boxes, AP 0.001 vs
    0.000 / 0.002 (`figures/subset_compare/subset_class_coverage.csv`, `results/<run>/eval_holdout40/per_class.csv`).
  - Roughly doubling Tractor examples did not raise Tractor AP. This is confounded the same way as §5.4: the smart
    subsets also carry more boxes overall.
- Caveat: holdout40 has only 35 Tractor and 29 Liquid boxes, so their AP values are noisy.

## E7: Results (4 Oct 2026; GPU kernel `aradhya1211/auric-e7-dota-frozen`, code `e5a115e`, ~1.20 GPU-h kernel time)

- **Freezing** (`results/e7_b1h_dota_frozen/trainable_params.json`):
  - Frozen layers 0–10 (the backbone, as pre-registered), plus layer 23's DFL convolution. Ultralytics always
    freezes `.dfl`, which has no learnable role; it is not a change from E3.
  - 3,987,727 of 9,429,727 parameters trainable; 5,442,000 frozen.
  - DOTA transfer 493/499 tensors, as in E3.

| | B1h | E3 | E7 |
|---|---|---|---|
| val mAP50 (95% CI) | 0.1065 (0.056–0.165) | 0.0881 | 0.0715 (0.047–0.105) |
| holdout40 mAP50 (95% CI) | 0.1507 (0.056–0.187) | 0.0817 | **0.1282 (0.049–0.166)** |
| train40 mAP50 | 0.378 | 0.729 | 0.599 |
| train40 − holdout40 | 0.227 | 0.647 | **0.471** |
| class-agnostic AP50, val / holdout40 / train40 | 0.255 / 0.362 / 0.649 | 0.227 / 0.251 / 0.877 | 0.236 / 0.308 / 0.793 |

Holdout40 curve: 0.128 / 0.132 / 0.124 / 0.135 / 0.128 at epochs 10–50, flat
(`results/e7_b1h_dota_frozen/checkpoint_curve_holdout/checkpoint_curve_holdout.csv`).

- **Verdict:**
  - vs E3: **supported.** Holdout40 is +0.046 above E3 (needed > 0.017), and the gap is 0.471 vs 0.647.
  - vs B1h: as predicted, no win. Holdout40 is 0.023 below, about one seed spread.
- **Conclusion:** freezing the DOTA backbone undoes much of E3's overfitting, which supports the hypothesis that E3
  overwrote transferable features. E7 still fits training images more than B1h does (0.599 vs 0.378) and generalises
  no better.
- **Decision rule:** E7 does not beat B1h on holdout40 by more than 0.017, so it does not enter E5.

### E6: run record (4 Oct 2026; GPU kernel `aradhya1211/auric-e6-rfs`, code `7ec3894`)

- **Repeat factors** (`rfs.json`, from the kernel output; t = 0.1, over 3439 training tiles):

  | class | Cargo | Box | Flatbed | Tractor | Liquid |
  |---|---|---|---|---|---|
  | f_c (share of tiles containing c) | 0.540 | 0.291 | 0.162 | 0.101 | 0.064 |
  | r_c | 1.0 | 1.0 | 1.0 | 1.0 | 1.247 |

  - List length 3496 (vs 3439 tiles), mean tile repeat 1.016.
  - 219 iterations per epoch, 49 epochs, 10,731 iterations (B1h 10,750).
  - Warmup 2.945 epochs = 645 steps, exactly B1h's.
  - close_mosaic 10 epochs = 2190 steps vs B1h's 2150: a **residual of +40 steps**.
- **Important for interpretation:** with t = 0.1 at tile level, every class except Liquid already appears in at least
  10% of tiles, so only Liquid tiles are repeated, about 1.25×. Only 1.6% more tile views were added in total. E6 is
  therefore a weak test of the hypothesis: Tractor is not up-weighted at all. This follows from the pre-registered
  t and the tile-level f_c; it is reported, not changed.
- **Crash:** training completed all 49 epochs (`train/results.csv`). `train.py` then crashed in
  `detlib/curves.summarize_training`, which expected the train entry to be a directory, not a list file. So the
  pre-registered evaluations did not run.
  - Fixed in `3226d48`.
  - The evaluations of the saved `last.pt` (val, holdout40, train40, class-agnostic, holdout40 checkpoint curve) run
    on CPU-only kernel `auric-e6-eval`. Same commands as `scripts/run_e34.sh`, on CPU; no retraining.

## E6b: repeat-factor sampling with t = 0.3 (pre-registered 4 Oct 2026, before launch; `configs/e6b_b1h_rfs_t03.yaml`)

- **Observation:** E6 (t = 0.1) repeated only Liquid tiles (r = 1.25). It added only 57 tile views (+1.6%), so it was
  **too weak to test the hypothesis**; Tractor and Flatbed were not up-weighted at all.
- **Hypothesis (author's):** rare classes are under-sampled. Showing their tiles more often helps them.
- **Changes vs E6** (only this): t = 0.3. Same step matching (about 10,750 iterations; warmup and close_mosaic matched
  to B1h's steps), `patience: 0`, `last.pt` only, holdout40 scored at every checkpoint (`scripts/run_e34.sh`).
- **Repeat factors at t = 0.3**, computed before launch (CPU-only kernel `auric-rfs-preview`, code `0c8fc4a`;
  `results/rfs_preview/rfs_preview.json`; the t = 0.1 row reproduces E6's 3496 entries):

  | class | Cargo | Box | Flatbed | Tractor | Liquid |
  |---|---|---|---|---|---|
  | f_c (share of tiles) | 0.540 | 0.291 | 0.162 | 0.101 | 0.064 |
  | r_c | 1.000 | 1.015 | 1.360 | 1.722 | 2.161 |
  | tiles containing c → list entries containing c | 1856 → 2275 | 1001 → 1303 | 558 → 855 | 348 → 622 | 221 → 478 |

  - The list has 4055 entries vs 3439 tiles: **+616 extra tile views (+17.9%)**, above the author's 10% bar.
- **Prediction (author's):** Tractor and Flatbed holdout40 AP rise slightly; overall holdout40 mAP50 changes by less
  than 0.017. Reasons:
  - In §5.4, about twice as many Tractor boxes did not raise Tractor AP (smart 302: 559 vs 290 boxes, AP 0.006 vs
    0.001 / 0.035).
  - The per-class learning curve projects no single-class gain above noise (Tractor b < 0; Flatbed +0.007 mAP50 at
    +500 instances).
- **Launch:** in E8's GPU slot when E8 finishes. It needs ≥ 2.9 h available (about 1.9 h plus a 1 h margin) and more
  than 5 h left after it; otherwise ask the author.

## E9: a larger COCO model, yolo11m (pre-registered 4 Oct 2026, before launch; `configs/e9_b1h_yolo11m.yaml`)

- **Observation:** E1–E3 show that fitting faster hurts held-out mAP50. Model capacity beyond YOLO11s was never
  tested (REPORT summary: "not tested").
- **Hypothesis (author's):** more capacity, with the same COCO pretraining, generalises better from 403 images.
- **Changes vs B1h** (only these): weights `yolo11m.pt` (COCO) instead of `yolo11s.pt`; `patience: 0`. Everything
  else is B1h's recipe: 50 epochs, 1024 tiles, batch 16 with `train.py`'s automatic fallback to 8 on out-of-memory,
  `last.pt` only, holdout40 scored at every 10-epoch checkpoint (`scripts/run_e34.sh`). If batch 8 is used, the
  batch actually used and the resulting iteration count (`training_summary.json`) are reported.
- **Prediction (author's):** holdout40 mAP50 does **not** beat B1h by more than 0.017, because E3 suggests stronger
  features memorise faster. mAP50 without Truck w/Liquid is also reported, as in the robustness check.
- **Budget (author's, for this run only):**
  - The quota floor is lowered from 5 h to 4 h.
  - `scripts/epoch_guard.py` runs inside the kernel. After epoch 2 it projects the training time (hours at epoch 2
    + 48 × epoch-2 duration) and kills the run if the projection exceeds **4.6 h**. That leaves about 0.4 h for
    setup and the evaluations, so the run stays under 5 GPU-h. The evaluation allowance is based on E3/E4 (kernel
    time minus training time ≈ 0.26–0.3 h). If the guard stops it, the author is told.
- **Launch:** after E8 and E6b finish.

## E10 steps 1–3: xView overlap, label comparison, extra data, FP re-scoring (4 Oct 2026; CPU-only kernel `aradhya1211/auric-xview-overlap` v2, code `59ef959`)

- **Step 1, data source:** Kaggle mirror `hassanmojab/xview-dataset`, 20.8 GB. It contains 846 xView train images
  (`train_images/`), 281 xView val images (`val_images/`, no public labels) and `train_labels/xView_train.geojson`
  (601,937 boxes in 847 images). Kaggle lists the licence as "other"; xView itself is CC BY-NC-SA 4.0.
- **Step 2, overlap** (stem and width × height; `results/xview_overlap/match_summary.json`, `match.csv`):

  | our split | images | same stem in xView | same stem and size | in xView train | in xView val |
  |---|---|---|---|---|---|
  | train | 443 | 443 | **443** | 443 | 0 |
  | holdout40 | 40 | 40 | **40** | 40 | 0 |
  | val | 22 | 22 | **18** | 18 | 0 |

  - The 4 val images whose size differs are exact rescalings of xView train images (`match.csv`):

    | val image | ours | xView | factor |
    |---|---|---|---|
    | 2308 | 7204×5458 | 3602×2729 | 2× |
    | 2391 | 7072×5932 | 3536×2966 | 2× |
    | 2384 | 1369×1334 | 2739×2668 | ≈0.5× |
    | 2460 | 1596×1598 | 3193×3196 | ≈0.5× |

  - **All 465 of our images are xView *train* images**, 4 of the val images rescaled. So any xView-pretrained model
    would have seen every val image with labels.
- **Step 2, labels:**
  - On the 461 images matched at the same size, 8771 of our 8810 boxes (99.6%) pair one-to-one with an xView box at
    IoU ≥ 0.5 (`label_summary.json`).
  - Our class vs the paired xView type (`label_pairs_crosstab.csv`):
    - The diagonal dominates: Cargo 4321, Box 2641, Tractor 645, Flatbed 677, Liquid 108.
    - 379 pairs (4.3%; 8771 − 8392 on the diagonal) have a different xView type, mostly our Tractor / Flatbed / Liquid
      → xView Cargo (52 / 55 / 48)
      and our Liquid → xView Box (32).
  - The supplied labels are therefore xView's boxes for these five types, with a small share of class changes. That
    count is pooled over train and val; the per-split breakdown and the 4 rescaled val images are pending (audit
    item 5).
- **Step 2, extra data:** 382 xView train images are not in our dataset (`extra_counts.csv`).

  | group | instances (images) |
  |---|---|
  | **our 5 classes** | Cargo 881 (20), Box 548 (19), Tractor 129 (17), Flatbed 134 (17), Liquid 23 (12): **1715 in about 20 images** |
  | excluded truck-like types | Truck 1969 (154), Utility Truck 530 (69), Trailer 476 (54), Dump Truck 330 (54), Pickup 227 (44), Haul Truck 164 (25), Crane Truck 36 (22), Cement Mixer 21 (9) |

  - The extra images contain few of our five classes, so E10 would mainly test hypothesis (b), look-alike negatives,
    rather than (a), more data.
  - For (a): +1715 instances is about 25% more than our 6880 training boxes.
- **Step 3, FP re-scoring with xView labels** (best-overlapping xView box at IoU ≥ 0.3; `fp_rescore_summary.json`):
  - **The 60 audited detections:**
    - 43 overlap an excluded truck-like xView box (37 "Truck", plus Trailer, Utility and Dump Truck).
    - 6 overlap one of our 5 types that is missing from our labels.
    - 3 overlap another xView class (Bus, Small Car).
    - 8 overlap nothing.
    - Of the 46 judged truck-like by eye, 34 are excluded types, 6 are our classes missing, 1 is another class and 5
      have no xView box.
  - **All 350 B1h holdout40 predictions with conf ≥ 0.25 and IoU < 0.1 to every GT box:**
    - 216 (62%) excluded truck-like (152 "Truck", 33 Dump, 21 Utility, 8 Trailer, …).
    - 12 our classes missing.
    - 55 other xView class (31 Bus, 14 Small Car, …).
    - 67 nothing.
  - **Conclusion:** most of B1h's confident "background" false positives are real trucks of types the dataset excludes.
    xView's generic "Truck" class dominates. Missing labels of our own five classes are rare (12 / 350).

## E10: extra xView data, val and holdout40 excluded (pre-registered 4 Oct 2026, before launch; `configs/e10_b1h_xview_extra.yaml`)

- **Observation:**
  - All 465 of our images are xView train images (E10 steps 1–2).
  - 62% of B1h's confident holdout40 false positives are trucks of types the dataset excludes; 12 of 350 are our own
    classes missing from the labels (step 3).
  - The 382 xView train images not in our dataset hold 1715 instances of our 5 classes (about 20 images) and 3753 of
    the excluded truck types.
- **Hypotheses (author's):**
  - (a) More data improves generalisation.
  - (b) Explicit labels for look-alike trucks reduce confident false positives in our five classes.
  - Because the extra images contain few of our classes (+25% instances, concentrated in about 20 images), E10
    mainly tests (b).
- **Changes vs B1h:**
  - **Training images:** our 403 (train minus holdout40) plus every labelled xView train image whose ID is not one
    of our 465 images. The 22 val and 40 holdout40 IDs are hard-excluded (`splits/e10_exclude.txt`). The builder
    asserts that none reaches the list, and `tests/test_e3e4.py::test_e10_exclusion` checks the list and the
    selection rule. The list is written to `e10_train_list.txt` in the kernel.
  - **Classes (13):** our 5, plus the 8 excluded truck types as extra classes on all training images, including our
    403: Pickup Truck, Utility Truck, Truck, Trailer, Crane Truck, Dump Truck, Haul Truck, Cement Mixer.
    - Our own 403 images keep their supplied 5-class labels unchanged.
    - The extra images use xView's boxes, mapped 24→Cargo, 25→Box, 28→Flatbed, 26→Tractor, 29→Liquid.
    - `tools/make_xview_dataset.py`.
  - **Evaluation:** only our 5 classes count. `eval.py` drops predicted classes ≥ 5 before merging; this is a no-op
    for 5-class models.
  - **Recipe:** B1h's (YOLO11s COCO, 1024 tiles with overlap 256, `patience: 0`, `last.pt` only, holdout40 at every
    checkpoint). Total iterations are matched to B1h's 10,750 (`target_iterations`; epochs set at run time), and
    warmup and close_mosaic are matched to B1h's steps (as E6). The actual epochs and iterations will be reported.
  - **Caveat:** some excluded-type boxes (for example haul trucks and trailers) may exceed the 256 px tile overlap
    and are then clipped or dropped by the tiler's min_vis 0.5 rule.
- **Predictions (author's direction; numbers from the §5.3 curve):**
  1. **Holdout40 mAP50.**
     - §5.3's power-law fit (a = 0.00424, b = 0.577; `figures/learning_curve/power_law_fit.csv`) at the added
       instance count: 6880 → 8595 instances is about 403 → 504 image-equivalents. Fit 0.135 → 0.154, **a gain of
       about +0.019**, i.e. holdout40 about 0.170.
     - That is right at the 0.017 noise bar. **The prediction is a gain at the noise boundary, not a clear win.**
     - E10 doubles as a direct test of §5.3. If the gain is well below +0.019, extra same-source instances are worth
       less than the curve suggests, for example because they come from only about 20 images.
     - mAP50 without Liquid and class-agnostic AP are also reported.
  2. **Background false positives shrink**, because the excluded truck types are now labelled:
     - (i) val TIDE Bkg error count and dAP below B1h's 9380 / +0.077 (`figures/b1h_tile1024_holdout40/errors/tide_dAP.csv`);
     - (ii) among E10's holdout40 predictions with conf ≥ 0.25 and IoU < 0.1 to every GT box, fewer overlap an
       excluded xView truck type than B1h's 216 (`analysis/xview_overlap.py`).
- **Decision rule:** if E10 beats B1h on holdout40 by more than 0.017, it becomes the final model, and §3 and the
  qualitative examples are redone on it. **E10 is the last training run;** afterwards the final model is frozen.
- **Launch:** in the next free GPU slot. E6b and E8 are running.

## E12: scale-robust inference, no training (pre-registered 4 Oct 2026, before any run; `analysis/e12_scale.py`)

- **Observation:** 4 val images are 2× / 0.5× rescaled copies of xView images (2308, 2391 at 2×; 2384, 2460 at
  0.5×). That puts val's object sizes outside the training distribution: for example, trucks about 9 px in 2384
  (visual review).
- **Hypothesis:** inference that adapts to object scale recovers rescaled images.
- **Changes:** none to the model (B1h `last.pt`). Two candidate rules, both using only the image itself (never xView
  metadata or the known factors):
  - **(a) multi:** union of sliced predictions at 0.5×, 1× and 2× (image resized bilinearly, boxes mapped back), then
    the usual class-wise NMS merge (IoS 0.6, max_det 902).
  - **(b) auto:** predict at 1×; take the median sqrt(area) of predictions with conf ≥ 0.25; choose s ∈ {0.5, 1, 2}
    closest in log scale to 22 px (the median training truck size, `figures/eda/summary.json`); use the predictions
    at that s.
- **Controlled test, holdout40 only:**
  - Three versions: the originals, plus bilinear 0.5× and 2× copies. Labels are normalised, so they are unchanged.
  - B1h is scored on each version with plain (1×), multi and auto.
- **Selection rule (fixed now):**
  - A rule qualifies if (i) on the original holdout40 it loses at most 0.017 mAP50 vs plain, and (ii) its mean mAP50
    over the 0.5× and 2× copies is above plain's mean over them.
  - Among qualifying rules, the one with the higher rescaled-copy mean is chosen. It is written down here before val
    is touched, and then val is scored **once** with it.
  - The 4 rescaled val images and the 18 others are reported separately; the list is used for reporting only.
  - If no rule qualifies, val is not re-scored and B1h's plain inference stays.
- **Prediction (Claude Code; no author prediction recorded):** auto recovers most of the 0.5× and 2× copies' loss
  without hurting the originals. Multi hurts the originals through extra low-confidence boxes, as TTA did.
- **Cost:** GPU, evaluation only (the 2× copies at 2× scale are 16× the pixels).

## Budget decisions (author, 4 Oct 2026, ~16:30 IST)

- `kaggle quota` showed 9.11 h remaining, with E6b and E8 running. E10 (~2.5 h) plus E12 (~1 h) would have left about
  2.5 h.
- **E8 is stopped for GPU budget.** It is not a result. Any holdout40 checkpoints it finished will be reported
  descriptively only, with no verdict against its pre-registration. The stop is done by the author in the Kaggle web
  UI, because the CLI (2.2.4) has no stop command; `delete` would destroy its output.
- **E9 (yolo11m) is cancelled** (never launched); it does not fit the budget.
- **E11 (photometric-robustness run) is cancelled** (never pre-registered).
- **Final candidate:** E10's model with E12's scale-robust inference, each only if it passes its own pre-registered
  test. Combining them is inference only, so it costs no extra training. Otherwise B1h, and/or plain inference.
- The quota floor is about 4.5 h after E10 + E12 (approved).

## Robustness check: mAP50 without Truck w/Liquid (post-hoc; 4 Oct 2026; CPU-only kernel `auric-oracle-noliq-figs`, `analysis/no_liquid.py`)

**Post-hoc, not pre-registered.** This is the mean AP50 of the other four classes, from each run's `per_class.csv`
(`results/no_liquid/no_liquid.csv`). Liquid has 29 holdout40 and 20 val boxes.

| run | holdout40 mAP50 | holdout40 without Liquid | val mAP50 | val without Liquid |
|---|---|---|---|---|
| B1h | 0.151 | 0.170 | 0.106 | 0.097 |
| B1h seed 1 | 0.133 | 0.167 | 0.063 | 0.078 |
| E1 | 0.092 | 0.108 | 0.068 | 0.072 |
| E2 | 0.111 | 0.136 | 0.062 | 0.058 |
| E3 | 0.082 | 0.102 | 0.088 | 0.097 |
| E4 | 0.130 | 0.163 | 0.076 | 0.095 |
| E7 | 0.128 | 0.153 | 0.072 | 0.085 |
| TTA, all four | 0.138 | 0.167 | — | — |

- The holdout40 seed spread is 0.017 with Liquid (0.151 vs 0.133) and 0.004 without it (0.170 vs 0.167). Liquid AP
  alone is 0.073 vs 0.000 across the two seeds, which is worth 0.015 of mAP50. **Most of the seed spread is Liquid.**
- Without Liquid, E4 (0.163) and TTA (0.167) are within 0.007 of B1h (0.170). They are **no better than B1h**,
  rather than worse.
  - The pre-registered verdicts stand unchanged (E4 partly supported; TTA not supported).
- **The E1–E3 conclusions stand:** they remain 0.034–0.068 below B1h without Liquid.

## GT-box oracle on holdout40: detector head vs crop classifier on the same boxes (4 Oct 2026; same kernel)

- **B1h's head at the true boxes** (pool mode): 0.690 accuracy on holdout40's 738 boxes, mean per class 0.411
  (`figures/b1h_tile1024_holdout40/gt_oracle_holdout/comparison.csv`).
- **The crop classifier on the same 738 boxes:** 0.610 (`results/crop_classifier/summary.json`).
- **Always "Cargo":** 321 / 738 = 0.435.
- **The detector head is better than the dedicated crop classifier**, by 0.080 on the same boxes.
- The main confusion is still Cargo ↔ Box: 42 Cargo called Box and 41 Box called Cargo (`confusion_pool.csv`).
- This replaces the earlier rough comparison of 60% (val oracle) vs 61% (holdout crop classifier) on different sets.

## E13: ensemble of existing models (pre-registered 4 Oct 2026, before any run; `analysis/e13_ensemble.py`, CPU only)

- **Observation:** no single change beats B1h. E4 and E7 make different trade-offs: E4 overfits less and finds more
  trucks, E7 keeps aerial features. Top xView solutions used ensembles, e.g. the first-place RFL solution (Reduced
  Focal Loss, arXiv 1903.01347).
- **Hypothesis (author's):** models with different errors fuse into a better detector.
- **Method:**
  - Each model's saved merged predictions, after its usual tile merge (class-wise NMS on IoS 0.6, max_det 902), are
    fused per image with `ensemble_boxes.weighted_boxes_fusion`.
  - Settings: **equal model weights, IoU threshold 0.55**, skip_box_thr 0.001 (= eval conf), default `conf_type`
    "avg". Then the top 902 boxes. These choices are fixed now and are not tuned on val.
  - Scored as usual, with mAP50 without Liquid as well.
- **Combos, holdout40 only:** B1h alone; {B1h, E4}; {B1h, E7}; {B1h, E4, E7}; and, once E10 finishes, every combo
  that adds E10.
- **Selection:** the ensemble with the highest holdout40 mAP50. Val is scored **once**, for that single ensemble,
  in the final run that includes E10.
  - A preview run without E10 and without val may run first. It is descriptive only and selects nothing.
- **Prediction (author's):** the best ensemble beats B1h on holdout40 by more than 0.017.
- **If it passes:**
  - The final submission is that ensemble, plus E12's inference rule if E12 passes.
  - `predict.py` must reproduce it, and the README lists every weights file it needs.
- **Amendments (author, 4 Oct 2026, before any E13 result):**
  1. Choosing the best of several combinations on holdout40 makes the chosen ensemble's holdout40 score optimistic.
     **Pass rule:** the chosen ensemble must beat B1h on holdout40 by more than 0.017 **and** beat B1h on holdout40
     mAP50 without Liquid. Both are reported, plus val once.
  2. For any combination including E10, E10's predictions of the 8 extra classes are dropped before fusion; only
     our 5 classes enter the ensemble. (`eval.py` already drops them; `e13_ensemble.py` filters `cls < 5`
     explicitly for every model.)

## Data-integrity audit: pixels and labels vs the xView originals (4 Oct 2026; CPU-only kernel `aradhya1211/auric-xview-pixels`, code `0c96e12`; `analysis/xview_pixels.py`)

**Pixels** (`results/xview_pixels/pixels.csv`, `pixels_summary.json`):

| split | images | byte-identical to xView | rescaled | other changes |
|---|---|---|---|---|
| train | 403 | **403** | 0 | 0 |
| holdout40 | 40 | **40** | 0 | 0 |
| val | 22 | **14** | 4 | **4** |

The 8 modified val images. Ratios are ours / xView, on grey levels; for the rescaled images, the measurements after
resizing xView with the best-matching method are given.

| image | change | evidence |
|---|---|---|
| 2308 | upscaled 2× | best match: Lanczos (residual MAE 1.73) |
| 2391 | upscaled 2× | best match: Lanczos (residual MAE 1.87) |
| 2384 | downscaled 0.5× | best match: area (residual MAE 0.68) |
| 2460 | downscaled 0.5× | best match: area (residual MAE 0.37) |
| 2292 | noise added | noise ratio 2.73, mean brightness change 0.01, contrast 0.92 |
| 2543 | blur + noise | Laplacian-variance ratio 0.66, noise ratio 1.72 |
| 1399 | contrast reduced, darker tones compressed | contrast ratio 0.50, gamma fit 0.52, Laplacian ratio 0.25 |
| 2139 | contrast boosted | contrast ratio 1.41, gamma fit 1.74, brightness +3.1 |

- 2470 and 2472 (the dark port images) are byte-identical to xView. Their darkness is in the original imagery.
- No image has its R and B channels swapped.

**Labels per split** (one-to-one IoU ≥ 0.5 vs xView's five types; the 4 rescaled val images are compared after
scaling our boxes; `label_compare_by_split.csv`, `label_changes_by_split.csv`):

| split | our boxes | xView boxes | paired, same class | paired, **changed class** | shifted (IoU < 0.95) | ours unpaired | xView unpaired |
|---|---|---|---|---|---|---|---|
| train | 6880 | 7235 | 6510 | **341 (5.0%)** | 378 | 29 | 384 (5.3%) |
| holdout40 | 738 | 786 | 695 | **38 (5.2%)** | 42 | 5 | 53 (6.7%) |
| val | 1552 | 1724 | 1546 | **0 (0%)** | 10 | 6 | 178 (10.3%) |

- **Val's classes are exactly xView's. About 5% of train and holdout40 boxes carry a different class than xView.**
- Most train changes move a box out of xView's Cargo or Box into a rarer class (ours ← xView):
  - Flatbed ← Cargo 54; Tractor ← Cargo 48; Liquid ← Cargo 46; Box ← Cargo 36.
  - Cargo ← Box 27; Flatbed ← Box 26; Liquid ← Box 26; Tractor ← Box 19.
- **Val's labels omit 10.3% of xView's boxes of the five types, vs 5.3% for train.**
- Train also has more boxes shifted relative to xView than val (378 vs 10).
- **Link to E6b:** the rare classes' training labels are partly flipped Cargo/Box boxes. For example, 54 of
  our training Flatbed boxes and 48 of our Tractor boxes are Cargo in xView. Repeating those tiles (E6b) therefore
  also repeats label noise, a plausible reason resampling did not help.
- **Disclosure:** The val alterations and label differences were found by comparing against the public xView originals. Val labels and pixels were never used for training or for selecting anything, and E12's rules are chosen on synthetically degraded holdout40 only.
- **Reading:** the training labels look like xView's with class noise added; val keeps xView's classes but drops
  about 1 in 10 boxes; and 8 of 22 val images were photometrically or geometrically altered.

## Holdout40 TIDE breakdown vs val (B1h; CPU-only kernel `auric-b1h-errors-holdout`, code `9bae85a`; `errors.py --holdout-list`)

| fix (dAP50) | val | holdout40 |
|---|---|---|
| Cls | +0.147 | **+0.177** |
| Bkg | +0.077 | **+0.130** (11684 errors) |
| Missed | +0.044 | +0.014 |
| Loc | +0.022 | +0.015 |

Sources: `figures/b1h_tile1024_holdout40/errors/tide_dAP.csv` (val), `errors_holdout/tide_dAP.csv` (holdout40).
- **Classification also dominates on holdout40.** It is the largest fix there too, even though the GT-box oracle is
  much better on holdout40 (0.690 vs 0.601): classification errors on detected boxes still cost the most AP.
- The background bin is larger on holdout40, where §3.3b / E10 step 3 found 62% of confident background errors to be
  excluded truck types.
- Misses matter less on holdout40 than on val.

## E6b: Results (4 Oct 2026; GPU kernel `aradhya1211/auric-e6b-rfs-t03`, code `0c8fc4a`, ~1.71 GPU-h kernel time)

- **Run:** 4055 list entries, 42 epochs, 10,668 iterations (B1h 10,750). The close_mosaic residual is −118 steps
  (`results/e6b_b1h_rfs_t03/rfs.json`).

| | B1h | E6b |
|---|---|---|
| holdout40 mAP50 (95% CI) | 0.1507 | 0.1363 (0.045–0.170) |
| val mAP50 (95% CI) | 0.1065 | 0.0908 (0.049–0.145) |
| train40 mAP50 | 0.378 | 0.569 |
| holdout40 AP50 Tractor / Flatbed / Liquid | 0.007 / 0.069 / 0.073 | 0.002 / 0.061 / 0.010 |

- Holdout40 curve: 0.101 / 0.129 / 0.137 / 0.139 / 0.136 at epochs 10 / 20 / 30 / 40 / 42.
- **Verdict:** the overall change is −0.014, within the 0.017 bar as predicted. But **Tractor and Flatbed did not rise**
  (0.007 → 0.002 and 0.069 → 0.061), so the class part of the prediction is **not supported**.
- Repeating rare-class tiles made the model fit its training images more (train40 0.569) without helping the rare
  classes on held-out images. This is consistent with the §5.4 Tractor evidence.

## E13 preview, holdout40 only, without E10 (descriptive; selects nothing; pre-amendment script; `results/e13_preview/`)

| combo | holdout40 mAP50 | without Liquid |
|---|---|---|
| B1h | 0.1507 | 0.1702 |
| B1h + E4 | 0.1627 | 0.1858 |
| B1h + E7 | 0.1660 | 0.1877 |
| B1h + E4 + E7 | 0.1723 | 0.1955 |

- The final E13 run, with E10 and val scored once, follows when E10 finishes.

## E15: train on the original xView labels (pre-registered 4 Oct 2026, before launch; `configs/e15_b1h_xview_labels.yaml`)

- **Observation:** our training labels deviate from their documented public source. 341 of 6851 paired train boxes
  (5.0%) carry a different class than xView, mostly xView Cargo/Box → our Flatbed, Tractor or Liquid, and 378 are
  shifted (IoU < 0.95). Val keeps xView's classes (Data-integrity audit).
- **Justification (author's, recorded before results):** this is a training-data quality fix. The labels are restored
  to their documented source; nothing is tuned toward val.
- **Changes vs B1h (only this):**
  - For our 403 training images, every supplied box that pairs one-to-one with an xView box of the five types
    (IoU ≥ 0.5) takes xView's original class **and** box. Unpaired supplied boxes keep their supplied label, and
    xView boxes without a supplied partner are not added (`tools/make_xview_relabel.py`).
  - Everything else is B1h's recipe: 50 epochs, seed 0, `patience: 0`, `last.pt`. Val and holdout40 are excluded
    from training, as for B1h.
- **Evaluation:**
  - holdout40 **twice**: with its supplied labels, and with xView-original labels relabelled by the same rule.
    **B1h and E15 are compared under both.**
  - val **once** (supplied labels, which equal xView's classes).
  - mAP50 without Liquid as well.
- **Prediction (author's):**
  - Under the xView-original holdout labels, E15 beats B1h by more than 0.017; under the supplied labels it may not.
  - Flatbed, Tractor and Liquid AP rise.
- **Cost:** about 2 GPU-h. Next free GPU slot; approved quota floor 3.5 h.

## E10: Results (4 Oct 2026; GPU kernel `aradhya1211/auric-e10-xview-extra`, code `23607fb`, ~1.90 GPU-h kernel time)

- **Data:** 403 of our images plus **202** extra xView images (`results/e10_b1h_xview_extra/e10_dataset_summary.json`,
  `e10_train_list.txt`, 605 entries).
  - *Note:* the pre-registration said 382 extra images. Only the 202 that contain a box of the 5 classes or the 8
    excluded truck types were used, as the builder's rule specifies; the other 180 carry none of the 13 classes.
  - The added instance counts are as pre-registered: 1715 of our classes, plus 3747 excluded-type boxes after clipping
    (pre-registration: 3753 before clipping).
  - **0 of the 62 excluded IDs (val + holdout40) are in the training list** (checked against `splits/e10_exclude.txt`).
- **Schedule:** 6601 tiles, 26 epochs, 10,738 iterations (B1h 10,750); close_mosaic residual −85 steps (`rfs.json`).

| | B1h | E10 |
|---|---|---|
| holdout40 mAP50 (95% CI) | 0.1507 | **0.1256 (0.032–0.157)** |
| holdout40 class-agnostic AP50 | 0.362 | 0.378 |
| val mAP50 (95% CI) | 0.1065 | 0.0820 (0.042–0.138) |
| val class-agnostic AP50 | 0.255 | 0.202 |
| train40 mAP50 | 0.378 | 0.234 |
| val TIDE Bkg (n / dAP) | 9380 / +0.077 | **4056 / +0.055** |
| val TIDE Missed (n / dAP) | 419 / +0.044 | 727 / +0.059 |
| holdout40 AP50 Cargo / Box / Flatbed / Tractor / Liquid | 0.096 / 0.509 / 0.069 / 0.007 / 0.073 | 0.105 / 0.496 / 0.026 / 0.001 / 0.000 |

Sources: `results/e10_b1h_xview_extra/{eval,eval_holdout40,eval_train40}/`, `figures/e10_b1h_xview_extra/errors/tide_dAP.csv`.
Holdout40 curve: 0.092 / 0.133 / 0.126 at epochs 10 / 20 / 26.

- **Verdicts:**
  1. **Holdout40 mAP50 vs the §5.3-based prediction (+0.019): not supported.** It moved −0.025, the opposite way, so
     E10 is not a direct confirmation of §5.3's curve.
     - Without Liquid: E10 is (0.105 + 0.496 + 0.026 + 0.001) / 4 = 0.157 vs B1h's 0.170, still below.
     - Liquid fell to 0 (29 boxes).
  2. **Background false positives shrink: supported on val** (Bkg errors 9380 → 4056, dAP +0.077 → +0.055). Misses
     rose (419 → 727). Part (ii), the xView re-scoring of E10's holdout40 FPs, is pending (CPU).
- **Decision:** E10 does not beat B1h on holdout40. It is not the final model on its own; it still enters E13 as an
  ensemble candidate.

## E12: scale test result (4 Oct 2026; GPU kernel `aradhya1211/auric-e12-scale`, code `23607fb`, ~0.51 GPU-h)

Holdout40 mAP50 by version and rule (`results/e12_scale/test_scores.csv`):

| version | plain | multi | auto |
|---|---|---|---|
| original (1×) | 0.1507 | 0.1414 | **0.1570** |
| 0.5× copy | 0.0927 | 0.1227 | 0.1217 |
| 2× copy | 0.0915 | 0.1193 | **0.1482** |
| mean of the copies | 0.0921 | 0.1210 | **0.1350** |

- **By the pre-registered rule, both scale rules qualify:** each loses less than 0.017 on the originals (multi −0.009;
  auto +0.006) and beats plain on the rescaled copies.
- **auto has the higher copy mean, so auto is the scale rule.** It chose 2× for 13 of the 40 0.5× copies and 0.5×
  for 35 of the 40 2× copies.
- **Val is NOT scored yet:** amendment D extends E12 with photometric rules, and val is scored once after the
  amended test.

### E12 amendment: robust inference (author, 4 Oct 2026; before any amended run and before any val scoring; `analysis/e12_robust.py`)

- **Why:** 8 of 22 val images are altered: 4 rescaled, plus noise, blur + noise, contrast ×0.5 and ×1.4
  (Data-integrity audit). Scale alone covers only half.
- **Rules (all use only the image itself; parameters fixed now):**
  - **plain:** no change.
  - **auto:** the scale rule chosen in the scale test.
  - **cnorm:** per image and per channel, a linear map to the training images' channel mean and std (403 train
    images, holdout40 excluded, computed once).
  - **denoise:** `cv2.bilateralFilter(d=5, sigmaColor=20, sigmaSpace=5)`.
  - **combo:** denoise, then cnorm, then auto.
- **Controlled test, holdout40 ONLY.** Synthetic, deterministic copies (seed 0):

  | copy | how it is made |
  |---|---|
  | clean | unchanged |
  | 0.5× / 2× | bilinear rescale |
  | Gaussian noise | σ = 10 |
  | blur + noise | Gaussian blur σ = 1.5, then noise σ = 8 |
  | contrast ×0.5 / ×1.4 | about each image's mean |

  Every rule is scored on every copy.
- **Pass rule:**
  - A rule passes if (i) clean holdout40 mAP50 is at most 0.017 below plain, and (ii) its mean over the 6 degraded
    copies is above plain's mean over them.
  - The chosen rule is the passing rule with the highest degraded mean (`chosen_rule.json`). Then val is scored
    **once** with it, reported for all 22 images and separately for the 14 unaltered, 4 rescaled and 4
    photometrically altered images.
  - If none passes: plain, and val is not re-scored.
- **Disclosure:** the degraded copies are synthetic and were made only from holdout40. The val alteration types,
  found via the xView comparison, motivated *which* degradations to simulate, but no val pixel or label is used to
  choose anything.
- **Prediction (Claude Code; no author prediction recorded):** auto and combo pass. cnorm alone helps the contrast
  copies but hurts clean images slightly.

## Amendments after external review (author, 4 Oct 2026; recorded before any E13-final, E15 or re-scoped E12 result)

1. **E13, multi-label output (inference only).**
   - B1h, E4, E7, E10 and E15 are re-run on holdout40 with multi-label output: each box may carry every class whose
     score is ≥ conf, as Ultralytics' DetectionValidator does (`non_max_suppression(..., multi_label=True)`). Then the
     usual tile merge and scorer.
   - **Motivating evidence:** given the true box, the correct class is in B1h's top 2 for 83.6% of val boxes and
     84.8% of holdout40 boxes (top 1: 60.1% / 69.0%; `figures/b1h_tile1024_holdout40/gt_oracle*/gt_scores.csv`). The
     val number is a diagnostic only; the holdout number motivates the change.
   - **Selection:** single-label vs multi-label is chosen on holdout40 only.
   - **Disclosure:** Ultralytics' own mAP uses multi-label.
2. **Clean holdout40 is the selection yardstick.**
   - Holdout40 is relabelled with xView's original classes and boxes, by the same rule as E15's training labels.
   - From now on selection uses **holdout40-clean**; supplied-label holdout40 is still reported.
   - All existing models are re-scored on both.
3. **Geographic decontamination before E10 counts.**
   - Each image's footprint is computed from the xView GeoTIFF tags.
   - Any extra E10 image whose footprint overlaps or touches a val or holdout40 image is reported. If E10 trained on
     one, E10 is either retrained without it or excluded from val scoring.
   - Train vs holdout40 and train vs val overlaps are reported too.
4. **E10's labels for our 5 classes on our 403 images were the SUPPLIED labels** (unchanged; `make_xview_dataset.py`
   writes our label file and only appends xView's excluded-type boxes). xView labels were used only for the extra
   images and for the 8 extra classes.
5. **E12 re-scoped:** the degraded-holdout copies are replaced with a generic suite, fixed now (seed 0).

   | corruption | severities |
   |---|---|
   | scale | 0.5, 0.75, 1.25, 1.5, 2.0 (bilinear) |
   | Gaussian noise | σ = 5, 10 |
   | Gaussian blur | σ = 1, 2 |
   | contrast | ×0.6, ×1.4 about the image mean |
   | brightness | −30, +30 |
   | JPEG | quality 50, 20 |

   - **Rules:** plain, auto, cnorm, denoise, combo (as in the first amendment).
   - **Selection:** on the suite average, over all 15 corruptions, on holdout40-clean; clean holdout40 must not drop
     by more than 0.017.
   - **Disclosure:** the audit motivated testing robustness, but the suite is generic and is not fitted to val's
     measured corruptions.
   - The first-amendment run (`auric-e12-robust`, launched about 20:10) is superseded. Its results were not read
     before this amendment and are not used.
6. **Checkpoint selection** is allowed for the final system only.
   - Rule (written now): among the saved checkpoints (`epoch010`–`epoch040`, `last.pt`) of the model(s) in the
     chosen final system, take the one with the highest holdout40-clean mAP50.
   - No new training. Val is scored once, with the final system.
- **E13 start gate (author, 4 Oct 2026):** E13 final does not start until (a) the geographic-overlap check
  (`analysis/review_checks.py`, kernel `auric-review-checks`) and (b) the E10 label-conflict check
  (`analysis/e10_label_conflicts.py`, kernel `auric-e10-conflicts`) have both finished and been recorded here. If the
  E12 suite parts still hold the CPU slots when E15 finishes, E13 waits. E13 is launched by hand, never by a watcher.

## E15: Results (5 Oct 2026; GPU kernel `aradhya1211/auric-e15-xview-labels`, code `1bb49a2`, ~1.76 GPU-h kernel time)

- **Relabelling** (`results/e15_b1h_xview_labels/relabel_summary.json`):
  - train: 6880 boxes paired with xView, 343 classes changed, 0 left unpaired;
  - holdout40: 738 paired, 38 changed.
  - (The earlier audit's 6851 paired used unclipped xView boxes; this builder clips them to the image first.)

| holdout40 mAP50 (95% CI) | B1h | E15 |
|---|---|---|
| supplied labels | 0.1507 (0.056–0.187) | 0.1372 (0.049–0.168) |
| **xView-original labels ("holdout40-clean")** | **0.1765** (0.066–0.225) | 0.1526 (0.053–0.193) |

| | B1h | E15 |
|---|---|---|
| val mAP50 | 0.1065 | 0.0908 |
| train40 mAP50 | 0.378 | 0.372 |
| holdout40 AP50, supplied (Cargo / Box / Flatbed / Tractor / Liquid) | 0.096 / 0.509 / 0.069 / 0.007 / 0.073 | 0.091 / 0.529 / 0.049 / 0.002 / 0.016 |
| holdout40-clean AP50 | 0.094 / 0.569 / 0.086 / 0.003 / 0.131 | 0.097 / 0.571 / 0.074 / 0.000 / 0.020 |

Holdout40 (supplied) curve: 0.111 / 0.106 / 0.129 / 0.143 / 0.137 at epochs 10–50. Sources:
`results/e15_b1h_xview_labels/{eval,eval_holdout40,eval_train40,xviewlabels_holdout40_e15,xviewlabels_holdout40_b1h}/per_class.csv`.

- **Verdict:**
  - **Not supported.** Under xView-original holdout labels, E15 is 0.024 *below* B1h (predicted: more than 0.017
    above). Under supplied labels it differs by −0.013, no detectable effect.
  - Flatbed, Tractor and Liquid did **not** rise; Liquid fell under both label sets.
  - Liquid alone explains most of the difference: 29 holdout boxes, 0.131 → 0.020 on clean labels.
- **New fact:** B1h itself scores **0.1765 on holdout40-clean** vs 0.1507 on supplied labels (+0.026). Scored against
  xView's original classes and boxes, the same predictions do better.
- **Decision:** E15 does not beat B1h, so it is not final alone. It enters E13 as a candidate.

## Gap breakdown on val (diagnostic only, never a claimed result; 5 Oct 2026; CPU-only kernel `auric-opgap`, code `c21e52e`)

| row | mAP50 | class-agnostic AP50 |
|---|---|---|
| 1 plain | 0.1065 | 0.255 |
| 2 excluded-type ignore | 0.1237 | 0.307 |
| 3 native scale (4 rescaled images) | 0.1122 | 0.290 |
| 4 = 2 + 3 | 0.1297 | 0.347 |
| 5 dropped-box ignore | 0.1112 | 0.272 |
| **6 = 2 + 3 + 5** | **0.1371** | **0.373** |

- Source: `results/gap_breakdown/gap_breakdown.csv`, `.json`.
- Row 2 removes 1176 predictions (1565 excluded-type xView boxes on val). Row 5 uses 176 dropped five-type xView boxes
  (≥ 0.5 IoU with no supplied box); the audit counted 178 unpaired one-to-one.
- xView labels are used here only to explain the gap, never for training or selection.

## Operating points, B1h (5 Oct 2026; same kernel; `results/operating_points/`, `figures/operating_points/`)

| threshold | val mAP50 | val P | val R | val F1 | holdout40 mAP50 | holdout40 P | holdout40 R | holdout40 F1 |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.106 | 0.065 | 0.489 | 0.115 | 0.151 | 0.041 | 0.710 | 0.077 |
| 0.05 | 0.091 | 0.199 | 0.314 | 0.244 | 0.139 | 0.167 | 0.519 | 0.253 |
| 0.1 | 0.082 | 0.254 | 0.265 | **0.259** | 0.131 | 0.219 | 0.463 | 0.298 |
| 0.25 | 0.062 | 0.323 | 0.155 | 0.210 | 0.107 | 0.346 | 0.332 | **0.339** |
| 0.5 | 0.039 | 0.338 | 0.051 | 0.088 | 0.058 | 0.551 | 0.161 | 0.249 |
| 0.8 | 0.004 | 0.400 | 0.001 | 0.003 | 0.003 | 0.500 | 0.004 | 0.008 |

- All 12 thresholds are in the CSVs.
- **F1-optimal threshold:** 0.10 on val (F1 0.259) and 0.25 on holdout40 (F1 0.339).
- mAP50 can only fall as the threshold rises (0.106 → 0.004 on val).
- **Precision is understated:** detections of excluded truck types and of the dropped val boxes count as false
  positives.

## Visual review (Claude chat), with supporting numbers (5 Oct 2026; CPU-only kernel `auric-val-review`, code `5204c51`; `results/val_review/`)

Claude chat reviewed `figures/inspect_val/` without seeing our interpretation (one viewer, not blind to the project).
Claude Code reproduced the numbers.

1. **Labels are not shifted.** A global (dx, dy) search over ±60 px (step 3), counting GT matched by conf ≥ 0.05
   predictions at IoU ≥ 0.5, finds the best at zero shift for **20 of 22** images. The other 2 gain at most 1 matched
   box (`r1_label_shift.csv`). *(The review said all 22; the two exceptions are a 1-box tie-level difference.)*
2. **Merging is not the bottleneck.** Holdout40 raw predictions re-merged (`r2_remerge.csv`), vs 0.1507 (reproduced
   exactly):

   | variant | change in mAP50 |
   |---|---|
   | NMS IoU 0.5 | +0.001 |
   | NMS IoU 0.7 | −0.005 |
   | drop interior-edge boxes | +0.001 |
   | cross-tile-only NMS | −0.003 |

3. **Truck w/Box concentration:** 224 of val's 493 Box labels are in 2470 + 2472 (`r3_val_class_counts.csv`, with
   per-image class counts).
4. **Observations by eye** (one viewer, not blind): about 16 of the 24 top val false positives show trucks; Cargo vs
   Box looks inconsistent between train and val; several of the worst-recall images look degraded.
   - The pixel audit confirms 2292 (noise), 2543 (blur + noise) and the rescaled 2384.
   - 2470 and 2472 are dark in the original imagery, not altered.
5. **Image sizes:** 2384 and 2460 are smaller than every train image, and 2308 and 2391 larger. Train's longest
   sides range from 2576 to 5119 px.

**Follow-ups:**
- **(a) Image quality vs recall** (`a_standardised.csv`, `a_correlations.csv`, `figures/val_review/a_*.png`).
  Re-weighting holdout40 recall to val's quartile distribution explains:

  | measure | share of the 0.665 vs 0.852 recall gap |
  |---|---|
  | brightness | 31% |
  | RMS contrast | 31% |
  | blur | −10% |
  | noise | 6% |

  - Per-image Spearman correlations with recall are weak (|ρ| ≤ 0.27).
  - Brightness and contrast explain about a third of the gap; blur and noise measured this way do not.
  - With 22 val images this is weak evidence.
- **(b) 2391.png** (`b_2391_missed.csv`, `figures/val_review/b_2391_tile.jpg`):
  - 68 of its 108 GT boxes have no prediction at IoU ≥ 0.5 at any confidence: 46 Box, 18 Cargo, 3 Tractor, 1 Flatbed.
  - The CSV gives oracle scores and nearby raw predictions per box.
  - 2391 is one of the 2× upscaled images.
- **(c) Geography** (GeoTIFF footprints, `c_geo_overlaps.csv`):
  - 30 train or holdout40 images touch or overlap 15 val images, but every overlap is ≤ 3.2% of the image's area: they
    are adjacent chips of the same scenes, not duplicated pixels.
  - 2 pairs involve holdout40.
  - 2459 (train) overlaps 2470 by 0.34% of its area.
- **(d) E11** (photometric robustness) was cancelled for budget. The E12 robust-inference suite covers the inference
  side.

## E12 (re-scoped): Results (5 Oct 2026; CPU-only kernels `auric-e12s-a`–`e` + `auric-e12-select`; `results/e12_robust/`)

Holdout40-clean mAP50 (xView-original labels), B1h weights (`robust_test_pivot.csv`):

| rule | clean holdout40 | mean over the 15 corruptions | passes |
|---|---|---|---|
| plain | 0.1765 | 0.1447 | — |
| **auto** | **0.1840** | **0.1451** | **yes** |
| denoise | 0.1712 | 0.1432 | no |
| combo | 0.1655 | 0.1353 | no |
| cnorm | 0.1605 | 0.1314 | no |

- **Selection by the pre-registered rule:** auto is the only passing rule, so it is chosen.
- **Its margin over plain on the suite average is tiny:** +0.0004 (0.1451 vs 0.1447). It is ahead on the clean images
  (+0.0076).
- **Val, scored once with auto** (`val_score.json`):

  | val subset | mAP50 with auto | plain |
  |---|---|---|
  | all 22 | **0.0906** | 0.1065 |
  | 14 unaltered | 0.1119 | not computed |
  | 4 rescaled | 0.0621 | not computed |
  | 4 photometric | 0.1021 | not computed |

- **On val, auto lowers mAP50 by 0.016 vs plain**, at the edge of the val seed spread (0.043) and well within it.
- **Auto's choices on val:** it chose 0.5× for 7 unaltered images, including 2470 and 2472, and 1× for all 4 rescaled
  images (`chosen_scales`). So it did not apply the intended correction to the rescaled images.
  - On the 2× upscaled images (2308, 2391) the detected boxes' median size evidently did not push the rule to 0.5×,
    plausibly because most of those trucks were missed in the first pass (see 2391: 68 of 108 boxes never matched).
- **Verdict:** the rule passed its pre-registered holdout40 test by a negligible margin and does not help on val. By
  the pre-registration it qualifies for the final system; whether to include it is the author's decision, recorded
  with this result. No other rule is tried on val.

## E10 label-conflict check (5 Oct 2026; CPU-only kernel `auric-e10-conflicts`; `results/e10_conflicts/`)

- **E10 conflicts:** in E10's training labels for our 403 images, **10** pairs where a supplied 5-class box and an
  appended excluded-type xView box overlap at IoU ≥ 0.5, i.e. the same truck labelled twice:

  | our class | xView excluded type | pairs |
  |---|---|---|
  | Cargo | Truck | 3 |
  | Box | Trailer | 2 |
  | Cargo | Dump Truck | 1 |
  | Box | Truck | 1 |
  | Flatbed | Utility Truck | 1 |
  | Tractor | Trailer | 1 |
  | Flatbed | Dump Truck | 1 |

- **Inside xView's own labels** on the same images there are **9** such pairs, with almost the same class pairs
  (`xview_internal_conflicts.csv`). The conflicts come from xView itself.
- **Conclusion:** a handful (10 among about 3700 appended boxes). It is not a material caveat on E10. E15-style
  xView-consistent labels would **not** avoid it, because xView double-labels these trucks itself.

## Clean-room reproducibility (5 Oct 2026; CPU-only kernel `auric-clean-repro` v2; `results/clean_repro/`)

- **What it ran:** the README's commands, from a fresh `git clone` of the public repo, into a new virtual environment
  (`.venv`, Python 3.13.15, torch 2.10.0+cpu, Ultralytics 8.4.171; verified with `sys.prefix`). The weights came by
  anonymous `curl` from the GitHub release.
- **Weights:** SHA-256 `3fa24066…7ffb`, as expected.
- **Result:** val mAP50 **0.1065**. The per-class AP50 values equal `results/b1h_tile1024_holdout40/eval/per_class.csv`
  exactly.
- **Runtime:** install 143 s, prediction on 22 val images 356 s (CPU), total 501 s.
- **Two Kaggle-specific deviations:**
  1. Kaggle's Python image lacks `ensurepip`, so the venv was made with `--without-pip` plus `get-pip.py`. The v1
     run, which skipped this, had silently fallen back to Kaggle's own packages.
  2. Kaggle mounts the dataset already extracted, so the data step is a symlink instead of an unzip.
  - Kaggle's `sitecustomize` prints a harmless `wrapt` warning inside the venv.

## Review checks (5 Oct 2026; CPU-only kernel `auric-review-checks` v3; `results/review_checks/review_checks.json`)

*Root cause of repeated "stale code" kernel failures, fixed in `scripts/kaggle_kernel_run.py`:* kernels that mounted
the B1h kernel's output also saw that kernel's old repo copy (commit `4a17eb1`), and the entry script could pick it.
It now prefers the `auric-cv-code` dataset. The guards caught every case; no result used stale code.

1. **max_det** (B1h holdout40, re-merged): 902 → 0.1507; 3000 → 0.1511; 10000 → 0.1511 (**+0.0004**).
   The pycocotools cross-check uses maxDets = the same 902 (`eval.coco_crosscheck`).
2. **Class-agnostic merging** (NMS ignoring class) lowers holdout40 to **0.1287** (vs 0.1507).
3. **IoU 0.1 vs 0.5:** holdout40 mAP 0.1666 at IoU 0.1 vs 0.1507 at 0.5. Tractor AP stays near 0: **0.0098** at IoU 0.1
   (0.0070 at 0.5), so Tractor's failure is not a box-offset issue.
4. **GT-box oracle, top-1 / top-2:**

   | split | boxes | top-1 | top-2 |
   |---|---|---|---|
   | val | 1552 | 0.601 | **0.836** |
   | holdout40 | 738 | 0.690 | **0.848** |

5. **Density, as boxes per megapixel** (*corrected 2026-10-05: replaces boxes per image*):

   | split | pooled | per-image median | boxes per image |
   |---|---|---|---|
   | train | 1.73 | 0.69 | 17.2 |
   | val | 6.01 | 4.80 | 70.5 |

   Val is about 3.5× denser per area pooled, 7× by median.
6. **Geography** (GeoTIFF footprints; "touch" includes shared edges; 667 footprints):

   | pair of sets | touching or overlapping pairs |
   |---|---|
   | E10 extra images vs val | **8** (e.g. 1216/1211, 2306 and 2309/2308, 2398/2384) |
   | E10 extra images vs holdout40 | **23** |
   | our train vs val | 44 (inherent to the supplied split) |
   | our train vs holdout40 | 56 |

   The earlier review measured these overlaps at ≤ 3.2% of an image's area, i.e. adjacent chips.
   - **Decision under amendment 3 (Claude Code, applying the pre-recorded rule):** E10 trained on extra images
     touching val and holdout40 images, so **E10 is excluded from E13** and from any val scoring. It is not
     retrained: about 2 GPU-h, and E10 showed no gain.
   - E10's holdout40 numbers are reported as possibly optimistic.

## E13 final: Results (5 Oct 2026; GPU kernel `aradhya1211/auric-e13-final`, code `3f340cf`, ~0.45 GPU-h; `results/e13_final/`)

- **Candidates:** B1h, E4, E7 and E15. E10 is excluded under amendment 3 (its extra images touch val and holdout40).
  All B1h-containing combinations are scored, single- and multi-label, on holdout40-clean (xView-original labels)
  and supplied labels (`holdout40_candidates.csv`).

| top candidates | mode | holdout40-clean | clean without Liquid | supplied |
|---|---|---|---|---|
| **B1h + E4 + E7** | **multi** | **0.2086** | **0.2225** | 0.1789 |
| B1h + E4 + E7 + E15 | multi | 0.2065 | 0.2244 | 0.1782 |
| B1h + E4 | multi | 0.2048 | 0.2092 | 0.1759 |
| B1h + E4 + E7 | single | 0.2004 | 0.2196 | 0.1723 |
| B1h alone | multi | 0.1866 | 0.1939 | 0.1597 |
| **B1h alone (reference)** | single | **0.1765** | **0.1879** | 0.1507 |

- **Selection:** multi-label B1h + E4 + E7, by the highest holdout40-clean mAP50.
- **Pass rule (amendment 1):** **passes.** It is +0.032 over B1h on holdout40-clean (> 0.017) and beats B1h without
  Liquid (0.2225 vs 0.1879).
  - Caveat: the choice is the best of 32 candidates on the same 40 images, so 0.2086 is optimistic.
- **Multi-label output** helps every candidate here, for example B1h alone 0.1765 → 0.1866.
- **Checkpoint rule (amendment 6):** for each member, the checkpoint with the highest holdout40-clean mAP50 alone (multi):

  | member | checkpoint scores (clean) | chosen |
  |---|---|---|
  | B1h | ep10 0.151, ep20 0.144, ep30 0.191, **ep40 0.200**, ep50 0.187 | **epoch040** |
  | E4 | 0.132, 0.159, 0.159, **0.171**, 0.171 | **epoch040** |
  | E7 | 0.165, **0.190**, 0.158, 0.158, 0.153 | **epoch020** |

  With these checkpoints the system scores 0.2364 on holdout40-clean. That is doubly optimistic (checkpoints and
  combination both picked on holdout40). Supplied labels: 0.1941.
- **Val, scored once:** **mAP50 0.1349** (without Liquid 0.1274), vs B1h's 0.1065, i.e. **+0.028**.
  - This is the claimed final result.
  - Seed noise on val is 0.043 (B1h seeds), so the gain is below the val seed spread. On holdout40-clean the gain is
    larger than the seed spread.
- **The final system:** three YOLO11s checkpoints, each with the B1h sliced pipeline and multi-label NMS, fused by WBF.
  - Weights: GitHub release `weights-final-v1`, with SHA-256 sums.
  - Reproduced by `predict.py --weights <3 files> --multi-label`.
  - E12's auto rule is **not** included: it passed its holdout test by +0.0004 but lowered val, and the author's
    decision on it is pending.

## Detection-cap check for the final system (rule recorded 5 Oct 2026, before running)

- **Observation:** with multi-label output, each box can carry several class copies, and 21 of 22 val images hit
  max_det = 902 in the final system. 902 was set before multi-label output existed.
- **Test, holdout40-clean only:** re-run the final ensemble with max_det = 3000 everywhere: per-tile prediction,
  per-model merge, and the top-k after fusion. Compare with max_det = 902. Count the images that hit the cap.
- **Rule:**
  - If holdout40-clean mAP50 improves by **less than 0.005**, keep 902 and note it as a limitation.
  - Otherwise adopt 3000 as a bug fix, score val once more, update the release notes and README, and report both val
    numbers with the reason.
- The final system's checkpoints (B1h ep40, E4 ep40, E7 ep20) are unchanged.
