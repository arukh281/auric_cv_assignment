# Technical report: overhead truck detection (draft)

**AI assistance.** This project was built with AI assistance: Claude Code for code, analysis and drafting, and a
chat assistant for experiment planning. Hypotheses, predictions and decision rules marked "author's" in
DETAILED_EXPERIMENTS.md were chosen and approved by the author before each run; other interpretations were drafted by
Claude Code and reviewed by the author. The brief allows this: "External resources and AI-assisted code are
permitted, but you remain responsible for understanding, documenting and defending the submitted system and
conclusions."

## Pretrained weights and val leakage

All models start from **COCO-pretrained** Ultralytics YOLO11s (`yolo11s.pt`, Ultralytics release asset). The
SHA-256 of the file actually loaded is recorded in each run's `init_weights.json`.

**xView-pretrained checkpoints are deliberately not used.** Our five classes (Cargo Truck, Truck w/Box,
Truck w/Flatbed, Truck Tractor, Truck w/Liquid) are xView classes, and the imagery looks like xView. A checkpoint
trained on xView may have seen our validation images or their source scenes during pretraining. That would
inflate validation mAP in a way we cannot measure or remove.

## Reproducibility details (per run)

| Item | Where it is recorded |
|---|---|
| Architecture, initialization | `configs/*.yaml` (`model`), `runs/<run>/init_weights.json` |
| Image size / preprocessing, tiling | `configs/*.yaml`, `runs/<run>/tiling_params.json` |
| Train/val usage | Train split for training only. **Corrected 2026-10-04:** an earlier version said `val: false`; the configs actually set `val: true` (`configs/*.yaml`). Ultralytics then computes its own per-epoch val metrics on un-sliced full images, used for training curves only; `last.pt` is always reported (`results/*/eval*/metrics.json`, field `weights`). One leak: Ultralytics EarlyStopping (default `patience=100`, unset in configs before E3/E4) watches that val fitness and stopped E1 at epoch 145 of 150 (DETAILED_EXPERIMENTS.md, "CHECK 0"). From E3/E4 on, configs set `patience: 0` and checkpoint curves are scored on holdout40 |
| Augmentations, optimizer, scheduler, batch, epochs, losses | `runs/<run>/train/args.yaml` (full Ultralytics args, defaults included) |
| Seed / determinism | `seed: 0`, `deterministic: true` in config and `args.yaml` |
| Inference thresholds / post-processing | `runs/<run>/eval/eval_args.json` (conf 0.001, NMS IoU 0.7, max_det, tile merge) |
| Software | `requirements.txt`, `runs/<run>/env/*_pip_freeze.txt` |
| Hardware / runtime | `runs/<run>/env/*_hardware.json` (GPU, CUDA, cuDNN, torch, git commit) |
| Exact commands | `runs/<run>/command.txt` (every invocation, including resumes) |

## Metric definition

- **mAP50** is the mean over the 5 classes of AP at IoU 0.5, with COCO 101-point interpolation. It is computed by
  `detlib/scoring.py` on full-resolution val images, identically for every model.
- The Ultralytics-style interpolation is also reported (`AP50_ultralytics`). It is higher whenever recall saturates
  below 1, and it caps a perfect AP at 0.995.
- 95% CIs come from a 1000-sample bootstrap over the 22 val images (seed 0).

## Error taxonomy (Phase 3, `analysis/errors.py`)

Error types follow TIDE (Bolya et al., "TIDE: A General Toolbox for Identifying Object Detection Errors",
ECCV 2020). The order and boundaries were checked against the reference implementation, `tidecv/quantify.py`,
`TIDERun._eval_image` (github.com/dbolya/tide, commit `49a5d2a`), and match it. A first draft used a different
order (Cls, Dupe, Loc, Both, Bkg); it was changed to the reference before any run was analysed.

Every prediction that is not a true positive at IoU 0.5 gets the first type that applies (fg = 0.5, bg = 0.1):

1. **Loc**: bg <= best same-class IoU <= fg.
2. **Cls**: best other-class IoU >= fg.
3. **Dupe**: IoU >= fg with a same-class GT already matched by a higher-confidence prediction.
4. **Bkg**: best IoU with any GT <= bg.
5. **Both**: everything else (TIDE's `OtherError`, short name "Both").

**Missed** is a GT that no true positive matched and that no Cls or Loc error can claim.

Oracle fixes follow the reference `fix()` methods. Cls relabels the prediction and Loc snaps its box to the GT; either
is dropped instead if that GT is already matched. Dupe, Bkg and Both are removed. Missed removes the GT. Each fix is
re-scored with `detlib/scoring.py`. Matching uses our scorer, not TIDE's COCO-API matcher.

Rates and confusion matrices are reported at two threshold sets, tagged in every table:

- A fixed confidence of 0.25.
- The per-class F1-optimal confidence. It is chosen on val, which is the only test set, so it is an optimistic
  diagnostic, not a deployable setting.

---

## Summary (2026-10-05)

**The target was missed by a wide margin.** Target: ≥ 0.75 mAP50 on the 22-image val set.
- **Final system:** a multi-label WBF ensemble of B1h ep40 + E4 ep40 + E7 ep20, selected on holdout40-clean (E13).
  Val mAP50 **0.1349**, scored once (`results/e13_final/e13_final.json`; reproduced in a clean environment,
  `results/clean_repro_final/`).
- **Single-model baseline B1h:** 0.1065 (95% CI 0.0563–0.1653). Its seed-1 repeat scored 0.0634.

Why, in order of evidence strength:
1. **The model does not generalise from 403 images.** The same B1h weights score 0.378 on 40 of their own training
   images, 0.151 on 40 held-out train images and 0.107 on val
   (`results/b1h_tile1024_holdout40/{eval_train40,eval_holdout40,eval}/metrics.json`). Training 3× longer (E1, E2)
   raised train40 to 0.774 / 0.906 but *lowered* holdout40 to 0.092 / 0.111 (`results/e{1,2}_*/eval*/metrics.json`).
2. **Classification is the ceiling, not localisation.** Ignoring class, B1h's val AP50 is 0.255 vs 0.107 class-aware
   (`results/b1h_tile1024_holdout40/eval/class_agnostic.json`). Placing a box perfectly on every GT, B1h names the
   right type 60% of the time vs 52% for always answering "Cargo Truck" (§5.1). Oracle-fixing classification errors
   adds +0.147 mAP50, localisation errors +0.022 (`figures/b1h_tile1024_holdout40/errors/tide_dAP.csv`).
3. **More data helps, but not enough.** The learning curve is still rising, but the (unreliable) power-law
   extrapolation to 903 images gives 0.215 holdout mAP50 (CI 0.076–0.276). The 500 extra *instances* the brief asks
   about (~29 images) project to about +0.006, within seed noise (§5.3).
4. **Other candidate causes, as since tested.** "Ruled out" means a discriminating test was run; "no evidence for" means
   a signal was looked for and not found.
   - **Domain shift / altered val (tested: real, small cost).**
     - The xView audit (§3.3c) found 8 of 22 val images altered: 4 rescaled 2× / 0.5×, plus noise, blur + noise and
       contrast changes. All training images are untouched.
     - B1h finds 0.665 of val trucks vs 0.852 of holdout40 trucks. Box size explains 3% of that gap; brightness and
       contrast about 31% each; blur and noise little (§3.4, `results/val_review/a_standardised.csv`).
     - Re-running the 4 rescaled images at native scale lifts val 0.1065 → 0.1122 (§6.5).
   - **Image quality (tested):** two of the altered images are degraded (noise; blur + noise), and two have changed
     contrast. Their effect is part of the scene-level gap above.
   - **Label completeness (tested: real, small cost).**
     - 216 of 350 confident holdout40 false positives are excluded truck types, and val's labels drop 10.3% of
       xView's boxes of the five types.
     - Ignoring both in diagnostic re-scoring lifts val to 0.1237 and 0.1112 respectively (§6.5).
     - Tiling preserves the labels (0 of 3439 tiles differ).
   - **Model capacity:** YOLO11s fits its training data (16-tile overfit AP50 1.000; E1/E2 train40 0.774 / 0.906).
     A larger model was not tested (E9 was cancelled for budget).
   - **Tile-merge settings: ruled out.** Alternative merges change B1h val mAP50 within 0.1047–0.1077 and holdout40
     by −0.005 to +0.001 (`figures/b1h_tile1024_holdout40/merge_sensitivity.csv`, `results/val_review/r2_remerge.csv`).
5. **Audit and gap breakdown.** All data corrections together lift B1h's val mAP50 only from 0.1065 to 0.1371 in
   diagnostic re-scoring (§6.5). The data issues are real, but they are not what keeps the score far from 0.75.
6. **What helped:** multi-label output and a 3-model ensemble (E13). Val 0.1065 → 0.1349; +0.032 on holdout40-clean.

## The four questions

**(1) Is the training the problem? Mostly no.**
- Training longer (E1, E2) or starting from aerial weights (E3) made the model fit its training images better and do
  worse on new ones: holdout40 0.092 / 0.111 / 0.082 vs B1h's 0.151.
- More augmentation (E4) and a frozen aerial backbone (E7) overfitted less, with **no detectable effect** on
  holdout40 vs B1h (0.130 / 0.128 vs 0.151; without Liquid 0.163 / 0.153 vs 0.170; seed spread 0.017).
- B1h's own holdout40 curve is flat from epoch 40 to 50.

**(2) Are the rare classes the problem? They are weak, but not because they are rare.**
- Showing rare-class tiles more often (E6b) had **no detectable effect** overall (0.136 vs 0.151) and did not
  raise Tractor or Flatbed AP (0.007 → 0.002, 0.069 → 0.061).
- Neither did doubling Tractor boxes in the §5.4 subsets.
- 500 targeted instances project below noise (§5.3).
- About 5% of training boxes carry a different class than xView. Training on xView's original labels instead (E15) did
  **not** help: holdout40-clean 0.153 vs 0.176, supplied 0.137 vs 0.151, val (once) 0.091 vs 0.107; single seed.
  Those label differences are not a main limiter.

**(3) Can smarter inference help? Modestly, yes.**
- Multi-label output plus a 3-model ensemble (B1h + E4 + E7, E13) is the only change that cleared the noise bar on
  holdout40-clean: +0.032 (0.2086 vs 0.1765).
- On val, scored once, it gives 0.107 → 0.135, which is within val's seed spread (0.043).
- TTA (0.138 vs 0.151), the crop classifier (0.100–0.117; 0.610 vs the head's 0.690 on the same boxes) and
  scale-adaptive inference (E12: +0.0004 on the corruption suite; val 0.091) did not help.

**(4) Is the data the problem? Partly.**
- **The data issues are real:**
  - all images come from xView;
  - 8 of 22 val images are altered (4 rescaled, 4 photometric);
  - val drops 10.3% of xView's boxes;
  - 216 of 350 confident holdout40 false positives are excluded truck types.
- **But they cost little measured mAP50.** Correcting all of them diagnostically lifts B1h's val score only from 0.107
  to 0.137 (§6.5).
- **Fixing them in training did not raise mAP50:**
  - E10 halved val background errors (9380 → 4056) without an mAP gain, and was excluded because its extra images
    touch val and holdout40 scenes;
  - E15 scored lower on holdout40-clean.

## 2. Dataset and baselines

### 2.1 Dataset

Full-dataset statistics (`figures/eda/summary.json`, `figures/eda/tables/*.csv`):

| Class | train inst. | val inst. | train imgs | val imgs | train share | val share |
|---|---|---|---|---|---|---|
| Cargo Truck | 3773 | 800 | 368 | 19 | 0.495 | 0.516 |
| Truck w/Box | 2366 | 493 | 261 | 17 | 0.311 | 0.318 |
| Truck w/Flatbed | 662 | 122 | 195 | 14 | 0.087 | 0.079 |
| Truck Tractor | 625 | 117 | 128 | 11 | 0.082 | 0.075 |
| Truck w/Liquid | 192 | 20 | 99 | 8 | 0.025 | 0.013 |

(`figures/eda/tables/class_counts.csv`)

- 443 train images / 7618 boxes; 22 val images / 1552 boxes (`figures/eda/summary.json`).
- **Val is much denser:** boxes per image mean 17.2 (median 7) in train vs 70.5 (median 76.5) in val
  (`figures/eda/tables/boxes_per_image_stats.csv`).
- **Images are large, objects tiny:** median width × height 3228 × 2891 px (train), 3114 × 2911.5 px (val)
  (`figures/eda/tables/images.csv`); median sqrt(box area) 22.05 px train, 25.50 px val (`figures/eda/summary.json`);
  77.2% of train boxes (5881/7618) are COCO-small (`figures/eda/tables/size_bins.csv`). Largest train box side 161 px
  (`figures/eda/tables/boxes.csv`).
- **Imbalance:** Cargo + Box are 81% of train instances; Liquid has 20 val instances, so its AP has very wide CIs.
- **Label issues:** 1 empty train image (1938.png), 36 boxes with a side < 4 px (train 10, val 26)
  (`figures/eda/tables/label_issues_summary.csv`). No near-duplicates within or across splits at pHash ≤ 8
  (`figures/eda/summary.json`).
- **Visual observations** (`analysis/notes/visual_inspection.md`, 42 images of a local 20-train/22-val subset, all
  marked uncertain by the author): haze (val 2460, 2470, 2472), strong blur (val 2292, 2308, 2543), an apparently
  finer or upsampled scale (val 2391), water speckle (val 1399, 1447, 1456), visible-but-unlabelled trucks in both
  splits, and possibly offset GT boxes in 2470/2472. Val has more ports and dense truck yards than the train sample.
  Truck w/Liquid median sqrt(area) is 41.5 px in val vs 25.5 px in train (from 20 val boxes;
  `figures/domain_shift/tables/box_sizes_full_dataset.csv`).

### 2.2 Baselines: B0 → B1 → B1h

All runs: COCO-pretrained YOLO11s, SGD lr0 0.01, 50 epochs, seed 0, `last.pt`, sliced eval for tiled runs
(DETAILED_EXPERIMENTS.md, B0/B1/B1h entries).

| Run | Input | Train imgs | Iterations | eval max_det | val mAP50 (95% CI) | holdout40 mAP50 (95% CI) |
|---|---|---|---|---|---|---|
| B0 `b0_full640` | whole image → 640 | 443 | 1400 | 334 | 0.0020 (0.0009–0.0045) | – |
| B1 `b1_tile1024` | 1024 tiles, overlap 256 | 443 | 11850 | 334 | 0.0715 (0.0419–0.1242) | – |
| B1h `b1h_tile1024_holdout40` | as B1, 40 train imgs held out | 403 | 10750 | 902 | 0.1065 (0.0563–0.1653) | 0.1507 (0.0558–0.1874) |

Sources: `results/<run>/eval/per_class.csv`, `results/<run>/training_summary.json`,
`results/b1h_tile1024_holdout40/eval_holdout40/per_class.csv`.

Per-class val AP50 (95% CI), `results/<run>/eval/per_class.csv`:

| Class | B0 | B1 | B1h |
|---|---|---|---|
| Cargo Truck | 0.002 (0.001–0.005) | 0.089 (0.047–0.150) | 0.131 (0.063–0.203) |
| Truck w/Box | 0.008 (0.003–0.020) | 0.130 (0.035–0.226) | 0.180 (0.034–0.323) |
| Truck w/Flatbed | 0.000 (no preds) | 0.053 (0.012–0.099) | 0.071 (0.007–0.155) |
| Truck Tractor | 0.000 (no preds) | 0.006 (0.001–0.019) | 0.006 (0.000–0.022) |
| Truck w/Liquid | 0.000 (no preds) | 0.079 (0.000–0.329) | 0.145 (0.005–0.416) |

Caveats:
- B0 vs B1 is equal epochs, not equal compute (1400 vs 11850 iterations; DETAILED_EXPERIMENTS.md B0).
- B1 vs B1h differ in training set (443 vs 403), max_det (334 vs 902) and platform (Colab vs Kaggle;
  `results/<run>/env/train_hardware.json`). B1h's seed-1 repeat scored 0.0634 val / 0.1333 holdout
  (`results/b1h_seed1/eval/per_class.csv`, `eval_holdout40/per_class.csv`): a seed spread of 0.043 val and 0.017
  holdout, as large as the B1→B1h difference. B1h's "best" status is within noise.
- Scorer: ours equals pycocotools on every run (`results/<run>/eval/scorer_comparison.csv`).

### 2.3 Qualitative predictions

Grids of TP / FN / FP (background, localisation, wrong class) and full-image overviews:
`figures/b1_tile1024/grid_*.png`, `overview_{1211,1929,2308,2472}.png`;
`figures/b1h_tile1024_holdout40/grid_*.png`, `overview_{1457,1929,20,2470}.png`; error crops in
`figures/<run>/errors/crops_*.png`. Observations recorded from B1's grids (`analysis/notes/visual_inspection.md`):
- Missed trucks include clearly visible white trailers in 2391 and visible tankers in 1362, as well as blurred (2543),
  hazy (2460, 2472) and very dark (1399) cases.
- Background FPs cluster in 1206, 1211, 2470, 2472, 2293 and 31; some of them look like unlabelled trucks (uncertain).
- Class confusions are mostly Box ↔ Cargo, Tractor ↔ Box and Flatbed ↔ Cargo.
- No duplicate errors at conf 0.25 (`crops_dupe.png` empty).

### 2.4 Inference settings and the operating point
- All reported mAP50 values use conf ≥ 0.001.
- **mAP50 integrates over all confidence thresholds, so raising the threshold can only lower it.**
  - On val: 0.106 at 0.001 → 0.082 at 0.1 → 0.062 at 0.25 → 0.004 at 0.8.
  - On holdout40: 0.151 → 0.003 (`results/operating_points/`).
- For a deployed detector, the F1-optimal threshold is **0.10 on val** (P 0.254, R 0.265, F1 0.259) and **0.25 on
  holdout40** (P 0.346, R 0.332, F1 0.339). Figures: `figures/operating_points/`.
- **Precision is understated:** detections of excluded truck types (§3.3b) and of the 178 xView boxes missing from
  val's labels count as false positives.
- **Final system on val** (`results/clean_repro_final/operating_points_final_val.csv`):
  - mAP50 0.135 at conf 0.001.
  - F1-optimal threshold **0.10** (P 0.263, R 0.298, F1 0.280).
  - mAP50 falls to 0 by conf 0.7: fused scores are averaged over models, so they are lower than single-model scores.

## 3. Failure diagnosis

### 3.1 Error bins (TIDE order, val)

Gain in mAP50 if each error type were fixed by the oracle (`figures/<run>/errors/tide_dAP.csv`):

| Fix | B0 n / dAP50 | B1 n / dAP50 | B1h n / dAP50 |
|---|---|---|---|
| Cls | 100 / +0.102 | 621 / **+0.190** | 723 / **+0.147** |
| Loc | 248 / +0.004 | 190 / +0.014 | 386 / +0.022 |
| Both | 324 / +0.000 | 247 / +0.002 | 422 / +0.003 |
| Dupe | 20 / +0.000 | 0 / 0 | 0 / 0 |
| Bkg | 6287 / +0.007 | 4701 / +0.053 | 9380 / +0.077 |
| Missed | 1205 / +0.005 | 584 / +0.041 | 419 / +0.044 |

For the tiled runs, classification errors are the largest single loss, followed by background FPs and misses.
Localisation is small.

### 3.2 Class-agnostic vs class-aware

| B1h | mAP50 (class-aware) | class-agnostic AP50 | trucks found (any class, conf ≥ 0.001) |
|---|---|---|---|
| val | 0.1065 | 0.2551 | 1032/1552 = 0.665 |
| holdout40 | 0.1507 | 0.3617 | 629/738 = 0.852 |

Sources: `results/b1h_tile1024_holdout40/eval/class_agnostic.json`, `eval_holdout40/class_agnostic.json` (via
DETAILED_EXPERIMENTS.md B1h). Ignoring the class roughly doubles AP. At conf 0.25 only 361/1552 (0.233) val GT are matched by
any class (`results/b1h_tile1024_holdout40/eval/class_agnostic.json`).

Class confusion at GT locations (B1h, pooled anchors; rows = GT, `figures/b1h_tile1024_holdout40/gt_oracle/confusion_pool.csv`):
Cargo → Box 192 of 800; Box → Cargo 171 of 493; Tractor → Cargo 47 of 117; Liquid predicted correctly 2 of 20.
**Classification, in particular Cargo vs Box, is the main bottleneck** (also §5.1).

### 3.3 Size, density, image quality

- **Size:** B1 misses 71–79% of val trucks in every 8–48 px bin (1434 of 1552 GT) at F1-optimal thresholds, vs 52%
  at 48–96 px (n = 73) (`figures/b1_tile1024/errors/slices.csv`). B1h at conf 0.25: miss rate 0.94 / 0.90 / 0.86 /
  0.76 / 0.56 for 8–16 / 16–24 / 24–32 / 32–48 / 48–96 px (`figures/b1h_tile1024_holdout40/errors/slices.csv`).
  Size matters at the extremes but most trucks are missed regardless of size.
- **Density:** B1h conf-0.25 miss rate 0.88 / 0.87 / 0.90 / 0.81 across per-image object-count bins
  (`figures/b1h_tile1024_holdout40/errors/slices.csv`); no clear trend.
- **Image quality: no evidence for an effect, from a weak test.**
  - B1's 10 visually unflagged val images score 0.079 (0.037–0.179); the 12 flagged ones (haze, blur, dark, speckle)
    score 0.063 (0.036–0.090). The CIs overlap (`figures/b1_tile1024/per_image/subset_map_val.csv`).
  - The flags are subjective, and 10 vs 12 images could only detect a large effect.
  - Not tested as a cause of the val/holdout40 gap (§3.4).
- **Domain shift (train vs val): no evidence for it from this test, which is weak.**
  - Domain-classifier AUCs: 0.336 (0.168–0.515) on image stats, 0.453 (0.334–0.564) on crop embeddings and
    0.368 (0.209–0.549) per image. Every CI includes 0.5 (`figures/domain_shift/tables/domain_auc.csv`).
  - It was computed on the local 20-train / 22-val subset only.
  - The recall-gap analysis (§3.4) later found a scene-level gap between val and holdout40 that this classifier did
    not detect, so domain shift is **not** ruled out.
- **Pipeline checks:** in-sample train40 mAP50 0.4027 for B1 (`results/b1_tile1024/eval_train40/metrics.json`);
  disabling tile merge drops B1 0.0715 → 0.0484 (`figures/b1_tile1024/merge_sensitivity.csv`); max_det 3000 adds
  0.006 (`figures/b1_tile1024_maxdet3000/merge_sensitivity.csv`).

### 3.3b Are the background false positives really false?
- **Audit:** of the 60 most confident holdout40 predictions that match no label (IoU < 0.1; conf 0.53–0.80),
  Claude Code judged by eye:

  | verdict | count |
  |---|---|
  | truck-like and unlabelled | 46 |
  | other vehicle | 3 |
  | background | 4 |
  | unclear | 7 |

  (`figures/b1h_tile1024_holdout40/fp_audit_sheet.png`, `results/fp_audit/fp_audit.csv`; DETAILED_EXPERIMENTS.md
  "Background false-positive audit".)
- **Two explanations the crops cannot separate:**
  - (a) vehicles of our five classes missing from the labels;
  - (b) truck types the dataset excludes. xView, whose classes ours match, also has Pickup Truck, Utility Truck,
    Trailer and a generic Truck class.
- Either way the metric penalises detections of real vehicles, so **measured mAP50 is a lower bound** on how well
  the model finds trucks.
- **Limits:**
  - One viewer, not blind to the hypothesis.
  - Low-resolution crops.
  - Only the top 60 of 11684 unmatched predictions.
  - The effect on mAP50 is not quantified.

### 3.3c Dataset construction audit (comparison with the public xView originals)

Source: DETAILED_EXPERIMENTS.md "Data-integrity audit"; `results/xview_pixels/`, `results/xview_overlap/`.

- **Provenance:** all 465 images are xView training images (stem and size; `match_summary.json`).
- **Pixels:**
  - All 403 train and 40 holdout40 images are byte-identical to xView.
  - **8 of 22 val images are altered:**

    | change | images |
    |---|---|
    | upscaled 2× | 2308, 2391 |
    | downscaled 0.5× | 2384, 2460 |
    | noise added | 2292 (noise ratio 2.73) |
    | blur plus noise | 2543 (Laplacian ratio 0.66, noise ratio 1.72) |
    | contrast halved | 1399 (contrast ratio 0.50) |
    | contrast boosted | 2139 (ratio 1.41) |

  - 2470 and 2472 are byte-identical; they are dark in the original imagery.
  - Source: `pixels.csv`.
- **Labels, one-to-one against xView's five types** (`label_compare_by_split.csv`, `label_changes_by_split.csv`):

  | split | class changes vs xView | xView boxes missing from our labels | other |
  |---|---|---|---|
  | train | 341 / 6851 paired (5.0%), mostly xView Cargo/Box → our Flatbed, Tractor or Liquid | 384 | 378 shifted (IoU < 0.95) |
  | holdout40 | 38 (5.2%) | 53 | |
  | val | **0** | **178 (10.3%)** | |

- **Link to E6b:** the rare classes' training labels are partly flipped Cargo/Box boxes. That is a plausible reason
  repeating them did not help.
- **Disclosure:** The val alterations and label differences were found by comparing against the public xView originals. Val labels and pixels were never used for training or for selecting anything, and E12's rules are chosen on synthetically degraded holdout40 only.

### 3.4 Generalisation and sanity checks

- **Seen vs unseen (B1h):** train40 0.378, holdout40 0.151, val 0.107
  (`results/b1h_tile1024_holdout40/{eval_train40,eval_holdout40,eval}/metrics.json`). Holdout and val mAP50
  CIs overlap. The main gap is between seen and unseen images; on top of it, val is harder than holdout40
  (*corrected 2026-10-04: previously "not a val-specific problem"*). Why val has fewer trucks found (66% vs 85%):
  - Re-weighting holdout40 to val's distribution of box size explains 3% of the gap, and image size 7%.
  - Val recall is lower in every size bin from 8 to 48 px (for example, 24–32 px: 0.73 vs 0.94).
  - In the densest bin (≥ 100 boxes per image) val is still 0.20 lower. Density cannot be compared at 60–100 boxes,
    where holdout has no images.
  - The gap is therefore a difference between the scenes themselves, not size, density or resolution, and these
    analyses cannot name it. Val has only 22 images.
  - Sources: `results/recall_gap/standardised.csv`, `recall_by_factor.csv`; DETAILED_EXPERIMENTS.md "Val vs holdout40 recall
    gap".
- **Overfit test** (16 tiles, all 5 classes, augmentation off): AP50 0.355 / 0.987 / 1.000 at epochs 50 / 100 / 300;
  final cls_loss 0.097 (`results/sanity/overfit/`). The model and pipeline can fit these labels.
- **Label check:** the first check flagged 85 of 3439 tiles (`results/sanity/label_check.json`). With the checker
  fixed to pair boxes by IoU, the re-run flags 0 of 3439 (`results/sanity/label_check_iou/label_check.json`). The IoU-paired
  re-check found all 3,278 boxes in those tiles identical (`results/sanity/label_mismatch/summary.json`): an artefact
  of coordinate-sort pairing in dense tiles, not a tiling error (annotation correctness was not tested here; see the FP audit). The author overrode the original NOT PASS before
  the re-check; with it, both PASS criteria hold (DETAILED_EXPERIMENTS.md, SANITY).
- Effective B1h settings: imgsz 1024, mosaic 1.0, scale 0.5, close_mosaic 10
  (`results/b1h_tile1024_holdout40/train/args.yaml`).

## 4. Experiment records

Full entries are in DETAILED_EXPERIMENTS.md under the headings named below. "Pre-registered" means written before results.

| Exp. | Observation | Hypothesis | Changes | Results | Conclusion | Next |
|---|---|---|---|---|---|---|
| **B0** (EXPERIMENTS "B0") | ~3200 px images, ~22 px trucks | Not recorded before the run; reconstructed afterwards: 640 letterbox makes trucks ~4–5 px, so B0 is a floor | whole image @640, 50 ep | val 0.0020 (0.0009–0.0045) | Almost no detection; tiling needed | B1 |
| **B1** ("B1") | as B0; max box side 161 px < overlap 256 | Pre-registered: 0.40–0.60 mAP50; Cls dominant; Liquid worst | 1024 tiles, overlap 256, sliced eval | val 0.0715 (0.0419–0.1242) | Far below prediction; Cls dominant (correct); rejection condition (b) largely met (size not the main limiter) | test generalisation |
| **B1h** ("B1h") | is val unusually hard? | No prediction written before this run | 40 train imgs held out; max_det 902 | val 0.1065, holdout 0.1507, train40 0.378 | Poor generalisation, not val-specific; B1→B1h gain within seed noise | learning curves |
| **LC** ("LC") | train/holdout gap | Pre-registered: curve still rising; 25% → 0.07–0.12 holdout | 25/50/75% subsets at equal iterations + seed 1 | §5.3 | Rising; 25% gave 0.061 | §5.4 |
| **S54** ("S54") | – | Selection and rule pre-registered (`b02413c`); prediction never provided | smart vs random subsets | §5.4 | rule computed; no subset ≥ 90% | – |
| **SANITY** ("SANITY") | train40 only 0.378 | settings / labels / capacity | args check, label check, overfit test | §3.4 | pipeline can fit; tiling preserves labels (re-check; annotation correctness not tested) | E1/E2 |
| **E1/E2** ("E1 / E2") | B1h losses still falling | Pre-registered: E1 undertraining; E2 scale 0.5 hurts small trucks | E1 150 ep; E2 150 ep + scale 0.2 | below | E1 not supported (overfitting); E2 inconclusive, leaning not supported | E3/E4 |
| **E3/E4** ("E3 / E4") | E1/E2 overfit | Pre-registered (author's): aerial pretraining (E3) / flipud 0.5 + mixup 0.1 (E4) reduce overfitting; holdout > 0.1507 + 0.017 | B1h recipe, 50 ep, `patience: 0`, holdout checkpoint curves | E3 0.082, E4 0.130 holdout40 | E3 rejected (reversed); E4 partly supported (gap shrank, no detectable mAP effect) | E7, E8 |

E1/E2 results (DETAILED_EXPERIMENTS.md "E1 / E2: Results"; `results/<run>/eval/per_class.csv`, `eval_holdout40/per_class.csv`,
`eval_train40/metrics.json`):

| | B1h (50 ep) | E1 (150 ep, early-stopped at 145) | E2 (150 ep, scale 0.2) |
|---|---|---|---|
| val mAP50 (95% CI) | 0.1065 (0.0563–0.1653) | 0.0680 (0.0355–0.1243) | 0.0620 (0.0326–0.1104) |
| holdout40 mAP50 (95% CI) | 0.1507 (0.0558–0.1874) | 0.0923 (0.0302–0.1384) | 0.1112 (0.0344–0.1391) |
| train40 mAP50 | 0.378 | 0.774 | 0.906 |
| final train cls_loss | 1.596 | 0.795 | 0.543 |

More training fits the training images far better while held-out mAP50 falls: overfitting, not undertraining.
Val checkpoint curves peak mid-training (E1 0.100 at epoch 90, E2 0.107 at 40–50;
`results/<run>/checkpoint_curve/checkpoint_curve.csv`); they were reported as curves only, never used to pick weights.
E1's early stop was chosen by Ultralytics val (see Reproducibility, corrected row).

### 4.1 Every model: val mAP50 and per-class AP50 (brief §2.3)

Sources: each row's `results/<run>/eval/per_class.csv` (final system: `results/clean_repro_final/per_class.csv`).

| model | val mAP50 | Cargo | Box | Flatbed | Tractor | Liquid | note |
|---|---|---|---|---|---|---|---|
| B0 | 0.0020 | 0.002 | 0.008 | 0.000 | 0.000 | 0.000 |  |
| B1 | 0.0715 | 0.089 | 0.130 | 0.053 | 0.006 | 0.079 |  |
| B1h | 0.1065 | 0.131 | 0.180 | 0.071 | 0.006 | 0.145 |  |
| B1h seed 1 | 0.0634 | 0.102 | 0.129 | 0.053 | 0.029 | 0.004 |  |
| b1h_f25 | 0.0169 | 0.035 | 0.028 | 0.022 | 0.000 | 0.000 |  |
| b1h_f50 | 0.0492 | 0.045 | 0.048 | 0.020 | 0.000 | 0.132 |  |
| b1h_f50 seed 1 | 0.0573 | 0.054 | 0.034 | 0.015 | 0.001 | 0.183 |  |
| b1h_f75 | 0.0846 | 0.119 | 0.188 | 0.040 | 0.007 | 0.069 |  |
| b1h_f75 seed 1 | 0.0736 | 0.095 | 0.166 | 0.035 | 0.003 | 0.070 |  |
| smart50 | 0.0562 | 0.081 | 0.132 | 0.032 | 0.004 | 0.032 |  |
| smart75 | 0.0832 | 0.090 | 0.153 | 0.051 | 0.003 | 0.119 |  |
| E1 | 0.0680 | 0.085 | 0.177 | 0.024 | 0.001 | 0.053 |  |
| E2 | 0.0620 | 0.062 | 0.115 | 0.037 | 0.018 | 0.078 |  |
| E3 | 0.0881 | 0.122 | 0.207 | 0.053 | 0.004 | 0.054 |  |
| E4 | 0.0761 | 0.112 | 0.110 | 0.120 | 0.039 | 0.000 |  |
| E6b | 0.0908 | 0.114 | 0.146 | 0.047 | 0.001 | 0.146 |  |
| E7 | 0.0715 | 0.090 | 0.165 | 0.070 | 0.014 | 0.018 |  |
| E10 | 0.0820 | 0.087 | 0.124 | 0.113 | 0.014 | 0.073 | caveat: 8 of its extra training images touch val images geographically |
| E15 | 0.0908 | 0.104 | 0.125 | 0.088 | 0.028 | 0.109 |  |
| **Final system (E13)** | 0.1349 | 0.171 | 0.215 | 0.096 | 0.028 | 0.165 | multi-label ensemble; val scored once |
| E6 | — | | | | | | not evaluated: superseded by E6b (E6 repeated only Liquid tiles, +1.6% views); its CPU evaluation failed twice and was not retried |
| E8 | — | | | | | | stopped at epoch 68 for GPU budget; never evaluated |
| E9, E11 | — | | | | | | cancelled for budget, never trained |
| E12 (auto rule on B1h) | 0.0906 | | | | | | inference rule, not adopted (see E12 entry); per-class not computed |
| TTA, crop classifier | — | | | | | | holdout40 only by their pre-registered rules (val only if they passed; neither did) |

Other dataset facts recorded with these runs:
- 44 train–val and 56 train–holdout40 image pairs touch geographically (adjacent chips;
  each overlap ≤ 3.2% of an image's area). This is a property of the supplied split.
- The E10 label-conflict check found 10 trucks labelled with both a 5-class box and an excluded-type box, vs 9 inside
  xView's own labels. That is negligible.

## 5. Research questions

### 5.1 If locations were perfect

Method (`analysis/gt_box_oracle.py`): at every val GT box, take the model's own class scores at the matching anchor
and check whether the top class is right. This removes detection and measures classification alone.

| Model | accuracy (1552 GT) | mean per-class accuracy | GT with an anchor at IoU ≥ 0.5 |
|---|---|---|---|
| B0 | 0.481 | 0.215 | 396 |
| B1 | 0.553 | 0.387 | 1449 |
| B1h | 0.601 | 0.436 | 1455 |

Sources: `figures/{b0_full640,b1_tile1024,b1h_tile1024_holdout40}/gt_oracle/comparison.csv`, `summary.json`.
Baselines: always "Cargo" = 0.516 (800/1552, `figures/eda/tables/class_counts.csv`); uniform guessing = 0.20 per class.

With perfect locations B1h names the type correctly 60% of the time, barely above always answering Cargo; per class it
is 44% vs 20% chance. The TIDE oracle agrees: fixing Cls adds +0.147, Loc +0.022, Missed +0.044
(`figures/b1h_tile1024_holdout40/errors/tide_dAP.csv`). **Classification is the ceiling.** Fixing only localisation errors gives
0.129 mAP50; fixing only classification errors gives 0.254; both are far from 0.75 (same file).

**What this experiment cannot establish.**
- **It does not capture the interaction between detection and classification.** The GT-box oracle reads the raw head's
  class scores, max-pooled over every anchor whose decoded box has IoU ≥ 0.5 with the GT (`pool` mode; the reported
  0.601). The detector's own box choice, NMS and tile merging therefore play no part. A
  detector whose boxes were perfect might still score classes differently, for example after a box survives NMS
  against a neighbour of another class. The TIDE oracle fixes errors one type at a time on real predictions, so it
  also cannot model fixing two error types jointly.
- **It measures this model's classifier, not the best achievable one.** 60% is what B1h's head does with perfect
  locations. It says nothing about what a dedicated classifier, more context or higher resolution could achieve. The
  crop-classifier experiment tested that: 0.610 on holdout40 GT crops vs the head's 0.690 on the same boxes.
- **It cannot separate ambiguity in the data from weakness of the model.** Some Cargo vs Box confusions may be
  genuinely ambiguous from above, or inconsistently labelled. The oracle cannot tell these apart from model errors.
- **Small sample.** It is measured on the 22 val images (1552 boxes, but only 20 Liquid and 117 Tractor). Accuracy is
  pooled over boxes and carries no CI, and per-class accuracy for the rare classes is unstable.
- **Matching rule.** "Matching anchor" means IoU ≥ 0.5 with the anchor's decoded box. 97 of 1552 val GT boxes (6%)
  had no such anchor for B1h (`comparison.csv`, "flagged") and are scored from the best available anchor.

### 5.2 Resisting examples

**Method** (`analysis/training_dynamics.py`):
- Every GT box of the 403 B1h training images is tracked across B1h's checkpoints (epochs 10–50).
- A box counts as detected when some prediction overlaps it at IoU ≥ 0.5 with conf ≥ 0.25, and as correct when the
  top such prediction has its class.
- Categories: learned-early, learned-late, forgotten, never-learned (detected, never correct) and never-detected.
- The images were split in half beforehand (`b02413c`): an inspect half for exploration and a test half kept unread.
  Five claims (C1–C5; C2 in two parts) from the inspect half were pre-registered (`95953f3`) and then tested on the test half.

**Test-half categories** (3369 boxes): learned-late 1356, never-detected 1270, forgotten 319, never-learned 280,
learned-early 144 (`results/s52_confirm/test/summary.json`). About 38% of training boxes are never detected even
though the model trains on them.

**Confirmed on the test half.**
- Judged only against the claims committed in `95953f3`, before the test half was read.
- That commit has five claims, C1–C5. C2 has two parts, C2a and C2b, each with its own criterion, so there are six
  rows. All six pass, meaning all five claims hold (`results/s52_confirm/test/claims.csv`; image-bootstrap 95% CIs).

| claim | finding | criterion | test half |
|---|---|---|---|
| C1 | Small boxes resist: never-detected share for boxes < 16 px minus for ≥ 32 px | > 0 | +0.631 (0.559–0.699) |
| C2a | Truck w/Box is learned early more than any other class (margin) | > 0 | +0.118 (0.061–0.157) |
| C2b | ...and forgotten more than any other class (margin) | > 0 | +0.157 (0.090–0.209) |
| C3 | Truck w/Liquid is "found but never classified right" more than any other class (margin) | > 0 | +0.281 (0.164–0.375) |
| C4 | Most never-detected boxes are "never confident", not invisible: share with an IoU ≥ 0.5 prediction at conf ≥ 0.001 | > 0.5 | 0.778 (0.716–0.848) |
| C5 | Cargo Truck boxes often never detected | > 0.3 | 0.474 (0.427–0.532) |

**Descriptive only (computed after the test half was opened; not pre-registered).** The rows below give the
never-detected boxes with an epoch-50 prediction of any class at IoU ≥ 0.5, by confidence floor. The chance control
places one random box per GT box (same image and size, IoU 0 with every GT box; seed 0). Sources:
`analysis/s52_floor.py`, CPU-only kernel `auric-s52-floor`, `results/s52_floor/{inspect,test}/floor.csv`.

| conf floor | inspect half (1288 boxes) | test half (1270 boxes) | chance, test half (3369 boxes) |
|---|---|---|---|
| ≥ 0.001 | 0.793 | 0.778 | 0.0009 |
| ≥ 0.05 | 0.538 | 0.543 | 0.0009 |
| ≥ 0.10 | 0.396 | 0.400 | 0.0006 |

The inspect column reproduces the unsaved 3 Oct session numbers. About 40% of never-detected training boxes have a
correctly placed prediction at conf ≥ 0.10, against well under 1% for random boxes of the same size. That is far
above chance, and the effect is the same in both halves.

**What resists learning** (interpretation written after the result; no author prediction recorded):
1. **Small trucks.** Under 16 px they are mostly never detected at the operating threshold, even in training images.
2. **Low confidence rather than invisibility.** Most of those boxes get a prediction, but below 0.25. This fits
   §3/§5.1: the model localises far better than it classifies.
3. **Unstable class boundaries.** Box is learned first and then forgotten most often, consistent with the Cargo ↔ Box
   confusion in §3.2.
4. **Rare classes.** Liquid (192 train boxes) is found but rarely named correctly.

### 5.3 Value of 500 more labels

Learning curve at equal iterations (~10,750), held-out = holdout40, seed 0 unless noted
(`figures/learning_curve/learning_curve.csv`, `power_law_fit.csv`):

| Train images | holdout mAP50 | holdout class-agnostic AP50 | val mAP50 |
|---|---|---|---|
| 101 (25%) | 0.061 | 0.160 | 0.017 |
| 202 (50%) | 0.095 | 0.232 | 0.049 |
| 302 (75%) | 0.105 | 0.293 | 0.085 |
| 403 (100%), seed 0 / 1 | 0.151 / 0.133 | 0.362 / 0.362 | 0.106 / 0.063 |

- The curve is still rising at 403 images (pre-registered prediction: correct; 25% point 0.061 vs predicted
  0.07–0.12, CI 0.012–0.090 overlaps; DETAILED_EXPERIMENTS.md "LC").
- Power-law fit on holdout mAP50 (403 point = mean of seeds, 0.142) extrapolates to **0.215 at 903 images (95% CI
  0.076–0.276)**, a gain of +0.080 vs seed spread 0.017. **Every fit is flagged unreliable** (2.24× beyond the data;
  several per-class fits non-monotone or failed) (`figures/learning_curve/power_law_fit.csv`). Only the direction is
  trustworthy.
- Per class (holdout, seed 0, 101 → 403 images; `figures/learning_curve/learning_curve.csv` via HANDOFF.md §6):
  Box 0.236 → 0.509, Cargo 0.018 → 0.096, Flatbed 0.034 → 0.069, Liquid 0.015 → 0.073, Tractor 0.000 → 0.007.
  Box benefits most; Tractor barely moves.
- **Answer:** more labels should help, but the optimistic projection is far below 0.75. Classification needs its own
  remedy.

**Correction 2026-10-04: instances vs images.**
- The brief asks about "500 additional labelled **instances**"; the extrapolation above adds 500 **images**
  (403 → 903).
- B1h trains on 7618 − 738 = 6880 boxes (`figures/eda/summary.json`;
  `results/b1h_tile1024_holdout40/eval_holdout40/class_agnostic.json`, `n_gt`). That is 17.1 boxes per image, so 500
  instances is about 7% more data, roughly 29 images (403 → 432).
- With the same power-law fit (holdout mAP50 = a·n^b, a = 0.00424, b = 0.577; `power_law_fit.csv`), the prediction
  is 0.1351 at 403 images and 0.1351 × (432/403)^0.577 ≈ 0.1406 at 432. That is a gain of about **+0.006**, below
  the 0.017 seed spread.
- **Answer to the question as asked:** 500 more instances from the same source would **not** materially improve the
  system, and the effect would not be measurable against seed noise. The 903-image figure answers a different
  question (500 more images, about 8,500 boxes).

**Assumptions behind these estimates**
1. **The power law holds beyond the measured range.** It was fitted on four points (101–403 images), with the 403
   point averaged over two seeds. For 500 images it is extrapolated 2.24× beyond the data, and every fit is flagged
   unreliable (`power_law_fit.csv`). For 500 instances it is only 7% beyond, which is much safer, but the gain is
   then within noise.
2. **New labels come from the same distribution:** the same scenes, sizes, class mix and labelling process as the
   current training set, as the brief says. The recall-gap analysis (§3.4) shows val scenes differ from train. Labels
   that resembled val, such as dense yards and ports, could help more than the curve predicts. Labels that repeat
   existing scenes could help less.
3. **Equal training iterations.** Every learning-curve point used about 10,750 iterations, so a larger set gets fewer
   epochs per image. A real +500 run might be trained longer, although E1/E2 suggest more epochs hurt held-out
   mAP50.
4. **Per-class rankings rest on few held-out boxes.** Holdout40 has 321 Cargo, 297 Box, 56 Flatbed, 35 Tractor and 29
   Liquid boxes (`results/b1h_tile1024_holdout40/eval_holdout40/per_class.csv`). The Tractor and Liquid curves are
   near zero and noisy, and the Liquid power-law fit failed. "Box benefits most, Tractor least" is supported for Box;
   for Tractor it means only that no effect is visible.
5. **The class mix of new labels matches the current one** (50% Cargo, 31% Box). Targeted labels for rare or confused
   classes were not modelled.
6. **Sample count vs other limits.**
   - The curve can only show limits that shrink with more data.
   - The parts that do not shrink are invisible to it: classification ceiling (§5.1), val scene shift (§3.4) and
     overfitting (E1/E2).
   - Class-agnostic holdout AP rises 0.160 → 0.362 from 101 to 403 images, while Cargo AP stays below 0.10. This
     suggests that detection is more sample-limited, and Cargo vs Box classification is limited by something other
     than count.

### 5.4 Smallest subset retaining ≥ 90%

**The smallest successful subset we tested is the full training set (403 images); no tested smaller subset (202 or 302 images, smart or random) reached 90% on holdout40 or on val.**

Pre-registered in commit `b02413c` before any S54 run (DETAILED_EXPERIMENTS.md "S54"): smart = class coverage (rarest class
first) then greedy k-center on DINOv2-small embeddings; "recovers 90%" = held-out mAP50 ≥ 0.90 × 0.1420 = 0.128
(0.1420 = mean of the two full-data seeds). **The author's prediction was never provided** (an unfilled placeholder);
nothing was written before the runs. Implementation note: the coverage step adds the image with the most boxes of
the needed class (`tools/make_smart_subsets.py`), which differs from the pre-registered wording "in order of rarest
class contained".

Held-out results (`figures/subset_compare/subset_compare.csv`):

| Run | Size | Selection | seed | held-out mAP50 (95% CI) | held-out class-agn. AP50 | % of full |
|---|---|---|---|---|---|---|
| b1h_f50 | 202 | random | 0 | 0.095 (0.026–0.127) | 0.232 | 66.8 |
| b1h_f50_seed1 | 202 | random | 1 | 0.082 (0.028–0.109) | 0.221 | 57.5 |
| b1h_smart50 | 202 | smart | 0 | 0.105 (0.028–0.142) | 0.284 | 74.1 |
| b1h_f75 | 302 | random | 0 | 0.105 (0.028–0.144) | 0.293 | 73.6 |
| b1h_f75_seed1 | 302 | random | 1 | 0.123 (0.038–0.165) | 0.306 | 86.6 |
| b1h_smart75 | 302 | smart | 0 | 0.104 (0.033–0.132) | 0.334 | 73.3 |

Pre-registered rule (`figures/subset_compare/decision_rule.csv`): smart beats random on class-agnostic AP50 at both
sizes (+0.052 at 202, +0.028 at 302, vs seed spreads 0.011 / 0.013), but **not** on mAP50 (+0.010 at 202 vs spread
0.013; −0.019 at 302).

**Answer and conclusion (written 2026-10-04, after the results; no prediction was recorded before the runs).**
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

**What is in the selected subsets, and what makes them useful?** (added 2026-10-04, after the results;
`analysis/subset_characterise.py`, CPU-only kernel `auric-s54-char-merge`; `figures/subset_compare/subset_characterisation.csv`,
`subset_class_coverage.csv`. The random seed-1 runs used the same image lists as seed 0.)

| subset | images | boxes | boxes / image (mean, median) | images with ≥ 30 boxes | box size < 16 / 16–32 / ≥ 32 px |
|---|---|---|---|---|---|
| full pool | 403 | 6880 | 17.1, 7 | 14.4% | 23% / 54% / 23% |
| random 202 | 202 | 3182 | 15.8, 6.5 | 11.4% | 23% / 55% / 23% |
| smart 202 | 202 | **4211** | 20.8, 5 | 16.8% | 20% / 55% / 26% |
| random 302 | 302 | 4842 | 16.0, 6 | 12.9% | 23% / 53% / 25% |
| smart 302 | 302 | **6022** | 19.9, 7 | 18.2% | 22% / 53% / 24% |

Class coverage, as boxes (images) and share of the pool's boxes of that class:

| class | pool | random 202 | smart 202 | random 302 | smart 302 |
|---|---|---|---|---|---|
| Cargo Truck | 3452 (330) | 1758 (168) 51% | 1883 (165) 55% | 2599 (250) 75% | 2870 (250) 83% |
| Truck w/Box | 2069 (238) | 848 (116) 41% | 1354 (111) 65% | 1457 (177) 70% | 1888 (178) 91% |
| Truck w/Flatbed | 606 (178) | 285 (93) 47% | 421 (91) 69% | 371 (129) 61% | 550 (142) 91% |
| Truck Tractor | 590 (117) | 205 (54) 35% | **438 (63) 74%** | 290 (85) 49% | **559 (97) 95%** |
| Truck w/Liquid | 163 (90) | 86 (45) 53% | 115 (49) 71% | 125 (67) 77% | 155 (82) 95% |

- **Smart subsets carry more labelled boxes for the same number of images:** +32% at 202 images, +24% at 302. Of
  the 302-image smart subset, 95% of all Tractor and Liquid boxes in the pool are included. The coverage step picks
  the image with the most boxes of a needed class, so it favours dense, multi-class images: 16.8% vs 11.4% of
  images with ≥ 30 boxes at 202.
- **Box sizes are essentially unchanged.** Selection did not favour larger or smaller trucks.
- **Scene coverage: no difference measurable with this proxy.** Every subset, random included, covers all 20
  k-means clusters of the DINOv2 embeddings. The proxy is too coarse to separate them.
- **What makes them useful.** Smart subsets win on class-agnostic AP50 (the pre-registered primary metric), and the
  simplest explanation is that they contain more labelled trucks, especially rare-class trucks, per image.
- **This is a confound, not a demonstration that the selection is clever.** A random subset matched on box count
  rather than image count was not run, so "smart" cannot be separated from "more boxes". It also did not help
  class-aware mAP50: more Tractor boxes (438 vs 205) did not lift Tractor AP (0.001 vs 0.000 at 202).

**The 90% rule on validation mAP50, as the brief specifies** (added 2026-10-04, after all results).
- **Why two versions exist:**
  - The brief defines success as 90% of the full-data *validation* mAP50.
  - The rule above was pre-registered on holdout40 so that no decision would be made on val, which is the only test
    set.
  - Both are reported. The holdout40 version is the pre-registered one.
- **Threshold:** 0.9 × 0.1065 (B1h, seed 0) = **0.0959**. Val mAP50 with image-bootstrap 95% CIs
  (`figures/subset_compare/subset_compare.csv`):

  | Run | Size | Selection | seed | val mAP50 (95% CI) | % of 0.1065 | ≥ 0.0959? |
  |---|---|---|---|---|---|---|
  | b1h_f50 | 202 | random | 0 | 0.049 (0.022–0.108) | 46% | no |
  | b1h_f50_seed1 | 202 | random | 1 | 0.057 (0.023–0.111) | 54% | no |
  | b1h_smart50 | 202 | smart | 0 | 0.056 (0.030–0.102) | 53% | no |
  | b1h_f75 | 302 | random | 0 | 0.085 (0.048–0.145) | 79% | no |
  | b1h_f75_seed1 | 302 | random | 1 | 0.074 (0.046–0.135) | 69% | no |
  | b1h_smart75 | 302 | smart | 0 | 0.083 (0.041–0.145) | 78% | no |
  | b1h_tile1024_holdout40 | 403 | full | 0 | 0.106 (0.056–0.165) | 100% | reference |
  | b1h_seed1 | 403 | full | 1 | 0.063 (0.042–0.096) | 60% | **no** |

- **Result on val:** no subset reaches 0.0959, the same answer as on holdout40.
- **This is weak evidence.**
  - Val's seed spread at full data is 0.043 (0.1065 vs 0.0634), four times the 0.011 gap between 90% and 100%.
  - The full-data seed-1 model itself fails the rule (60%).
  - Every subset's CI contains 0.0959.
  - On val, the rule mostly measures which seed was drawn.

**Next.** Fractions above 75% (for example 85% and 95%, about 343 and 383 images) were not tested.
- The answer lies there, since 75% falls short (best 0.123) and 100% passes (0.151 / 0.133).
- Each size needs three runs (smart plus two random seeds).
- They were not run because E1/E2 showed the bottleneck is generalisation and classification, not the choice of
  training images (§3.4). The remaining GPU budget (20.44 h until 10 Oct, before E3/E4) went to experiments that
  could move mAP50 rather than refine a subset size.


## 6. Final analysis and next experiment

**Final system:** a multi-label WBF ensemble of B1h ep40 + E4 ep40 + E7 ep20 (E13). Val mAP50 **0.1349**, scored once.
- On holdout40-clean, the ensemble before checkpoint choice scores 0.2086, vs B1h's 0.1765.
- The checkpoints were chosen on holdout40, so holdout numbers after that choice are optimistic.
- The val gain over B1h (+0.028) is within val's seed spread (0.043).
- No single-model change beat B1h.

### 6.1 Dominant limitations, with evidence

**(1) Generalisation from 403 images.** Fitting faster hurts; regularising helps detection, not classification.

| run | change vs B1h | train40 | holdout40 | gap | holdout40 class-agnostic |
|---|---|---|---|---|---|
| B1h | none | 0.378 | **0.151** | 0.227 | 0.362 |
| E1 | 150 epochs | 0.774 | 0.092 | 0.682 | 0.259 |
| E2 | 150 epochs, scale 0.2 | 0.906 | 0.111 | 0.795 | 0.278 |
| E3 | DOTA init | 0.729 | 0.082 | 0.647 | 0.251 |
| E7 | DOTA init, frozen backbone | 0.599 | 0.128 | 0.471 | 0.308 |
| E4 | flipud + mixup | 0.266 | 0.130 | 0.135 | **0.400** |

Sources: `results/<run>/{eval_train40,eval_holdout40}/{metrics.json,class_agnostic.json}`.
holdout values are in DETAILED_EXPERIMENTS.md "E1 / E2: Results". Picture: `figures/story/scoreboard.png`.

- **Fitting faster hurts.** E1, E2 and E3 all raised train40 and lowered holdout40, by more than the seed noise each
  time.
- **Freezing helps relative to E3.** E7 recovered +0.046 of E3's loss; vs B1h there is no detectable effect
  (−0.023, without Liquid −0.017).
- **Augmentation helps detection, not naming.** E4 overfits least, and it is the only run that finds *more* held-out
  trucks than B1h (class-agnostic 0.400 vs 0.362). Its class-aware mAP50 differs by −0.020, which is within seed noise: **no detectable effect** (without Liquid
  0.163 vs 0.170).
- **B1h stops near the right point.** Its own holdout curve is flat from epoch 40 to 50 (0.154 → 0.151;
  `results/b1h_tile1024_holdout40/checkpoint_curve_holdout/`).
- Each run is a single seed.

**(2) Classification between look-alike types: weak on val, partly val-specific (plausible).**
*Corrected 2026-10-04: previously framed as a general ceiling.*

| at the true boxes (GT-box oracle, B1h head, pool) | val (1552 boxes) | holdout40 (738 boxes) |
|---|---|---|
| accuracy | 0.601 | 0.690 |
| always "Cargo" | 0.515 (800/1552) | 0.435 (321/738) |
| margin over always "Cargo" | +0.086 | +0.255 |

Sources: `figures/b1h_tile1024_holdout40/gt_oracle/comparison.csv`, `gt_oracle_holdout/comparison.csv`.
- **Naming is decent on held-out images that resemble the training set and much weaker on val.**
- That fits what was found about val:
  - 4 rescaled images (2× / 0.5×).
  - A scene-level recall gap (§3.4).
  - Class changes relative to xView: 4.3% of pairs, pooled over train and val.
- Graded **plausible** until the pixel comparison and the per-split class-change counts are in.
- **A dedicated crop classifier is worse than the detector head on the same 738 holdout40 boxes** (0.610 vs 0.690;
  `results/crop_classifier/summary.json`). Re-labelling detections with it lowers holdout40 mAP50 to 0.100 / 0.117.
- TTA did not help either (0.138 vs 0.151; without Liquid 0.167 vs 0.170).
- TIDE on holdout40 confirms that classification also dominates there: Cls +0.177, Bkg +0.130, Missed +0.014, Loc
  +0.015 (`figures/b1h_tile1024_holdout40/errors_holdout/tide_dAP.csv`).

**(3) Labels: the metric undercounts.**
- 46 of the 60 most confident unmatched holdout40 predictions look like real, unlabelled trucks (§3.3b).
- They are either our classes missing from the labels or excluded truck types (xView's pickups, utility trucks,
  trailers, generic trucks).
- Either way, measured mAP50 is a lower bound on detection quality, and the false-positive bin is partly an artefact
  of incomplete labels.
- Caveats: one viewer, not blind, low resolution, top 60 only.

**(4) Val scenes differ from training scenes.**
- B1h finds 0.665 of val trucks vs 0.852 of holdout40 trucks, a gap of 0.187 (CI 0.043–0.307).
- Box size explains 3% of the gap and image size 7%; density does not explain it where it can be compared (§3.4).
- The remaining difference is at scene level, so val is harder than our own held-out images.

**(5) More data alone will not close the gap.**
- 500 more instances in the current class mix project about +0.006 holdout mAP50.
- 500 targeted instances of any single class project at most +0.016 (Liquid, an unreliable 4.1× extrapolation;
  Box +0.011 among reliable fits). Both are below the 0.017 noise (§5.3).
- Independently, roughly doubling Tractor boxes in the §5.4 subsets did not raise Tractor AP.

Also confirmed: small trucks are mostly "never confident" rather than invisible (§5.2 C1, C4, pre-registered and
confirmed on the unseen half).

### 6.2 Strength of each conclusion

| Conclusion | Status | Evidence |
|---|---|---|
| The binding limit is generalisation; fitting faster hurts | **strongly supported** (four runs, single seed each) | E1, E2, E3 vs B1h; E7 vs E3 |
| Classification, not localisation, is the main loss on val | **strongly supported on val**; holdout40 margin much larger (§6.1 (2)) | §3.1 TIDE, §3.2 class-agnostic, §5.1 oracle |
| A crop classifier or TTA does not fix classification | **supported** (single settings) | crop classifier 0.100 / 0.117; TTA 0.138 |
| Small trucks are "never confident", not invisible | **strongly supported** (pre-registered, test half) | §5.2 C1, C4 |
| Labels are incomplete, or exclude real truck types, and the score undercounts | **supported, size unquantified** | FP audit 46/60 |
| Val has a scene-level shift from train | **plausible; mechanism unknown** | recall gap 0.187 |
| 500 more instances would not materially help | **supported as projected; extrapolation caveats** | §5.3 |
| Augmentation improves detection but not class-aware mAP50 | **plausible** (one run, two settings at once) | E4 |
| Smart subsets beat random on detection | **supported by the pre-registered rule, confounded with box count** | §5.4 |
| *Rejected:* B1 reaches 0.40–0.60 (pre-registered) | rejected (0.0715) | B1 |
| *Rejected:* B1h is undertrained (E1) | rejected | E1 |
| *Rejected:* aerial pretraining reduces overfitting (E3) | rejected, reversed | E3 |
| *Rejected:* TTA improves holdout40 (pre-registered) | rejected: no detectable effect (−0.013, within noise) | TTA |
| *Weakened:* scale 0.5 hurts small trucks (E2) | inconclusive, leaning not supported | E2 |
| *Weakened:* "size isn't what limits detections" (B1) | weakened: size matters most below 16 px | §5.2 C1; DETAILED B1 note |
| *Weakened:* "no domain shift" (B1 diagnosis) | weakened by the recall gap | §3.4 |
| Rare-class resampling helps Tractor/Flatbed (E6b) | not supported (Tractor 0.007 → 0.002, Flatbed 0.069 → 0.061; overall no detectable effect) | E6b; E6 was too weak a test (+1.6% tile views) |
| Longer training helps when augmentation limits overfitting (E8) | not tested: E8 was stopped at epoch 68 for GPU budget | DETAILED "Budget decisions" |

### 6.3 What changed most between the initial and final system
- **B0 → B1, tiling: the only large change** (0.0020 → 0.0715 val mAP50). At 640 px a 22 px truck shrinks to a few
  pixels. With 1024 tiles, 1449 of 1552 val trucks have a matching anchor, vs 396 at 640 px (§5.1).
- **B1 → B1h:** 0.0715 → 0.1065, but within seed noise (B1h seed 1: 0.0634). The same recipe is kept.
- **B1h → final system:** multi-label NMS and a WBF ensemble of B1h ep40 + E4 ep40 + E7 ep20 (E13). Holdout40-clean
  0.1765 → 0.2086 for the ensemble before checkpoint choice; val 0.1065 → 0.1349. That is the only change after
  tiling that cleared the noise bar on holdout40-clean.
- *Corrected 2026-10-05: an earlier version said the final system was the B1 recipe without the held-out images.*

### 6.4 Single highest-priority next step (one working day)

**A label audit and completion pass on val and holdout40, then re-score B1h.**
- Re-annotate missing trucks of the five classes.
- Mark excluded truck types (pickups, utility trucks, trailers, generic trucks) as *ignore* regions, so detections on
  them count neither as true nor as false positives.
- Re-score B1h with the same scorer.

**Why this beats the alternatives:**
- About three quarters of the most confident "false alarms" look like real vehicles (46/60). Until the labels are
  fixed, every model comparison is measured against a target that punishes correct detections, and no model change
  can be measured properly.
- It is cheap (62 images, no GPU) and it tells us how far the true score is from 0.107.
- It also tells us whether the 0.75 target is reachable with these labels at all.
- The alternatives are weaker:
  - More model changes: four runs already show no gain.
  - A better classifier: the crop classifier did not help.
  - More labels: projected below noise.
  - All of them would be measured on the same flawed labels.
  - Ensembling: the top xView solutions combined several detectors, for example the first-place RFL (Reduced Focal
    Loss) solution, arXiv 1903.01347. E13's multi-label ensemble of existing models raised val to 0.1349 with no new
    training. It is the final system.

### 6.5 Why 0.75 was not reached

The following points together suggest that 0.75 mAP50 is out of reach for this data and protocol, and that part of
the measured gap is in the evaluation data rather than in the model.

1. **Published benchmark.** On 19 small, visually similar xView vehicle classes, the best reported result is 0.3065
   mAP (arXiv 2104.11854, Table IV). Our five classes are a subset of that kind of problem.
2. **The class-agnostic ceiling.** Ignoring class entirely, B1h reaches 0.255 AP50 on val and 0.362 on holdout40
   (`results/b1h_tile1024_holdout40/{eval,eval_holdout40}/class_agnostic.json`). Even perfect type naming would start
   from there, far below 0.75.
3. **What the "false positives" are.** 216 of B1h's 350 confident holdout40 false positives overlap xView boxes of
   truck types outside the five classes (`results/xview_overlap/fp_rescore_summary.json`). The metric counts real
   trucks as errors.
4. **Altered val images.** 8 of 22 val images are altered versions of their xView originals: 4 rescaled 2× / 0.5×,
   plus noise, blur + noise, and two contrast changes (§3.3c). Val also drops 10.3% of xView's boxes of the five
   types.
5. **xView leakage path.** All 465 images are xView training images, so any xView-pretrained model would have seen
   every val image with labels. We therefore used COCO-pretrained weights only.
6. **Levers tried.**
   - Without detectable gains: longer training (E1/E2), aerial pretraining (E3, E7), augmentation (E4), resampling
     (E6/E6b), extra xView data with excluded types (E10), xView-original labels (E15), TTA, and a crop classifier.
   - Helped modestly: a multi-label ensemble (E13, val 0.1065 → 0.1349).
   - Robust inference (E12) passed its holdout test by a negligible margin and lowered val (0.0906).

**Error-repair sum (an indication, not a bound).** Fixing each TIDE error type on val in turn adds +0.147 (Cls),
+0.077 (Bkg), +0.044 (Missed) and +0.022 (Loc) to 0.1065, about 0.40 in total
(`figures/b1h_tile1024_holdout40/errors/tide_dAP.csv`). The bins can overlap, so the sum is not a strict bound. It
still lies well below 0.75.

**Gap breakdown** (diagnostic re-scoring of B1h's val predictions; xView labels are used only here, never for
training or selection; `results/gap_breakdown/gap_breakdown.csv`):

| row | val mAP50 | class-agnostic AP50 |
|---|---|---|
| plain | 0.1065 | 0.255 |
| predictions on excluded truck types ignored | 0.1237 | 0.307 |
| the 4 rescaled images run at their native scale | 0.1122 | 0.290 |
| predictions on the val boxes dropped from xView's labels ignored | 0.1112 | 0.272 |
| all three | **0.1371** | **0.373** |

- Correcting the evaluation data's known issues raises B1h's val mAP50 by about 0.03.
- Scored against xView's original holdout labels, B1h reaches 0.1765 (E15 entry).
- Neither comes close to 0.75. The remaining distance is the detection and classification difficulty of about
  22-pixel, look-alike trucks.
- Resolution is a known lever: super-resolving 30 cm imagery to 15 cm improved mAP by 13–36% in Shermeyer & Van Etten
  (arXiv 1812.04098). We did not test it.

**E12 (robust inference) was not adopted.**
- Its holdout40 suite gain (+0.0004) is below the noise threshold.
- Val (0.0906 vs 0.1065) had been scored before the decision; the pre-registration lacked a minimum-gain requirement.
- It failed because per-image truck size varies as much across normal images as the rescaling does, so the rule kept
  the rescaled images at 1× and shrank 7 normal ones (DETAILED_EXPERIMENTS.md "E12: decision").

**Verified non-issues** (`results/review_checks/review_checks.json`):
- Raising max_det from 902 to 3000 / 10000 changes holdout40 mAP50 by +0.0004 **for single-label B1h**. The
  pycocotools cross-check uses the same maxDets (902). For the multi-label final system see the detection-cap check
  (TODO-MAXDET).
- Class-agnostic merging lowers holdout40 to 0.129.
- Tractor AP stays near 0 even at IoU 0.1 (0.010), so it is not a box-offset issue.
- **Density, corrected to boxes per megapixel:** val 6.0 vs train 1.7 pooled (medians 4.8 vs 0.7).

### 6.6 Retrospective
- Every model choice after B1h was made on holdout40, never on val.
- In hindsight, auditing the data's provenance first would still have saved time:
  - we spent effort explaining val-specific anomalies (rescaled and altered images, the recall gap) that a source
    comparison reveals in minutes;
  - we could have tested E10 and E15 earlier.
- It would not have changed the main conclusion: the measured cost of the data issues is about 0.03 mAP50 (§6.5).
- In the first hour of a similar project, we would check:
  1. **The data's source:** match images to public datasets by name, size and hash.
  2. **Per-split image statistics:** size, brightness, contrast, blur and noise, train vs val.
  3. **Label consistency against the source:** class and box agreement per split.

## 7. Deliverables

See `SUBMISSION_CHECKLIST.md` and the README sections "Final model and prediction" and "Reproduce everything".
Final weights: GitHub release `weights-final-v1` (3 checkpoints; the ensemble). The baseline B1h is in
`weights-b1h-v1`.
`predict.py` on CPU reproduced B1h's saved predictions for 2 val images (all paired; max confidence difference 3e-6;
`results/predict_test/compare.json`).
