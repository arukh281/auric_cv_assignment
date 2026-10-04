# Technical report: overhead truck detection (draft)

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
| Train/val usage | Train split for training only. **Corrected 2026-10-04:** an earlier version said `val: false`; the configs actually set `val: true` (`configs/*.yaml`). Ultralytics then computes its own per-epoch val metrics on un-sliced full images, used for training curves only; `last.pt` is always reported (`results/*/eval*/metrics.json`, field `weights`). One leak: Ultralytics EarlyStopping (default `patience=100`, unset in configs before E3/E4) watches that val fitness and stopped E1 at epoch 145 of 150 (EXPERIMENTS.md, "CHECK 0"). From E3/E4 on, configs set `patience: 0` and checkpoint curves are scored on holdout40 |
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

## Summary (2026-10-04)

**The target was missed by a wide margin.** Target: ≥ 0.75 mAP50 on the 22-image val set. Best result: **0.1065**
(95% CI 0.0563–0.1653), run `b1h_tile1024_holdout40` (`results/b1h_tile1024_holdout40/eval/per_class.csv`), about
one seventh of the target. Its repeat with seed 1 scored 0.0634 (`results/b1h_seed1/eval/per_class.csv`).

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
   extrapolation to 903 images gives 0.215 holdout mAP50 (CI 0.076–0.276) (§5.3).
4. **Other candidate causes.** Terms used: "ruled out" means a discriminating test was run; "no evidence for" means
   a signal was looked for and not found; "not tested" means neither. *Corrected 2026-10-04: this point previously
   said all of the following were "ruled out".*
   - **Train/val domain shift: not ruled out; the leading explanation for val < holdout40.** B1h finds 0.665 of val
     trucks vs 0.852 of holdout40 trucks, a gap of 0.187 (CI 0.043–0.307). Box size explains 3% of it and image size
     7%; density does not explain it where it can be compared. That leaves a scene-level difference
     (`results/recall_gap/standardised.csv`, §3.4). The earlier domain classifier found no separation (AUC CIs include
     0.5; §3.3), but it ran on only 42 local images and is a weak test.
   - **Image quality: not tested as a cause of the gap; possible contributor to the scene difference.** The visual
     notes report haze and blur in val (`analysis/notes/visual_inspection.md`, author-marked uncertain). The only
     related analysis is a 10-vs-12-image split of val by visual flags. Its CIs overlap (§3.3), so it gives no
     evidence for an effect, but it could only have detected a large one.
   - **Label errors: partly tested.** The label check verified that tiling preserves the labels: 0 of 3439 tiles differ
     from their source (`results/sanity/label_check_iou/label_check.json`). It did **not** test whether the annotations
     are correct or complete. The background-FP audit (`auric-fp-crop`) tests missing labels: **pending**.
   - **Model capacity:** YOLO11s has enough capacity to fit the training data. A 16-tile overfit test reaches AP50 1.000
     (§3.4), and E1/E2 reach train40 0.774 / 0.906. **Not tested:** whether a larger model would generalise better.
   - **Tile-merge settings: ruled out as a main cause (for B1).** Re-scoring B1's saved raw predictions with NMS on IoS
     0.5/0.6/0.7 and IoU 0.5 gives val mAP50 between 0.0707 and 0.0722. Removing the merge entirely costs
     0.0715 → 0.0484. max_det 3000 adds 0.006 (`figures/b1_tile1024/merge_sensitivity.csv`,
     `figures/b1_tile1024_maxdet3000/merge_sensitivity.csv`). This was not repeated on B1h.

Experiments still running when this was written: E3 (aerial pretraining), E4 (flipud + mixup), a crop classifier on
GT crops, and a background false-positive audit (§5.2 test-half confirmation done: all five claims hold).

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
(EXPERIMENTS.md, B0/B1/B1h entries).

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
- B0 vs B1 is equal epochs, not equal compute (1400 vs 11850 iterations; EXPERIMENTS.md B0).
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
EXPERIMENTS.md B1h). Ignoring the class roughly doubles AP. At conf 0.25 only 361/1552 (0.233) val GT are matched by
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
  - Sources: `results/recall_gap/standardised.csv`, `recall_by_factor.csv`; EXPERIMENTS.md "Val vs holdout40 recall
    gap".
- **Overfit test** (16 tiles, all 5 classes, augmentation off): AP50 0.355 / 0.987 / 1.000 at epochs 50 / 100 / 300;
  final cls_loss 0.097 (`results/sanity/overfit/`). The model and pipeline can fit these labels.
- **Label check:** the first check flagged 85 of 3439 tiles (`results/sanity/label_check.json`). With the checker
  fixed to pair boxes by IoU, the re-run flags 0 of 3439 (`results/sanity/label_check_iou/label_check.json`). The IoU-paired
  re-check found all 3,278 boxes in those tiles identical (`results/sanity/label_mismatch/summary.json`): an artefact
  of coordinate-sort pairing in dense tiles, not a tiling error (annotation correctness was not tested here; see the FP audit). The author overrode the original NOT PASS before
  the re-check; with it, both PASS criteria hold (EXPERIMENTS.md, SANITY).
- Effective B1h settings: imgsz 1024, mosaic 1.0, scale 0.5, close_mosaic 10
  (`results/b1h_tile1024_holdout40/train/args.yaml`).

## 4. Experiment records

Full entries are in EXPERIMENTS.md under the headings named below. "Pre-registered" means written before results.

| Exp. | Observation | Hypothesis | Changes | Results | Conclusion | Next |
|---|---|---|---|---|---|---|
| **B0** (EXPERIMENTS "B0") | ~3200 px images, ~22 px trucks | Not recorded before the run; reconstructed afterwards: 640 letterbox makes trucks ~4–5 px, so B0 is a floor | whole image @640, 50 ep | val 0.0020 (0.0009–0.0045) | Almost no detection; tiling needed | B1 |
| **B1** ("B1") | as B0; max box side 161 px < overlap 256 | Pre-registered: 0.40–0.60 mAP50; Cls dominant; Liquid worst | 1024 tiles, overlap 256, sliced eval | val 0.0715 (0.0419–0.1242) | Far below prediction; Cls dominant (correct); rejection condition (b) largely met (size not the main limiter) | test generalisation |
| **B1h** ("B1h") | is val unusually hard? | No prediction written before this run | 40 train imgs held out; max_det 902 | val 0.1065, holdout 0.1507, train40 0.378 | Poor generalisation, not val-specific; B1→B1h gain within seed noise | learning curves |
| **LC** ("LC") | train/holdout gap | Pre-registered: curve still rising; 25% → 0.07–0.12 holdout | 25/50/75% subsets at equal iterations + seed 1 | §5.3 | Rising; 25% gave 0.061 | §5.4 |
| **S54** ("S54") | – | Selection and rule pre-registered (`b02413c`); prediction never provided | smart vs random subsets | §5.4 | rule computed; no subset ≥ 90% | – |
| **SANITY** ("SANITY") | train40 only 0.378 | settings / labels / capacity | args check, label check, overfit test | §3.4 | pipeline can fit; tiling preserves labels (re-check; annotation correctness not tested) | E1/E2 |
| **E1/E2** ("E1 / E2") | B1h losses still falling | Pre-registered: E1 undertraining; E2 scale 0.5 hurts small trucks | E1 150 ep; E2 150 ep + scale 0.2 | below | E1 not supported (overfitting); E2 inconclusive, leaning not supported | E3/E4 |
| **E3/E4** ("E3 / E4") | E1/E2 overfit | Pre-registered (author's): aerial pretraining (E3) / flipud 0.5 + mixup 0.1 (E4) reduce overfitting; holdout > 0.1507 + 0.017 | B1h recipe, 50 ep, `patience: 0`, holdout checkpoint curves | **pending (launched)** | pending | pending |

E1/E2 results (EXPERIMENTS.md "E1 / E2: Results"; `results/<run>/eval/per_class.csv`, `eval_holdout40/per_class.csv`,
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
  crop-classifier experiment (`auric-fp-crop`, pending) is the first test of that.
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

**Descriptive only (computed after the test half was opened; not pre-registered):** the chance-box rate and the
never-detected share at conf ≥ 0.05 / ≥ 0.10, from `analysis/s52_floor.py` (CPU kernel `auric-s52-floor`). TODO-FINAL:
numbers from `results/s52_floor/test/summary.json` when the kernel finishes. Inspect-half values from the unsaved
3 Oct session analysis were 0.002 (chance) and 0.396 (≥ 0.10).

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
  0.07–0.12, CI 0.012–0.090 overlaps; EXPERIMENTS.md "LC").
- Power-law fit on holdout mAP50 (403 point = mean of seeds, 0.142) extrapolates to **0.215 at 903 images (95% CI
  0.076–0.276)**, a gain of +0.080 vs seed spread 0.017. **Every fit is flagged unreliable** (2.24× beyond the data;
  several per-class fits non-monotone or failed) (`figures/learning_curve/power_law_fit.csv`). Only the direction is
  trustworthy.
- Per class (holdout, seed 0, 101 → 403 images; `figures/learning_curve/learning_curve.csv` via HANDOFF.md §6):
  Box 0.236 → 0.509, Cargo 0.018 → 0.096, Flatbed 0.034 → 0.069, Liquid 0.015 → 0.073, Tractor 0.000 → 0.007.
  Box benefits most; Tractor barely moves.
- **Answer:** more labels should help, but the optimistic projection is far below 0.75. Classification needs its own
  remedy.

### 5.4 Smallest subset retaining ≥ 90%

Pre-registered in commit `b02413c` before any S54 run (EXPERIMENTS.md "S54"): smart = class coverage (rarest class
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

*Skeleton (2026-10-04). Items marked TODO-FINAL wait for E3/E4, the background-FP audit and the crop classifier.*

### 6.1 Dominant limitations of the final detector
1. **Classification between truck types, mainly Cargo vs Box.** Evidence: class-agnostic 0.255 vs class-aware 0.107
   (val); the GT-box oracle gets 60% right vs 52% for always "Cargo"; the Cls oracle fix adds +0.147 (§3.2, §5.1).
2. **Generalisation from 403 images.** Evidence: train40 0.378 vs holdout40 0.151; E1/E2 raise train40 to
   0.77 / 0.91 while holdout40 falls (§4).
3. **Low confidence on small trucks.** Evidence: §5.2 C1 and C4 confirmed on the test half; 78% of never-detected boxes
   have a prediction below 0.25.
4. **Val scenes are harder than train scenes.** Evidence: recall 0.665 vs 0.852 on holdout40, not explained by size,
   resolution or density (§3.4).
5. TODO-FINAL: missing labels / out-of-scope vehicles as a ceiling (background-FP audit).

### 6.2 Strength of each conclusion
| Conclusion | Status | Evidence |
|---|---|---|
| Classification, not localisation, is the main loss | strongly supported | §3.1, §3.2, §5.1 (three independent analyses) |
| The detector overfits; longer training does not help | strongly supported (one seed each) | E1, E2 vs B1h |
| Small trucks are mostly "never confident", not invisible | strongly supported | §5.2 C1, C4, pre-registered and confirmed |
| More labels help but cannot reach 0.75 | plausible, unresolved in size | §5.3; power-law fit flagged unreliable |
| Val is harder because of a scene-level shift | plausible; mechanism unknown | §3.4 recall gap |
| Smart subset selection helps detection, not classification | supported by the pre-registered rule, single seed for smart | §5.4 |
| Aerial pretraining reduces overfitting (E3) | TODO-FINAL | |
| Overhead augmentation reduces overfitting (E4) | TODO-FINAL | |
| Missing labels cap the score | TODO-FINAL (FP audit) | |
| A second-stage crop classifier fixes Cargo vs Box | TODO-FINAL | |
| **Weakened or rejected:** B1 would reach 0.40–0.60 (pre-registered) | rejected (0.0715) | EXPERIMENTS.md B1 |
| **Weakened or rejected:** B1h is undertrained (E1) | rejected | E1 |
| **Weakened or rejected:** scale 0.5 hurts small trucks (E2) | inconclusive, leaning not supported | E2 |

### 6.3 What changed most between the initial and final system
- B0 → B1: native-resolution tiling plus sliced evaluation, 0.0020 → 0.0715 val mAP50. Explanation: at 640 px a
  22-px truck becomes about 4 px. Supported by B0's oracle: only 396/1552 GT have a matching anchor, vs 1449/1552 for
  B1 (§5.1).
- B1 → B1h: 0.0715 → 0.1065, but this is within seed noise (B1h seed 1: 0.0634).
- TODO-FINAL: E3/E4/E5 or crop-classifier change, if any passes its rule.

### 6.4 Single highest-priority next step (one working day)
TODO-FINAL: choose after E3/E4 and the audits. The candidates and what would decide between them:

| Candidate | Targets limitation | Information value | Performance value | Decided by |
|---|---|---|---|---|
| (a) Two-stage: detector proposals + crop classifier trained on more context and higher resolution | 1 | tests whether Cargo vs Box is separable at all from pixels | high if the classifier works | crop-classifier result (`auric-fp-crop`) |
| (b) Re-label audit of a val sample (and train), including missing trucks | 4, 5 | tells whether the 0.75 target is reachable with these labels | none directly | FP-audit share of unlabelled trucks |
| (c) Aerial-pretrained or larger backbone, with the best augmentation | 2 | tests capacity/generalisation | moderate | E3/E4 outcome |
| (d) More labels targeted at Cargo/Box confusion and val-like scenes (ports, yards) | 1, 2, 4 | direct test of §5.3 | moderate (projection 0.215) | §5.3 + recall gap |
| (e) Lower operating threshold / calibration study for small trucks | 3 | low | small for mAP (mAP already integrates over thresholds) | §5.2 C4 |


## 7. Deliverables

See `SUBMISSION_CHECKLIST.md` and the README sections "Final model and prediction" and "Reproduce everything".
Final weights: GitHub release `weights-b1h-v1` (TODO-FINAL: replace if E5 or a second stage changes the final model).
`predict.py` on CPU reproduced B1h's saved predictions for 2 val images (all paired; max confidence difference 3e-6;
`results/predict_test/compare.json`).
