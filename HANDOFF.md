# HANDOFF: Auric CV assignment (5-class overhead truck detector)

Written 2026-10-03 from read-only inspection of the repo at commit `78337f2` (working tree clean). Every number below
gives the file it was read from. "UNKNOWN" means it could not be established from files. This file replaces an older
HANDOFF.md (commit `fc1e4b1`). It has not been committed.

**One-line status:** the target is ≥ 0.75 mAP50 on val. The best measured val mAP50 is **0.1065** (run
`b1h_tile1024_holdout40`). The target is not met, and every analysis so far points at classification and
generalisation, not the eval pipeline.

---

## 0. CURRENT STATE (updated 2026-10-04 afternoon IST; supersedes older statements below where they differ)

**Best model is still `b1h_tile1024_holdout40`.** Val mAP50 0.1065, holdout 0.1507.
- **Weights:** GitHub release `weights-b1h-v1` (file `b1h_tile1024_holdout40_last.pt`, SHA-256 `3fa24066…7ffb`,
  matching `results/b1h_tile1024_holdout40/eval/metrics.json`). Also in the output of Kaggle kernel
  `aradhya1211/auric-b1h` v1.

**Done on 4 Oct (commits after `727d7c7`):**
- **Check 0:** every reported score used `last.pt`. The training `data.yaml` uses the official val set, but only
  for Ultralytics' own curves. One leak: early stopping (patience 100) watched that val score and cut E1 at epoch 145.
  From E3/E4 on, configs set `patience: 0` and checkpoints are scored on holdout40 (`EXPERIMENTS.md` "CHECK 0").
- **E3** (DOTA `yolo11s-obb.pt` init) and **E4** (flipud 0.5 + mixup 0.1): pre-registered, launched as GPU kernels
  `auric-e3-dota` and `auric-e4-flipud-mixup` (code `734299d`, `scripts/run_e34.sh`). Results pending.
- **CPU kernels:**
  - `auric-fp-crop` (FP audit + crop classifier): pending.
  - `auric-e1-figs`: done; E1 `figures/` + `errors/` committed, E1 size-recall column verified.
  - `auric-s52-confirm`: v1 failed on stale code before reading anything; v2 pending.
  - `auric-recall-gap`: pending.
  - `auric-sanity-labels`: label check with the IoU-paired checker; pending.
  - `auric-predict-test`: `predict.py` on 2 val images vs the saved predictions; pending.
- **§5.2 test-half confirmation** pre-registered (claims C1–C5) before the test half was read (`95953f3`).
- **EXPERIMENTS.md audit** (`ba8da62`): index table, six fields per entry, numbers checked against sources,
  visible corrections. The E1 no-mosaic epoch count changed 4 → 5, and the E1/E2 launch and sanity dates changed
  2 Oct → 3 Oct.
- **REPORT.md:**
  - §2–§4, §5.1, §5.3 and §5.4 drafted (`07213e4`), then reviewed line by line against the sources; no
    misstatements found.
  - Summary of the 0.75 miss added; §5.4 conclusion and Next written.
  - Still to write: §5.2, §6, §7.
- **Deliverables:**
  - `predict.py` (single predict/evaluate entry point).
  - `requirements-lock.txt` (exact B1h environment).
  - README "Final model" and "Reproduce everything" sections.
- **`analysis/sanity_check.py`** now pairs labels by IoU (`--labels-only` for a CPU run).

**Kernel gotcha:** after `kaggle_cli_package.sh`, the dataset can report `ready` before kernels mount the new
version (s52-confirm v1 got `734299d`). New script kernels now start with `grep -q <commit> CODE_COMMIT || exit 3`.

**Done since the original handoff (all in EXPERIMENTS.md, committed):**

1. **§5.4 smart vs random** (overnight 2–3 Oct): `figures/subset_compare/`.
   - Smart beats both random seeds on held-out class-agnostic AP50 at 202 and 302 images (margins +0.052 / +0.028
     vs seed spreads 0.011 / 0.013).
   - It does not on held-out mAP50.
   - No 50% or 75% subset reaches 0.128. No conclusion written yet.
2. **§5.2 training dynamics** (inspect half only): `figures/s52/inspect/`, `results/s52/`.
   - Never-detected confidence-floor numbers are in §5 below. They are still not saved to a file.
3. **SANITY** (kernel `auric-sanity`, 3 Oct; corrected 2026-10-04 from "2 Oct", per the code commit `f2388f1` date):
   - B1h `args.yaml` copied to `results/b1h_tile1024_holdout40/train/args.yaml`: imgsz 1024, mosaic 1.0, scale 0.5,
     close_mosaic 10, rect false.
   - B1h losses still falling at epoch 50 (`figures/sanity/b1h_loss_curves.png`).
   - 16-tile overfit test with augmentation off: AP50 0.355 / 0.987 / **1.000** at epochs 50 / 100 / 300; final
     cls_loss 0.097.
   - The label check flagged 85 tiles. The CPU re-check (`auric-label-mismatch`) found **all 3,278 boxes identical**,
     so it is a **checker artefact** (coordinate-sort pairing in dense tiles), not a label problem
     (`results/sanity/label_mismatch/`).
   - The author overrode the NOT PASS before the re-check; with the re-check, both PASS criteria hold.
4. **E1 / E2** (pre-registered; kernels `auric-e1-b1h-150ep`, `auric-e2-b1h-150ep-scale02`, code `e72d3b8`):

| | B1h | E1 (150 ep; EarlyStopping stopped it at 145) | E2 (150 ep, scale 0.2) |
|---|---|---|---|
| val mAP50 (CI) | 0.1065 (0.056–0.165) | 0.0680 (0.036–0.124) | 0.0620 (0.033–0.110) |
| holdout40 mAP50 (CI) | 0.1507 (0.056–0.187) | 0.0923 (0.030–0.138) | 0.1112 (0.034–0.139) |
| train40 mAP50 | 0.378 | 0.774 | 0.906 |
| final train cls_loss | 1.60 | 0.80 | 0.54 |

   - Val peaks mid-training: E1 0.100 at epoch 90, E2 0.107 at epochs 40–50.
   - Recall < 16 px at conf 0.25: 0.057 / 0.061 / 0.061 (B1h / E1 / E2).
   - **Verdicts:** E1 (undertraining) **not supported**, and the pattern looks like overfitting. E2 (scale)
     **inconclusive, leaning not supported**.
   - Sources: `results/e1_b1h_150ep/`, `results/e2_b1h_150ep_scale02/` (incl. `eval_train40/` from the CPU kernel
     `auric-e1e2-train40`), `figures/e2_b1h_150ep_scale02/`.
   - E1's `figures/` and `errors/` were missing from its Kaggle output (cause unknown); regenerated on 4 Oct by CPU kernel `auric-e1-figs`.

**GPU budget:** `kaggle quota` shows 9.56 h used, **20.44 h remaining**, refresh 2026-10-10 (~05:30 local).
- Rule from the author: never launch a GPU kernel projected over 10 h without asking.
- CPU-only work goes in CPU-only Kaggle kernels: `scripts/kaggle_cli_script_kernel.sh <slug> '<cmd>' --cpu --push`.
- **No compute on the Mac** (author's explicit rule), and no Colab for training.

**Open items**
- ~~`patience` bug~~ fixed for E3/E4 (`patience: 0`); older configs are unchanged by design.
- Next experiment not chosen. The evidence points at generalisation and overfitting, not training length. Candidates
  in EXPERIMENTS.md: stronger augmentation or regularisation, a crop classifier for Cargo vs Box (planned as a
  CPU-only kernel), more data.
- Write-ups still missing: §5.2 interpretation (and confirmation on the test half), §5.3 per-class "which classes
  benefit", §5.4 conclusion, §6 final analysis, the full REPORT.md.
- `STATUS.md` is stale. (`REPORT.md`'s `val: false` line was corrected on 4 Oct.)
- ~~`sanity_check.check_labels` coordinate-sort pairing~~ fixed (IoU pairing); re-run result pending.

---

## 1. Project map

### Directory tree (tracked files, without raw images or caches)

```
auric_cv_assignment/
├── train.py                 Train one config with Ultralytics (resumable; tiling; checkpoints at epochs 10/20/…; OOM fallback)
├── eval.py                  Val/train/holdout images -> sliced or full inference -> merge -> scorer -> metrics (+ pycocotools cross-check)
├── requirements.txt         Pinned non-torch deps (see §2)
├── README.md                How to run (Colab Terminal, Colab notebook, Kaggle notebook, Kaggle via CLI, local)
├── REPORT.md                Technical report, IN PROGRESS (pretrained-weights choice, reproducibility table, metric, error taxonomy)
├── EXPERIMENTS.md           Experiment log: B0, B1, B1h, Plan, LC (§5.3), §5.1, S54 (§5.4) pre-registration
├── STATUS.md                STALE: Phase-2 status from 2 Oct (pre-training)
├── MORNING.md               Overnight report 2–3 Oct (§5.2 inspect-half counts, §5.4 runs, GPU hours, commits)
├── HANDOFF.md               This file
├── configs/
│   ├── data.yaml            Ultralytics dataset config (5 classes)
│   ├── b0.yaml              B0: whole image letterboxed to 640
│   ├── b1.yaml              B1: native-resolution 1024 tiles, overlap 256
│   ├── b1h.yaml             B1h: B1 with 40 train images held out (holdout_list), eval max_det 902
│   ├── b1h_seed1.yaml       B1h, training seed 1
│   ├── b1h_f25/f50/f75.yaml     Learning-curve subsets (train_list), equal total iterations
│   ├── b1h_f50_seed1.yaml, b1h_f75_seed1.yaml   Random subsets, seed 1 (§5.4 noise)
│   └── b1h_smart50.yaml, b1h_smart75.yaml       "Smart" subsets (§5.4)
├── detlib/                  Shared library
│   ├── scoring.py           COCO-101 AP50 scorer (+ Ultralytics-style interpolation, bootstrap)
│   ├── tiling.py            Tile windows / box clipping
│   ├── merge.py             Cross-tile merge (class-wise NMS on IoS/IoU, WBF)
│   ├── data.py              Class map, YOLO label I/O, EDA-table helpers
│   ├── envlog.py            Writes env/ (pip freeze, GPU, torch, git commit or CODE_COMMIT)
│   └── curves.py            Training-curve plots
├── tools/
│   ├── make_tiles.py        Tiler (--exclude-list / --include-list keep original image indices)
│   ├── make_holdout.py      Builds splits/holdout40_seed0.txt; replicates tile counts from EDA tables
│   ├── make_subsets.py      Nested stratified LC subsets + equal-iteration configs
│   └── make_smart_subsets.py   §5.4 DINOv2 embeddings (embed) + coverage/k-center selection (select)
├── analysis/
│   ├── eda.py               Phase-1 EDA (DO NOT re-run locally: data/ points at a subset)
│   ├── pred_review.py       Qualitative TP/FP/FN grids, confusion matrix, overviews
│   ├── checkpoint_curve.py  Real (sliced) metric for each kept checkpoint
│   ├── merge_sensitivity.py Re-scores raw tile predictions under merge settings (no re-inference)
│   ├── errors.py            TIDE-style error bins + oracle dAP50, sliced FN/FP rates, confusion, crops
│   ├── gt_box_oracle.py     §5.1 classification at GT boxes (pooled / best-anchor modes)
│   ├── class_agnostic.py    Class-agnostic AP50 and recall from saved predictions
│   ├── per_image.py         Per-image metrics, visual-inspection flags, subset mAP
│   ├── domain_shift.py      Train-vs-val image stats, crop montages, domain classifiers (data_small)
│   ├── learning_curve.py    §5.3 curves + power-law extrapolation with reliability flags
│   ├── subset_compare.py    §5.4 smart vs random table + pre-registered rule output
│   ├── training_dynamics.py §5.2 per-GT-box categories across B1h checkpoints
│   └── notes/visual_inspection.md   Hand-written visual observations (42 images)
├── scripts/
│   ├── _env.sh              Platform paths (Kaggle auto-detected, else Colab)
│   ├── _run.sh              train -> eval -> pred_review -> merge_sensitivity -> checkpoint_curve
│   ├── run_b0.sh, run_b1.sh, run_b1h.sh, run_lc.sh, run_analysis.sh, status.sh
│   ├── colab_setup.sh, kaggle_setup.sh, kaggle_push_results.sh
│   └── kaggle_cli_package.sh, kaggle_cli_kernel.sh, kaggle_cli_script_kernel.sh, kaggle_kernel_run.py   (Kaggle via CLI)
├── notebooks/colab_train.ipynb, notebooks/kaggle_train.ipynb
├── splits/                  holdout40_seed0, train_f{25,50,75}_seed0, train_smart{50,75}_seed0, s52_{inspect,test}_seed0, *.json
├── tests/                   10 files, 47 known-answer tests (test_pipeline, test_analysis, test_diagnostics, test_domain_shift,
│                            test_holdout, test_learning_curve(_setup), test_per_image, test_smart_subsets, test_training_dynamics)
├── results/<run>/           Per-run CSV/JSON/PNG copied from Colab/Kaggle (no weights): eval*/, train/, env/, checkpoint_curve/
│   ├── s52/                 §5.2 per-box tables (inspect + test halves) and args
│   └── s54/                 §5.4 DINOv2 embeddings (embeddings.npz) and embed args
└── figures/
    ├── eda/                 Phase-1 EDA figures + tables/ (full dataset)
    ├── <run>/               pred_review grids, confusion, checkpoint_curve, merge_sensitivity, errors/, gt_oracle/, per_image/
    ├── b1_tile1024_maxdet3000/merge_sensitivity.csv   B1 re-scored with a larger max_det (see §8)
    ├── domain_shift/, learning_curve/, subset_compare/, s52/inspect/
```

**Untracked or ignored local items:**
- `data -> data_small`: a symlink to a 20-train / 22-val subset. The full dataset is only on Drive/Kaggle.
- `data_small/`, `cv_dataset.zip`, and `Computer_Vision_Assignment_Auric_AI_Technologies__1_.pdf` (the brief).
- `kaggle_build/`: CLI build folders.
- `runs/`: empty on the Mac.

### Results folders

| folder | runs inside |
|---|---|
| `results/` | b0_full640, b1_tile1024, b1h_tile1024_holdout40, b1h_seed1, b1h_f25, b1h_f50, b1h_f75, b1h_f50_seed1, b1h_f75_seed1, b1h_smart50, b1h_smart75, s52, s54 |
| `figures/` | the same runs, plus eda, domain_shift, learning_curve, subset_compare, s52, b1_tile1024_maxdet3000 |

### Git

- Branch `main`; remote GitHub `arukh281/auric_cv_assignment` (private). 45 commits, HEAD `78337f2` (2026-10-03 06:59 +0530).
- Working tree clean at the time of writing, apart from this HANDOFF.md.
- Most recent commits (newest first):

```
78337f2 MORNING.md: overnight report
7deafdb Results: b1h_f75_seed1; S54 subset comparison table + pre-registered rule output
3140888 Results: b1h_smart75, b1h_f50_seed1
40067b8 Results: b1h_smart50
e8633a6 Results: §5.2 training dynamics (per-box tables both halves, inspect-half summaries and crops only)
f296537 S54 smart subsets (DINOv2-small) and configs
5807e3e analysis/subset_compare.py
883d884 §5.2 training_dynamics.py, §5.4 make_smart_subsets.py, script-mode Kaggle kernels
b02413c S54 pre-registration (before any S54 run); §5.2 inspect/test halves (before any §5.2 computation)
3e4b9c8 §5.1 GT-box oracle conclusion
c6188ca Learning-curve conclusion (§5.3)
089e471 Results: learning curves + learning_curve analysis
...      (full list: git log --oneline)
```

---

## 2. Environment

- **Framework:** Ultralytics `8.4.171` (`requirements.txt`; also `env/train_hardware.json` of every run). Model: YOLO11s, COCO-pretrained `yolo11s.pt`.
- **`requirements.txt`** exists and pins ultralytics 8.4.171, ensemble-boxes 1.0.9, imagehash 4.3.2, pandas 3.0.6, matplotlib 3.11.2,
  pillow 12.3.0, pyyaml 6.0.3, scipy 1.17.1, pycocotools 2.0.11, scikit-learn 1.9.1. Torch, torchvision, numpy and opencv are
  deliberately unpinned: they are the platform's own builds, installed as constraints.
- **Runtimes, per `results/<run>/env/train_hardware.json`:**

| runs | platform | GPU | torch | Python |
|---|---|---|---|---|
| b0_full640, b1_tile1024 | Colab | 1 × Tesla T4 | 2.11.0+cu130 | 3.13.15 |
| b1h_tile1024_holdout40, b1h_seed1, b1h_f25/f50/f75 (2 Oct) | Kaggle | 2 × Tesla T4 (training used CUDA:0) | 2.10.0+cu128 | 3.12.13 |
| b1h_smart50/75, b1h_f50_seed1, b1h_f75_seed1 (3 Oct) | Kaggle | 2 × Tesla T4 | 2.11.0+cu128 | 3.13.15 |

- The `cuda` field in these JSONs is null. The CUDA version is UNKNOWN beyond the torch build tag.
- **Local Mac:** `.venv` with Python 3.11, CPU/MPS only. Used for tests and CPU analyses, not for training.
- **Git commit of each run:** recorded in `env/train_hardware.json` (`git_commit`). Kaggle runs use a `CODE_COMMIT` file because the packaged code has no .git.
  - b1h: `4a17eb1`
  - b1h_seed1 and f25/f50/f75: `ab32af2`
  - smart50/75 and f50/f75_seed1: `f296537`
  - b1: `bebdd2b`
  - b0: `e0885bf`

---

## 3. Dataset

Sources: `figures/eda/summary.json` and `figures/eda/tables/*.csv`, computed on the full dataset.

| ID | Class | train instances | val instances | train images | val images | train share | val share |
|---|---|---|---|---|---|---|---|
| 0 | Cargo Truck | 3773 | 800 | 368 | 19 | 0.4953 | 0.5155 |
| 1 | Truck w/Box | 2366 | 493 | 261 | 17 | 0.3106 | 0.3177 |
| 2 | Truck w/Flatbed | 662 | 122 | 195 | 14 | 0.0869 | 0.0786 |
| 3 | Truck Tractor | 625 | 117 | 128 | 11 | 0.0820 | 0.0754 |
| 4 | Truck w/Liquid | 192 | 20 | 99 | 8 | 0.0252 | 0.0129 |

(`figures/eda/tables/class_counts.csv`; class IDs from `configs/data.yaml`.)

- **Image and box totals:** train 443 images / 7618 boxes; val 22 images / 1552 boxes (`summary.json`).
- **Boxes per image** (`boxes_per_image_stats.csv`): train mean 17.2, median 7, max 451; val mean 70.5, median 76.5, min 8, max 167.
- **Image size** (`images.csv`):
  - Train: width median 3228 (2576–5119), height median 2891 (2426–3325).
  - Val: width median 3114 (1369–7204), height median 2911.5 (1334–5932).
- **Box size:**
  - Median sqrt(area): train 22.05 px, val 25.50 px (`summary.json`).
  - Largest train box side: 161 px (`configs/b1.yaml` comment; `figures/eda/tables/boxes.csv`).
  - COCO small/medium/large counts per class are in `size_bins.csv`. Train Cargo: 3474 / 298 / 1; train Box: 1229 / 1136 / 1.
- **Label issues** (`label_issues_summary.csv`): 1 empty train image (1938.png), and 36 boxes with a side < 4 px (train 10, val 26).
- **Near-duplicates (pHash ≤ 8):** 0 cross-split pairs and 0 within-split pairs (`summary.json`).
- **Recorded observations:**
  - `EXPERIMENTS.md` (B0/B1 Observation): letterboxing ~3200 px to 640 shrinks a median truck to ~4–5 px (derived, not measured).
  - `analysis/notes/visual_inspection.md`: 42 images of data_small. It covers haze (val 2460, 2470, 2472), blur (val 2292, 2308, 2543), apparent scale difference (val 2391, 2308), water speckle (val 1399, 1447, 1456), visible-but-unlabelled trucks in both splits, and possibly offset GT boxes in 2470/2472. All of these are marked as uncertain.

### Splits (none of these split val; val is always the full 22 images)

| split | file | how made | used for |
|---|---|---|---|
| holdout40 | `splits/holdout40_seed0.txt` | 40 **train** images; seed 0; stratified by each image's rarest class (`tools/make_holdout.py`); 1938.png excluded | never trained on by any B1h-family run; evaluated as `eval_holdout40/` |
| LC subsets | `splits/train_f{25,50,75}_seed0.txt` | nested, class-stratified prefixes of one seed-0 ordering of the 403 non-holdout train images (`tools/make_subsets.py`) | §5.3 |
| smart subsets | `splits/train_smart{50,75}_seed0.txt` | class coverage then greedy k-center on DINOv2-small embeddings (`tools/make_smart_subsets.py`) | §5.4 |
| §5.2 halves | `splits/s52_inspect_seed0.txt` (202 images), `splits/s52_test_seed0.txt` (201 images) | seed-0 permutation of the 403 non-holdout **train** images, committed before any §5.2 computation (`b02413c`) | inspect = explore / summarise; test = reserved for confirming §5.2 findings |

**Reserved, do not open:** `results/s52/per_box_test.csv`, which holds the §5.2 per-box table for the test half. It exists and
has not been opened or summarised. The §5.2 prediction files on Kaggle (`/kaggle/working/s52/preds/*.csv.gz` in kernel
`aradhya1211/auric-s52`) also contain test-half images. Only inspect-half rows were ever read from them.

---

## 4. Training runs

Common to every run (from `configs/*.yaml`):
- **Model:** YOLO11s initialised from COCO-pretrained `yolo11s.pt` (xView weights deliberately not used: REPORT.md).
- **Optimiser:** SGD, lr0 0.01, momentum 0.937. AMP on, `cache: disk`, `deterministic: true`, batch 16.
- **Augmentation:** Ultralytics 8.4.171 defaults; no class weighting or resampling in any config. The defaults seen in the B1h
  smoke-kernel log during this session include hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, fliplr 0.5, mosaic 1.0, scale 0.5,
  translate 0.1, erasing 0.4, mixup 0. That log is not in the repo. (Corrected 2026-10-04: `train/args.yaml` has since been copied to `results/b1h_tile1024_holdout40/train/args.yaml`.)
- **Evaluation:** `last.pt` is evaluated (no checkpoint is chosen on val), conf 0.001, NMS IoU 0.7. Tiled runs use sliced eval
  with tile 1024, overlap 256, and class-wise NMS on IoS 0.6.

| Run | imgsz / input | epochs | train images → tiles | iters (it/ep × ep) | seed | eval max_det | val mAP50 (95% CI) | Cargo | Box | Flatbed | Tractor | Liquid | holdout40 mAP50 (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0_full640 | 640, whole image | 50 | 443 images | 1400 (28×50) | 0 | 334 | 0.0020 (0.0009–0.0045) | 0.0022 | 0.0077 | 0.0000 | 0.0000 | 0.0000 | – |
| b1_tile1024 | 1024 tiles | 50 | 443 → 3785 | 11850 (237×50) | 0 | 334 | 0.0715 (0.0419–0.1242) | 0.0889 | 0.1299 | 0.0530 | 0.0064 | 0.0792 | – |
| **b1h_tile1024_holdout40** | 1024 tiles | 50 | 403 → 3439 | 10750 (215×50) | 0 | 902 | **0.1065 (0.0563–0.1653)** | 0.1306 | 0.1797 | 0.0711 | 0.0062 | 0.1447 | 0.1507 (0.0558–0.1874) |
| b1h_seed1 | 1024 tiles | 50 | 403 → 3439 | 10750 | 1 | 902 | 0.0634 (0.0421–0.0959) | 0.1017 | 0.1292 | 0.0533 | 0.0290 | 0.0037 | 0.1333 (0.0506–0.1661) |
| b1h_f25 | 1024 tiles | 199 | 101 → 861 | 10746 (54×199) | 0 | 902 | 0.0169 (0.0106–0.0292) | 0.0345 | 0.0282 | 0.0219 | 0.0001 | 0.0000 | 0.0606 (0.0123–0.0899) |
| b1h_f50 | 1024 tiles | 101 | 202 → 1690 | 10706 (106×101) | 0 | 902 | 0.0492 (0.0215–0.1078) | 0.0453 | 0.0484 | 0.0203 | 0.0005 | 0.1316 | 0.0948 (0.0263–0.1274) |
| b1h_f75 | 1024 tiles | 67 | 302 → 2546 | 10720 (160×67) | 0 | 902 | 0.0846 (0.0477–0.1448) | 0.1189 | 0.1878 | 0.0403 | 0.0070 | 0.0691 | 0.1045 (0.0284–0.1437) |
| b1h_f50_seed1 | 1024 tiles | 101 | 202 → 1690 | 10706 | 1 | 902 | 0.0573 (0.0227–0.1105) | 0.0537 | 0.0345 | 0.0147 | 0.0009 | 0.1827 | 0.0816 (0.0278–0.1089) |
| b1h_f75_seed1 | 1024 tiles | 67 | 302 → 2546 | 10720 | 1 | 902 | 0.0736 (0.0460–0.1349) | 0.0948 | 0.1658 | 0.0346 | 0.0027 | 0.0701 | 0.1230 (0.0383–0.1648) |
| b1h_smart50 | 1024 tiles | 104 | 202 → 1646 | 10712 (103×104) | 0 | 902 | 0.0562 (0.0298–0.1024) | 0.0814 | 0.1324 | 0.0320 | 0.0037 | 0.0316 | 0.1052 (0.0282–0.1418) |
| b1h_smart75 | 1024 tiles | 66 | 302 → 2594 | 10758 (163×66) | 0 | 902 | 0.0832 (0.0414–0.1451) | 0.0903 | 0.1532 | 0.0508 | 0.0025 | 0.1192 | 0.1040 (0.0328–0.1319) |

**Sources:**
- Val numbers: `results/<run>/eval/per_class.csv` (column `AP50_coco`, CIs `AP50_coco_ci95_lo/hi`) and `metrics.json`.
- Holdout numbers: `results/<run>/eval_holdout40/per_class.csv`.
- Iterations, tiles and seed: `results/<run>/training_summary.json`.
- max_det: `results/<run>/eval/eval_args.json` (B0/B1) or `metrics.json` (`max_det`, B1h family).
- In-sample train40 numbers (not in the table): b1_tile1024 0.4027 (0.3175–0.4985), `results/b1_tile1024/eval_train40/per_class.csv`;
  b1h_tile1024_holdout40 0.3778 (0.2997–0.4609), `results/b1h_tile1024_holdout40/eval_train40/per_class.csv`.
- Scorer cross-check (`results/<run>/eval/scorer_comparison.csv`):
  - Ours and pycocotools agree to < 1e-15 on every run (no flag raised).
  - The Ultralytics val cross-check exists only for B0: 0.0024 vs ours 0.0020 (`results/b0_full640/eval/ultralytics_crosscheck.json`).

**Extra columns per run:**
- warmup_epochs / close_mosaic, scaled so every run covers the same iterations as B1h (645 / 2150):
  - f25 11.94 / 40; f50 and f50_seed1 6.08 / 20; f75 and f75_seed1 4.03 / 13
  - smart50 6.26 / 21; smart75 3.96 / 13
  - B0, B1, B1h and b1h_seed1 use the Ultralytics defaults, 3 / 10.
  - Source: `configs/*.yaml`.
- Holdout list: `holdout_list` = `splits/holdout40_seed0.txt` for every b1h* run. Subset runs also have `train_list`.

**Commands:**
- Colab (B0, B1): `bash scripts/run_b1.sh` / `bash scripts/run_b0.sh`. These call `scripts/_run.sh <config> <name>`:
  `train.py --config ... --oom-fallback-batch 8`, then `eval.py --config ...`, `pred_review.py`, `merge_sensitivity.py`
  and `checkpoint_curve.py`.
- Kaggle via CLI (B1h family):
  - `bash scripts/kaggle_cli_package.sh` (code → dataset `aradhya1211/auric-cv-code`)
  - then `bash scripts/kaggle_cli_kernel.sh full configs/<cfg>.yaml --push`
  - The kernel runs `scripts/run_b1h.sh` (B1h) or `scripts/run_lc.sh <cfg>` (all others), then `scripts/run_analysis.sh <name>`.
- The exact `command.txt` files were not copied into `results/` (.txt is excluded), so the literal command lines are
  UNKNOWN beyond these scripts.

**Current best:** `b1h_tile1024_holdout40`, with val mAP50 0.1065 and holdout 0.1507.
- Its seed-1 repeat scored 0.0634 on val, a spread of 0.043, so "best" is within run-to-run noise of several other runs.
- Weights: Kaggle kernel `aradhya1211/auric-b1h` v1 output, `/kaggle/working/runs/b1h_tile1024_holdout40/train/weights/last.pt`
  (also `best.pt` and `epoch010…050.pt`; checkpoint sha256 prefixes in `results/s52/training_dynamics_args.json`).
- Weights for B0 and B1 should be on Google Drive under `/content/drive/MyDrive/auric/runs/<run>/train/weights/`
  (`scripts/_env.sh` defaults). Their presence is UNKNOWN (not verifiable from the Mac).
- Other Kaggle runs: `/kaggle/working/runs/<run>/train/weights/` in kernels `aradhya1211/auric-b1h-<f25|f50|f75|seed1|smart50|smart75|f50-seed1|f75-seed1>`.
- No weights are in git.

---

## 5. Analyses done so far

All analyses below use **val** (22 images) unless stated otherwise. The only use of the §5.2 halves is in the §5.2 rows.

| Analysis | Script | Inputs → outputs | Key numbers (source) | Conclusion recorded (EXPERIMENTS.md unless noted) |
|---|---|---|---|---|
| EDA | `analysis/eda.py` | full dataset → `figures/eda/` | §3 above | B0/B1 Observations |
| Checkpoint curve | `analysis/checkpoint_curve.py` | kept checkpoints → `figures/<run>/checkpoint_curve.csv` | B1 ep10/20/30/40/50: 0.036/0.058/0.055/0.056/0.071; B1h: 0.043/0.072/0.067/0.097/0.106; B0: ≤0.002 throughout | none written |
| Scorer sanity | `eval.py --split train` | 40 seen train images | B1 in-sample mAP50 0.4027 (`results/b1_tile1024/eval_train40/metrics.json`) | "Scoring bug? Unlikely" |
| Tile-merge sensitivity | `analysis/merge_sensitivity.py` | `predictions_raw.csv` | B1 (max_det 334): IoS 0.6 0.0715, no merge 0.0484 (`figures/b1_tile1024/merge_sensitivity.csv`); B1h (902): IoS 0.6 0.1065, no merge 0.0805 | "Tile merging helps" |
| max_det sensitivity | merge_sensitivity with a larger max_det | B1 raw predictions | IoS 0.6 at max_det 3000: 0.0776 vs 0.0715 (`figures/b1_tile1024_maxdet3000/merge_sensitivity.csv`) | "Detection limit: +0.006" |
| Error taxonomy (TIDE order) | `analysis/errors.py` | predictions.csv → `figures/<run>/errors/` | B1 gain if fixed: cls +0.190, bkg +0.053, missed +0.041, loc +0.014 (`figures/b1_tile1024/errors/tide_dAP.csv`); B1h: cls +0.147, bkg +0.077, missed +0.044, loc +0.022; B0: cls +0.102 | §5.1: "classification, not localization, is the main ceiling" |
| Sliced FN/FP rates | `errors.py` (`slices.csv`) | B1 | 8–48 px trucks missed 71–79% at F1-opt thresholds; 48–96 px 52% (`figures/b1_tile1024/errors/slices.csv`); density bins 72/70/77/72% | B1 conclusion + condition (b) verdict |
| GT-box oracle (§5.1) | `analysis/gt_box_oracle.py` | weights + val GT → `figures/<run>/gt_oracle/` | pooled accuracy / mean per-class: B0 0.481/0.215, B1 0.553/0.387, B1h 0.601/0.436; "unflagged" (anchor at IoU ≥ 0.5): B0 396/1552, B1 1449/1552, B1h 1455/1552 (`comparison.csv`, `summary.json`) | §5.1 entry; B0 conclusion "26% vs 93% aligned" = 396/1552 vs 1449/1552 |
| Class-agnostic | `analysis/class_agnostic.py` | predictions → `class_agnostic.json` | B1 val agnostic AP50 0.2374, recall@0.001 0.5838; B1h val 0.2551 / 0.665, holdout 0.3617 / 0.852 | B1h Results ("Trucks found") |
| Per-image + flags | `analysis/per_image.py` | B1 val → `figures/b1_tile1024/per_image/` | unflagged 10 images mAP50 0.0791 (0.037–0.179), flagged 12 images 0.0631 (0.036–0.090) (`subset_map_val.csv`) | "Poor image quality? Not the main cause" |
| Domain shift | `analysis/domain_shift.py` | data_small (20 train / 22 val) → `figures/domain_shift/` | domain AUC: image stats 0.336 (0.168–0.515); crop embeddings 0.453 (0.334–0.564); per-image 0.368 (0.209–0.549) (`tables/domain_auc.csv`) | "Domain shift? No clear evidence" |
| B1h generalisation | eval on val/holdout/train40 | B1h | 0.107 / 0.151 / 0.378 | B1h Conclusion |
| Learning curves (§5.3) | `analysis/learning_curve.py` | 5 runs → `figures/learning_curve/` | holdout mAP50 101/202/302/403 imgs: 0.061/0.095/0.105/0.151 (seed1 0.133); +500 extrapolation 0.215 (0.076–0.276), every fit flagged unreliable (`power_law_fit.csv`) | LC Conclusion |
| §5.4 smart vs random | `analysis/subset_compare.py` | 8 runs → `figures/subset_compare/` | see §6 (5.4) | **none written yet** |
| §5.2 training dynamics | `analysis/training_dynamics.py` | B1h checkpoints, 403 train images | **inspect half only:** 3511 boxes: learned-early 169, learned-late 1547, forgotten 266, never-learned 241, never-detected 1288 (`results/s52/training_dynamics_args.json`, `figures/s52/inspect/counts_by_*.csv`) | **none written yet** |
| §5.2 never-detected confidence floor | ad-hoc session script (see below) | `figures/s52` inputs + Kaggle epoch-50 predictions, **inspect half only** | see below | none written |

**§5.2 never-detected confidence-floor analysis** (inspect half; run in-session on 2026-10-03; **not saved to any file in the repo**):
- **Definitions:**
  - Best prediction: the highest-IoU epoch-50 prediction of any class above the confidence floor.
  - Chance boxes: 3,511 boxes with the same image, size and class as the GT boxes, placed at random with IoU 0 to every GT box (seed 0).
- **Results:**

| conf floor | never-detected with best IoU ≥ 0.5 | chance boxes with best IoU ≥ 0.5 |
|---|---|---|
| ≥ 0.001 | 0.793 (median conf 0.084, p90 0.210) | 0.002 |
| ≥ 0.05 | 0.538 | 0.000 |
| ≥ 0.10 | 0.396 | 0.000 |

- Per class at ≥ 0.001: Cargo 0.801, Box 0.790, Flatbed 0.821, Tractor 0.674, Liquid 0.862.
- To make it citable, re-run it and save the output. The epoch-50 predictions are in kernel `aradhya1211/auric-s52`'s output.

**Inspect vs test halves:** every §5.2 summary, figure and the confidence-floor analysis used the **inspect half only**.
The test half is untouched apart from its per-box table, which was written and committed but never opened.

---

## 6. Status against the assignment

| Section | Status | Evidence | Missing |
|---|---|---|---|
| 2. Dataset observations, baseline, mAP + per-class for every model, qualitative predictions | PARTIAL | EDA (`figures/eda/`), visual notes, configs/scripts/tests, `per_class.csv` for all 11 runs, grids/overviews (`figures/<run>/grid_*.png`, `overview_*.png`, `errors/crops_*.png`) | Written prose in REPORT.md; EXPERIMENTS B0/B1 "Results" fields still say "Pending Colab" |
| 3. Failure diagnosis | PARTIAL | TIDE bins with dAP50, sliced FN/FP rates, confusion matrices, competing explanations tested (scoring, merge, max_det, domain shift, image quality, generalisation via B1h) | Consolidated write-up in REPORT.md; FP population (background FPs 4701 B1 / 9380 B1h) not discussed in prose |
| 4. Experiment records | PARTIAL | EXPERIMENTS.md entries for B0, B1, B1h, LC, §5.1, S54 | B0 Hypothesis "TODO (me)"; B0/B1 "Results: Pending" and "Next step: Pending results"; B1h lacks Observation/Hypothesis/Changes fields; S54 has no Results/Conclusion |
| 5.1 Perfect locations | DONE (draft) | EXPERIMENTS §5.1, `gt_oracle/` for B0/B1/B1h | Move into REPORT.md |
| 5.2 Resisting examples | PARTIAL | categories + counts + crops for the inspect half; confidence-floor analysis (unsaved) | Interpretation; confirmatory check on the test half; save the confidence-floor numbers |
| 5.3 500 more labels | PARTIAL | LC entry with conclusion; per-class holdout AP50 in `learning_curve.csv` | "Which classes benefit most/least" not written (per-class holdout curves: Box 0.236→0.509, Cargo 0.018→0.096, Flatbed 0.034→0.069, Tractor 0.000→0.007, Liquid 0.015→0.073 from 101→403 images, seed 0) |
| 5.4 Smallest subset ≥ 90% | PARTIAL | pre-registration (b02413c), runs, `figures/subset_compare/` | Conclusion; the author's prediction was never written (EXPERIMENTS S54 says "none provided"); no tested subset reached 0.128 |
| 6. Final analysis + next experiment | NOT STARTED | – | everything |
| 7. Deliverables | PARTIAL | eval/train code, configs, requirements, README, tests | REPORT.md incomplete; final weights not in the repo (Kaggle output only); STATUS.md stale; target 0.75 not reached |

§5.4 numbers (`figures/subset_compare/decision_rule.csv`, held-out):

| size | metric | smart | random s0 | random s1 | seed spread | smart − best random | rule: exceeds both by > spread |
|---|---|---|---|---|---|---|---|
| 202 | class-agnostic AP50 | 0.2841 | 0.2324 | 0.2213 | 0.0111 | +0.0517 | True |
| 202 | mAP50 | 0.1052 | 0.0948 | 0.0816 | 0.0132 | +0.0103 | False |
| 302 | class-agnostic AP50 | 0.3343 | 0.2934 | 0.3062 | 0.0128 | +0.0281 | True |
| 302 | mAP50 | 0.1040 | 0.1045 | 0.1230 | 0.0185 | −0.0189 | False |

The 90% threshold is 0.128 held-out mAP50. No 50% or 75% run reached it; the highest was f75_seed1 at 0.1230
(`figures/subset_compare/subset_compare.csv`).

---

## 7. Open hypotheses and planned next steps (as recorded)

- **B1h "Next steps"** (EXPERIMENTS.md): learning curves (since done), stronger data augmentation, and a separate classifier on
  cropped trucks for Cargo vs Box.
- **B1h conclusion:** the gap in "trucks found" on val vs holdout (66% vs 85%) has an unknown cause. Density is a candidate,
  but val miss rates did not vary with density, so this remains untested.
- **LC conclusion:** "More data should help, but additional labels alone are unlikely to achieve the target.
  Classification needs a separate improvement strategy."
- **Plan table** (EXPERIMENTS.md "Plan from here"): item 4, "Final analysis + report + README", is outstanding.
- **REPORT.md** is labelled "Technical report (in progress)". Sections "dataset observations, baselines, experiment chain,
  failure analysis, research investigations 5.1–5.4, final analysis" are noted there as to come.
- **§5.2:** the test half is reserved for confirming whatever is found on the inspect half. No hypothesis is written yet.
- **`analysis/notes/visual_inspection.md`:** possible label noise (unlabelled trucks; offset GT in 2470/2472), recorded as observations, not tested.
- **Code comments:** no TODO/FIXME in code. The only TODO is `EXPERIMENTS.md` line 37 (B0 Hypothesis).

---

## 8. Known issues, inconsistencies, unverified items

1. **The target is not met.** The best val mAP50 is 0.1065 against the 0.75 target.
2. **Run-to-run noise is large.** b1h_tile1024_holdout40 vs b1h_seed1 differ by 0.043 on val mAP50 and 0.017 on holdout. Single-run comparisons (e.g. B1 0.071 vs B1h 0.107) are within noise.
3. **B1 vs B1h are not a clean comparison.** They differ in platform/env (Colab torch 2.11 cu130 vs Kaggle 2.10 cu128), training set (443 vs 403 images), and eval max_det (334 vs 902).
4. **The Kaggle environment changed between 2 Oct and 3 Oct runs** (torch 2.10.0 / Python 3.12.13 → 2.11.0 / 3.13.15). The §5.4 smart and seed-1 runs therefore ran on a different stack from the random f50/f75 seed-0 runs they are compared with.
5. **B0 vs B1 is "equal epochs, not equal compute"** (1400 vs 11850 iterations; EXPERIMENTS.md).
6. **Ultralytics val cross-check** was only ever done on B0, where AP is near zero. Our scorer matches pycocotools on every run.
7. **Per-epoch Ultralytics val** during tiled training runs on un-sliced full images. It is curves only, not comparable with the reported metric.
8. **REPORT.md contradicts the configs.** It says `val: false`, but the configs set `val: true` (per-epoch curves only; `last.pt` is still reported).
9. **STATUS.md is stale** (Phase 2, pre-training).
10. **EXPERIMENTS.md has placeholder text:** B0 Hypothesis "TODO (me)"; B0/B1 "Results: Pending Colab"; "Next step: Pending results".
11. **`figures/b1_tile1024_maxdet3000/merge_sensitivity.csv` predates the merge_sensitivity fix.** Its `default` column is all False and it has no `max_det` column. The max_det used (3000) is inferred from the folder name; the exact command is UNKNOWN.
12. **The §5.2 never-detected confidence-floor numbers are not saved in any repo file** (see §5).
13. **§5.2 categories depend on the conf 0.25 threshold.** 79.3% of inspect-half "never-detected" boxes have an IoU ≥ 0.5 prediction at a lower confidence (unsaved analysis above).
14. **No §5.4 prediction was pre-registered.** The instruction contained an unfilled placeholder. smart50 results existed by 04:24 on 3 Oct, before any prediction could be added.
15. **Smart-selection coverage step.** "Add the unselected image with the most class-c boxes" is an implementation choice (`tools/make_smart_subsets.py` docstring). The pre-registration said "add images in order of rarest class contained". The runs used the implemented rule; the mismatch is disclosed in REPORT.md §5.4 and the S54 entry (4 Oct), and no author prediction was recorded.
16. **All learning-curve power-law fits are flagged unreliable.** The extrapolation to 903 images is 2.24× beyond the data, and some upper interval limits exceed 1 (`figures/learning_curve/power_law_fit.csv`).
17. **Kaggle GPU accounting is uncertain.** `kaggle quota` reported 0.00 h used on the morning of 3 Oct (refresh 2026-10-10). MORNING.md's ~8.3 GPU-h for the night is an upper-bound estimate from poll times.
18. **Weights are not in the repo.** Drive (B0/B1) availability is UNKNOWN. Kaggle kernel outputs hold the rest.
19. **The local dataset is a subset.** `data` points at `data_small` (20 train / 22 val). Do not re-run `analysis/eda.py` locally (it would overwrite full-data tables). Several tests use these local files.
20. **Exact command lines (`command.txt`) were not copied** into `results/`. (Corrected 2026-10-04: B1h's `train/args.yaml` is now at `results/b1h_tile1024_holdout40/train/args.yaml` and confirms the augmentation values.)
21. **Possible label issues** (`analysis/notes/visual_inspection.md`): unlabelled trucks, possibly offset GT boxes. Not quantified.
