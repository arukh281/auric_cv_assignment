# Auric CV assignment: 5-class truck detector

YOLO11s (COCO-pretrained) on a 5-class overhead truck dataset. Target: mAP50 ≥ 0.75 on the provided val set.

**Result:** the target was not reached. Best val mAP50 is 0.1065 (95% CI 0.0563–0.1653), from B1h. Why: see
`REPORT.md` (summary at the top) and `DETAILED_EXPERIMENTS.md` (one entry per run, index at the top).

## Final model and prediction (deliverable)

- **Weights:** GitHub release
  [`weights-b1h-v1`](https://github.com/arukh281/auric_cv_assignment/releases/tag/weights-b1h-v1), file
  `b1h_tile1024_holdout40_last.pt`. SHA-256 `3fa2406665706b9f44a155ca8b411b73e1c9eb4ffdc58ab03c4cb0e5bae37ffb`, the
  same as `weights_sha256` in `results/b1h_tile1024_holdout40/eval/metrics.json`. Weights are not in git history.
- **Environment:** `requirements.txt` is the installable set used by the setup scripts. `requirements-lock.txt`
  holds the exact versions of the final model's training session: Ultralytics 8.4.171, torch 2.10.0+cu128,
  torchvision 0.25.0, numpy 2.0.2, Python 3.12.13, plus the full `pip freeze`.

### Reproduce the reported val mAP50 (0.1065) from scratch

You need Python 3.11–3.13, `git`, `curl`, about 3 GB of disk, and the assignment's dataset (`cv_dataset.zip`; it
is not redistributed here). No GPU is needed. Every command below runs from a terminal, and none needs a GitHub login.

```bash
# 1. code
git clone https://github.com/arukh281/auric_cv_assignment.git
cd auric_cv_assignment

# 2. environment (CPU wheels; on an NVIDIA GPU use --index-url https://download.pytorch.org/whl/cu128 instead)
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. final weights (public GitHub release, no login) and checksum
curl -L -o b1h_tile1024_holdout40_last.pt \
  https://github.com/arukh281/auric_cv_assignment/releases/download/weights-b1h-v1/b1h_tile1024_holdout40_last.pt
sha256sum b1h_tile1024_holdout40_last.pt   # macOS: shasum -a 256
# expected: 3fa2406665706b9f44a155ca8b411b73e1c9eb4ffdc58ab03c4cb0e5bae37ffb

# 4. data: unzip the dataset so that data/ contains classmap.txt, val/images/*.png and val/labels/*.txt
unzip -q /path/to/cv_dataset.zip -d data_unzipped
ln -s "$(dirname "$(find "$PWD/data_unzipped" -name classmap.txt | head -1)")" data
ls data   # expected: classmap.txt  data.yaml  train  val

# 5. predict + score all 22 val images
python predict.py --weights b1h_tile1024_holdout40_last.pt --images data/val/images --labels data/val/labels \
  --out out_val --device cpu
```

**Expected output:**
- The last line printed is `[predict] mAP50 0.1065 (COCO 101-point, conf >= 0.001)`.
- `out_val/per_class.csv`'s `AP50_coco` column equals `results/b1h_tile1024_holdout40/eval/per_class.csv`:
  Cargo 0.1306, Box 0.1797, Flatbed 0.0711, Tractor 0.0062, Liquid 0.1447.
- `out_val/predictions.csv` holds image, cls, conf, x1, y1, x2, y2 in full-image pixels;
  `out_val/predictions_raw.csv` holds the tile predictions before merging; `out_val/metrics.json` holds the summary.

`predict.py` uses exactly the tiling, merge and scorer of every reported number: the `eval:` section of
`configs/b1h.yaml`, that is 1024 tiles with overlap 256, conf 0.001, class-wise NMS on IoS 0.6 and max_det 902. It
computes no bootstrap CIs; those come from `eval.py`. To get predictions only, for any folder of images, drop
`--labels`.

**Clean-room check:** the CPU-only Kaggle kernel `auric-clean-repro` ran exactly these commands in a fresh venv.
- Result: mAP50 0.1065, with per-class AP50 identical to the reported values.
- Time: 8.4 min in total, about 6 min of it for prediction on CPU.
- Kaggle-only adjustments: its Python lacks `ensurepip`, so step 2 used
  `python3 -m venv --without-pip .venv && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python`; and its
  dataset arrives already unzipped.
- Details: `results/clean_repro/`, DETAILED_EXPERIMENTS.md "Clean-room reproducibility".
- If `python3 -m venv` fails on your machine with an `ensurepip` error, use the same workaround.

## Reproduce everything

Every compute step ran on Kaggle, launched from a laptop with the Kaggle CLI.
- Code goes up as a dataset: `bash scripts/kaggle_cli_package.sh`, which packages committed HEAD.
- Each step is a kernel: `bash scripts/kaggle_cli_kernel.sh full <config> --push` (GPU training), or
  `bash scripts/kaggle_cli_script_kernel.sh <slug> '<command>' [--source <user>/<kernel>] [--cpu] --push` (any
  command; `--source` mounts an earlier kernel's output, such as its weights, under `/kaggle/input`).
- Each kernel runs `scripts/kaggle_setup.sh` first: data to `/kaggle/tmp/data`, all `tests/test_*.py`.
- Paths inside a kernel come from `scripts/_env.sh`: `$DATA_ROOT`, `$RUNS`, `$FIGS`, `$PY`. The same commands run on any
  machine with `DATA_ROOT` pointing at the dataset root (`classmap.txt train/ val/`).

| Step | Command (inside the repo, after setup) | Outputs |
|---|---|---|
| Splits (committed) | `python tools/make_holdout.py --n 40 --seed 0`; `python tools/make_subsets.py --seed 0 --write-configs` | `splits/` |
| EDA | `python analysis/eda.py` (**full dataset only**) | `figures/eda/` |
| B0 / B1 (Colab) | `bash scripts/run_b0.sh`, `bash scripts/run_b1.sh` | `runs/b0_full640`, `runs/b1_tile1024` |
| B1h (final model) | `bash scripts/run_b1h.sh` then `bash scripts/run_analysis.sh b1h_tile1024_holdout40` | train, val/holdout40 evals, checkpoint curve, errors, GT-box oracle |
| train40 eval | `python eval.py --config configs/b1h.yaml --data-root "$DATA_ROOT" --runs-root "$RUNS" --split train --max-images 40 --sample-seed 0 --out "$RUNS/b1h_tile1024_holdout40/eval_train40"` | `eval_train40/` |
| Learning curve (§5.3) | `bash scripts/run_lc.sh configs/b1h_{f25,f50,f75,seed1}.yaml` (one kernel each), then `python analysis/learning_curve.py` | `figures/learning_curve/` |
| §5.4 smart subsets | GPU: `python tools/make_smart_subsets.py embed --data-root "$DATA_ROOT" --out <dir>`; CPU: `python tools/make_smart_subsets.py select --embeddings results/s54/embeddings.npz --write-configs`; `bash scripts/run_lc.sh configs/b1h_{smart50,smart75,f50_seed1,f75_seed1}.yaml`; `python analysis/subset_compare.py` | `figures/subset_compare/` |
| §5.1 GT-box oracle | `python analysis/gt_box_oracle.py --config configs/b1h.yaml --runs-root "$RUNS" --data-root "$DATA_ROOT"` | `figures/<run>/gt_oracle/` |
| §5.2 training dynamics | `python analysis/training_dynamics.py --weights-dir <B1h weights dir> --data-root "$DATA_ROOT" --out <dir>`; confirmation: `python analysis/s52_confirm.py --per-box <dir>/per_box_{inspect,test}.csv --preds <dir>/preds/predictions_ep050.csv.gz --out <dir2>` | `results/s52/`, `figures/s52/` |
| Error bins (TIDE) | `python analysis/errors.py --preds <run>/eval/predictions.csv --name <run> --data-root "$DATA_ROOT"` | `figures/<run>/errors/` |
| Class-agnostic AP | `python analysis/class_agnostic.py --preds <run>/<eval dir>/predictions.csv --data-root "$DATA_ROOT"` | `class_agnostic.json` |
| Merge sensitivity | `python analysis/merge_sensitivity.py --raw <run>/eval/predictions_raw.csv --name <run>` | `figures/<run>/merge_sensitivity.csv` |
| Per-image / domain shift | `python analysis/per_image.py`; `python analysis/domain_shift.py --data-root data_small` | `figures/<run>/per_image/`, `figures/domain_shift/` |
| Sanity checks | `python analysis/sanity_check.py --data-root "$DATA_ROOT" --tiles <tiles dir> --out <dir>` (GPU; `--labels-only` for the CPU label check); `python analysis/label_mismatch.py ...` | `results/sanity/`, `figures/sanity/` |
| E1 / E2 | `bash scripts/run_lc.sh configs/e1_b1h_150ep.yaml` (and `e2_...`) + `run_analysis.sh` | `results/e{1,2}_*` |
| E3 / E4 | `bash scripts/run_e34.sh configs/e3_b1h_dota.yaml` (and `e4_b1h_flipud_mixup.yaml`) | val/holdout40/train40 on `last.pt`, holdout checkpoint curve |
| FP audit / crop classifier | `python analysis/fp_audit.py --preds <B1h>/eval_holdout40/predictions.csv --data-root "$DATA_ROOT" --out <dir>`; `python analysis/crop_classifier.py --data-root "$DATA_ROOT" --holdout splits/holdout40_seed0.txt --preds <B1h>/eval_holdout40/predictions.csv --out <dir>` | contact sheet, classifier results |
| Val vs holdout recall gap | `python analysis/recall_gap.py --val-preds <B1h>/eval/predictions.csv --holdout-preds <B1h>/eval_holdout40/predictions.csv --holdout splits/holdout40_seed0.txt --data-root "$DATA_ROOT" --out <dir>` | `standardised.csv`, `recall_by_factor.csv` |

Each kernel used is named in its `DETAILED_EXPERIMENTS.md` entry, together with its code commit. Results (CSV/JSON/PNG, no
weights) are copied into `results/<run>/` and `figures/<run>/`.

## Layout
| Path | What |
|---|---|
| `train.py` | Train one config. Resumable: re-running the same command continues from `last.pt` |
| `eval.py` | Val images → predictions → metrics (full-image or sliced). Same scorer for every model |
| `tools/make_tiles.py` | Tiles the TRAIN split at native resolution |
| `analysis/eda.py` | Phase 1 dataset analysis. **Needs the full dataset; do not re-run on a subset** (it overwrites `figures/eda/`) |
| `analysis/pred_review.py` | TP/FP/FN grids, confusion matrix with background |
| `detlib/` | Shared tiling, scoring, merging, environment logging |
| `configs/` | `b0.yaml`, `b1.yaml`, `data.yaml` |
| `notebooks/colab_train.ipynb` | Colab driver for B0 and B1 |
| `tests/test_pipeline.py` | Known-answer tests: scoring, tiling, merging |
| `tests/test_analysis.py` | Known-answer tests: TIDE error types and oracle fixes, GT-box oracle |
| `analysis/errors.py` | Phase 3: TIDE-style error bins + oracle dAP50, sliced FN/FP rates, confusion, crops (reads saved predictions) |
| `analysis/gt_box_oracle.py` | 5.1: classify GT boxes from the raw head's class scores, pooled and best-anchor modes (needs weights + GPU) |
| `scripts/run_analysis.sh` | Runs both analysis scripts for a finished run on Colab or Kaggle; log in `<run>/analysis.log` |
| `scripts/kaggle_setup.sh`, `scripts/kaggle_push_results.sh`, `notebooks/kaggle_train.ipynb` | Kaggle: setup, push of small result files, end-to-end notebook |
| `runs/<run>/` | Config, command, env, metrics, training curves per run. Not in git (`.gitignore`); kept on Drive with the weights |
| `analysis/merge_sensitivity.py` | Re-scores saved raw tile predictions under other merge settings (no re-inference) |
| `DETAILED_EXPERIMENTS.md`, `REPORT.md` | Experiment log and report |

## Dataset
Expected layout (the `cv_dataset.zip` root):
```
classmap.txt  data.yaml  train/images  train/labels  val/images  val/labels
```
Labels are YOLO txt (`cls xc yc w h`, normalized). Locally, `data/` is a symlink to the dataset folder.

## Run on Colab from the Terminal (main workflow)
Paths default to `/content/drive/MyDrive/auric/{cv_dataset.zip,runs}` (see `scripts/_env.sh`; override any by
exporting it first).

1. In a notebook cell, once per session (approve the popup): `from google.colab import drive; drive.mount('/content/drive')`.
2. Open the Terminal (left bar) and clone the repo. Colab secrets are not visible in the terminal, so paste a
   GitHub token at the silent prompt. It is not echoed, and it is removed from the git config straight after cloning:
   ```bash
   cd /content && read -rsp "GitHub token: " T && echo && \
     git clone -q "https://$T@github.com/arukh281/auric_cv_assignment.git" repo; unset T; \
     cd repo && git remote set-url origin https://github.com/arukh281/auric_cv_assignment.git && git log -1 --oneline
   ```
   If `/content/repo` already exists in this session: `cd /content/repo && git log -1 --oneline` (pulling also needs the token).
3. Setup: installs the requirements without touching Colab's torch, prepares `/content/data` from the Drive zip,
   writes `runs/colab_env_lock.txt` and runs the tests. It is safe to re-run; it ends with `SETUP OK`.
   ```bash
   bash scripts/colab_setup.sh
   ```
4. Train and evaluate, B1 first. `nohup` keeps the run alive if the terminal tab closes. Each script trains (or resumes
   from `last.pt` on Drive, or skips training if the run already finished), then runs `eval.py`, the prediction
   review, merge sensitivity (B1) and `checkpoint_curve.py`. The log goes to `runs/<run>/run.log` on Drive.
   ```bash
   nohup bash scripts/run_b1.sh > /dev/null 2>&1 &
   bash scripts/status.sh            # epoch, losses, time left, live batch progress, GPU, running processes
   tail -f /content/drive/MyDrive/auric/runs/b1_tile1024/run.log
   # after B1 finishes:
   nohup bash scripts/run_b0.sh > /dev/null 2>&1 &
   ```
5. **After a disconnect or the ~4h50m free-tier cap:** start a new session, mount Drive (step 1), then re-run steps
   2–4. Setup skips what is already done, and the run script resumes from the last saved epoch.

## Run on Colab (notebook cells, alternative)
**Environment:** a Colab **T4 GPU (16 GB)** runtime on **Python 3.13**.
- Colab's preinstalled CUDA torch/torchvision are used as-is and **never reinstalled**. The install cell runs
  `pip install -r requirements.txt -c <constraints pinned to Colab's own torch, torchvision, numpy, opencv>`.
- The exact environment is saved twice:
  - `DRIVE_RUNS/colab_env_lock.txt` (full `pip freeze`)
  - each run's `env/*_pip_freeze.txt` and `env/*_hardware.json` (Python, torch, CUDA, cuDNN, GPU name e.g.
    "Tesla T4", GPU memory, Ultralytics version, git commit)
- To reproduce: use the same Colab runtime, install with that lock file, and keep torch as shipped.

**Training:**
- AMP (fp16) is on.
- `last.pt` is written to Drive every epoch, so a disconnect loses at most one epoch.
- If batch 16 does not fit on the T4, `train.py` retries at batch 8 and records the batch actually used.

**Steps:**
1. Upload `cv_dataset.zip` to Google Drive (the notebook default is `MyDrive/auric/cv_dataset.zip`). Check that it
   shows the full size (~6.1 GB).
2. Push this repo to GitHub. For a private repo, add a GitHub token as a Colab secret named `GH_TOKEN`
   (key icon in the left bar), and turn on notebook access for this notebook.
3. Open `notebooks/colab_train.ipynb` in Colab and choose Runtime → Change runtime type → **T4 GPU**.
4. Edit the first cell: `DRIVE_DATA`, `DRIVE_RUNS`, `REPO_URL`, `BRANCH`.
5. Run the setup cells, then the tests. The tests must print `tests passed`.
6. Run the **1-epoch timing probe** and its estimate cell. It reports seconds per epoch, the estimated hours for
   100 epochs of B0 and of B1, and the number of tiles B1 trains on. **If B1 is over ~8 h, stop and choose a
   schedule first.**
7. Run B0, then B1 (each: train cell, then eval cell). Expected outputs under `DRIVE_RUNS`:
   - `b0_full640/` and `b1_tile1024/`: `config.yaml`, `command.txt`, `env/`, `train/weights/last.pt`,
     `training_summary.json`, `training_curves.png`, `eval/metrics.json`, `eval/per_class.csv`,
     `eval/scorer_comparison.csv`, `eval/bootstrap_ci.csv`, `eval/pr_curves.png`
   - `figures/<run>/`: TP/FP/FN grids, confusion matrix, and `merge_sensitivity.csv` (B1)
   - `../runs_export.zip`: everything except weights
8. **After a disconnect:** reconnect, re-run the setup cells, then re-run the same training cell. It resumes from
   the last saved epoch on Drive. B1 tiles are rebuilt identically from the seed.

## Run on Kaggle (end to end, background mode)

`scripts/_env.sh` picks the platform by itself. It uses Kaggle paths if `KAGGLE_KERNEL_RUN_TYPE` is set or
`/kaggle/input` exists, and the Colab paths otherwise, which are unchanged.

| | Colab | Kaggle |
|---|---|---|
| dataset source | `/content/drive/MyDrive/auric/cv_dataset.zip` | `/kaggle/input/<dataset>/cv_dataset.zip`, or the extracted folder Kaggle makes from it |
| `DATA_ROOT` | `/content/data` | `/kaggle/tmp/data` |
| `RUNS` (`FIGS` = `$RUNS/figures`) | `/content/drive/MyDrive/auric/runs` | `/kaggle/working/runs` (saved as notebook output) |
| `WORK_DIR` | `/content/work` | `/kaggle/tmp/work` |
| setup | `scripts/colab_setup.sh` | `scripts/kaggle_setup.sh` |
| env lock | `$RUNS/colab_env_lock.txt` | `$RUNS/kaggle_env_lock.txt` |

**Steps:**
1. Create a Kaggle notebook from `notebooks/kaggle_train.ipynb`.
2. Add the dataset with **Add Data**.
3. Add a secret `GH_TOKEN` (a GitHub token with push access) under **Add-ons → Secrets** and attach it to the
   notebook.
4. Set Accelerator = GPU and Internet = On.
5. Set `RUN = "b0"` or `"b1"` in the first code cell.
6. Choose **Save Version → Save & Run All**.

**What the notebook does:**
1. Clones the repo to `/kaggle/tmp/repo`.
2. Runs `scripts/kaggle_setup.sh`, which:
   - `git pull`s with the secret;
   - installs `requirements.txt` with Kaggle's torch / torchvision / numpy / opencv pinned as constraints, the same
     logic as Colab;
   - prepares `/kaggle/tmp/data`;
   - writes the env lock;
   - runs every `tests/test_*.py` and stops unless all pass.
3. Runs `scripts/run_<RUN>.sh`, which trains and evaluates as on Colab.
4. Optionally runs `scripts/run_analysis.sh`.
5. Runs `scripts/kaggle_push_results.sh <run>`, which copies CSV/JSON/PNG files (no `weights/`, nothing over 20 MB)
   into `results/<run>/` and `figures/<run>/`, commits, and pushes.

The token is read with `kaggle_secrets` inside each command and is never printed or written to `.git/config`.
Weights stay in the notebook's output, under `/kaggle/working/runs/<run>/train/weights/`.

**Limits:** a Kaggle session is capped at 12 h. A new session starts with an empty `/kaggle/working`, so `last.pt`
resume across sessions only works if the previous version's output is attached as input and copied to
`/kaggle/working/runs` first.

## Kaggle via CLI (from the Mac, no GitHub token on Kaggle)

1. Install the CLI with `uv pip install -p .venv -U kaggle`. It authenticates from `~/.kaggle`; the token is never
   copied into this repo or a kernel.
2. Data: the private dataset `aradhya1211/auric-cv-dataset`. `scripts/kaggle_setup.sh` copies its extracted folder
   (or unzips `cv_dataset.zip`) to `/kaggle/tmp/data`.
3. Code: `bash scripts/kaggle_cli_package.sh`. It refuses uncommitted changes. It `git archive`s HEAD without `results/`
   and with only `figures/eda` of `figures/`, adds a `CODE_COMMIT` file, and uploads `code.tar.gz` as the private
   dataset `aradhya1211/auric-cv-code`. The first time it is created; after that a new version is made with the
   commit hash as the message.
4. Kernel: `bash scripts/kaggle_cli_kernel.sh smoke|full --push` builds `kaggle_build/kernel_<mode>/` (gitignored)
   and pushes it.
   - Kernels: `aradhya1211/auric-b1h-smoke` or `aradhya1211/auric-b1h`.
   - Settings: private, GPU `machine_shape: NvidiaTeslaT4` (T4 x2), internet on, both datasets attached.
5. What the kernel's entry point (`scripts/kaggle_kernel_run.py`) does:
   - copies the code to `/kaggle/working/repo`;
   - runs `kaggle_setup.sh` with `SKIP_PULL=1`;
   - `smoke`: 1 epoch on the tiles of the first 10 train images (47 tiles expected);
   - `full`: `scripts/run_b1h.sh` then `scripts/run_analysis.sh b1h_tile1024_holdout40`.
   - The commit goes to `<run>/code_commit.txt`, and `env/*_hardware.json` records it from `CODE_COMMIT`.
6. Monitor with `.venv/bin/kaggle kernels status aradhya1211/auric-b1h`. When it finishes, run
   `.venv/bin/kaggle kernels output aradhya1211/auric-b1h -p <scratch folder outside the repo>`. Copy only
   CSV/JSON/PNG files under 20 MB into `results/` and `figures/`. Weights stay in the kernel output, under
   `runs/b1h_tile1024_holdout40/train/weights/`.

Without `.git` on Kaggle, the byte-identical `eval.py` test in `tests/test_diagnostics.py` skips; it needs the
old commit's history.

## Run locally
```bash
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt torch torchvision
.venv/bin/python tests/test_pipeline.py                     # known-answer tests
.venv/bin/python tests/test_analysis.py                     # Phase 3 analysis known-answer tests
# after a run: python analysis/errors.py --preds runs/<name>/eval/predictions.csv --name <name>
#              python analysis/gt_box_oracle.py --config configs/<cfg>.yaml --runs-root <runs>
.venv/bin/python train.py --config configs/b0.yaml          # data/ = dataset root, outputs to runs/b0_full640
.venv/bin/python eval.py  --config configs/b0.yaml          # evaluates runs/b0_full640/train/weights/last.pt
.venv/bin/python eval.py  --config configs/b1.yaml --weights path/to/last.pt   # any checkpoint, sliced
.venv/bin/python analysis/pred_review.py --preds runs/b0_full640/eval/predictions.csv --name b0_full640
```

### After a run finishes: Phase 3 analysis and bringing results back

Run these in the Colab Terminal after `run.log` shows `<run> DONE`. Drive must be mounted from a notebook cell, and
`/content/data` must be prepared (re-run `bash scripts/colab_setup.sh` in a new session). Example for `b1_tile1024`:

```bash
# (a) update the clone (the token is used once, never stored), then run the analysis
cd /content/repo && read -rsp "GitHub token: " T && echo && \
  git pull -q "https://$T@github.com/arukh281/auric_cv_assignment.git" main; unset T; git log -1 --oneline
nohup bash scripts/run_analysis.sh b1_tile1024 > /dev/null 2>&1 &
tail -f /content/drive/MyDrive/auric/runs/b1_tile1024/analysis.log      # Ctrl-C stops watching, not the job

# (b) after "analysis b1_tile1024 DONE": copy CSV/JSON/PNG (no weights, nothing over 20 MB) into the repo, push
cd /content/repo && R=b1_tile1024 && A=/content/drive/MyDrive/auric/runs && \
  rsync -a --prune-empty-dirs --max-size=20m --exclude='weights/' --include='*/' \
    --include='*.csv' --include='*.json' --include='*.png' --exclude='*' "$A/$R/" "results/$R/" && \
  rsync -a --prune-empty-dirs --max-size=20m --include='*/' \
    --include='*.csv' --include='*.json' --include='*.png' --exclude='*' "$A/figures/$R/" "figures/$R/" && \
  find "$A/$R" "$A/figures/$R" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.png' \) -size +20M \
    -printf 'SKIPPED (over 20 MB): %p\n' && du -sh "results/$R" "figures/$R"
git add "results/$R" "figures/$R" && \
  git -c user.name="aradhya khandelwal" -c user.email="arukhandelwal281@gmail.com" \
    commit -q -m "Results: $R (CSV/JSON/PNG from Drive)" && git log -1 --oneline
read -rsp "GitHub token: " T && echo && \
  git push -q "https://$T@github.com/arukh281/auric_cv_assignment.git" HEAD:main; unset T
```
Then `git pull` locally. Nothing on Drive is deleted or modified; rsync only reads from it.

Re-score saved predictions without inference: `eval.py --config ... --from-preds <predictions.csv>`.
