# Auric CV assignment: 5-class truck detector

YOLO11s (COCO-pretrained) on a 5-class overhead truck dataset. Target: mAP50 ≥ 0.75 on the provided val set.

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
| `EXPERIMENTS.md`, `REPORT.md` | Experiment log and report |

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
