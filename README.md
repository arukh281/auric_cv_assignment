# Finding trucks in satellite photos

## 1. What this is

I built a program that finds trucks in overhead satellite photos and sorts each one into five types: cargo, box,
flatbed, tractor and liquid tanker. The goal was a score of 0.75 (out of 1) on the official test of 22 photos. My final
system scores 0.135 (exactly 0.1349), so I did not reach the goal. The main reason is that, from only 403 training photos,
the program finds many of the trucks but often names the wrong type on photos it hasn't seen.

## 2. Read this first

| Document | Read it when you want… |
|---|---|
| [REPORT.md](REPORT.md) | The short report: about 10 minutes, every section linked to the full evidence |
| [EXPERIMENTS.md](EXPERIMENTS.md) | The story in plain words, with simple charts |
| [DETAILED_EXPERIMENTS.md](DETAILED_EXPERIMENTS.md) | The full log: every run, with its prediction written down *before* the result |
| [docs/REPORT_FULL.md](docs/REPORT_FULL.md) | All the evidence: full tables, sources and caveats |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Every command, for Colab, Kaggle and a laptop |

## 3. How I approached it

- **Looked at the data first:** photo sizes, truck sizes (most are about 22 pixels long), and how many of each type.
- **Built a simple baseline**, then a better one that cuts each photo into tiles so small trucks stay visible.
- **Found out why it failed:** I measured which mistakes cost the most. The biggest one was naming the wrong truck
  type.
- **Tested one idea at a time:** each experiment changed one thing compared with the baseline.
- **Wrote my prediction down before each run**, with a rule for what would count as success.
- **Chose the final model on a practice set** of 40 training photos I held back, and scored the official test only
  once for it.

## 4. Codebase map

```
.
├── README.md           this file
├── REPORT.md           the short report
├── EXPERIMENTS.md      the story, in plain words
├── DETAILED_EXPERIMENTS.md   the full experiment log
├── requirements.txt    packages to install (requirements-lock.txt: exact versions used)
├── train.py            train one model from a config file
├── eval.py             score a model: photos → predictions → score
├── predict.py          run the final system (or any model) on a folder of photos
├── configs/            one settings file per experiment (b1h.yaml is the baseline)
├── detlib/             shared code: tiling, merging, scoring
├── analysis/           the diagnosis scripts (mistake types, charts, audits)
├── tools/              data preparation: tiles, practice-set split, subsets
├── scripts/            run scripts for Colab and Kaggle
├── tests/              known-answer tests for scoring, tiling and analysis
├── splits/             the fixed lists of held-back and subset photos
├── results/            numbers from every run (CSV/JSON, no model files)
├── figures/            charts; figures/simple/ holds the plain-language ones
├── notebooks/          Colab and Kaggle notebooks
└── docs/               full report, command reference, submission checklist, working notes
```

| Where to look for… | Files |
|---|---|
| Training | `train.py` + `configs/` |
| Scoring | `eval.py` + `detlib/scoring.py` |
| The final model | `predict.py` + section 5 below (weights and checksums) |
| A specific experiment | `configs/<name>.yaml` + `results/<name>/` + its entry in DETAILED_EXPERIMENTS.md |
| The diagnosis | `analysis/` |
| Charts | `figures/simple/` |
| Tests | `tests/` |

## 5. Run it

**Setup.** You need Python 3.11–3.13, about 3 GB of disk and the assignment's dataset (`cv_dataset.zip`, not
included here). No graphics card is needed.

```bash
git clone https://github.com/arukh281/auric_cv_assignment.git && cd auric_cv_assignment
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt     # requirements-lock.txt lists the exact versions I used (Python 3.12.13)
unzip -q /path/to/cv_dataset.zip -d data_unzipped
ln -s "$(dirname "$(find "$PWD/data_unzipped" -name classmap.txt | head -1)")" data
```

**Run the final system** (three models voting; weights from the public release `weights-final-v1`):

```bash
U=https://github.com/arukh281/auric_cv_assignment/releases/download/weights-final-v1
for f in final_b1h_epoch040.pt final_e4_epoch040.pt final_e7_epoch020.pt SHA256SUMS.txt; do curl -L -o $f $U/$f; done
sha256sum -c SHA256SUMS.txt          # macOS: shasum -a 256 -c SHA256SUMS.txt
python predict.py --weights final_b1h_epoch040.pt final_e4_epoch040.pt final_e7_epoch020.pt --multi-label \
  --images data/val/images --labels data/val/labels --out out_final --device cpu
```

**Expected output:** the last line is `[predict] mAP50 0.1349 (COCO 101-point, conf >= 0.001)`. "mAP50" is the
score: the average, over the five types, of how well boxes are found and named. Per type (`out_final/per_class.csv`):
Cargo 0.1713, Box 0.2145, Flatbed 0.0960, Tractor 0.0277, Liquid 0.1651. On a CPU this takes about 22 minutes; a
clean-room run reproduced these numbers exactly (`results/clean_repro_final/`).

**Retrain one model** (the baseline, B1h). This needs a GPU; it took about 1.4–1.9 hours on a Kaggle T4:

```bash
DATA_ROOT=$PWD/data bash scripts/run_b1h.sh     # train, then score on the official test and the practice set
```

Other experiments run the same way from their config; see [docs/REPRODUCE.md](docs/REPRODUCE.md).

## 6. What I did not do, and limits

- **The target was missed by a wide margin:** 0.135 against 0.75.
- **Each experiment was trained once.** Training the same model twice moves the score by about 0.02 on the practice
  set and 0.04 on the official test, so small differences are not meaningful.
- **The official test has only 22 photos**, and 8 of them were altered (resized, noised, blurred or contrast-changed)
  compared with their public originals.
- **I did not try** a bigger model, a different kind of detector, or learned super-resolution, mostly because of the
  GPU budget (about 30 hours a week).
- **I did not use models pretrained on xView**, the public dataset these photos come from, because they would already
  have seen the official test's answers.

## 7. How AI was used

I built this with AI help: Claude Code wrote code, ran analyses and drafted text, and a chat assistant helped plan
experiments. Predictions and decision rules marked "author's" were mine, set before each run; other interpretations
were drafted by Claude Code and reviewed by me. Details are in [docs/REPORT_FULL.md](docs/REPORT_FULL.md).
