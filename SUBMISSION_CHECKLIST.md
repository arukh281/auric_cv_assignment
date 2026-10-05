# Submission checklist

Source: `Computer_Vision_Assignment_Auric_AI_Technologies__1_.pdf` (in the repo folder, not tracked by git). All
quotes are verbatim from it. Status as of 2026-10-04 (*Update 5 Oct: all rows below re-checked; all experiments final*): **done** / **partial** / **pending** (waiting on a running
kernel or a later write-up) / **missing** (nothing exists yet).

**Format, length and packaging.** The brief says only "Concise technical report" and "Short README". It sets no
page or word limit, no file format, and no packaging or delivery method. UNKNOWN beyond that; ask the assessor if
it matters.

## Primary objective

| Requirement (verbatim) | Status | Where |
|---|---|---|
| "Develop an object detector that achieves at least 0.75 mAP50 on the provided validation set." | **missing (not met)**: best 0.1065 (*Update 5 Oct:* final system 0.1349) | REPORT.md summary; `results/b1h_tile1024_holdout40/eval/per_class.csv` |
| "the quality of your diagnosis, experimental reasoning and reproducibility will be evaluated alongside the score." | done (*Update 5 Oct*) | REPORT.md, DETAILED_EXPERIMENTS.md, README.md |

## 1. Dataset and Task

| Requirement (verbatim) | Status | Where |
|---|---|---|
| "Any external weights, code or tools must be documented clearly enough for us to reproduce the result." | done for B1h: COCO `yolo11s.pt` with sha256 in `init_weights.json`; Ultralytics 8.4.171. **pending** for E3 (DOTA `yolo11s-obb.pt`), whose sha256 is recorded by the kernel. *Update 5 Oct:* done, `results/e3_b1h_dota/init_weights.json` | REPORT.md "Pretrained weights"; `requirements-lock.txt`; `results/*/init_weights.json` |

## 2. Baseline, Dataset Observations and Reproducibility

| Requirement (verbatim) | Status | Where |
|---|---|---|
| "Study the supplied dataset before or alongside model development. Report the observations that materially influence your modeling or evaluation decisions." | done | REPORT.md §2.1; `figures/eda/` |
| "Establish a reasonable initial detection baseline and document the complete training setup required to reproduce it." | done | REPORT.md §2.2; `configs/b0.yaml`, `configs/b1.yaml`, `configs/b1h.yaml`; README "Reproduce everything" |
| "For every reported model, provide overall mAP50 and per-class AP50 on the validation set." | partial: B0/B1/B1h per-class in REPORT §2.2; E1/E2 per-class only in DETAILED_EXPERIMENTS.md; LC/S54 runs' per-class val AP50 in `figures/learning_curve/learning_curve.csv` and `results/<run>/eval/per_class.csv` but not in REPORT; E3/E4 **pending**. *Update 5 Oct:* done, every model in REPORT §4.1 | REPORT.md §2.2, §4; DETAILED_EXPERIMENTS.md index |
| "Include representative qualitative predictions and misclassification details. Do not limit the report to successful examples." | partial: grids and crops exist and are referenced; no images are embedded in REPORT.md | REPORT.md §2.3; `figures/<run>/grid_*.png`, `errors/crops_*.png`, confusion matrices |
| "Minimum reproducibility information: architecture and initialization, image size/preprocessing, train/validation usage, augmentations, optimizer and scheduler, batch size, epochs, sampling strategy, losses or class weighting if modified, random seed where applicable, inference thresholds/post-processing, software dependencies and hardware/runtime details." | done (by pointer) | REPORT.md "Reproducibility details"; `results/b1h_tile1024_holdout40/train/args.yaml`; `eval_args.json`; `env/*_hardware.json`; `requirements-lock.txt` |

## 3. Diagnose What the Detector Is Really Getting Wrong

| Requirement (verbatim) | Status | Where |
|---|---|---|
| "Investigate false positives and false negatives as populations, not only as isolated examples." | partial: TIDE bins, size/density/brightness slices, recall gap done; background-FP audit **pending** (`auric-fp-crop`). *Update 5 Oct:* done, §3.3b (`results/fp_audit/fp_audit.csv`) | REPORT.md §3.1, §3.3, §3.4; `figures/<run>/errors/slices.csv` |
| "Develop a useful failure taxonomy and quantify the dominant failure modes." | done | REPORT.md "Error taxonomy", §3.1 (`tide_dAP.csv`) |
| "Report class-confusion / misclassification behavior and identify any systematic patterns." | done | REPORT.md §3.2 (Cargo ↔ Box); `figures/*/gt_oracle/confusion_pool.csv`, `errors/confusion_matrix_*.csv` |
| "Where several explanations are plausible, state the alternatives and design an analysis or experiment that helps discriminate between them." | partial: recall-gap (size / density / resolution / scene), E1 vs overfitting, E2 scale; missing-labels hypothesis **pending** (FP audit). *Update 5 Oct:* done, §3.3b–c, §6.5 | REPORT.md summary point 4, §3.4; DETAILED_EXPERIMENTS.md |
| "Evidence standard: … Show how you established the claim, how large the effect is, and what competing explanation could produce a similar observation." | partial: applied in the summary and §3 (ruled out / no evidence for / not tested); not yet re-checked for §5.3–§5.4 wording | REPORT.md |

## 4. Experimentation Toward the Target

| Requirement (verbatim) | Status | Where |
|---|---|---|
| "Each experiment should follow a traceable research chain rather than being an isolated hyperparameter trial." | done up to E1/E2; E3/E4 **pending**. *Update 5 Oct:* done through E16 | REPORT.md §4; DETAILED_EXPERIMENTS.md |
| Experiment record fields: "Observation", "Hypothesis", "Changes vs. previous run", "Results", "Conclusion", "Next experiment setup (if required)" | done in DETAILED_EXPERIMENTS.md (six fields per entry, audit `ba8da62`; E3/E4 results in); E6–E8 pending (*Update 5 Oct:* E6b, E7 done; E6 superseded; E8 stopped, no result). A 5-minute human-readable summary is in EXPERIMENTS.md | DETAILED_EXPERIMENTS.md; EXPERIMENTS.md; REPORT.md §4 table |
| "Results: Quantitative and qualitative results, including relevant class-wise behavior." | partial: class-wise yes; qualitative only by figure pointers | DETAILED_EXPERIMENTS.md |

## 5. Research Investigations

"Use the validation set for the investigations below. … explicitly state the limits of what each experiment can
establish."

| Requirement (verbatim) | Status | Where |
|---|---|---|
| 5.1 "Design and perform an experiment that answers this question." | done (GT-box oracle on val + TIDE oracle) | REPORT.md §5.1; `figures/*/gt_oracle/` |
| 5.1 "Quantify how much error remains once localization uncertainty is removed or controlled." | done | REPORT.md §5.1 |
| 5.1 "Explain what the experiment still cannot establish." | done | REPORT.md §5.1 "What this experiment cannot establish" |
| 5.2 "Find a principled way to identify training instances that the system repeatedly struggles to learn." | done | REPORT.md §5.2; `analysis/training_dynamics.py` |
| 5.2 "Investigate whether these examples form one population or several distinct populations." | partial: categories by size/class; not framed explicitly as one-vs-several populations | REPORT.md §5.2 |
| 5.2 "Provide evidence for the mechanisms you believe are responsible. 'Hard examples' by itself is not an explanation." | partial: confidence floor (C4), size (C1), class instability (C2); descriptive floor numbers **pending** (`auric-s52-floor`). *Update 5 Oct:* done, REPORT §5.2 | REPORT.md §5.2 |
| 5.2 "Where possible, use the observed training behavior to make a prediction about examples you have not manually inspected, then test that prediction." | done: pre-registered C1–C5 (`95953f3`), all hold on the test half | `results/s52_confirm/test/claims.csv` |
| 5.3 "Estimate whether 500 additional labels would materially improve the system." | done | REPORT.md §5.3 |
| 5.3 "State the assumptions behind your estimate and the uncertainty in the conclusion." | done (six assumptions listed; instances-vs-images correction: +500 instances ≈ +0.006, within noise) | REPORT.md §5.3 |
| 5.3 "Identify which of the five classes are likely to benefit most and least, and support the ranking with evidence from your experiments." | done (Box most, Tractor least) | REPORT.md §5.3 |
| 5.3 "Distinguish, as far as the available evidence permits, between limited sample count and other possible performance limitations." | done (assumption 6) | REPORT.md §5.3 |
| 5.4 "Goal: Find the smallest training subset you can demonstrate recovers at least 90% of the full-data validation mAP50." | done (negative): no tested subset reaches 90% on val (0.0959) or on the pre-registered holdout40 rule (0.128); both are reported and the difference is disclosed; the val answer is weak (seed spread 0.043 > gap 0.011) | REPORT.md §5.4 |
| 5.4 "Declare the subset-selection method before training the reduced-data model." | done (`b02413c`; the implemented rule differs from its wording, disclosed) | REPORT.md §5.4 |
| 5.4 "Compare the selected subset against a random subset of the same size under a comparable training setup." | done (2 random seeds per size; torch-version caveat disclosed) | REPORT.md §5.4 |
| 5.4 "Characterize the examples that survive selection. What makes them useful?" | done: more boxes per image and rare-class-rich, same box sizes; confounded with box count (disclosed) | REPORT.md §5.4; `figures/subset_compare/subset_characterisation.csv` |
| 5.4 "Report subset size, class coverage and achieved performance." | done | REPORT.md §5.4 (class-coverage table) |
| 5.4 "Report the smallest successful subset you actually tested. Do not claim a mathematically proven minimum unless you can establish one." | done: none of the tested subsets succeeded | REPORT.md §5.4 |

## 6. Final Analysis

| Requirement (verbatim) | Status | Where |
|---|---|---|
| "What are the dominant limitations of the final detector?" | skeleton; TODO-FINAL (*Update 5 Oct:* done) | REPORT.md §6 |
| "Which conclusions are strongly supported, which remain plausible but unresolved, and which initial hypotheses were weakened or rejected?" | skeleton; TODO-FINAL (*Update 5 Oct:* done) | REPORT.md §6 |
| "What changed most between the initial and final system, and what evidence supports your explanation for that change?" | skeleton; TODO-FINAL (*Update 5 Oct:* done) | REPORT.md §6 |
| "What is the single highest-priority next experiment or data action you would take if given one additional working day? Explain why it has higher information or performance value than the alternatives." | skeleton with candidates; TODO-FINAL (*Update 5 Oct:* done) | REPORT.md §6 |

## 7. What to Submit

| Deliverable (verbatim) | Expectation (verbatim) | Status | Where |
|---|---|---|---|
| Concise technical report | "Dataset observations, baseline, experiment records, quantitative and qualitative results, failure analysis, research investigations, conclusions and prioritized next step." | partial (§6 and §7 pending; gaps above). *Update 5 Oct:* done | `REPORT.md` |
| Final model weights | "The exact checkpoint corresponding to your claimed validation result." | done: final ensemble in release `weights-final-v1` (3 checkpoints + SHA256SUMS); baseline B1h in `weights-b1h-v1` | README "Final system" |
| Inference and evaluation code | "A reproducible path from provided validation images to predictions and reported metrics." | done; reproducibility check done: CPU run on 2 val images reproduces the saved predictions (`results/predict_test/compare.json`) | `predict.py`, `eval.py` |
| Training code/configuration | "All configurations, scripts and commands required to reproduce the final training setup." | done | `train.py`, `configs/b1h.yaml`, `scripts/run_b1h.sh`, README |
| Environment/dependencies | "Requirements file, environment specification or equivalent, including key package versions." | done | `requirements.txt`, `requirements-lock.txt` |
| Run instructions | "Short README with commands and expected inputs/outputs." | done, but the README is long (older Colab/Kaggle sections follow the short ones) | `README.md` |

"Submission standard: A reviewer should be able to reproduce your claimed validation predictions and metrics without
reconstructing undocumented choices." Partial: the full path is documented. An end-to-end check (2 val images, CPU) reproduced the saved
predictions to within 3e-6 confidence (`results/predict_test/`). A full 22-image re-run has not been done. *Update 5 Oct:* done for the final system in a clean environment, val 0.1349 reproduced (`results/clean_repro_final/`).
