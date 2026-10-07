# Morning report: overnight run of 2–3 Oct 2026

Facts only. Times are IST, taken from my 5-minute status polls, so a "done" time is when a poll first saw COMPLETE
(the real finish is up to 5 min earlier).

## What ran

| Kaggle kernel | Purpose | Code | Running | Done | Est. GPU h |
|---|---|---|---|---|---|
| auric-s52-smoke | §5.2 smoke, 5 images | 883d884 | ~02:12 | ≤02:29 | ≤0.3 |
| auric-s54-embed | §5.4 DINOv2-small embeddings, 402 images | 883d884 | ~02:12 | ≤02:29 | ≤0.3 |
| auric-s52 | §5.2 full: 403 images × 5 B1h checkpoints | 883d884 | 02:32 | 03:13 | ≤0.7 |
| auric-b1h-smart50-smoke | §5.4 1-epoch smoke (41 tiles) | f296537 | 02:34 | 02:39 | ≤0.1 |
| auric-b1h-smart50 | §5.4 full | f296537 | 02:43 | 04:24 | ≤1.7 |
| auric-b1h-smart75 | §5.4 full | f296537 | 03:13 | 05:05 | ≤1.9 |
| auric-b1h-f50-seed1 | §5.4 full | f296537 | 04:24 | 06:01 | ≤1.6 |
| auric-b1h-f75-seed1 | §5.4 full | f296537 | 05:05 | 06:50 | ≤1.75 |

- Every log contains SETUP OK and the run's DONE marker, with no Traceback and no FAILED line.
- All tests (47 in 10 files) passed in each kernel's setup and locally before each push.
- GPU was 2 × Tesla T4 (machine_shape NvidiaTeslaT4).

**Kaggle GPU hours this week**
- `kaggle quota` at ~07:00 IST: GPU 0.00 h used, 30.00 h remaining, refresh 2026-10-10T00:00:00. Tonight's runs started
  on 2 Oct UTC, so the quota week appears to have rolled over at 3 Oct 00:00 UTC. I cannot confirm that from the CLI.
- My estimate for tonight is about 8.3 GPU-hours (upper bounds above). That is above the ~5–6 h you estimated:
  each of the four training runs took 1.6–1.9 h, plus about 1.4 h for §5.2, the smokes and the embeddings.

## What failed or needed attention (no GPU run failed)

1. **S54 prediction.** The instructions contained the unfilled placeholder `<<< WRITE YOUR GUESS HERE ... >>>`, so the
   pre-registration records "Prediction: none provided".
   - At ~02:35 you asked to add your prediction. I asked for the text and have not received it, so nothing was
     added.
   - b1h_smart50 finished at 04:24. Its results were committed (40067b8) but not viewed by the author.
   - A prediction added now can no longer be labelled "before any §5.4 result existed".
2. **Background poller.** It hit the 2 h background limit 3 times. Each time I restarted it from its state file, and
   no kernel was affected.
3. **My fetch script.** It failed once because of my quoting; I fixed it and reran. No data was affected.
4. **Kernel code finder.** Mounting B1h's output (kernel_sources) also mounts B1h's old repo/ copy. The kernel now
   prefers the packaged code (code.tar.gz), and the logs confirm the intended commits ran.

## Part A: §5.2 training dynamics (inspect half only)

- **Images:** 403 non-holdout train images, split before any computation into inspect (202) and test (201), seed 0
  (b02413c). The test half has per-box rows in results/s52/per_box_test.csv and is not summarised or plotted.
- **Inference:** the B1h checkpoints (epochs 10/20/30/40/50, sha256 in results/s52/training_dynamics_args.json), with
  eval.py's sliced inference and merge, max_det 902.
- **Category per GT box:** derived from detected (any class, IoU ≥ 0.5, conf ≥ 0.25) and correct (the
  top-confidence such prediction has the GT class).

Inspect half, 3,511 GT boxes:

| | learned-early | learned-late | forgotten | never-learned | never-detected | total |
|---|---|---|---|---|---|---|
| all | 169 | 1547 | 266 | 241 | 1288 | 3511 |
| Cargo Truck | 1 | 831 | 45 | 66 | 846 | 1789 |
| Truck w/Box | 167 | 385 | 183 | 46 | 195 | 976 |
| Truck w/Flatbed | 1 | 137 | 12 | 60 | 123 | 333 |
| Truck Tractor | 0 | 188 | 26 | 40 | 95 | 349 |
| Truck w/Liquid | 0 | 6 | 0 | 29 | 29 | 64 |

| box size (sqrt area, px) | learned-early | learned-late | forgotten | never-learned | never-detected | total |
|---|---|---|---|---|---|---|
| [0, 8) | 0 | 0 | 0 | 0 | 17 | 17 |
| [8, 16) | 0 | 258 | 12 | 25 | 591 | 886 |
| [16, 24) | 0 | 604 | 65 | 78 | 440 | 1187 |
| [24, 32) | 10 | 372 | 57 | 70 | 168 | 677 |
| [32, 48) | 118 | 285 | 116 | 61 | 60 | 640 |
| [48, 96) | 41 | 27 | 16 | 7 | 11 | 102 |
| [96, inf) | 0 | 1 | 0 | 0 | 1 | 2 |

| GT objects per image | learned-early | learned-late | forgotten | never-learned | never-detected | total |
|---|---|---|---|---|---|---|
| < 2 | 0 | 12 | 1 | 3 | 20 | 36 |
| 2–6 | 3 | 72 | 10 | 11 | 104 | 200 |
| 7–17 | 11 | 247 | 31 | 38 | 250 | 577 |
| ≥ 18 | 155 | 1216 | 224 | 189 | 914 | 2698 |

**Files**
- figures/s52/inspect/counts_by_{class,size,objects,image}.csv and categories.png
- figures/s52/inspect/crops_<category>__<class>.png: 22 grids of up to 24 seeded crops. Green = GT, red =
  highest-IoU prediction at epoch 50. Combinations with no boxes have no grid, e.g. learned-early Truck Tractor.
- results/s52/per_box_inspect.csv, results/s52/per_box_test.csv, results/s52/training_dynamics_args.json
- Per-checkpoint predictions (.csv.gz, 37 MB) stay in the Kaggle output of aradhya1211/auric-s52, under
  /kaggle/working/s52/preds/.

## Part B: §5.4 smart vs random subsets

**Selection** (tools/make_smart_subsets.py; splits/smart_subsets_seed0.json):
- Embedding: facebook/dinov2-small via transformers (CLS token). The fallback was not needed.
- 1938.png was never included.

| | smart50 | random f50 | smart75 | random f75 |
|---|---|---|---|---|
| images | 202 | 202 | 302 | 302 |
| coverage-step images (+ k-center) | 34 (+168) | – | 91 (+211) | – |
| overlap with random | 98 | – | 222 | – |
| tiles / epochs / iterations | 1646 / 104 / 10712 | 1690 / 101 / 10706 | 2594 / 66 / 10758 | 2546 / 67 / 10720 |
| boxes Cargo / Box / Flatbed / Tractor / Liquid | 1883 / 1354 / 421 / 438 / 115 | 1758 / 848 / 285 / 205 / 86 | 2870 / 1888 / 550 / 559 / 155 | 2599 / 1457 / 371 / 290 / 125 |

**Results** (figures/subset_compare/subset_compare.csv; 95% bootstrap CIs over images, 1000 resamples; max_det 902;
"% full" = held-out mAP50 / 0.1420):

| run | size | sel. | seed | val mAP50 | val class-agn. AP50 | held-out mAP50 | held-out class-agn. AP50 | % full | ≥ 0.128 |
|---|---|---|---|---|---|---|---|---|---|
| b1h_f50 | 202 | random | 0 | 0.049 (0.022–0.108) | 0.123 (0.087–0.169) | 0.095 (0.026–0.127) | 0.232 (0.077–0.388) | 66.8 | no |
| b1h_f50_seed1 | 202 | random | 1 | 0.057 (0.023–0.111) | 0.127 (0.088–0.174) | 0.082 (0.028–0.109) | 0.221 (0.072–0.384) | 57.5 | no |
| b1h_smart50 | 202 | smart | 0 | 0.056 (0.030–0.102) | 0.179 (0.119–0.255) | 0.105 (0.028–0.142) | 0.284 (0.095–0.449) | 74.1 | no |
| b1h_f75 | 302 | random | 0 | 0.085 (0.048–0.145) | 0.231 (0.155–0.316) | 0.105 (0.028–0.144) | 0.293 (0.107–0.469) | 73.6 | no |
| b1h_f75_seed1 | 302 | random | 1 | 0.074 (0.046–0.135) | 0.227 (0.153–0.310) | 0.123 (0.038–0.165) | 0.306 (0.116–0.476) | 86.6 | no |
| b1h_smart75 | 302 | smart | 0 | 0.083 (0.041–0.145) | 0.220 (0.149–0.309) | 0.104 (0.033–0.132) | 0.334 (0.139–0.490) | 73.3 | no |
| b1h_tile1024_holdout40 | 403 | full | 0 | 0.106 (0.056–0.165) | 0.255 (0.184–0.341) | 0.151 (0.056–0.187) | 0.362 (0.133–0.542) | 106.1 | yes |
| b1h_seed1 | 403 | full | 1 | 0.063 (0.042–0.096) | 0.259 (0.179–0.348) | 0.133 (0.051–0.166) | 0.362 (0.142–0.533) | 93.9 | yes |

**Pre-registered rule, computed mechanically** (figures/subset_compare/decision_rule.csv):

| size | metric | smart | random s0 | random s1 | seed spread | smart − best random | exceeds by more than spread |
|---|---|---|---|---|---|---|---|
| 202 | held-out class-agn. AP50 (primary) | 0.2841 | 0.2324 | 0.2213 | 0.0111 | +0.0517 | True |
| 202 | held-out mAP50 (secondary) | 0.1052 | 0.0948 | 0.0816 | 0.0132 | +0.0103 | False |
| 302 | held-out class-agn. AP50 (primary) | 0.3343 | 0.2934 | 0.3062 | 0.0128 | +0.0281 | True |
| 302 | held-out mAP50 (secondary) | 0.1040 | 0.1045 | 0.1230 | 0.0185 | −0.0189 | False |

**Recovers 90%:** no 50% or 75% subset reaches held-out mAP50 ≥ 0.128, smart or random.

**Other files**
- figures/subset_compare/subset_compare.png
- results/b1h_{smart50,smart75,f50_seed1,f75_seed1}/ and figures/b1h_{...}/ (CSV/JSON/PNG under 20 MB, no weights)
- results/s54/embeddings.npz and embed_args.json

**Weights:** in each kernel's output at /kaggle/working/runs/b1h_<smart50|smart75|f50_seed1|f75_seed1>/train/weights/.

## Commits made overnight

| commit | content |
|---|---|
| b02413c | S54 pre-registration (before any S54 run); §5.2 inspect/test halves (before any §5.2 computation) |
| 883d884 | analysis/training_dynamics.py, tools/make_smart_subsets.py, script-mode kernels with kernel_sources; tests |
| 5807e3e | analysis/subset_compare.py |
| f296537 | smart50/75 lists, embeddings, configs b1h_smart50/75 and b1h_f50/f75_seed1; config tests |
| e8633a6 | §5.2 results (per-box tables for both halves, inspect-half figures) |
| 40067b8 | b1h_smart50 results |
| 3140888 | b1h_smart75, b1h_f50_seed1 results |
| 7deafdb | b1h_f75_seed1 results; S54 comparison table and rule output |

No conclusions were written into EXPERIMENTS.md or docs/REPORT_FULL.md. The only overnight change to EXPERIMENTS.md is the S54
pre-registration entry in b02413c.
