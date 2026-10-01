# Technical report (in progress)

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
| Train/val usage | Train split for training only. Val is used only for final evaluation (`val: false`, `last.pt` reported, no epoch selection on val) |
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

Sections to come: dataset observations, baselines, experiment chain, failure analysis, research investigations
5.1-5.4, final analysis.
