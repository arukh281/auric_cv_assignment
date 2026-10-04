# Visual inspection pack: val

Rendered by `analysis/inspect_val.py` from B1h's saved val predictions (`results/b1h_tile1024_holdout40/eval/predictions.csv`) and B1h's GT-box oracle scores (`figures/b1h_tile1024_holdout40/gt_oracle/gt_scores.csv`). No new inference. GT solid, prediction dashed, one colour per class (legend on each sheet). Crops: square, 3x the box's longer side, nearest-neighbour upscale to 320 px (scale in each caption). Matching: one-to-one, class-agnostic, IoU >= 0.5.

| file | selection rule |
|---|---|
| 01_overview_worst.jpg | 2 val images with the lowest recall (any class, IoU >= 0.5, conf >= 0.001); predictions at conf >= 0.25; unmatched GT yellow halo, unmatched predictions magenta halo |
| 02_overview_random.jpg | 2 random val images, numpy default_rng(0).choice over sorted names; same style |
| 03_zoom_worst.jpg | the 4 tiles (1024, overlap 256) of the lowest-recall image with the most GT boxes fully inside; predictions conf >= 0.25 labelled |
| 04_cargo_called_box.jpg | 24 val GT Cargo Truck boxes that the GT-box oracle (pool) assigns to Truck w/Box, highest Box score first |
| 05_box_called_cargo.jpg | 24 val GT Truck w/Box boxes assigned to Cargo Truck, highest Cargo score first |
| 06_val_fp_top.jpg | 24 highest-confidence val predictions with IoU < 0.1 to every GT box |
| 07_val_missed.jpg | 24 random (default_rng(0)) val GT boxes with no prediction of any class at IoU >= 0.5, any confidence |
| 08_train_vs_val_box.jpg | 18 random train vs 18 random val GT Truck w/Box crops (default_rng(0)) |
| 09_train_vs_val_cargo.jpg | same for Cargo Truck |
| 10_class_reference.jpg | 8 random train GT crops per class (default_rng(0)) |

## Per-image recall (val, any class, IoU >= 0.5, conf >= 0.001)

| image | GT boxes | matched | recall |
|---|---|---|---|
| 2543.png | 97 | 29 | 0.299 |
| 2292.png | 30 | 10 | 0.333 |
| 2391.png | 108 | 40 | 0.370 |
| 2384.png | 90 | 39 | 0.433 |
| 2460.png | 86 | 42 | 0.488 |
| 1447.png | 43 | 22 | 0.512 |
| 1181.png | 33 | 18 | 0.545 |
| 1929.png | 77 | 44 | 0.571 |
| 1399.png | 28 | 17 | 0.607 |
| 1211.png | 167 | 105 | 0.629 |
| 1206.png | 97 | 67 | 0.691 |
| 1362.png | 16 | 12 | 0.750 |
| 31.png | 35 | 27 | 0.771 |
| 2470.png | 153 | 126 | 0.824 |
| 2293.png | 145 | 122 | 0.841 |
| 2472.png | 101 | 85 | 0.842 |
| 2308.png | 76 | 66 | 0.868 |
| 20.png | 43 | 38 | 0.884 |
| 1530.png | 14 | 13 | 0.929 |
| 2139.png | 92 | 89 | 0.967 |
| 1456.png | 8 | 8 | 1.000 |
| 1457.png | 13 | 13 | 1.000 |
