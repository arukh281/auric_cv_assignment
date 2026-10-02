"""Cut TRAIN images into overlapping tiles at native resolution (val is never tiled; eval.py slices it
on the fly and scores against the original full-image labels).

Usage:
  python tools/make_tiles.py --data-root data --out work/tiles_1024_256 \
      --tile 1024 --overlap 256 --empty-keep 0.2 --min-vis 0.5 --seed 0

--exclude-list FILE (one image file name per line, e.g. splits/holdout40_seed0.txt) leaves those train images out of
tiling entirely. Every other image keeps its index in the full train list, so its tiles (including the seeded
empty-tile choices) are identical to a run without the list.

Outputs in --out: train/images, train/labels, data.yaml (train = tiles, val = full-res val images),
tiling_params.json (params + counts), tiles_index.csv (tile -> source image and offset), _COMPLETE.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import (eda_max_box_side, list_images, load_classes, read_yolo_labels,  # noqa: E402
                         write_resolved_data_yaml, write_yolo_labels)
from detlib.tiling import clip_boxes, tile_windows  # noqa: E402

STAT_KEYS = ("kept_whole", "kept_clipped", "dropped_partial")


def read_list(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def tile_one(job):
    idx, img_path, lbl_dir, out, a = job
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # same decode as Ultralytics' loader
    if img is None:
        raise RuntimeError(f"cannot read {img_path}")
    h, w = img.shape[:2]
    cls, xyxy = read_yolo_labels(Path(lbl_dir) / f"{img_path.stem}.txt", w, h)
    rng = np.random.default_rng([a["seed"], idx])  # per-image stream: identical output with any worker count
    rows, tot = [], dict.fromkeys(STAT_KEYS, 0)
    for win in tile_windows(w, h, a["tile"], a["overlap"]):
        c, b, st = clip_boxes(cls, xyxy, win, a["min_vis"])
        empty = len(c) == 0
        keep = (not empty) or rng.random() < a["empty_keep"]
        for k in STAT_KEYS:
            tot[k] += st[k]
        name = f"{img_path.stem}__x{win[0]}_y{win[1]}"
        rows.append(dict(tile=name, source=img_path.name, x0=win[0], y0=win[1], x1=win[2], y1=win[3],
                         n_boxes=len(c), is_empty=empty, kept=keep, **st))
        if not keep:
            continue
        tw, th = win[2] - win[0], win[3] - win[1]
        cv2.imwrite(str(out / "train" / "images" / f"{name}{a['ext']}"), img[win[1]:win[3], win[0]:win[2]])
        write_yolo_labels(out / "train" / "labels" / f"{name}.txt", c, b, tw, th)
    return rows, tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--empty-keep", type=float, default=0.2, help="fraction of box-free tiles kept")
    ap.add_argument("--min-vis", type=float, default=0.5, help="min visible area fraction to keep a clipped box")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ext", default=".png", help="tile format; .png keeps pixels identical to the source")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=None, help="testing only: tile the first N train images")
    ap.add_argument("--allow-small-overlap", action="store_true")
    ap.add_argument("--exclude-list", help="file with train image names to leave out (held-out experiments)")
    ap.add_argument("--include-list", help="file with the only train image names to tile (learning-curve subsets)")
    a = ap.parse_args()

    max_side = eda_max_box_side("train")
    print(f"[tiles] max train box side from EDA table: {max_side:.1f} px; overlap = {a.overlap} px")
    if a.overlap <= max_side and not a.allow_small_overlap:
        sys.exit(f"overlap {a.overlap} must exceed the max box side {max_side:.1f} px so every box fits whole in some tile")

    root, out = Path(a.data_root), Path(a.out)
    if (out / "_COMPLETE").exists():
        print(f"[tiles] {out} already complete, skipping")
        return
    (out / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    imgs = list_images(root / "train" / "images")[: a.max_images]
    excl = read_list(a.exclude_list) if a.exclude_list else []
    if a.include_list:  # tile only the listed images; everything else counts as excluded (indices unchanged)
        incl = set(read_list(a.include_list))
        excl = sorted(set(excl) | {p.name for p in imgs if p.name not in incl})
    params = dict(tile=a.tile, overlap=a.overlap, empty_keep=a.empty_keep, min_vis=a.min_vis, seed=a.seed,
                  ext=a.ext, max_images=a.max_images, max_train_box_side_px=max_side,
                  data_root=str(root.resolve()), n_source_images=sum(p.name not in set(excl) for p in imgs))
    if a.include_list:
        params.update(include_list=str(a.include_list), n_included_found=sum(p.name in incl for p in imgs))
    if a.exclude_list:
        params.update(exclude_list=str(a.exclude_list), excluded_images=excl,
                      n_excluded_found=sum(p.name in set(excl) for p in imgs))
    # index = position in the full list, so excluding images never changes the other images' random streams
    jobs = [(i, p, root / "train" / "labels", out, params) for i, p in enumerate(imgs) if p.name not in set(excl)]
    rows, tot = [], dict.fromkeys(STAT_KEYS, 0)
    with ProcessPoolExecutor(a.workers) as ex:
        for r, t in ex.map(tile_one, jobs):
            rows += r
            for k in STAT_KEYS:
                tot[k] += t[k]
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "tiles_index.csv", index=False)
    params["counts"] = dict(
        tiles_total=len(idx), tiles_written=int(idx.kept.sum()), tiles_with_boxes=int((~idx.is_empty).sum()),
        empty_tiles_kept=int((idx.is_empty & idx.kept).sum()), empty_tiles_dropped=int((idx.is_empty & ~idx.kept).sum()),
        box_instances_in_written_tiles=int(idx[idx.kept].n_boxes.sum()), **{f"box_tile_{k}": v for k, v in tot.items()})
    (out / "tiling_params.json").write_text(json.dumps(params, indent=2))
    write_resolved_data_yaml(out / "data.yaml", out, "train/images", (root / "val" / "images").resolve(),
                             load_classes(root))
    (out / "_COMPLETE").touch()
    print(json.dumps(params["counts"], indent=2))


if __name__ == "__main__":
    main()
