"""Choose a held-out set of train images (B1h) from the full-dataset EDA tables. Deterministic, no image I/O.

Usage:
  python tools/make_holdout.py [--n 40] [--seed 0] [--out splits/holdout40_seed0.txt]

Stratification: every train image with boxes is assigned to the rarest class it contains (rarity = number of
train images containing the class). Strata get round(n * share) images (largest remainder, at least 1 each), sampled
with numpy default_rng(seed) in sorted-name order, so every class appears in the holdout. 1938.png (no boxes) is never
chosen. Prints per-class GT box and image counts for holdout vs rest, and the expected tile counts for B1-style
tiling with and without the holdout (computed from figures/eda/tables/{images,boxes}.csv with the same windows,
clipping and empty-tile random draws as tools/make_tiles.py, image indices = position in the full train list).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import EDA_TABLES  # noqa: E402
from detlib.tiling import clip_boxes, tile_windows  # noqa: E402

EMPTY_IMAGE = "1938.png"


def choose_holdout(boxes, n=40, seed=0):
    tr = boxes[boxes.split == "train"]
    imgs_per_class = tr.groupby("class_name").image.nunique()
    rank = {c: r for r, c in enumerate(imgs_per_class.sort_values(kind="mergesort").index)}  # 0 = rarest
    stratum = tr.groupby("image").class_name.agg(lambda s: min(s.unique(), key=lambda c: rank[c]))
    stratum = stratum.drop(EMPTY_IMAGE, errors="ignore").sort_index()
    sizes = stratum.value_counts()
    raw = n * sizes / sizes.sum()
    k = np.maximum(np.floor(raw).astype(int), 1)
    for c in (raw - np.floor(raw)).sort_values(ascending=False, kind="mergesort").index:
        if k.sum() >= n:
            break
        k[c] += 1
    while k.sum() > n:  # only if the minimum of 1 per stratum overshoots
        k[k.idxmax()] -= 1
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted(k.index, key=lambda c: rank[c]):
        pool = sorted(stratum[stratum == c].index)
        out += [pool[i] for i in sorted(rng.choice(len(pool), int(k[c]), replace=False))]
    return sorted(out)


def expected_tiles(images, boxes, exclude=(), tile=1024, overlap=256, empty_keep=0.2, min_vis=0.5, seed=0):
    """Replicates tools/make_tiles.py counts from the EDA tables (no pixels)."""
    tr = images[images.split == "train"].sort_values("image", key=lambda s: s.map(_sort_key))
    bx = {k: d for k, d in boxes[boxes.split == "train"].groupby("image")}
    excl = set(exclude)
    written = with_boxes = 0
    for idx, r in enumerate(tr.itertuples()):
        if r.image in excl:
            continue
        d = bx.get(r.image)
        cls = d.cls.to_numpy(int) if d is not None else np.zeros(0, int)
        xyxy = d[["x1", "y1", "x2", "y2"]].to_numpy(float) if d is not None else np.zeros((0, 4))
        rng = np.random.default_rng([seed, idx])
        for win in tile_windows(int(r.width), int(r.height), tile, overlap):
            c, _, _ = clip_boxes(cls, xyxy, win, min_vis)
            empty = len(c) == 0
            if (not empty) or rng.random() < empty_keep:
                written += 1
                with_boxes += not empty
    return dict(tiles_written=written, tiles_with_boxes=with_boxes, empty_tiles_kept=written - with_boxes,
                n_source_images=len(tr) - len(excl & set(tr.image)))


def _sort_key(name):
    return name  # list_images sorts Path objects by full path; same directory, so by file name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="splits/holdout40_seed0.txt")
    a = ap.parse_args()
    boxes = pd.read_csv(EDA_TABLES / "boxes.csv")
    images = pd.read_csv(EDA_TABLES / "images.csv")
    hold = choose_holdout(boxes, a.n, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(hold) + "\n")
    tr = boxes[boxes.split == "train"]
    h = tr[tr.image.isin(hold)]
    rest = tr[~tr.image.isin(hold)]
    t = pd.DataFrame(dict(holdout_boxes=h.class_name.value_counts(), rest_boxes=rest.class_name.value_counts(),
                          holdout_images=h.groupby("class_name").image.nunique(),
                          rest_images=rest.groupby("class_name").image.nunique())).fillna(0).astype(int)
    t.loc["total"] = [len(h), len(rest), len(hold), rest.image.nunique()]
    print(f"[holdout] {len(hold)} images -> {out}")
    print(t.to_string())
    full = expected_tiles(images, boxes)
    ho = expected_tiles(images, boxes, hold)
    print("[holdout] expected tiles (from EDA tables):", json.dumps(dict(b1_all=full, b1h_without_holdout=ho)))


if __name__ == "__main__":
    main()
