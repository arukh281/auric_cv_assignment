"""Repeat-factor sampling (LVIS, Gupta et al. 2019) over training tiles, implemented by repeating list entries.

For each class c, f_c = fraction of training tiles (the images Ultralytics trains on) containing c. Class repeat
factor r_c = max(1, sqrt(t / f_c)); tile repeat factor r = max over the tile's classes of r_c (1 for an empty tile).
The static list holds floor(r) copies of each tile plus one more with probability r - floor(r) (seeded), so the
expected number of copies is r. Returns the list and a summary with per-class f_c and r_c.
"""
import math
from pathlib import Path

import numpy as np


def tile_classes(label_dir, tiles):
    out = {}
    for t in tiles:
        f = Path(label_dir) / f"{t}.txt"
        out[t] = {int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()
    return out


def repeat_factors(classes_of, nc, t=0.1):
    n = len(classes_of)
    f = np.array([sum(c in s for s in classes_of.values()) / n for c in range(nc)])
    rc = np.array([max(1.0, math.sqrt(t / fc)) if fc > 0 else 1.0 for fc in f])
    rt = {k: max([rc[c] for c in s], default=1.0) for k, s in classes_of.items()}
    return f, rc, rt


def build_list(image_paths_by_tile, rt, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(image_paths_by_tile):
        r = rt[k]
        n = int(math.floor(r)) + int(rng.random() < r - math.floor(r))
        out += [str(image_paths_by_tile[k])] * n
    return out
