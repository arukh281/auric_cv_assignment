"""Nested, class-stratified training subsets for the learning curves (§5.3), plus their configs. No image I/O.

Usage:
  python tools/make_subsets.py [--seed 0] [--write-configs]

Pool: every train image not in splits/holdout40_seed0.txt (403 images, including the box-free 1938.png).
Stratification: as tools/make_holdout.py, each image belongs to the rarest class it contains ("empty" for 1938.png).
Nesting: one ordering of the pool is built by shuffling each stratum (numpy default_rng(seed)) and then repeatedly
taking the next image from the stratum furthest below its proportional share. Subset f = the first round(f * 403)
images of that ordering, so every subset is stratified and contains every smaller one.

Equal compute: each subset's tile count is computed with tools/make_holdout.expected_tiles (the same replica that
reproduces B1/B1h's tile counts exactly), iterations per epoch = ceil(tiles / 16) and
epochs = round(10750 / iterations_per_epoch), B1h's total. Iteration-based schedule parts are held equal too:
warmup_epochs = 3 * 215 / it_per_epoch (B1h: 3 epochs x 215 it) and close_mosaic = round(10 * 215 / it_per_epoch);
checkpoint_every = epochs // 5.

Writes splits/train_f{25,50,75}_seed0.txt, splits/learning_curve_seed0.json (sizes, per-class counts, tiles, epochs,
iterations), and with --write-configs configs/b1h_f{25,50,75}.yaml and configs/b1h_seed1.yaml (copies of b1h.yaml
with only the listed fields changed).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from detlib.data import EDA_TABLES  # noqa: E402
from make_holdout import expected_tiles  # noqa: E402

FRACTIONS = (25, 50, 75)
B1H_ITERS, B1H_IT_PER_EPOCH, BATCH = 10750, 215, 16
HOLDOUT = REPO / "splits" / "holdout40_seed0.txt"


def strata(images, boxes, pool):
    tr = boxes[boxes.split == "train"]
    rank = {c: r for r, c in enumerate(tr.groupby("class_name").image.nunique().sort_values(kind="mergesort").index)}
    rare = tr.groupby("image").class_name.agg(lambda s: min(s.unique(), key=lambda c: rank[c]))
    return {im: rare.get(im, "empty") for im in pool}


def nested_order(stratum_of, seed=0):
    rng = np.random.default_rng(seed)
    groups = {}
    for im in sorted(stratum_of):
        groups.setdefault(stratum_of[im], []).append(im)
    queues = {k: [v[i] for i in rng.permutation(len(v))] for k, v in sorted(groups.items())}
    n = sum(len(v) for v in queues.values())
    taken = dict.fromkeys(queues, 0)
    order = []
    for _ in range(n):
        # stratum furthest below its share of what has been taken so far (ties: name order)
        k = min((k for k in queues if taken[k] < len(queues[k])),
                key=lambda k: ((taken[k] + 0.5) / len(queues[k]), k))
        order.append(queues[k][taken[k]])
        taken[k] += 1
    return order


def plan(seed=0):
    images, boxes = pd.read_csv(EDA_TABLES / "images.csv"), pd.read_csv(EDA_TABLES / "boxes.csv")
    hold = set(HOLDOUT.read_text().split())
    pool = sorted(set(images[images.split == "train"].image) - hold)
    order = nested_order(strata(images, boxes, pool), seed)
    tr = boxes[boxes.split == "train"]
    out = dict(seed=seed, pool_size=len(pool), holdout=str(HOLDOUT.relative_to(REPO)), batch=BATCH,
               b1h_total_iterations=B1H_ITERS, b1h_iterations_per_epoch=B1H_IT_PER_EPOCH,
               epochs_formula="round(10750 / ceil(tiles / 16))", subsets={})
    all_train = set(images[images.split == "train"].image)
    for f in FRACTIONS + (100,):
        sub = order[: int(round(f / 100 * len(pool)))]
        t = expected_tiles(images, boxes, sorted(all_train - set(sub)))
        ipe = math.ceil(t["tiles_written"] / BATCH)
        ep = int(round(B1H_ITERS / ipe))
        b = tr[tr.image.isin(sub)]
        out["subsets"][f"f{f}"] = dict(
            n_images=len(sub), list=f"splits/train_f{f}_seed0.txt" if f != 100 else None,
            tiles=t["tiles_written"], iterations_per_epoch=ipe, epochs=ep, total_iterations=ep * ipe,
            warmup_epochs=round(3 * B1H_IT_PER_EPOCH / ipe, 4), close_mosaic=int(round(10 * B1H_IT_PER_EPOCH / ipe)),
            checkpoint_every=max(1, ep // 5),
            boxes_per_class=b.class_name.value_counts().sort_index().to_dict(),
            images_per_class=b.groupby("class_name").image.nunique().sort_index().to_dict(), images=sub)
    return out


def write_configs(p):
    base = (REPO / "configs" / "b1h.yaml").read_text()
    cfg0 = yaml.safe_load(base)
    for f in FRACTIONS:
        s = p["subsets"][f"f{f}"]
        cfg = yaml.safe_load(base)
        cfg["name"] = f"b1h_f{f}"
        cfg["train_list"] = s["list"]
        cfg["epochs"] = s["epochs"]
        cfg["checkpoint_every"] = s["checkpoint_every"]
        cfg["train_args"]["warmup_epochs"] = s["warmup_epochs"]
        cfg["train_args"]["close_mosaic"] = s["close_mosaic"]
        head = (f"# b1h_f{f}: configs/b1h.yaml with training restricted to {s['n_images']} of the 403 non-holdout images\n"
                f"# ({s['list']}, nested + class-stratified, tools/make_subsets.py seed 0) and the epoch count set so the total\n"
                f"# iterations match B1h's 10750: epochs = round(10750 / ceil({s['tiles']} tiles / 16)) = {s['epochs']}"
                f" -> {s['total_iterations']} iterations.\n"
                f"# warmup_epochs / close_mosaic / checkpoint_every rescaled so they cover the same iterations as in B1h.\n")
        (REPO / "configs" / f"b1h_f{f}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
    cfg = dict(cfg0, name="b1h_seed1", seed=1)
    (REPO / "configs" / "b1h_seed1.yaml").write_text(
        "# b1h_seed1: exact copy of configs/b1h.yaml with training seed 1 (tiling seed unchanged), to measure\n"
        "# run-to-run noise.\n" + yaml.safe_dump(cfg, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-configs", action="store_true")
    a = ap.parse_args()
    p = plan(a.seed)
    for f in FRACTIONS:
        s = p["subsets"][f"f{f}"]
        (REPO / s["list"]).write_text("\n".join(sorted(s["images"])) + "\n")
    (REPO / "splits" / f"learning_curve_seed{a.seed}.json").write_text(json.dumps(p, indent=1))
    if a.write_configs:
        write_configs(p)
    rows = []
    for k, s in p["subsets"].items():
        r = dict(subset=k, images=s["n_images"], tiles=s["tiles"], it_per_epoch=s["iterations_per_epoch"],
                 epochs=s["epochs"], total_iters=s["total_iterations"], warmup_epochs=s["warmup_epochs"],
                 close_mosaic=s["close_mosaic"])
        r.update({f"boxes {c}": n for c, n in s["boxes_per_class"].items()})
        r.update({f"imgs {c}": n for c, n in s["images_per_class"].items()})
        rows.append(r)
    print(pd.DataFrame(rows).set_index("subset").T.to_string())


if __name__ == "__main__":
    main()
