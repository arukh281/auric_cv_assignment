"""§5.2 confidence floor, saved to a file (was an unsaved in-session analysis on 3 Oct, inspect half only). CPU only.

For the never-detected boxes of one half (results/s52 per-box table, category == never-detected): share that have an
epoch-50 prediction of any class with IoU >= 0.5 at conf >= 0.001 / 0.05 / 0.10. Chance control (as the 3 Oct analysis): one random box per GT
box of the half, any category (same image, same width and height, uniform position, IoU 0 with every GT box of the image; seed 0;
up to 1000 tries, else dropped), scored the same way.

Usage: python analysis/s52_floor.py --per-box <s52>/per_box_test.csv --preds <s52>/preds/predictions_ep050.csv.gz \
           --out <dir>/test
Writes <out>/floor.csv and summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import EDA_TABLES  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402

FLOORS = (0.001, 0.05, 0.10)


def best_iou(boxes, preds, conf):
    p = preds[preds.conf >= conf][["x1", "y1", "x2", "y2"]].to_numpy(float)
    return box_iou(boxes, p).max(1) if len(boxes) and len(p) else np.zeros(len(boxes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-box", required=True); ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(a.per_box); preds = pd.read_csv(a.preds)
    im = pd.read_csv(EDA_TABLES / "images.csv").query("split == 'train'")
    size = {r.image: (int(r.width), int(r.height)) for r in im.itertuples()}
    rng = np.random.default_rng(0)
    rows, dropped = [], 0
    for img, g in t.groupby("image"):
        allgt = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        nd = g[g.category == "never-detected"]
        p = preds[preds.image == img]
        W, H = size[img]
        nb = nd[["x1", "y1", "x2", "y2"]].to_numpy(float)
        ch = []
        for b in allgt:
            w, h = b[2] - b[0], b[3] - b[1]
            for _ in range(1000):
                x0, y0 = rng.uniform(0, max(W - w, 1)), rng.uniform(0, max(H - h, 1))
                c = np.array([[x0, y0, x0 + w, y0 + h]])
                if box_iou(c, allgt).max() == 0:
                    ch.append(c[0]); break
            else:
                ch.append(None); dropped += 1
        for f in FLOORS:
            gi = best_iou(nb, p, f)
            keep = np.array([c is not None for c in ch])
            cb = np.array([c for c in ch if c is not None]) if keep.any() else np.zeros((0, 4))
            ci = best_iou(cb, p, f)
            rows.append(dict(image=img, floor=f, n=len(nb), hit=int((gi >= 0.5).sum()), n_chance=len(cb),
                             chance_hit=int((ci >= 0.5).sum()), cls=None))
    r = pd.DataFrame(rows).groupby("floor")[["n", "hit", "n_chance", "chance_hit"]].sum()
    r["share"] = r.hit / r.n; r["chance_share"] = r.chance_hit / r.n_chance
    r.to_csv(out / "floor.csv")
    s = dict(per_box=a.per_box, never_detected=int(r.n.iloc[0]), chance_boxes_dropped=dropped,
             by_floor={f"{k:g}": dict(share=float(v.share), chance_share=float(v.chance_share)) for k, v in r.iterrows()})
    (out / "summary.json").write_text(json.dumps(s, indent=2)); print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
