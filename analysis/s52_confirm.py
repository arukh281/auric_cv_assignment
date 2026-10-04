"""§5.2 confirmation: test the pre-registered claims C1-C5 (EXPERIMENTS.md, "§5.2 test-half confirmation") on one
half of the §5.2 per-box table, with a 2000-sample bootstrap over images (seed 0). Run it on the inspect half first
(must reproduce the inspect numbers the claims came from), then on the test half. CPU only.

Usage: python analysis/s52_confirm.py --per-box <s52>/per_box_test.csv --preds <s52>/preds/predictions_ep050.csv.gz \
           --data-root /kaggle/tmp/data --out <dir>/test
A claim PASSES if its point estimate meets the criterion and so does the bootstrap 2.5th percentile.
Writes <out>/claims.csv and summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from detlib.scoring import box_iou  # noqa: E402


def floor_hit(t, preds, conf=0.001, iou=0.5):
    """Per row: some prediction of any class with conf >= conf and IoU >= iou at the last checkpoint."""
    out = np.zeros(len(t), bool)
    p = preds[preds.conf >= conf]
    for img, g in t.groupby("image"):
        q = p[p.image == img]
        if len(q):
            m = box_iou(g[["x1", "y1", "x2", "y2"]].to_numpy(float), q[["x1", "y1", "x2", "y2"]].to_numpy(float))
            out[t.index.get_indexer(g.index)] = m.max(1) >= iou
    return out


def share(t, mask, cat):
    s = t[mask]
    return (s.category == cat).mean() if len(s) else np.nan


def stats(t):
    nd = t.category == "never-detected"
    cls = sorted(t.class_name.unique())
    by = {c: t.class_name == c for c in cls}
    sh = lambda cat: {c: share(t, by[c], cat) for c in cls}
    early, forg, nl = sh("learned-early"), sh("forgotten"), sh("never-learned")
    oth = lambda d, k: max(v for c, v in d.items() if c != k and not np.isnan(v))
    return {
        "C1 never-detected share <16px minus >=32px": share(t, t["size"] < 16, "never-detected") - share(t, t["size"] >= 32, "never-detected"),
        "C2a Box learned-early share minus max other class": early["Truck w/Box"] - oth(early, "Truck w/Box"),
        "C2b Box forgotten share minus max other class": forg["Truck w/Box"] - oth(forg, "Truck w/Box"),
        "C3 Liquid never-learned share minus max other class": nl["Truck w/Liquid"] - oth(nl, "Truck w/Liquid"),
        "C4 never-detected with any-class IoU>=0.5 pred at conf>=0.001": t.loc[nd, "floor_hit"].mean() if nd.any() else np.nan,
        "C5 Cargo never-detected share": share(t, by["Cargo Truck"], "never-detected"),
    }


CRIT = {"C1": 0.0, "C2a": 0.0, "C2b": 0.0, "C3": 0.0, "C4": 0.5, "C5": 0.3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-box", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(a.per_box).reset_index(drop=True)
    t["floor_hit"] = floor_hit(t, pd.read_csv(a.preds))
    point = stats(t)
    imgs = t.image.unique(); groups = {i: g for i, g in t.groupby("image")}
    rng = np.random.default_rng(0); boots = []
    for _ in range(a.n_boot):
        s = pd.concat([groups[i] for i in rng.choice(imgs, len(imgs))], ignore_index=True)
        boots.append(stats(s))
    b = pd.DataFrame(boots)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(b[k], [2.5, 97.5]); thr = CRIT[k.split()[0]]
        rows.append(dict(claim=k, criterion=f"> {thr}", estimate=v, ci_lo=lo, ci_hi=hi, passes=bool(v > thr and lo > thr)))
    r = pd.DataFrame(rows); r.to_csv(out / "claims.csv", index=False)
    s = dict(per_box=a.per_box, n_boxes=len(t), n_images=len(imgs), halves=sorted(t.half.unique()),
             category_counts=t.category.value_counts().to_dict(), n_boot=a.n_boot)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s)); print(r.to_string(index=False))


if __name__ == "__main__":
    main()
