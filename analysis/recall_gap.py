"""Why does B1h find fewer val trucks (class-agnostic recall 0.665) than holdout40 trucks (0.852)? CPU only, from saved
predictions; no inference.

Usage (Kaggle CPU kernel, after scripts/kaggle_setup.sh):
  python analysis/recall_gap.py --val-preds <B1h>/eval/predictions.csv \
      --holdout-preds <B1h>/eval_holdout40/predictions.csv --holdout splits/holdout40_seed0.txt \
      --data-root /kaggle/tmp/data --out <dir>

Per GT box: matched = some prediction of ANY class with IoU >= 0.5 and conf >= the threshold, one-to-one greedy by
confidence (class-agnostic, as analysis/class_agnostic.py; the totals are checked against its 1032/1552 and 629/738).
Recall is broken down by
  box size     sqrt(area) px bins [0,8,16,24,32,48,96,inf)
  density      GT boxes in the box's image, bins [0,10,30,60,100,inf)
  image size   image megapixels, bins [0,6,9,12,inf)
for val and holdout at conf 0.001 and 0.25, and a standardised comparison: holdout recall re-weighted to val's
distribution of each factor (per-bin holdout recall x val's share of boxes in that bin). If re-weighting by a factor
moves holdout recall most of the way to val's, that factor's distribution shift can explain the gap; if not, the gap
is within bins (val scenes are harder at the same size/density/resolution). CIs: 2000-sample bootstrap over images.
Writes <out>/per_gt.csv, recall_by_factor.csv, standardised.csv, summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from detlib.data import EDA_TABLES, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402

BINS = {"size": [0, 8, 16, 24, 32, 48, 96, np.inf], "density": [0, 10, 30, 60, 100, np.inf],
        "megapixels": [0, 6, 9, 12, np.inf]}
CONFS = (0.001, 0.25)


def match(gb, pb, ps, conf, thr=0.5):
    """Greedy one-to-one class-agnostic matching by descending confidence. Returns bool per GT."""
    keep = ps >= conf
    pb, ps = pb[keep], ps[keep]
    hit = np.zeros(len(gb), bool)
    if not len(gb) or not len(pb):
        return hit
    iou = box_iou(pb[np.argsort(-ps)], gb)
    for row in iou:
        row = np.where(hit, -1, row)
        j = int(np.argmax(row))
        if row[j] >= thr:
            hit[j] = True
    return hit


def per_gt(preds, images, label_dir, sizes, split):
    rows = []
    g = {k: v for k, v in preds.groupby("image")}
    for name in images:
        W, H = sizes[name]
        gc, gb = read_yolo_labels(label_dir / f"{Path(name).stem}.txt", W, H)
        d = g.get(name)
        pb = d[["x1", "y1", "x2", "y2"]].to_numpy(float) if d is not None else np.zeros((0, 4))
        ps = d.conf.to_numpy(float) if d is not None else np.zeros(0)
        hits = {c: match(gb, pb, ps, c) for c in CONFS}
        for j in range(len(gb)):
            rows.append(dict(split=split, image=name, cls=int(gc[j]),
                             size=float(np.sqrt((gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1]))),
                             density=len(gb), megapixels=W * H / 1e6,
                             **{f"hit_{c:g}": bool(hits[c][j]) for c in CONFS}))
    return rows


def standardise(t, f, col):
    v, h = t[t.split == "val"], t[t.split == "holdout"]
    rv = v.groupby(f"{f}_bin", observed=False)[col].mean()
    rh = h.groupby(f"{f}_bin", observed=False)[col].mean()
    w = v[f"{f}_bin"].value_counts(normalize=True)
    common = [b for b in w.index if not np.isnan(rh.get(b, np.nan))]
    cover = float(w[common].sum())
    return float((rh[common] * w[common]).sum() / cover), cover


def boot(t, fn, n, seed=0):
    rng = np.random.default_rng(seed)
    groups = {s: {i: g for i, g in d.groupby("image")} for s, d in t.groupby("split")}
    out = []
    for _ in range(n):
        parts = [g[i] for s, g in groups.items() for i in rng.choice(list(g), len(g))]
        out.append(fn(pd.concat(parts, ignore_index=True)))
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-preds", required=True)
    ap.add_argument("--holdout-preds", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    im = pd.read_csv(EDA_TABLES / "images.csv")
    sz = {(r.split, r.image): (int(r.width), int(r.height)) for r in im.itertuples()}
    val_imgs = sorted(im[im.split == "val"].image)
    hold = [l.strip() for l in Path(a.holdout).read_text().splitlines() if l.strip()]
    rows = per_gt(pd.read_csv(a.val_preds), val_imgs, root / "val" / "labels",
                  {k[1]: v for k, v in sz.items() if k[0] == "val"}, "val")
    rows += per_gt(pd.read_csv(a.holdout_preds), hold, root / "train" / "labels",
                   {k[1]: v for k, v in sz.items() if k[0] == "train"}, "holdout")
    t = pd.DataFrame(rows)
    for f, b in BINS.items():
        t[f"{f}_bin"] = pd.cut(t[f], b, right=False).astype(str)
    t.to_csv(out / "per_gt.csv", index=False)

    rec = []
    for f in BINS:
        for (s, b), g in t.groupby(["split", f"{f}_bin"]):
            rec.append(dict(factor=f, bin=b, split=s, n_gt=len(g), n_images=g.image.nunique(),
                            **{f"recall_{c:g}": g[f"hit_{c:g}"].mean() for c in CONFS}))
    pd.DataFrame(rec).sort_values(["factor", "bin", "split"]).to_csv(out / "recall_by_factor.csv", index=False)

    st = []
    for c in CONFS:
        col = f"hit_{c:g}"
        rv = lambda x: x[x.split == "val"][col].mean()
        rh = lambda x: x[x.split == "holdout"][col].mean()
        base = dict(conf=c, val=rv(t), holdout=rh(t))
        lo, hi = boot(t, lambda x: rh(x) - rv(x), a.n_boot)
        st.append(dict(base, factor="none (raw gap)", holdout_standardised=rh(t), val_bins_covered=1.0,
                       gap=rh(t) - rv(t), gap_ci_lo=lo, gap_ci_hi=hi, share_of_gap_explained=0.0))
        for f in BINS:
            hs, cover = standardise(t, f, col)
            lo, hi = boot(t, lambda x: standardise(x, f, col)[0] - rv(x), a.n_boot)
            st.append(dict(base, factor=f, holdout_standardised=hs, val_bins_covered=cover, gap=hs - rv(t),
                           gap_ci_lo=lo, gap_ci_hi=hi,
                           share_of_gap_explained=(rh(t) - hs) / (rh(t) - rv(t)) if rh(t) != rv(t) else np.nan))
    pd.DataFrame(st).to_csv(out / "standardised.csv", index=False)
    s = dict(n_gt={k: int(v) for k, v in t.split.value_counts().items()},
             matched_conf0001={k: int(g["hit_0.001"].sum()) for k, g in t.groupby("split")},
             check_against_class_agnostic_json={"val": "1032/1552", "holdout": "629/738"},
             images={k: int(g.image.nunique()) for k, g in t.groupby("split")},
             median_size={k: float(g["size"].median()) for k, g in t.groupby("split")},
             median_density_per_box={k: float(g.density.median()) for k, g in t.groupby("split")},
             median_megapixels_per_box={k: float(g.megapixels.median()) for k, g in t.groupby("split")})
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s)); print(pd.DataFrame(st).to_string(index=False))


if __name__ == "__main__":
    main()
