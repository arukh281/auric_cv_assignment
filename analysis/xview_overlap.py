"""Is our dataset a subset of xView? CPU only, evaluation only (no training).

  python analysis/xview_overlap.py --xview <mirror root> --data-root /kaggle/tmp/data \
      --b1h-holdout-preds <B1h>/eval_holdout40/predictions.csv --out <dir>

Mirror: Kaggle hassanmojab/xview-dataset (train_images/train_images/*.tif, val_images/val_images/*.tif,
train_labels/xView_train.geojson). xView is CC BY-NC-SA 4.0.

1. Match: each of our images (figures/eda/tables/images.csv: 443 train + 22 val) is matched to an xView image with the
   same stem (102.png <-> 102.tif) AND the same width x height (TIFF header). Reported per split, plus holdout40.
2. Labels: for matched images in the xView TRAIN set, xView boxes (bounds_imcoords, pixel coords) are paired with our
   boxes one-to-one by IoU (greedy, >= 0.5, class-agnostic). Reported: counts, paired share of each side, and a
   crosstab of our class vs xView type_id for the pairs, which shows the class mapping empirically.
3. Extra data: per xView type_id, instance counts in xView train images NOT in our dataset (our 5 classes, mapped
   from step 2, and the excluded truck-like types).
4. FP re-scoring: for the 60 audited detections (analysis/inputs/xview/fp_audit.csv) and for all B1h holdout40
   predictions with conf >= 0.25 and IoU < 0.1 to every one of our GT boxes, the best-overlapping xView box
   (IoU >= 0.3) and its type_id, or "none".
Writes <out>/match.csv, match_summary.json, label_pairs_crosstab.csv, label_summary.json, extra_counts.csv,
fp_rescore_audit60.csv, fp_rescore_holdout_conf025.csv, fp_rescore_summary.json.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, load_classes, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
# xView type ids (xview_class_labels.txt); checked empirically by the crosstab in step 2
XVIEW = {17: "Passenger Vehicle", 18: "Small Car", 19: "Bus", 20: "Pickup Truck", 21: "Utility Truck", 23: "Truck",
         24: "Cargo Truck", 25: "Truck w/Box", 26: "Truck Tractor", 27: "Trailer", 28: "Truck w/Flatbed",
         29: "Truck w/Liquid", 32: "Crane Truck", 53: "Engineering Vehicle", 60: "Dump Truck", 61: "Haul Truck",
         62: "Scraper/Tractor", 65: "Cement Mixer"}
TRUCKLIKE_EXCLUDED = [20, 21, 23, 27, 32, 60, 61, 65]


def greedy_pairs(a, b, thr):
    if not len(a) or not len(b):
        return []
    iou = box_iou(a, b); out, ua, ub = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[i, j] < thr:
            break
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); out.append((i, j, float(iou[i, j])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xview", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--b1h-holdout-preds", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xv, root, out = Path(a.xview), Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = load_classes(root)
    hold = set(l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip())

    xfiles = {}
    for sub in ("train", "val"):
        for p in (xv / f"{sub}_images" / f"{sub}_images").glob("*.tif"):
            xfiles[p.stem] = (sub, p)
    ours = pd.read_csv(EDA_TABLES / "images.csv")
    rows = []
    for r in ours.itertuples():
        stem = Path(r.image).stem
        m = xfiles.get(stem)
        if m is None:
            rows.append(dict(split=r.split, image=r.image, width=r.width, height=r.height, xview_set=None,
                             xview_w=None, xview_h=None, match=False)); continue
        with Image.open(m[1]) as im:
            w, h = im.size
        rows.append(dict(split=r.split, image=r.image, width=r.width, height=r.height, xview_set=m[0], xview_w=w,
                         xview_h=h, match=bool(w == r.width and h == r.height)))
    mt = pd.DataFrame(rows); mt["holdout40"] = mt.image.isin(hold)
    mt.to_csv(out / "match.csv", index=False)
    ms = {}
    for key, g in (("train", mt[mt.split == "train"]), ("val", mt[mt.split == "val"]), ("holdout40", mt[mt.holdout40]),
                   ("train_minus_holdout40", mt[(mt.split == "train") & ~mt.holdout40])):
        ms[key] = dict(n=len(g), stem_found=int(g.xview_set.notna().sum()), matched_stem_and_size=int(g.match.sum()),
                       in_xview_train=int(((g.xview_set == "train") & g.match).sum()),
                       in_xview_val=int(((g.xview_set == "val") & g.match).sum()))
    (out / "match_summary.json").write_text(json.dumps(ms, indent=2)); print("[match]", json.dumps(ms), flush=True)

    feats = json.load(open(xv / "train_labels" / "xView_train.geojson"))["features"]
    xb = defaultdict(list)
    for f in feats:
        p = f["properties"]
        try:
            b = [float(v) for v in p["bounds_imcoords"].split(",")]
        except Exception:
            continue
        xb[Path(p["image_id"]).stem].append((int(p["type_id"]), b))
    del feats
    print(f"[geojson] {sum(len(v) for v in xb.values())} boxes in {len(xb)} images", flush=True)

    ct, lab = Counter(), dict(our_boxes=0, xview_boxes_all_types=0, xview_boxes_5_trucktypes=0, pairs_iou05=0,
                             images=0)
    matched = mt[mt.match & (mt.xview_set == "train")]
    for r in matched.itertuples():
        stem = Path(r.image).stem
        gc, gb = read_yolo_labels(root / r.split / "labels" / f"{stem}.txt", r.width, r.height)
        x = xb.get(stem, [])
        xt = np.array([t for t, _ in x]); xbx = np.array([b for _, b in x]).reshape(-1, 4)
        lab["images"] += 1; lab["our_boxes"] += len(gc); lab["xview_boxes_all_types"] += len(xt)
        lab["xview_boxes_5_trucktypes"] += int(np.isin(xt, [24, 25, 26, 28, 29]).sum())
        for i, j, _ in greedy_pairs(gb, xbx, 0.5):
            ct[(names[int(gc[i])], int(xt[j]))] += 1; lab["pairs_iou05"] += 1
    cross = pd.Series(ct).unstack(fill_value=0) if ct else pd.DataFrame()
    cross.columns = [f"{c} {XVIEW.get(c, '?')}" for c in cross.columns] if len(cross) else []
    cross.to_csv(out / "label_pairs_crosstab.csv")
    lab["our_boxes_paired_share"] = lab["pairs_iou05"] / max(lab["our_boxes"], 1)
    lab["xview_5type_boxes_paired_share_upper_bound"] = lab["pairs_iou05"] / max(lab["xview_boxes_5_trucktypes"], 1)
    (out / "label_summary.json").write_text(json.dumps(lab, indent=2)); print("[labels]", json.dumps(lab), flush=True)
    print(cross.to_string(), flush=True)

    ours_stems = set(Path(i).stem for i in ours.image)
    extra = Counter(); extra_imgs = Counter()
    for stem, v in xb.items():
        if stem in ours_stems:
            continue
        ts = [t for t, _ in v]
        for t in set(ts):
            extra_imgs[t] += 1
        extra.update(ts)
    keep = [24, 25, 26, 28, 29] + TRUCKLIKE_EXCLUDED
    ec = pd.DataFrame([dict(type_id=t, name=XVIEW.get(t, "?"), instances=extra[t], images=extra_imgs[t],
                            group="our 5 classes" if t in (24, 25, 26, 28, 29) else "truck-like, excluded")
                       for t in keep])
    n_extra_imgs = len([s for s in xb if s not in ours_stems])
    ec.to_csv(out / "extra_counts.csv", index=False)
    print(f"[extra] xView train images not in our dataset: {n_extra_imgs}"); print(ec.to_string(index=False), flush=True)

    sizes = {Path(r.image).stem: (r.width, r.height) for r in ours.itertuples()}
    def rescore(df, tag):
        res = []
        for r in df.itertuples():
            stem = Path(r.image).stem
            x = xb.get(stem, []); b = np.array([[r.x1, r.y1, r.x2, r.y2]], float)
            best_t, best_i = None, 0.0
            if x:
                iou = box_iou(b, np.array([bb for _, bb in x]))[0]
                k = int(iou.argmax()); best_i = float(iou[k]); best_t = x[k][0]
            hit = best_i >= 0.3
            res.append(dict(image=r.image, cls=int(r.cls), conf=float(r.conf), best_xview_iou=best_i,
                            xview_type=best_t if hit else None, xview_name=XVIEW.get(best_t, "?") if hit else "none",
                            group=("excluded truck-like" if best_t in TRUCKLIKE_EXCLUDED else
                                   "one of our 5 (missing in our labels)" if best_t in (24, 25, 26, 28, 29) else
                                   "other xView class") if hit else "no xView box",
                            **({"verdict": r.verdict} if hasattr(r, "verdict") else {})))
        t = pd.DataFrame(res); t.to_csv(out / f"fp_rescore_{tag}.csv", index=False)
        return t
    aud = rescore(pd.read_csv(REPO / "analysis" / "inputs" / "xview" / "fp_audit.csv"), "audit60")
    p = pd.read_csv(a.b1h_holdout_preds); p = p[p.conf >= 0.25]
    fps = []
    for img, g in p.groupby("image"):
        W, H = sizes[Path(img).stem]
        _, gb = read_yolo_labels(root / "train" / "labels" / f"{Path(img).stem}.txt", W, H)
        bx = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        m = box_iou(bx, gb).max(1) if len(gb) else np.zeros(len(bx))
        fps.append(g[m < 0.1])
    fp = rescore(pd.concat(fps), "holdout_conf025")
    summ = dict(audit60=aud.group.value_counts().to_dict(),
                audit60_by_verdict={f"{k[0]} | {k[1]}": int(v) for k, v in aud.groupby(["verdict", "group"]).size().items()},
                audit60_xview_types=aud.xview_name.value_counts().to_dict(),
                holdout_conf025_n=len(fp), holdout_conf025=fp.group.value_counts().to_dict(),
                holdout_conf025_xview_types=fp.xview_name.value_counts().to_dict(), iou_threshold=0.3)
    (out / "fp_rescore_summary.json").write_text(json.dumps(summ, indent=2)); print("[fp]", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
