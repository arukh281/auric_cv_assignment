"""Checks requested by the external review (CPU only; diagnostics, not results).

  python analysis/review_checks.py --data-root "$DATA_ROOT" --b1h <B1h run dir> --xview <mirror root> \
      --e10-list <e10_train_list.txt> --out <dir>

1. max_det: B1h holdout40 raw tile predictions re-merged (class-wise NMS, IoS 0.6) with max_det 902 / 3000 / 10000.
2. class-agnostic merge: NMS on IoS 0.6 ignoring class (each kept box keeps its own class), max_det 902.
3. AP50 per class at IoU 0.1 vs 0.5 on holdout40 (merged predictions as saved).
4. top-1 / top-2 accuracy of the GT-box oracle (pool scores), val and holdout40 (from the gt_scores.csv copies in
   analysis/inputs/review/).
5. density: GT boxes per megapixel, train vs val (figures/eda/tables): pooled and per-image median.
6. geography: footprints from GeoTIFF tags (ModelTiepoint + ModelPixelScale) for our 465 images and every extra E10
   image; overlapping or touching pairs: E10-extra vs val, E10-extra vs holdout40, train vs val, train vs holdout40.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES  # noqa: E402
from detlib.merge import nms  # noqa: E402
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from eval import collect, finalize, image_sizes  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
NAMES = ["Cargo Truck", "Truck w/Box", "Truck w/Flatbed", "Truck Tractor", "Truck w/Liquid"]


def mAP(preds, imgs, L, sizes, iou=0.5):
    aps, m, _ = ap_from_records(score_images(collect(preds[preds.conf >= 0.001], imgs, L, sizes), 5, iou_thr=iou), 5, "coco")
    return float(m), [float(x) for x in aps]


def footprint(p):
    with Image.open(p) as t:
        tg = t.tag_v2; tp, sc = tg.get(33922), tg.get(33550); w, h = t.size
    if not tp or not sc:
        return None
    return (tp[3], tp[4] - sc[1] * h, tp[3] + sc[0] * w, tp[4])  # lon0, lat0, lon1, lat1


def touch(a, b):
    return min(a[2], b[2]) >= max(a[0], b[0]) and min(a[3], b[3]) >= max(a[1], b[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--data-root", "--b1h", "--xview", "--e10-list", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    root, B, xv, out = Path(a.data_root), Path(a.b1h), Path(a.xview), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hold = [l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip()]
    imgs = [root / "train" / "images" / n for n in hold]; sizes = image_sizes(imgs); L = root / "train" / "labels"
    res = {}
    raw = pd.read_csv(B / "eval_holdout40" / "predictions_raw.csv")
    res["1_max_det"] = {str(md): mAP(finalize(raw, sizes, "nms", 0.6, "ios", md), imgs, L, sizes)[0] for md in (902, 3000, 10000)}
    rows = []
    for n, g in raw.groupby("image"):
        xy, cf, cl = g[["x1", "y1", "x2", "y2"]].to_numpy(float), g.conf.to_numpy(float), g.cls.to_numpy(int)
        k = nms(xy, cf, np.zeros_like(cl), 0.6, "ios")[:902]
        rows += [dict(image=n, cls=int(cl[i]), conf=float(cf[i]), x1=xy[i, 0], y1=xy[i, 1], x2=xy[i, 2], y2=xy[i, 3]) for i in k]
    res["2_class_agnostic_merge"] = mAP(pd.DataFrame(rows), imgs, L, sizes)[0]
    merged = pd.read_csv(B / "eval_holdout40" / "predictions.csv")
    m5, a5 = mAP(merged, imgs, L, sizes, 0.5); m1, a1 = mAP(merged, imgs, L, sizes, 0.1)
    res["3_iou"] = dict(iou05=dict(mAP50=m5, per_class=dict(zip(NAMES, a5))), iou01=dict(mAP=m1, per_class=dict(zip(NAMES, a1))))
    top = {}
    for tag, f in (("val", "val_gt_scores.csv"), ("holdout40", "holdout_gt_scores.csv")):
        g = pd.read_csv(REPO / "analysis" / "inputs" / "review" / f)
        sc = g[[f"pool_{n}" for n in NAMES]].to_numpy(); order = np.argsort(-sc, 1)
        top[tag] = dict(n=len(g), top1=float((order[:, 0] == g.gt_cls).mean()),
                        top2=float(((order[:, 0] == g.gt_cls) | (order[:, 1] == g.gt_cls)).mean()))
    res["4_oracle_topk"] = top
    im = pd.read_csv(EDA_TABLES / "images.csv"); bx = pd.read_csv(EDA_TABLES / "boxes.csv")
    cnt = bx.groupby(["split", "image"]).size().rename("n").reset_index()
    im = im.merge(cnt, on=["split", "image"], how="left").fillna({"n": 0}); im["mp"] = im.width * im.height / 1e6
    res["5_density_boxes_per_MP"] = {s: dict(pooled=float(g.n.sum() / g.mp.sum()), per_image_median=float((g.n / g.mp).median()),
                                             boxes_per_image_mean=float(g.n.mean()))
                                     for s, g in im.groupby("split")}
    tif = lambda s: xv / "train_images" / "train_images" / f"{s}.tif"
    hs = set(Path(h).stem for h in hold); vs = set(Path(i).stem for i in im[im.split == "val"].image)
    ts = set(Path(i).stem for i in im[im.split == "train"].image) - hs
    ex = set(Path(l.strip()).stem for l in Path(a.e10_list).read_text().splitlines() if l.strip().endswith(".tif"))
    F = {s: footprint(tif(s)) for s in hs | vs | ts | ex}
    def pairs(A, Bs):
        return sorted((x, y) for x in A for y in Bs if F.get(x) and F.get(y) and touch(F[x], F[y]))
    geo = dict(footprints=sum(v is not None for v in F.values()), n_e10_extra=len(ex))
    for name, A_, B_ in (("e10extra_vs_val", ex, vs), ("e10extra_vs_holdout40", ex, hs), ("train_vs_val", ts, vs), ("train_vs_holdout40", ts, hs)):
        p = pairs(A_, B_); geo[name] = dict(n_pairs=len(p), pairs=p[:200])
    res["6_geography"] = geo
    (out / "review_checks.json").write_text(json.dumps(res, indent=2)); print(json.dumps({k: v for k, v in res.items() if k != "6_geography"}, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else dict(n_pairs=v["n_pairs"], first=v["pairs"][:10])) for k, v in geo.items()}, indent=2))


if __name__ == "__main__":
    main()
