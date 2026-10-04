"""Our images and labels vs the xView originals, all 465 images (train, holdout40, val). CPU only, evaluation only.

  python analysis/xview_pixels.py --xview <mirror root> --data-root /kaggle/tmp/data --out <dir>

1. Pixels: our PNG vs xView's TIFF (same stem). If the sizes differ, ours is resized to xView's size (bilinear) for
   the statistics, and xView is also resized to OUR size with nearest / bilinear / bicubic / area / lanczos (cv2) to
   identify how the copy was made (lowest mean absolute difference wins). Per image: identical (all pixels equal),
   MAE, max |diff|, share of equal pixels, mean brightness difference (ours - xView, grey), RMS-contrast ratio,
   Laplacian-variance ratio (blur), Immerkaer noise ratio, per-channel mean shift, a gamma fit (ours = xView^g on
   normalised grey, least squares in log space), and whether swapping R and B lowers the MAE.
2. Labels per split: our boxes vs xView boxes of the five types (24 Cargo, 25 Box, 26 Tractor, 28 Flatbed,
   29 Liquid), one-to-one by IoU >= 0.5 (class-agnostic, greedy). For the 4 rescaled val images, our boxes are scaled
   to xView coordinates. Per split: paired same class, paired changed class (with direction), shifted (paired with
   IoU < 0.95), ours unpaired ("extra"), xView unpaired ("missing").
Writes <out>/pixels.csv, pixels_summary.json, label_compare_by_split.csv, label_changes_by_split.csv.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402

XV2OURS = {24: 0, 25: 1, 28: 2, 26: 3, 29: 4}
NAMES = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]
INTERP = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC,
          "area": cv2.INTER_AREA, "lanczos": cv2.INTER_LANCZOS4}


def immerkaer(g):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = g.shape
    return float(np.abs(cv2.filter2D(g.astype(np.float64), -1, k))[1:-1, 1:-1].sum() * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def stats(o, x):
    """o, x: RGB uint8, same shape."""
    go, gx = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY), cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    d = o.astype(np.int16) - x.astype(np.int16)
    sw = np.abs(o[..., ::-1].astype(np.int16) - x.astype(np.int16)).mean()
    m = (gx > 5) & (gx < 250) & (go > 0)
    gam = float(np.polyfit(np.log(gx[m] / 255.0), np.log(go[m] / 255.0), 1)[0]) if m.sum() > 100 else np.nan
    lap = lambda g: cv2.Laplacian(g, cv2.CV_64F).var()
    return dict(identical=bool((d == 0).all()), mae=float(np.abs(d).mean()), max_abs=int(np.abs(d).max()),
                share_equal=float((d == 0).all(-1).mean()), brightness_diff=float(go.mean() - gx.mean()),
                contrast_ratio=float(go.std() / max(gx.std(), 1e-9)), blur_ratio=float(lap(go) / max(lap(gx), 1e-9)),
                noise_ratio=float(immerkaer(go) / max(immerkaer(gx), 1e-9)),
                shift_r=float(d[..., 0].mean()), shift_g=float(d[..., 1].mean()), shift_b=float(d[..., 2].mean()),
                gamma=gam, rb_swap_lowers_mae=bool(sw < np.abs(d).mean()))


def pairs(a, b, thr=0.5):
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
    ap.add_argument("--xview", required=True); ap.add_argument("--data-root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xv, root, out = Path(a.xview), Path(a.data_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hold = set(l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip())
    im = pd.read_csv(EDA_TABLES / "images.csv")
    feats = json.load(open(xv / "train_labels" / "xView_train.geojson"))["features"]
    xb = defaultdict(list)
    for f in feats:
        p = f["properties"]; t = int(p["type_id"])
        if t in XV2OURS:
            try:
                xb[Path(p["image_id"]).stem].append((XV2OURS[t], [float(v) for v in p["bounds_imcoords"].split(",")]))
            except Exception:
                pass
    del feats
    prow, lab, chg = [], defaultdict(Counter), defaultdict(Counter)
    for r in im.itertuples():
        split = "val" if r.split == "val" else ("holdout40" if r.image in hold else "train")
        stem = Path(r.image).stem
        x = cv2.cvtColor(cv2.imread(str(xv / "train_images" / "train_images" / f"{stem}.tif"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        o = cv2.cvtColor(cv2.imread(str(root / r.split / "images" / r.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rec = dict(split=split, image=r.image, ours_w=o.shape[1], ours_h=o.shape[0], xview_w=x.shape[1], xview_h=x.shape[0])
        rescaled = o.shape[:2] != x.shape[:2]
        if rescaled:
            best = None
            for nm, f in INTERP.items():
                e = float(np.abs(cv2.resize(x, (o.shape[1], o.shape[0]), interpolation=f).astype(np.int16) - o.astype(np.int16)).mean())
                rec[f"mae_if_xview_resized_{nm}"] = e
                if best is None or e < best[1]:
                    best = (nm, e)
            rec["best_resize_method"] = best[0]
            xr = cv2.resize(x, (o.shape[1], o.shape[0]), interpolation=INTERP[best[0]])
            rec.update({f"after_best_resize_{k}": v for k, v in stats(o, xr).items()})
            o_cmp = cv2.resize(o, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            o_cmp = o
        rec.update(stats(o_cmp, x)); rec["rescaled"] = rescaled
        prow.append(rec)
        gc, gb = read_yolo_labels(root / r.split / "labels" / f"{stem}.txt", o.shape[1], o.shape[0])
        if rescaled:
            gb = gb * np.array([x.shape[1] / o.shape[1], x.shape[0] / o.shape[0]] * 2)
        xx = xb.get(stem, []); xc = np.array([c for c, _ in xx], int); xbb = np.array([b for _, b in xx]).reshape(-1, 4)
        pr = pairs(gb, xbb)
        L = lab[split]; L["our_boxes"] += len(gb); L["xview_boxes"] += len(xbb); L["images"] += 1
        for i, j, iou in pr:
            if gc[i] == xc[j]:
                L["paired_same_class"] += 1
            else:
                L["paired_changed_class"] += 1; chg[split][f"ours {NAMES[gc[i]]} <- xView {NAMES[xc[j]]}"] += 1
            if iou < 0.95:
                L["shifted_iou_lt_0.95"] += 1
        L["extra_ours_unpaired"] += len(gb) - len(pr); L["missing_xview_unpaired"] += len(xbb) - len(pr)
        print(f"[px] {r.image} {split} identical={rec['identical']} mae={rec['mae']:.2f}", flush=True)
    P = pd.DataFrame(prow); P.to_csv(out / "pixels.csv", index=False)
    LC = pd.DataFrame(lab).T.fillna(0).astype(int)
    LC["changed_class_rate"] = LC.paired_changed_class / (LC.paired_same_class + LC.paired_changed_class)
    LC.to_csv(out / "label_compare_by_split.csv")
    CH = pd.DataFrame(chg).fillna(0).astype(int); CH.to_csv(out / "label_changes_by_split.csv")
    summ = {}
    for sp, g in P.groupby("split"):
        summ[sp] = dict(n=len(g), identical=int(g.identical.sum()), rescaled=int(g.rescaled.sum()),
                        mae_median=float(g.mae.median()), mae_max=float(g.mae.max()),
                        brightness_diff_median=float(g.brightness_diff.median()),
                        non_identical_images=g[~g.identical].image.tolist()[:50])
    focus = ["2543.png", "2292.png", "2470.png", "2472.png", "2308.png", "2391.png", "2384.png", "2460.png"]
    summ["focus"] = P[P.image.isin(focus)].to_dict("records")
    (out / "pixels_summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "focus"}, indent=2)); print(LC.to_string()); print(CH.to_string())


if __name__ == "__main__":
    main()
