"""E10 label check (CPU): in E10's training labels for our 403 images, pairs where one of our 5-class (supplied) boxes
and an appended excluded-type xView box overlap at IoU >= 0.5, i.e. one truck labelled twice with conflicting classes.
Counted per (our class, xView type), one-to-one greedy. For comparison, the same count inside xView's own labels
(five-type box vs excluded-type box) on the same images: what E15-style xView-consistent labels would contain.

  python analysis/e10_label_conflicts.py --xview <mirror root> --data-root "$DATA_ROOT" --out <dir>
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402

FIVE = {24: "Cargo Truck", 25: "Truck w/Box", 28: "Truck w/Flatbed", 26: "Truck Tractor", 29: "Truck w/Liquid"}
EXCL = {20: "Pickup Truck", 21: "Utility Truck", 23: "Truck", 27: "Trailer", 32: "Crane Truck", 60: "Dump Truck",
        61: "Haul Truck", 65: "Cement Mixer"}
OURS = ["Cargo Truck", "Truck w/Box", "Truck w/Flatbed", "Truck Tractor", "Truck w/Liquid"]


def pairs(a, b, thr=0.5):
    if not len(a) or not len(b):
        return []
    iou = box_iou(a, b); out, ua, ub = [], set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if iou[i, j] < thr:
            break
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); out.append((int(i), int(j)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xview", required=True); ap.add_argument("--data-root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xv, root, out = Path(a.xview), Path(a.data_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hold = set(l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip())
    feats = json.load(open(xv / "train_labels" / "xView_train.geojson"))["features"]
    xb = defaultdict(list)
    for f in feats:
        p = f["properties"]; t = int(p["type_id"])
        if t in FIVE or t in EXCL:
            try:
                xb[Path(p["image_id"]).stem].append((t, [float(v) for v in p["bounds_imcoords"].split(",")]))
            except Exception:
                pass
    del feats
    im = pd.read_csv(EDA_TABLES / "images.csv").query("split == 'train'")
    e10, xvi, n_img = Counter(), Counter(), 0
    for r in im.itertuples():
        if r.image in hold:
            continue
        n_img += 1; s = Path(r.image).stem; W, H = int(r.width), int(r.height)
        gc, gb = read_yolo_labels(root / "train" / "labels" / f"{s}.txt", W, H)
        ex = [(t, b) for t, b in xb.get(s, []) if t in EXCL]; fv = [(t, b) for t, b in xb.get(s, []) if t in FIVE]
        eb = np.clip(np.array([b for _, b in ex]).reshape(-1, 4), 0, [W, H, W, H]); et = [t for t, _ in ex]
        for i, j in pairs(gb, eb):
            e10[(OURS[int(gc[i])], EXCL[et[j]])] += 1
        fb = np.array([b for _, b in fv]).reshape(-1, 4); ft = [t for t, _ in fv]
        for i, j in pairs(fb, eb):
            xvi[(FIVE[ft[i]], EXCL[et[j]])] += 1
    T = pd.DataFrame([dict(our_class=k[0], xview_excluded_type=k[1], e10_conflicts=v) for k, v in e10.items()])
    X = pd.DataFrame([dict(xview_five_type=k[0], xview_excluded_type=k[1], xview_internal_conflicts=v) for k, v in xvi.items()])
    T.sort_values("e10_conflicts", ascending=False).to_csv(out / "e10_conflicts.csv", index=False)
    X.sort_values("xview_internal_conflicts", ascending=False).to_csv(out / "xview_internal_conflicts.csv", index=False)
    s = dict(images=n_img, e10_conflict_pairs=int(sum(e10.values())), xview_internal_conflict_pairs=int(sum(xvi.values())))
    (out / "summary.json").write_text(json.dumps(s, indent=2)); print(json.dumps(s)); print(T.to_string(index=False)); print(X.to_string(index=False))


if __name__ == "__main__":
    main()
