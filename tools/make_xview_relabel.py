"""E15 data: our images with xView's ORIGINAL class and box for every supplied box that pairs with an xView box of the
five types (one-to-one, IoU >= 0.5, class-agnostic, greedy); unpaired supplied boxes keep their supplied label;
xView boxes with no supplied partner are NOT added. Applied to all 443 train images (the 40 holdout40 images are
relabelled the same way, for evaluation only: training excludes them through the config's holdout_list).

  python tools/make_xview_relabel.py --xview <mirror root> --data-root /kaggle/tmp/data --out /kaggle/tmp/xvr_data

Writes <out>/classmap.txt (our 5 classes), train/images (symlinks), train/labels (relabelled), val -> our val
(unchanged, never relabelled), relabel_summary.json (per split: paired, class changed, box replaced, unpaired kept).
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402

XV2OURS = {24: 0, 25: 1, 28: 2, 26: 3, 29: 4}


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


def relabel(gc, gb, xc, xb):
    """Returns new (cls, boxes) and counts."""
    nc, nb = gc.copy(), gb.copy(); c = Counter()
    for i, j in pairs(gb, xb):
        c["paired"] += 1
        if nc[i] != xc[j]:
            c["class_changed"] += 1
        nc[i], nb[i] = xc[j], xb[j]
    c["unpaired_kept"] = len(gc) - c["paired"]
    return nc, nb, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xview", required=True); ap.add_argument("--data-root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xv, root, out = Path(a.xview), Path(a.data_root), Path(a.out)
    (out / "train" / "images").mkdir(parents=True, exist_ok=True); (out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "classmap.txt", out / "classmap.txt")
    if not (out / "val").exists():
        os.symlink(root / "val", out / "val")
    hold = set(l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip())
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
    im = pd.read_csv(EDA_TABLES / "images.csv").query("split == 'train'")
    tot = defaultdict(Counter)
    for r in im.itertuples():
        s = Path(r.image).stem; W, H = int(r.width), int(r.height)
        gc, gb = read_yolo_labels(root / "train" / "labels" / f"{s}.txt", W, H)
        x = xb.get(s, []); xc = np.array([c for c, _ in x], int); xbb = np.clip(np.array([b for _, b in x]).reshape(-1, 4), 0, [W, H, W, H])
        nc, nb, c = relabel(gc, gb, xc, xbb)
        tot["holdout40" if r.image in hold else "train"].update(c)
        lines = [f"{int(k)} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}" for k, b in zip(nc, nb)]
        (out / "train" / "labels" / f"{s}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        os.symlink(root / "train" / "images" / r.image, out / "train" / "images" / r.image)
    summ = {k: dict(v) for k, v in tot.items()}
    (out / "relabel_summary.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
