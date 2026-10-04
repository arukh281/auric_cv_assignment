"""Compare two predictions.csv files on the images they share (reproducibility check for predict.py).

  python tools/compare_preds.py --a new/predictions.csv --b saved/predictions.csv [--conf 0.001] --out cmp.json

Per image: prediction counts at conf >= --conf, and a one-to-one pairing (same class, IoU >= 0.9, greedy by a's
confidence). Reports the share of predictions paired, max |conf| and max coordinate differences among pairs, and the
same for the top-100 predictions by confidence (the part that drives AP). Writes the summary to --out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.scoring import box_iou  # noqa: E402


def pair(a, b):
    ia, ib = [], []
    used = set()
    A = a[["x1", "y1", "x2", "y2"]].to_numpy(float); B = b[["x1", "y1", "x2", "y2"]].to_numpy(float)
    iou = box_iou(A, B) if len(A) and len(B) else np.zeros((len(A), len(B)))
    same = a.cls.to_numpy()[:, None] == b.cls.to_numpy()[None, :]
    iou = np.where(same, iou, 0)
    for i in np.argsort(-a.conf.to_numpy()):
        row = iou[i].copy(); row[list(used)] = 0
        j = int(np.argmax(row)) if len(row) else -1
        if j >= 0 and row[j] >= 0.9:
            used.add(j); ia.append(i); ib.append(j)
    return np.array(ia, int), np.array(ib, int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--conf", type=float, default=0.001); ap.add_argument("--out", required=True)
    o = ap.parse_args()
    A, B = pd.read_csv(o.a), pd.read_csv(o.b)
    res = {}
    for img in sorted(set(A.image) & set(B.image)):
        r = {}
        for tag, k in (("all", None), ("top100", 100)):
            a = A[(A.image == img) & (A.conf >= o.conf)].sort_values("conf", ascending=False).head(k).reset_index(drop=True)
            b = B[(B.image == img) & (B.conf >= o.conf)].sort_values("conf", ascending=False).head(k).reset_index(drop=True)
            ia, ib = pair(a, b)
            d = np.abs(a.loc[ia, ["x1", "y1", "x2", "y2"]].to_numpy() - b.loc[ib, ["x1", "y1", "x2", "y2"]].to_numpy())
            r[tag] = dict(n_a=len(a), n_b=len(b), paired=len(ia), paired_share_of_a=len(ia) / max(len(a), 1),
                          max_abs_conf_diff=float(np.abs(a.conf.to_numpy()[ia] - b.conf.to_numpy()[ib]).max()) if len(ia) else None,
                          max_abs_coord_diff_px=float(d.max()) if len(ia) else None)
        res[img] = r
    Path(o.out).write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
