"""E13: ensemble of existing models by weighted boxes fusion (WBF). CPU only, no training.

  python analysis/e13_ensemble.py --data-root "$DATA_ROOT" --run b1h=<dir> --run e4=<dir> --run e7=<dir> [--run e10=<dir>] \
      --out <dir> [--score-val]

Inputs: each run's saved merged predictions (<dir>/eval_holdout40/predictions.csv, <dir>/eval/predictions.csv), i.e.
after that model's usual tile merge (class-wise NMS, IoS 0.6, max_det 902). Fusion per image: ensemble_boxes
weighted_boxes_fusion over the models with EQUAL weights, iou_thr 0.55, skip_box_thr 0.001 (= eval conf), default
conf_type "avg" (a box found by fewer models gets a proportionally lower score); then top 902 by score. Coordinates are
normalised by the image size for WBF and mapped back. Scoring: eval.collect + detlib.scoring (COCO 101-point mAP50,
classes 0-4); mAP50 without Truck w/Liquid as well.
Combos: every set that contains b1h and at least one other run, plus b1h alone. Selection (pre-registered): the combo
with the highest holdout40 mAP50 among the ensembles. With --score-val, val is scored once, for that combo only (and
b1h alone is already known: 0.1065).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, list_images  # noqa: E402
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from eval import collect, image_sizes  # noqa: E402

COLS = ["image", "cls", "conf", "x1", "y1", "x2", "y2"]


def fuse(pred_list, sizes, iou=0.55, skip=0.001, max_det=902):
    from ensemble_boxes import weighted_boxes_fusion
    rows = []
    for n, (W, H) in sizes.items():
        bl, sl, ll = [], [], []
        for p in pred_list:
            q = p[p.image == n]
            b = q[["x1", "y1", "x2", "y2"]].to_numpy(float) / np.array([W, H, W, H], float)
            bl.append(np.clip(b, 0, 1).tolist()); sl.append(q.conf.tolist()); ll.append(q.cls.tolist())
        if not any(len(x) for x in bl):
            continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=[1] * len(pred_list), iou_thr=iou, skip_box_thr=skip)
        o = np.argsort(-s)[:max_det]
        rows += [dict(image=n, cls=int(l[k]), conf=float(s[k]), x1=b[k][0] * W, y1=b[k][1] * H, x2=b[k][2] * W, y2=b[k][3] * H) for k in o]
    return pd.DataFrame(rows, columns=COLS)


def score(preds, imgs, ldir, sizes):
    recs = score_images(collect(preds[preds.conf >= 0.001], imgs, ldir, sizes), 5)
    aps, m, _ = ap_from_records(recs, 5, "coco")
    return float(m), float(np.mean(aps[:4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True); ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--score-val", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = dict(r.split("=", 1) for r in a.run)
    hold = [l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip()]
    himgs = [root / "train" / "images" / n for n in hold]; hs = image_sizes(himgs)
    five = lambda d: d[d.cls < 5]  # amendment 2: only our 5 classes enter the ensemble (E10's extra classes dropped)
    H = {k: five(pd.read_csv(Path(v) / "eval_holdout40" / "predictions.csv")) for k, v in runs.items()}
    others = [k for k in runs if k != "b1h"]
    combos = [("b1h",)] + [("b1h",) + c for r in range(1, len(others) + 1) for c in itertools.combinations(others, r)]
    rows = []
    for c in combos:
        p = H["b1h"] if c == ("b1h",) else fuse([H[k] for k in c], hs)
        m, nl = score(p, himgs, root / "train" / "labels", hs)
        rows.append(dict(combo="+".join(c), holdout40_mAP50=m, holdout40_mAP50_no_liquid=nl)); print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / "holdout40_combos.csv", index=False)
    ens = R[R.combo != "b1h"]; best = ens.sort_values("holdout40_mAP50", ascending=False).iloc[0]
    res = dict(chosen=best.combo, chosen_holdout40=float(best.holdout40_mAP50),
               b1h_holdout40=float(R[R.combo == "b1h"].holdout40_mAP50.iloc[0]))
    res["margin_vs_b1h"] = res["chosen_holdout40"] - res["b1h_holdout40"]
    res["chosen_holdout40_no_liquid"] = float(best.holdout40_mAP50_no_liquid)
    res["b1h_holdout40_no_liquid"] = float(R[R.combo == "b1h"].holdout40_mAP50_no_liquid.iloc[0])
    # amendment 1: beat B1h by > 0.017 on holdout40 AND beat B1h on holdout40 mAP50 without Liquid
    res["passes"] = bool(res["margin_vs_b1h"] > 0.017 and res["chosen_holdout40_no_liquid"] > res["b1h_holdout40_no_liquid"])
    if a.score_val:
        vimgs = list_images(root / "val" / "images"); vs = image_sizes(vimgs)
        V = [five(pd.read_csv(Path(runs[k]) / "eval" / "predictions.csv")) for k in best.combo.split("+")]
        pv = fuse(V, vs); pv.to_csv(out / "val_preds_chosen.csv", index=False)
        m, nl = score(pv, vimgs, root / "val" / "labels", vs); res.update(val_mAP50=m, val_mAP50_no_liquid=nl)
    (out / "e13_result.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
