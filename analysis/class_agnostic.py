"""Class-agnostic re-scoring of saved predictions: every class becomes "truck". No GPU, no inference.

Usage:
  python analysis/class_agnostic.py --preds runs/b1_tile1024/eval/predictions.csv [--data-root data]

Images and split are those of the eval that wrote the predictions: if metrics.json next to --preds lists
`sampled_images` (eval.py --split train), those train images; otherwise the val images (first --max-images of
eval_args.json, if that was set). Scorer: detlib/scoring.py, IoU 0.5, COCO 101-point AP, as in eval.py.

Writes class_agnostic.json next to --preds:
  mAP50_per_class           mean over classes of per-class AP50 (same as eval.py's headline on the same CSV)
  AP50_class_agnostic       AP50 with every prediction and GT set to one class
  recall_agnostic_conf0.25  / recall_agnostic_conf0.001: matched GT / all GT at IoU 0.5, conf >= threshold
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import list_images, load_classes  # noqa: E402
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from eval import collect  # noqa: E402

RECALL_CONFS = (0.25, 0.001)


def agnostic(per):
    return [dict(d, p_cls=np.zeros(len(d["p_cls"]), int), g_cls=np.zeros(len(d["g_cls"]), int)) for d in per]


def recall(per, conf):
    recs = score_images([dict(d, p_xyxy=d["p_xyxy"][d["p_conf"] >= conf], p_cls=d["p_cls"][d["p_conf"] >= conf],
                              p_conf=d["p_conf"][d["p_conf"] >= conf]) for d in per], 1)
    n_gt = int(sum(r["n_gt"][0] for r in recs))
    n_tp = int(sum(r["tp"].sum() for r in recs))
    return n_tp, n_gt, (n_tp / n_gt if n_gt else float("nan"))


def scores(per, nc):
    """(per-class AP array, per-class mAP50, class-agnostic AP50, {conf: (n_tp, n_gt, recall)})."""
    aps, m, _ = ap_from_records(score_images(per, nc), nc, "coco")
    ag = agnostic(per)
    _, m_ag, _ = ap_from_records(score_images(ag, 1), 1, "coco")
    return aps, m, m_ag, {c: recall(ag, c) for c in RECALL_CONFS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="predictions.csv written by eval.py")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    src = Path(a.preds).parent
    root = Path(a.data_root)
    mj = json.loads((src / "metrics.json").read_text()) if (src / "metrics.json").exists() else {}
    ea = json.loads((src / "eval_args.json").read_text()) if (src / "eval_args.json").exists() else {}
    if mj.get("sampled_images"):
        split = "train"
        images = [root / "train" / "images" / n for n in mj["sampled_images"]]
    else:
        split = "val"
        images = list_images(root / "val" / "images")[: ea.get("max_images")]
    names = load_classes(root)
    conf_floor = ea.get("conf", 0.001)
    preds = pd.read_csv(a.preds)
    preds = preds[preds.conf >= conf_floor]
    per = collect(preds, images, root / split / "labels")
    aps, m, m_ag, rec = scores(per, len(names))
    res = dict(input=str(a.preds), split=split, n_images=len(images), conf_floor=conf_floor, iou=0.5,
               ap_method="coco 101-point (detlib/scoring.py)",
               mAP50_per_class=m, per_class_AP50={names[c]: aps[c] for c in names}, AP50_class_agnostic=m_ag,
               eval_metrics_mAP50=mj.get("mAP50"))
    for c, (tp, n, r) in rec.items():
        res[f"recall_agnostic_conf{c:g}"] = r
        res[f"n_matched_gt_conf{c:g}"] = tp
    res["n_gt"] = rec[RECALL_CONFS[0]][1]
    (src / "class_agnostic.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
