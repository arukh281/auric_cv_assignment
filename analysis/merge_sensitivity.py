"""Re-score saved raw (pre-merge) tile predictions under different cross-tile merge settings. No re-inference.

Usage:
  python analysis/merge_sensitivity.py --raw runs/b1_tile1024/eval/predictions_raw.csv --name b1_tile1024

Writes figures/<name>/merge_sensitivity.csv: one row per setting with mAP50 (COCO 101-pt, our scorer),
per-class AP50 and the number of predictions kept. Settings: class-wise NMS on IoS 0.5/0.6/0.7 and IoU 0.5,
plus "none" (no cross-tile merge) as a reference. Post-processing is eval.finalize, identical to eval.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import eda_max_boxes_per_image, list_images, load_classes  # noqa: E402
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from eval import collect, finalize, image_sizes  # noqa: E402

SETTINGS = [("nms", "ios", 0.5), ("nms", "ios", 0.6), ("nms", "ios", 0.7), ("nms", "iou", 0.5), ("none", "-", None)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="predictions_raw.csv written by eval.py")
    ap.add_argument("--name", required=True, help="figures/<name>/")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, help="default: same rule as eval.py")
    ap.add_argument("--out-root", default="figures")
    a = ap.parse_args()
    max_det = a.max_det or max(300, 2 * eda_max_boxes_per_image("val"))
    root, out = Path(a.data_root), Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    names = load_classes(root)
    images = list_images(root / "val" / "images")
    sizes = image_sizes(images)
    raw = pd.read_csv(a.raw)
    raw = raw[raw.conf >= a.conf]  # every val image is scored, so images with no predictions count their GT as FN

    rows = []
    for method, metric, thr in SETTINGS:
        preds = finalize(raw, sizes, method, thr, metric if metric != "-" else "iou", max_det)
        recs = score_images(collect(preds, images, root / "val" / "labels", sizes), len(names))
        aps, m, _ = ap_from_records(recs, len(names), "coco")
        row = dict(merge=method, metric=metric, thr=thr, n_raw=len(raw), n_kept=len(preds), mAP50=m)
        row.update({f"AP50 {names[c]}": aps[c] for c in names})
        rows.append(row)
        print(f"[merge] {method} {metric} {thr}: mAP50={m:.4f} kept={len(preds)}")
    t = pd.DataFrame(rows)
    t.insert(0, "default", (t.merge == "nms") & (t.metric == "ios") & (t.thr == 0.6))
    t.to_csv(out / "merge_sensitivity.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[merge] wrote {out / 'merge_sensitivity.csv'}")


if __name__ == "__main__":
    main()
