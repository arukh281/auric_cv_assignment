"""Background false-positive audit (CPU only): the N highest-confidence predictions on the holdout40 images that
overlap NO ground-truth box (class-agnostic IoU < --max-iou with every GT box), cropped with context into one sheet.

Usage (Kaggle CPU kernel, after scripts/kaggle_setup.sh):
  python analysis/fp_audit.py --preds <B1h>/eval_holdout40/predictions.csv --data-root /kaggle/tmp/data --out <dir>

Crop: square, side max(--min-side, --context x the box's longer side), centred on the box, clipped to the image,
resized to --cell px; the prediction is drawn in red, GT boxes inside the crop in green (so a near-miss is visible).
Caption: rank, predicted class, confidence. Writes <out>/fp_audit_sheet.png and fp_audit.csv (one row per crop, with
an empty `verdict` column for the manual count: truck / other_vehicle / background / unclear).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from detlib.data import EDA_TABLES, load_classes, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SHORT = {0: "Cargo", 1: "Box", 2: "Flatbed", 3: "Tractor", 4: "Liquid"}


def unmatched(preds, gts, max_iou):
    """preds: DataFrame (image, cls, conf, x1..y2); gts: {image: xyxy array}. Adds max_gt_iou; keeps < max_iou."""
    out = []
    for img, p in preds.groupby("image"):
        g = gts.get(img, np.zeros((0, 4)))
        b = p[["x1", "y1", "x2", "y2"]].to_numpy(float)
        m = box_iou(b, g).max(1) if len(g) else np.zeros(len(b))
        out.append(p.assign(max_gt_iou=m))
    t = pd.concat(out) if out else preds.assign(max_gt_iou=[])
    return t[t.max_gt_iou < max_iou].sort_values("conf", ascending=False)


def crop_window(b, W, H, context, min_side):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(min_side, context * max(b[2] - b[0], b[3] - b[1]))
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    return x0, y0, int(min(W, x0 + s)), int(min(H, y0 + s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-iou", type=float, default=0.1)
    ap.add_argument("--context", type=float, default=4.0)
    ap.add_argument("--min-side", type=int, default=96)
    ap.add_argument("--cell", type=int, default=224)
    ap.add_argument("--cols", type=int, default=10)
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(a.preds)
    im = pd.read_csv(EDA_TABLES / "images.csv").query("split == 'train'")
    sizes = {r.image: (int(r.width), int(r.height)) for r in im.itertuples()}
    gts = {}
    for img in preds.image.unique():
        W, H = sizes[img]
        gts[img] = read_yolo_labels(root / "train" / "labels" / f"{Path(img).stem}.txt", W, H)[1]
    top = unmatched(preds, gts, a.max_iou).head(a.n).reset_index(drop=True)
    rows = (len(top) + a.cols - 1) // a.cols
    cap = 16
    sheet = Image.new("RGB", (a.cols * a.cell, rows * (a.cell + cap)), "white")
    recs, cache = [], {}
    for i, r in top.iterrows():
        if r.image not in cache:
            cache = {r.image: Image.open(root / "train" / "images" / r.image).convert("RGB")}
        img = cache[r.image]
        b = np.array([r.x1, r.y1, r.x2, r.y2], float)
        x0, y0, x1, y1 = crop_window(b, *img.size, a.context, a.min_side)
        c = img.crop((x0, y0, x1, y1))
        k = a.cell / max(c.size)
        c = c.resize((max(1, round(c.size[0] * k)), max(1, round(c.size[1] * k))))
        d = ImageDraw.Draw(c)
        for g in gts[r.image]:
            if g[2] > x0 and g[0] < x1 and g[3] > y0 and g[1] < y1:
                d.rectangle([(g[0] - x0) * k, (g[1] - y0) * k, (g[2] - x0) * k, (g[3] - y0) * k], outline=(0, 230, 0), width=1)
        d.rectangle([(b[0] - x0) * k, (b[1] - y0) * k, (b[2] - x0) * k, (b[3] - y0) * k], outline=(255, 0, 0), width=2)
        X, Y = (i % a.cols) * a.cell, (i // a.cols) * (a.cell + cap)
        sheet.paste(c, (X, Y + cap))
        ImageDraw.Draw(sheet).text((X + 2, Y + 2), f"#{i + 1} {SHORT.get(int(r.cls), r.cls)} {r.conf:.2f}", fill=(0, 0, 0))
        recs.append(dict(rank=i + 1, image=r.image, cls=int(r.cls), conf=float(r.conf), max_gt_iou=float(r.max_gt_iou),
                         x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, box_w=r.x2 - r.x1, box_h=r.y2 - r.y1,
                         crop=f"{x0},{y0},{x1},{y1}", verdict=""))
    sheet.save(out / "fp_audit_sheet.png")
    pd.DataFrame(recs).to_csv(out / "fp_audit.csv", index=False)
    n_un = len(unmatched(preds, gts, a.max_iou))
    print(f"[fp_audit] {len(preds)} predictions, {n_un} with IoU < {a.max_iou} to every GT; sheet of top {len(top)}; "
          f"conf range {top.conf.min():.3f}-{top.conf.max():.3f}", flush=True)


if __name__ == "__main__":
    main()
