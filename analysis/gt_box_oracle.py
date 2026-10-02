"""Investigation 5.1, GT-box oracle: if every truck were found and boxed perfectly, how well does the detector
tell the five classes apart?

Usage:
  python analysis/gt_box_oracle.py --config configs/b1.yaml --runs-root runs [--weights .../last.pt]

For every val GT box, the detector's per-class scores at that box are read from the raw (pre-NMS,
pre-threshold) detection head: sigmoid class scores of every anchor whose decoded box has IoU >= --iou with
the GT, max-pooled per class. If no anchor reaches --iou, the anchor with the highest IoU is used and the
box is flagged (no_anchor_at_iou). The predicted class is the argmax. Localization and recall are taken out,
so what remains is classification only.

Input to the network, same geometry as eval.py:
  sliced (B1): the val image is cut with detlib.tiling.tile_windows(tile, overlap); each GT is read from the
               tile that contains it whole with the largest margin to the tile edge. Tiles go in at imgsz.
  full (B0):   the whole image, letterboxed to imgsz.
  Letterbox: aspect-preserving resize to imgsz, centered gray (114) padding, RGB / 255, float32.

Writes figures/<name>/gt_oracle/: gt_scores.csv (one row per GT: scores, argmax, IoU of the best anchor),
confusion.csv / confusion_norm.csv / confusion.png (rows = GT class, cols = argmax class), per_class.csv
(n, correct, accuracy = recall of the class; precision of the argmax), summary.json, oracle_args.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import list_images, load_classes, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402
from detlib.tiling import tile_windows  # noqa: E402


def letterbox(img, size):
    """img HxWx3 BGR -> (size x size x 3 RGB float32 in [0, 1], scale r, pad (px, py)). Inverse: (x - pad) / r."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = img
    return out[:, :, ::-1].astype(np.float32) / 255.0, r, (px, py)


def head_outputs(net, crops, size, device="cpu"):
    """Raw head of an Ultralytics detection model for a list of BGR crops.

    Returns per crop (xyxy of every anchor in crop pixels (A, 4), class scores (A, nc)).
    In eval mode the Detect head returns (B, 4 + nc, A): xywh in input pixels, then sigmoid class scores.
    """
    import torch
    lb = [letterbox(c, size) for c in crops]
    x = torch.from_numpy(np.stack([l[0] for l in lb]).transpose(0, 3, 1, 2).copy()).to(device)
    with torch.no_grad():
        y = net(x)
    y = (y[0] if isinstance(y, (list, tuple)) else y).float().cpu().numpy()
    out = []
    for k, (_, r, (px, py)) in enumerate(lb):
        xc, yc, bw, bh = y[k, 0], y[k, 1], y[k, 2], y[k, 3]
        xyxy = np.stack([xc - bw / 2 - px, yc - bh / 2 - py, xc + bw / 2 - px, yc + bh / 2 - py], 1) / r
        out.append((xyxy, y[k, 4:].T))
    return out


def scores_at_boxes(anchors_xyxy, anchor_scores, gt_xyxy, iou_thr=0.5):
    """Per GT: class scores max-pooled over anchors with IoU >= iou_thr (else the best-IoU anchor),
    best anchor IoU, number of anchors at iou_thr."""
    iou = box_iou(gt_xyxy, anchors_xyxy)
    nc = anchor_scores.shape[1]
    s = np.zeros((len(gt_xyxy), nc))
    best = iou.max(1) if iou.size else np.zeros(len(gt_xyxy))
    n_at = (iou >= iou_thr).sum(1) if iou.size else np.zeros(len(gt_xyxy), int)
    for g in range(len(gt_xyxy)):
        sel = iou[g] >= iou_thr
        s[g] = anchor_scores[sel].max(0) if sel.any() else anchor_scores[int(iou[g].argmax())]
    return s, best, n_at


def assign_tiles(gt_xyxy, wins):
    """Index of the window that holds each GT whole with the largest edge margin (else the most visible one)."""
    out = []
    for b in gt_xyxy:
        margin = np.array([min(b[0] - x0, b[1] - y0, x1 - b[2], y1 - b[3]) for x0, y0, x1, y1 in wins])
        if (margin >= 0).any():
            out.append(int(margin.argmax()))
        else:
            vis = [max(0, min(b[2], x1) - max(b[0], x0)) * max(0, min(b[3], y1) - max(b[1], y0)) for x0, y0, x1, y1 in wins]
            out.append(int(np.argmax(vis)))
    return np.array(out, int)


def image_scores(net, img, gt_xyxy, mode, imgsz, tile, overlap, iou_thr, batch=8, device="cpu"):
    h, w = img.shape[:2]
    wins = tile_windows(w, h, tile, overlap) if mode == "sliced" else [(0, 0, w, h)]
    asg = assign_tiles(gt_xyxy, wins)
    nc_scores, best, n_at = None, np.zeros(len(gt_xyxy)), np.zeros(len(gt_xyxy), int)
    used = sorted(set(asg.tolist()))
    for s in range(0, len(used), batch):
        chunk = used[s:s + batch]
        outs = head_outputs(net, [img[wins[k][1]:wins[k][3], wins[k][0]:wins[k][2]] for k in chunk], imgsz, device)
        for k, (xyxy, sc) in zip(chunk, outs):
            g = np.where(asg == k)[0]
            local = gt_xyxy[g] - np.array([wins[k][0], wins[k][1], wins[k][0], wins[k][1]], float)
            sg, bg, ng = scores_at_boxes(xyxy, sc, local, iou_thr)
            if nc_scores is None:
                nc_scores = np.zeros((len(gt_xyxy), sc.shape[1]))
            nc_scores[g], best[g], n_at[g] = sg, bg, ng
    return (nc_scores if nc_scores is not None else np.zeros((0, 0))), best, n_at


def oracle_tables(gt_cls, scores, names):
    """Confusion (GT x argmax), per-class accuracy/precision and overall accuracy."""
    nc = len(names)
    pred = scores.argmax(1) if len(scores) else np.zeros(0, int)
    cm = np.zeros((nc, nc), int)
    np.add.at(cm, (gt_cls, pred), 1)
    rows = []
    for c, n in names.items():
        tot, col = cm[c].sum(), cm[:, c].sum()
        rows.append(dict(cls=c, class_name=n, n_gt=int(tot), n_correct=int(cm[c, c]),
                         accuracy=cm[c, c] / tot if tot else np.nan, n_predicted=int(col),
                         precision=cm[c, c] / col if col else np.nan))
    per = pd.DataFrame(rows)
    summary = dict(n_gt=int(cm.sum()), accuracy=float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
                   mean_class_accuracy=float(per.accuracy.mean(skipna=True)))
    return cm, per, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/*.yaml; the `eval` section gives mode/imgsz/tile/overlap")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--weights", help="default: <runs-root>/<name>/train/weights/last.pt")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-root", default="figures")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--max-images", type=int, help="testing only")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    cfg = yaml.safe_load(open(a.config))
    ev = {**dict(mode="full", imgsz=640, tile=1024, overlap=256), **(cfg.get("eval") or {})}
    name = cfg.get("name", "adhoc")
    weights = a.weights or str(Path(a.runs_root) / name / "train" / "weights" / "last.pt")
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = YOLO(weights).model.float().eval().to(device)
    root, out = Path(a.data_root), Path(a.out_root) / name / "gt_oracle"
    out.mkdir(parents=True, exist_ok=True)
    names = load_classes(root)

    rows = []
    for i, p in enumerate(list_images(root / "val" / "images")[: a.max_images]):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        gc, gb = read_yolo_labels(root / "val" / "labels" / f"{p.stem}.txt", w, h)
        if not len(gc):
            continue
        s, best, n_at = image_scores(net, img, gb, ev["mode"], ev["imgsz"], ev["tile"], ev["overlap"], a.iou,
                                     a.batch, device)
        for j in range(len(gc)):
            rows.append(dict(image=p.name, gt_idx=j, gt_cls=int(gc[j]), pred_cls=int(s[j].argmax()),
                             best_anchor_iou=best[j], n_anchors_at_iou=int(n_at[j]),
                             no_anchor_at_iou=bool(n_at[j] == 0), **{f"s_{names[c]}": s[j, c] for c in names}))
        print(f"[oracle] {i + 1} {p.name}: {len(gc)} GT", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "gt_scores.csv", index=False)
    scores = t[[f"s_{names[c]}" for c in names]].to_numpy()
    cm, per, summary = oracle_tables(t.gt_cls.to_numpy(int), scores, names)
    summary.update(n_no_anchor_at_iou=int(t.no_anchor_at_iou.sum()), weights=weights, mode=ev["mode"])
    lab = list(names.values())
    pd.DataFrame(cm, index=[f"gt {n}" for n in lab], columns=[f"pred {n}" for n in lab]).to_csv(out / "confusion.csv")
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    pd.DataFrame(norm, index=[f"gt {n}" for n in lab], columns=[f"pred {n}" for n in lab]).to_csv(
        out / "confusion_norm.csv")
    per.to_csv(out / "per_class.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "oracle_args.json").write_text(json.dumps(dict(vars(a), weights=weights, eval=ev), indent=2))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v}\n{norm[r, c]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(lab)), lab, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(lab)), lab, fontsize=7)
    ax.set(xlabel="argmax class at GT box", ylabel="GT class",
           title=f"GT-box oracle: accuracy {summary['accuracy']:.3f} (n={summary['n_gt']})")
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=120); plt.close(fig)
    print(per.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[oracle] {json.dumps(summary)}\n[oracle] wrote {out}")


if __name__ == "__main__":
    main()
