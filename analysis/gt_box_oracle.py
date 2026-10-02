"""Investigation 5.1, GT-box oracle: if every truck were found and boxed perfectly, how well does the detector
tell the five classes apart?

Usage:
  python analysis/gt_box_oracle.py --config configs/b1.yaml --runs-root runs [--weights .../last.pt]

For every val GT box, the detector's per-class scores at that box are read from the raw (pre-NMS,
pre-threshold) detection head (sigmoid class scores per anchor, with the anchor's decoded box). Two modes:
  pool  max-pooled per class over every anchor whose box has IoU >= --iou with the GT; if none reaches --iou,
        the best-IoU anchor is used
  best  the single anchor with the highest IoU with the GT
A GT is flagged (no_anchor_at_iou) when no anchor reaches --iou; then both modes use the same anchor.
The predicted class is the argmax. Localization and recall are taken out, so what remains is classification only.
Accuracy is reported per mode for all, unflagged and flagged GT (with counts), plus the agreement between modes.

Input to the network, same geometry as eval.py:
  sliced (B1): the val image is cut with detlib.tiling.tile_windows(tile, overlap); each GT is read from the
               tile that contains it whole with the largest margin to the tile edge. Tiles go in at imgsz.
  full (B0):   the whole image, letterboxed to imgsz.
  Letterbox: aspect-preserving resize to imgsz, centered gray (114) padding, RGB / 255, float32.

Writes figures/<name>/gt_oracle/: gt_scores.csv (one row per GT: both modes' scores and argmax, best-anchor IoU,
flag), confusion_<mode>.csv / confusion_<mode>_norm.csv (rows = GT class, cols = argmax class), confusion.png
(both modes side by side), per_class.csv (per mode and subset: n, correct, accuracy = recall of the class,
precision of the argmax), comparison.csv (mode x subset all/unflagged/flagged: n, accuracy, mean class accuracy),
summary.json (incl. mode agreement), oracle_args.json.
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
    """Per GT: class scores max-pooled over anchors with IoU >= iou_thr (else the best-IoU anchor), class scores
    of the best-IoU anchor alone, best anchor IoU, number of anchors at iou_thr."""
    iou = box_iou(gt_xyxy, anchors_xyxy)
    nc = anchor_scores.shape[1]
    pool, best_s = np.zeros((len(gt_xyxy), nc)), np.zeros((len(gt_xyxy), nc))
    best = iou.max(1) if iou.size else np.zeros(len(gt_xyxy))
    n_at = (iou >= iou_thr).sum(1) if iou.size else np.zeros(len(gt_xyxy), int)
    for g in range(len(gt_xyxy)):
        sel = iou[g] >= iou_thr
        best_s[g] = anchor_scores[int(iou[g].argmax())]
        pool[g] = anchor_scores[sel].max(0) if sel.any() else best_s[g]
    return pool, best_s, best, n_at


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
    pool = best_s = None
    best, n_at = np.zeros(len(gt_xyxy)), np.zeros(len(gt_xyxy), int)
    used = sorted(set(asg.tolist()))
    for s in range(0, len(used), batch):
        chunk = used[s:s + batch]
        outs = head_outputs(net, [img[wins[k][1]:wins[k][3], wins[k][0]:wins[k][2]] for k in chunk], imgsz, device)
        for k, (xyxy, sc) in zip(chunk, outs):
            g = np.where(asg == k)[0]
            local = gt_xyxy[g] - np.array([wins[k][0], wins[k][1], wins[k][0], wins[k][1]], float)
            sp, sb, bi, ng = scores_at_boxes(xyxy, sc, local, iou_thr)
            if pool is None:
                pool, best_s = np.zeros((len(gt_xyxy), sc.shape[1])), np.zeros((len(gt_xyxy), sc.shape[1]))
            pool[g], best_s[g], best[g], n_at[g] = sp, sb, bi, ng
    empty = np.zeros((0, 0))
    return (pool if pool is not None else empty), (best_s if best_s is not None else empty), best, n_at


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


MODES = ("pool", "best")


def compare_modes(gt_cls, scores, flagged, names):
    """scores: {mode: (n, nc)}. Per mode x subset (all / unflagged / flagged): confusion, per-class rows, summary
    rows; plus the agreement of the argmax between modes."""
    subsets = {"all": np.ones(len(gt_cls), bool), "unflagged": ~flagged, "flagged": flagged}
    cms, per_rows, comp = {}, [], []
    for mode in scores:
        for sub, m in subsets.items():
            cm, per, summ = oracle_tables(gt_cls[m], scores[mode][m].reshape(-1, len(names)), names)
            if sub == "all":
                cms[mode] = cm
            per.insert(0, "subset", sub); per.insert(0, "mode", mode)
            per_rows.append(per)
            comp.append(dict(mode=mode, subset=sub, n_gt=summ["n_gt"], accuracy=summ["accuracy"],
                             mean_class_accuracy=summ["mean_class_accuracy"]))
    a, b = (scores[k].argmax(1) if len(scores[k]) else np.zeros(0, int) for k in MODES)
    agree = dict(n=int(len(a)), n_agree=int((a == b).sum()), agreement=float((a == b).mean()) if len(a) else float("nan"),
                 n_agree_unflagged=int((a == b)[~flagged].sum()), n_unflagged=int((~flagged).sum()))
    return cms, pd.concat(per_rows, ignore_index=True), pd.DataFrame(comp), agree


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
        sp, sb, best, n_at = image_scores(net, img, gb, ev["mode"], ev["imgsz"], ev["tile"], ev["overlap"], a.iou,
                                          a.batch, device)
        for j in range(len(gc)):
            rows.append(dict(image=p.name, gt_idx=j, gt_cls=int(gc[j]), pred_pool=int(sp[j].argmax()),
                             pred_best=int(sb[j].argmax()), best_anchor_iou=best[j], n_anchors_at_iou=int(n_at[j]),
                             no_anchor_at_iou=bool(n_at[j] == 0), **{f"pool_{names[c]}": sp[j, c] for c in names},
                             **{f"best_{names[c]}": sb[j, c] for c in names}))
        print(f"[oracle] {i + 1} {p.name}: {len(gc)} GT", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "gt_scores.csv", index=False)
    gt_cls, flagged = t.gt_cls.to_numpy(int), t.no_anchor_at_iou.to_numpy(bool)
    scores = {m: t[[f"{m}_{names[c]}" for c in names]].to_numpy() for m in MODES}
    cms, per, comp, agree = compare_modes(gt_cls, scores, flagged, names)
    lab = list(names.values())
    for mode, cm in cms.items():
        idx, cols = [f"gt {n}" for n in lab], [f"pred {n}" for n in lab]
        pd.DataFrame(cm, index=idx, columns=cols).to_csv(out / f"confusion_{mode}.csv")
        pd.DataFrame(cm / np.maximum(cm.sum(1, keepdims=True), 1), index=idx, columns=cols).to_csv(
            out / f"confusion_{mode}_norm.csv")
    per.to_csv(out / "per_class.csv", index=False)
    comp.to_csv(out / "comparison.csv", index=False)
    summary = dict(modes={r.mode + "/" + r.subset: dict(n_gt=int(r.n_gt), accuracy=r.accuracy,
                                                       mean_class_accuracy=r.mean_class_accuracy)
                          for r in comp.itertuples()},
                   agreement_pool_vs_best=agree, n_flagged_no_anchor_at_iou=int(flagged.sum()),
                   n_unflagged=int((~flagged).sum()), iou=a.iou, weights=weights, mode=ev["mode"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "oracle_args.json").write_text(json.dumps(dict(vars(a), weights=weights, eval=ev), indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mode in zip(axes, MODES):
        cm = cms[mode]
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}\n{norm[r, c]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(lab)), lab, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(lab)), lab, fontsize=7)
        acc = comp[(comp["mode"] == mode) & (comp.subset == "all")].accuracy.iloc[0]
        ax.set(xlabel="argmax class at GT box", ylabel="GT class", title=f"{mode}: accuracy {acc:.3f} (n={len(t)})")
    fig.suptitle(f"GT-box oracle; agreement pool vs best {agree['agreement']:.3f}; "
                 f"{int(flagged.sum())} GT without an anchor at IoU >= {a.iou}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=120); plt.close(fig)
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[oracle] agreement {json.dumps(agree)}\n[oracle] wrote {out}")


if __name__ == "__main__":
    main()
