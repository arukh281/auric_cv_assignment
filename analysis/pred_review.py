"""Qualitative review of one run's val predictions at an operating confidence threshold.

Usage: python analysis/pred_review.py --preds runs/b0_full640/eval/predictions.csv --name b0_full640 [--conf 0.25]

Outputs in figures/<name>/:
  counts_at_conf.csv            TP / FP (by kind) / FN per class at --conf
  confusion_matrix.csv/.png     rows = GT class + background, cols = predicted class + background
  grid_TP.png, grid_FP_<kind>.png, grid_FN.png   16 random examples each (seeded), crops with context
  overview_*.png                4 random val images with every mark drawn
Marks: green = TP, red = FP, orange = GT of the class an FP was confused with / poorly localized, cyan = FN.

FP kinds (simple rules, Phase 3 errors.py does the full taxonomy):
  wrong_class  IoU >= 0.5 with a GT of another class
  duplicate    IoU >= 0.5 with a same-class GT already matched by a higher-confidence prediction
  localization 0.1 <= best same-class IoU < 0.5
  background   everything else
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import list_images, load_classes, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou, match_image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
COL = dict(TP=(0, 200, 0), FP=(230, 0, 0), REF=(255, 150, 0), FN=(0, 220, 255))


def classify(preds, gc, gb):
    """Per prediction: kind in {TP, wrong_class, duplicate, localization, background}; per GT: matched bool."""
    pb, pc, ps = preds[["x1", "y1", "x2", "y2"]].to_numpy(float), preds.cls.to_numpy(int), preds.conf.to_numpy(float)
    tp, gidx, _ = match_image(pb, pc, ps, gb, gc, 0.5)
    iou = box_iou(pb, gb)
    kinds, ref = [], []
    for i in range(len(pb)):
        if tp[i]:
            kinds.append("TP"); ref.append(gidx[i]); continue
        same = np.where(gc == pc[i], iou[i], 0) if len(gb) else np.zeros(0)
        other = np.where(gc != pc[i], iou[i], 0) if len(gb) else np.zeros(0)
        if len(gb) and other.max() >= 0.5:
            kinds.append("wrong_class"); ref.append(int(other.argmax()))
        elif len(gb) and same.max() >= 0.5:
            kinds.append("duplicate"); ref.append(int(same.argmax()))
        elif len(gb) and same.max() >= 0.1:
            kinds.append("localization"); ref.append(int(same.argmax()))
        else:
            kinds.append("background"); ref.append(-1)
    matched = np.zeros(len(gb), bool)
    matched[gidx[tp]] = True
    return np.array(kinds), np.array(ref), matched


def confusion(preds, gc, gb, nc):
    """GT-centric matrix at IoU 0.5: one-to-one greedy by IoU regardless of class; leftovers vs background."""
    m = np.zeros((nc + 1, nc + 1), int)
    pb, pc = preds[["x1", "y1", "x2", "y2"]].to_numpy(float), preds.cls.to_numpy(int)
    iou = box_iou(gb, pb)
    used_g, used_p = set(), set()
    if iou.size:
        for gi, pi in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
            if iou[gi, pi] < 0.5:
                break
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi); used_p.add(pi)
            m[gc[gi], pc[pi]] += 1
    for gi in set(range(len(gb))) - used_g:
        m[gc[gi], nc] += 1
    for pi in set(range(len(pb))) - used_p:
        m[nc, pc[pi]] += 1
    return m


def crop(img, box, marks, size=200):
    x1, y1, x2, y2 = box
    pad = max(48, 1.5 * max(x2 - x1, y2 - y1))
    W, H = img.size
    win = (int(max(x1 - pad, 0)), int(max(y1 - pad, 0)), int(min(x2 + pad, W)), int(min(y2 + pad, H)))
    c = img.crop(win).convert("RGB")
    d = ImageDraw.Draw(c)
    for b, col in marks:
        d.rectangle([b[0] - win[0], b[1] - win[1], b[2] - win[0], b[3] - win[1]], outline=col, width=1)
    s = size / max(c.size)
    return c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))), Image.NEAREST)


def grid(items, title, path):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (im, cap) in zip(axes.flat, items):
        ax.imshow(im); ax.set_title(cap, fontsize=7)
    fig.suptitle(f"{title} (n shown={len(items)})")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--name", required=True, help="figures/<name>/")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold for TP/FP/FN counting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="figures")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    root, out = Path(a.data_root), Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    names = load_classes(root)
    nc = len(names)
    allp = pd.read_csv(a.preds)
    allp = allp[allp.conf >= a.conf]

    events, cm = [], np.zeros((nc + 1, nc + 1), int)
    images = list_images(root / "val" / "images")
    for p in images:
        with Image.open(p) as im:
            w, h = im.size
        gc, gb = read_yolo_labels(root / "val" / "labels" / f"{p.stem}.txt", w, h)
        pr = allp[allp.image == p.name].reset_index(drop=True)
        kinds, ref, matched = classify(pr, gc, gb)
        cm += confusion(pr, gc, gb, nc)
        for i, r in pr.iterrows():
            events.append(dict(image=p.name, kind=kinds[i], cls=int(r.cls), conf=r.conf, box=(r.x1, r.y1, r.x2, r.y2),
                               ref_box=tuple(gb[ref[i]]) if ref[i] >= 0 else None,
                               ref_cls=int(gc[ref[i]]) if ref[i] >= 0 else -1))
        for j in np.where(~matched)[0]:
            events.append(dict(image=p.name, kind="FN", cls=int(gc[j]), conf=np.nan, box=tuple(gb[j]),
                               ref_box=None, ref_cls=-1))
    ev = pd.DataFrame(events)
    kinds = ["TP", "wrong_class", "duplicate", "localization", "background", "FN"]
    counts = ev.groupby(["cls", "kind"]).size().unstack(fill_value=0).reindex(columns=kinds, fill_value=0)
    counts = counts.reindex(range(nc), fill_value=0)
    counts.index = [names[c] for c in counts.index]
    counts.loc["all"] = counts.sum()
    counts.insert(0, "conf_threshold", a.conf)
    counts.to_csv(out / "counts_at_conf.csv")

    labels = [names[c] for c in range(nc)] + ["background"]
    pd.DataFrame(cm, index=[f"GT {l}" for l in labels], columns=[f"pred {l}" for l in labels]).to_csv(out / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(np.log1p(cm), cmap="Blues")
    for i in range(nc + 1):
        for j in range(nc + 1):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(nc + 1), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(nc + 1), labels, fontsize=8)
    ax.set(xlabel="predicted", ylabel="ground truth", title=f"Confusion @ IoU 0.5, conf >= {a.conf} (bottom-right unused)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=120); plt.close(fig)

    cache = {}

    def img(n):
        if n not in cache:
            cache.clear()  # keep one full-res image in memory at a time
            cache[n] = Image.open(root / "val" / "images" / n)
        return cache[n]

    for kind in kinds:
        sub = ev[ev.kind == kind]
        if not len(sub):
            continue
        pick = sub.iloc[sorted(rng.sample(range(len(sub)), min(16, len(sub))))].sort_values("image")
        items = []
        for r in pick.itertuples():
            col = COL["TP"] if kind == "TP" else COL["FN"] if kind == "FN" else COL["FP"]
            marks = [(r.box, col)] + ([(r.ref_box, COL["REF"])] if r.ref_box is not None and kind != "TP" else [])
            cap = f"{r.image} {names[r.cls]}" + (f" {r.conf:.2f}" if kind != "FN" else "")
            if r.ref_cls >= 0 and kind == "wrong_class":
                cap += f"\nGT: {names[r.ref_cls]}"
            items.append((crop(img(r.image), r.box, marks), cap))
        fname = "grid_TP.png" if kind == "TP" else "grid_FN.png" if kind == "FN" else f"grid_FP_{kind}.png"
        grid(items, f"{a.name}: {kind} @ conf>={a.conf}", out / fname)

    for n in rng.sample([p.name for p in images], min(4, len(images))):
        im = img(n).convert("RGB")
        d = ImageDraw.Draw(im)
        for r in ev[ev.image == n].itertuples():
            col = COL["TP"] if r.kind == "TP" else COL["FN"] if r.kind == "FN" else COL["FP"]
            d.rectangle(list(r.box), outline=col, width=4)
        im.thumbnail((1600, 1600))
        im.save(out / f"overview_{Path(n).stem}.png")
    print(counts.to_string())
    print(f"[review] wrote {out}")


if __name__ == "__main__":
    main()
