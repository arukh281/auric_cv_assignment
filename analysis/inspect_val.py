"""Visual inspection pack for val (figures/inspect_val/). CPU only; renders B1h's saved val predictions and B1h's
GT-box oracle scores (no inference). Facts only on the sheets: GT class, predicted class, confidence, IoU.

  python analysis/inspect_val.py --data-root /kaggle/tmp/data --preds <B1h>/eval/predictions.csv --out <dir>

Inputs: --preds (merged val predictions), analysis/inputs/inspect_val/val_gt_scores.csv (copy of
figures/b1h_tile1024_holdout40/gt_oracle/gt_scores.csv; gt_idx = row order of the label file).
Matching (recall, unmatched): one-to-one greedy by confidence, class-agnostic, IoU >= 0.5.
Colours: one per class (legend on every sheet). GT solid, prediction dashed.
Crops: square, side 3 x the box's longer side, upscaled with nearest neighbour to a fixed cell (scale stated).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import list_images, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402
from detlib.tiling import tile_windows  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
NAMES = ["Cargo Truck", "Truck w/Box", "Truck w/Flatbed", "Truck Tractor", "Truck w/Liquid"]
SHORT = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]
COL = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180)]
CELL, MAXW = 320, 2000


def dashed(d, b, col, w=1, dash=6):
    x1, y1, x2, y2 = b
    for (a0, b0, a1, b1) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        L = max(abs(a1 - a0), abs(b1 - b0)); n = max(int(L // (2 * dash)), 1)
        for k in range(n):
            t0, t1 = 2 * k * dash / max(L, 1), min((2 * k + 1) * dash / max(L, 1), 1)
            d.line([a0 + (a1 - a0) * t0, b0 + (b1 - b0) * t0, a0 + (a1 - a0) * t1, b0 + (b1 - b0) * t1], fill=col, width=w)


def legend(width, rule, extra=""):
    im = Image.new("RGB", (width, 46), "white"); d = ImageDraw.Draw(im)
    x = 6
    for c, n in zip(COL, SHORT):
        d.rectangle([x, 6, x + 14, 18], outline=c, width=2); d.text((x + 18, 6), n, fill=(0, 0, 0)); x += 90
    d.text((x + 10, 6), "GT = solid   prediction = dashed" + extra, fill=(0, 0, 0))
    d.text((6, 26), "Selection: " + rule, fill=(0, 0, 0))
    return im


def stack(parts):
    w = max(p.width for p in parts); h = sum(p.height for p in parts)
    out = Image.new("RGB", (w, h), "white"); y = 0
    for p in parts:
        out.paste(p, (0, y)); y += p.height
    if out.width > MAXW:
        out = out.resize((MAXW, round(out.height * MAXW / out.width)), Image.LANCZOS)
    return out


def save(im, path):
    q = 88
    while True:
        im.save(path, "JPEG", quality=q)
        if path.stat().st_size < 2 * 1024 * 1024 or q < 40:
            return
        q -= 8


def match(gb, pb, ps, thr=0.5):
    hit_g = np.zeros(len(gb), bool); hit_p = np.zeros(len(pb), bool)
    if len(gb) and len(pb):
        iou = box_iou(pb, gb)
        for i in np.argsort(-ps):
            row = np.where(hit_g, -1, iou[i]); j = int(np.argmax(row))
            if row[j] >= thr:
                hit_g[j] = True; hit_p[i] = True
    return hit_g, hit_p


class Data:
    def __init__(s, root, preds):
        s.root = Path(root); s.p = pd.read_csv(preds); s.cache = {}
    def img(s, split, name):
        k = (split, name)
        if k not in s.cache:
            s.cache = {k: Image.open(s.root / split / "images" / name).convert("RGB")}
        return s.cache[k]
    def gt(s, split, name):
        im = s.img(split, name)
        return read_yolo_labels(s.root / split / "labels" / f"{Path(name).stem}.txt", *im.size)
    def preds(s, name, conf=0.001):
        q = s.p[(s.p.image == name) & (s.p.conf >= conf)]
        return q.cls.to_numpy(int), q[["x1", "y1", "x2", "y2"]].to_numpy(float), q.conf.to_numpy(float)


def crop_cell(im, b, gtcls=None, pred=None, caption="", scale_note=True):
    """pred = (cls, box) drawn dashed; gtcls drawn solid on b."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = max(8.0, 3 * max(b[2] - b[0], b[3] - b[1]))
    x0, y0 = cx - side / 2, cy - side / 2
    c = im.crop((int(x0), int(y0), int(x0 + side), int(y0 + side)))
    k = CELL / c.width
    c = c.resize((CELL, CELL), Image.NEAREST)
    d = ImageDraw.Draw(c)
    tb = lambda bb: [(bb[0] - int(x0)) * k, (bb[1] - int(y0)) * k, (bb[2] - int(x0)) * k, (bb[3] - int(y0)) * k]
    if gtcls is not None:
        d.rectangle(tb(b), outline=COL[gtcls], width=1)
    if pred is not None:
        dashed(d, tb(pred[1]), COL[pred[0]], 1)
    cell = Image.new("RGB", (CELL, CELL + 28), "white"); cell.paste(c, (0, 28))
    ImageDraw.Draw(cell).text((3, 2), caption + (f"  (x{k:.1f})" if scale_note else ""), fill=(0, 0, 0))
    return cell


def grid(cells, cols):
    rows = (len(cells) + cols - 1) // cols
    g = Image.new("RGB", (cols * CELL + (cols - 1) * 4, rows * (CELL + 32)), "white")
    for i, c in enumerate(cells):
        g.paste(c, ((i % cols) * (CELL + 4), (i // cols) * (CELL + 32)))
    return g


def overview(D, name, title):
    im = D.img("val", name).copy(); gc, gb = D.gt("val", name)
    pc, pb, ps = D.preds(name, 0.25)
    hg, hp = match(gb, pb, ps)
    d = ImageDraw.Draw(im); lw = max(2, im.width // 900)
    for c, b, h in zip(gc, gb, hg):
        if not h:
            d.rectangle([b[0] - 3 * lw, b[1] - 3 * lw, b[2] + 3 * lw, b[3] + 3 * lw], outline=(255, 255, 0), width=lw)
        d.rectangle(list(b), outline=COL[c], width=lw)
    for c, b, h in zip(pc, pb, hp):
        if not h:
            d.rectangle([b[0] - 3 * lw, b[1] - 3 * lw, b[2] + 3 * lw, b[3] + 3 * lw], outline=(255, 0, 255), width=lw)
        dashed(d, b, COL[c], lw, 8)
    s = MAXW / im.width
    im = im.resize((MAXW, round(im.height * s)), Image.LANCZOS)
    hdr = Image.new("RGB", (MAXW, 22), "white")
    ImageDraw.Draw(hdr).text((6, 4), f"{title}: {name} | GT {len(gb)}, unmatched GT {int((~hg).sum())} | "
                                      f"preds conf>=0.25 {len(pb)}, unmatched {int((~hp).sum())}", fill=(0, 0, 0))
    return stack([hdr, im])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True); ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = Data(a.data_root, a.preds)
    vals = [p.name for p in list_images(Path(a.data_root) / "val" / "images")]
    rec = []
    for n in vals:
        gc, gb = D.gt("val", n); pc, pb, ps = D.preds(n)
        hg, _ = match(gb, pb, ps)
        rec.append(dict(image=n, n_gt=len(gb), matched=int(hg.sum()), recall=hg.mean() if len(gb) else np.nan))
    R = pd.DataFrame(rec).sort_values(["recall", "image"]); R.to_csv(out / "val_recall.csv", index=False)
    UNM = " | unmatched GT: yellow halo; unmatched prediction: magenta halo"

    worst = list(R.image[:2])
    save(stack([legend(MAXW, "2 val images with the lowest recall (any class, IoU>=0.5, conf>=0.001); "
                             "predictions shown at conf>=0.25", UNM)] + [overview(D, n, "lowest recall") for n in worst]),
         out / "01_overview_worst.jpg")
    rng = np.random.default_rng(0); rnd = list(rng.choice(sorted(vals), 2, replace=False))
    save(stack([legend(MAXW, "2 random val images (numpy default_rng(0).choice over sorted names); predictions at conf>=0.25", UNM)]
               + [overview(D, n, "random") for n in rnd]), out / "02_overview_random.jpg")

    n = worst[0]; im = D.img("val", n); gc, gb = D.gt("val", n); pc, pb, ps = D.preds(n, 0.25)
    wins = tile_windows(im.width, im.height, 1024, 256)
    cnt = [int(((gb[:, 0] >= x0) & (gb[:, 2] <= x1) & (gb[:, 1] >= y0) & (gb[:, 3] <= y1)).sum()) for x0, y0, x1, y1 in wins]
    tiles = []
    for i in np.argsort(cnt, kind="stable")[::-1][:4]:
        x0, y0, x1, y1 = wins[i]; t = im.crop((x0, y0, x1, y1)).copy(); d = ImageDraw.Draw(t)
        for c, b in zip(gc, gb):
            d.rectangle([b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0], outline=COL[c], width=1)
        for c, b, s in zip(pc, pb, ps):
            bb = [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]; dashed(d, bb, COL[c], 1, 4)
            d.text((bb[0], max(0, bb[1] - 10)), f"{SHORT[c]} {s:.2f}", fill=COL[c])
        hdr = Image.new("RGB", (t.width, 18), "white")
        ImageDraw.Draw(hdr).text((4, 3), f"{n} tile x{x0} y{y0}: {cnt[i]} GT whole in tile", fill=(0, 0, 0))
        tiles.append(stack([hdr, t]))
    w = tiles[0].width; g = Image.new("RGB", (2 * w + 4, 2 * tiles[0].height + 4), "white")
    for k, t in enumerate(tiles):
        g.paste(t, ((k % 2) * (w + 4), (k // 2) * (tiles[0].height + 4)))
    save(stack([legend(MAXW, f"4 1024x1024 tiles (overlap 256) of {n} with the most GT boxes fully inside; "
                             "predictions conf>=0.25 with class and confidence"), g]), out / "03_zoom_worst.jpg")

    gs = pd.read_csv(REPO / "analysis" / "inputs" / "inspect_val" / "val_gt_scores.csv")
    def oracle_sheet(gtc, prc, fname):
        q = gs[(gs.gt_cls == gtc) & (gs.pred_pool == prc)].sort_values(f"pool_{NAMES[prc]}", ascending=False).head(24)
        cells = []
        for r in q.itertuples():
            gc, gb = D.gt("val", r.image); b = gb[int(r.gt_idx)]
            sc = getattr(r, f"pool_{NAMES[prc].replace(' ', '_').replace('/', '_')}", None)
            sc = gs.loc[r.Index, f"pool_{NAMES[prc]}"]
            cells.append(crop_cell(D.img("val", r.image), b, gtc, None,
                                   f"GT: {SHORT[gtc]} | pred: {SHORT[prc]} {sc:.2f} | {r.image}"))
        save(stack([legend(6 * CELL + 20, f"24 val GT boxes labelled {NAMES[gtc]} that B1h's GT-box oracle (pool mode) "
                                          f"assigns to {NAMES[prc]}, highest {NAMES[prc]} score first"), grid(cells, 6)]),
             out / fname)
    oracle_sheet(0, 1, "04_cargo_called_box.jpg"); oracle_sheet(1, 0, "05_box_called_cargo.jpg")

    fp = []
    for n in vals:
        gc, gb = D.gt("val", n); pc, pb, ps = D.preds(n)
        m = box_iou(pb, gb).max(1) if len(gb) and len(pb) else np.zeros(len(pb))
        for i in np.where(m < 0.1)[0]:
            fp.append((ps[i], n, pc[i], pb[i], m[i]))
    fp.sort(key=lambda x: -x[0])
    cells = [crop_cell(D.img("val", n), b, None, (c, b), f"pred: {SHORT[c]} {s:.2f} | max IoU {m:.2f} | {n}")
             for s, n, c, b, m in fp[:24]]
    save(stack([legend(6 * CELL + 20, "24 highest-confidence val predictions with IoU < 0.1 to every GT box"), grid(cells, 6)]),
         out / "06_val_fp_top.jpg")

    miss = []
    for n in vals:
        gc, gb = D.gt("val", n); pc, pb, ps = D.preds(n)
        m = box_iou(gb, pb).max(1) if len(gb) and len(pb) else np.zeros(len(gb))
        miss += [(n, int(gc[j]), gb[j], m[j]) for j in np.where(m < 0.5)[0]]
    idx = np.random.default_rng(0).choice(len(miss), min(24, len(miss)), replace=False)
    cells = [crop_cell(D.img("val", miss[i][0]), miss[i][2], miss[i][1], None,
                       f"GT: {SHORT[miss[i][1]]} | best IoU any pred {miss[i][3]:.2f} | {miss[i][0]}") for i in sorted(idx)]
    save(stack([legend(6 * CELL + 20, f"24 of {len(miss)} val GT boxes with no prediction of any class at IoU>=0.5 "
                                      "(any confidence), random, default_rng(0)"), grid(cells, 6)]), out / "07_val_missed.jpg")

    def gt_pool(split, cls):
        names = [p.name for p in list_images(Path(a.data_root) / split / "images")]
        pool = []
        for n in names:
            im = Image.open(Path(a.data_root) / split / "images" / n); W, H = im.size
            gc, gb = read_yolo_labels(Path(a.data_root) / split / "labels" / f"{Path(n).stem}.txt", W, H)
            pool += [(n, b) for c, b in zip(gc, gb) if c == cls]
        return pool
    def sample(split, cls, k):
        pool = gt_pool(split, cls); idx = np.random.default_rng(0).choice(len(pool), min(k, len(pool)), replace=False)
        return [crop_cell(D.img(split, pool[i][0]), pool[i][1], cls, None, f"{split} GT: {SHORT[cls]} | {pool[i][0]}")
                for i in sorted(idx)]
    for cls, fname in ((1, "08_train_vs_val_box.jpg"), (0, "09_train_vs_val_cargo.jpg")):
        left, right = grid(sample("train", cls, 18), 3), grid(sample("val", cls, 18), 3)
        both = Image.new("RGB", (left.width + right.width + 20, max(left.height, right.height)), "white")
        both.paste(left, (0, 0)); both.paste(right, (left.width + 20, 0))
        save(stack([legend(both.width, f"left: 18 random train GT {NAMES[cls]}; right: 18 random val GT {NAMES[cls]} "
                                       "(default_rng(0) over all GT boxes of the class in the split)"), both]), out / fname)
    rows = [grid(sample("train", c, 8), 8) for c in range(5)]
    save(stack([legend(rows[0].width, "8 random train GT crops per class, one row per class in legend order (default_rng(0))")] + rows),
         out / "10_class_reference.jpg")

    lines = ["# Visual inspection pack: val", "",
             "Rendered by `analysis/inspect_val.py` from B1h's saved val predictions "
             "(`results/b1h_tile1024_holdout40/eval/predictions.csv`) and B1h's GT-box oracle scores "
             "(`figures/b1h_tile1024_holdout40/gt_oracle/gt_scores.csv`). No new inference. GT solid, prediction dashed, "
             "one colour per class (legend on each sheet). Crops: square, 3x the box's longer side, nearest-neighbour "
             "upscale to 320 px (scale in each caption). Matching: one-to-one, class-agnostic, IoU >= 0.5.", "",
             "| file | selection rule |", "|---|---|",
             "| 01_overview_worst.jpg | 2 val images with the lowest recall (any class, IoU >= 0.5, conf >= 0.001); predictions at conf >= 0.25; unmatched GT yellow halo, unmatched predictions magenta halo |",
             "| 02_overview_random.jpg | 2 random val images, numpy default_rng(0).choice over sorted names; same style |",
             "| 03_zoom_worst.jpg | the 4 tiles (1024, overlap 256) of the lowest-recall image with the most GT boxes fully inside; predictions conf >= 0.25 labelled |",
             "| 04_cargo_called_box.jpg | 24 val GT Cargo Truck boxes that the GT-box oracle (pool) assigns to Truck w/Box, highest Box score first |",
             "| 05_box_called_cargo.jpg | 24 val GT Truck w/Box boxes assigned to Cargo Truck, highest Cargo score first |",
             "| 06_val_fp_top.jpg | 24 highest-confidence val predictions with IoU < 0.1 to every GT box |",
             "| 07_val_missed.jpg | 24 random (default_rng(0)) val GT boxes with no prediction of any class at IoU >= 0.5, any confidence |",
             "| 08_train_vs_val_box.jpg | 18 random train vs 18 random val GT Truck w/Box crops (default_rng(0)) |",
             "| 09_train_vs_val_cargo.jpg | same for Cargo Truck |",
             "| 10_class_reference.jpg | 8 random train GT crops per class (default_rng(0)) |", "",
             "## Per-image recall (val, any class, IoU >= 0.5, conf >= 0.001)", "", "| image | GT boxes | matched | recall |", "|---|---|---|---|"]
    lines += [f"| {r.image} | {r.n_gt} | {r.matched} | {r.recall:.3f} |" for r in R.itertuples()]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    for f in sorted(out.iterdir()):
        print(f.name, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
