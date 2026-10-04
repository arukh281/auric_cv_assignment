"""Sanity checks on B1h's training data and on whether YOLO11s can fit it at all (GPU for the overfit step).

Usage (Kaggle, after scripts/kaggle_setup.sh):
  python analysis/sanity_check.py --data-root /kaggle/tmp/data --tiles /kaggle/tmp/work/tiles_b1h --out <dir>

1. Tiles: tools/make_tiles.py with B1h's settings (tile 1024, overlap 256, empty_keep 0.2, min_vis 0.5, seed 0,
   .png, --exclude-list splits/holdout40_seed0.txt). Same code and indices as B1h, so the same tiles.
2. Label check, all written tiles: every tile label file is re-derived from the source image's YOLO labels with
   detlib.tiling.clip_boxes (the tiler's own rule) and compared (class id equal, coordinates within 0.5 px);
   class ids must be 0-4 and normalised coordinates inside [0, 1]. -> label_check.json
3. Rendering: 12 tiles (seed 0): 4 random tiles with boxes, 4 edge tiles (flush with the right or bottom image border,
   x0 > 0 or y0 > 0), 2 from the largest and 2 from the smallest source image (pixel area, non-holdout).
   Labels drawn with class names. -> label_tiles/*.png
4. Overfit: 16 tiles with boxes that together contain all 5 classes (one seeded tile per class, rarest first, then
   seeded random fill). YOLO11s (yolo11s.pt), imgsz 1024 (= tile size), 300 epochs, batch 16, every augmentation off
   (mosaic 0, scale 0, translate 0, fliplr 0, flipud 0, hsv_h/s/v 0, erasing 0, mixup 0, close_mosaic 0),
   SGD lr0 0.01 momentum 0.937 (as B1h), seed 0, deterministic, val off. Checkpoints kept at epochs 50/100/.../300.
   Evaluated on the same 16 tiles with eval.py (full mode, imgsz 1024; our scorer) at epochs 50, 100 and 300.
   Final-epoch boxes not matched at conf >= 0.25 (IoU >= 0.5, right class) are listed and drawn.
   -> overfit/{tiles.txt, results.csv, eval_epNNN/, failures.csv, render_*.png, summary.json}
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, load_classes, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou, match_image  # noqa: E402
from detlib.tiling import clip_boxes  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
COLORS = [(255, 60, 60), (60, 220, 60), (60, 140, 255), (255, 220, 0), (255, 0, 255)]
CKPTS = (50, 100, 300)


def make_tiles(data_root, tiles):
    if (tiles / "_COMPLETE").exists():
        return
    subprocess.run([sys.executable, str(REPO / "tools/make_tiles.py"), "--data-root", str(data_root), "--out", str(tiles),
                    "--tile", "1024", "--overlap", "256", "--empty-keep", "0.2", "--min-vis", "0.5", "--seed", "0",
                    "--exclude-list", str(REPO / "splits/holdout40_seed0.txt")], check=True)


def tile_labels(tiles, name, w, h):
    return read_yolo_labels(tiles / "train" / "labels" / f"{name}.txt", w, h)


def labels_match(gc, gb, ec, eb, tol=0.5):
    """True if the tile labels (gc, gb) equal the re-derived ones (ec, eb): same count, and a one-to-one pairing by
    class-agnostic IoU (greedy, highest first) in which every pair has the same class and all coordinates within tol px.
    Corrected 2026-10-04: this used to pair boxes by sorting coordinates (np.lexsort), which mis-pairs boxes with
    near-equal coordinates in dense tiles and flagged 85 identical tiles (results/sanity/label_mismatch/summary.json)."""
    if len(gc) != len(ec):
        return False
    if not len(gc):
        return True
    iou = box_iou(eb, gb)
    used_e, used_g = set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
        if i in used_e or j in used_g:
            continue
        if iou[i, j] <= 0:
            break
        used_e.add(i); used_g.add(j)
        if gc[j] != ec[i] or np.abs(gb[j] - eb[i]).max() > tol:
            return False
    return len(used_e) == len(ec)


def check_labels(data_root, tiles, idx, sizes):
    """Re-derive every written tile's labels from the source labels and compare."""
    src_cache, bad, n_box, raw_bad = {}, [], 0, []
    for r in idx[idx.kept].itertuples():
        if r.source not in src_cache:
            W, H = sizes[r.source]
            src_cache = {r.source: read_yolo_labels(data_root / "train" / "labels" / f"{Path(r.source).stem}.txt", W, H)}
        sc, sb = src_cache[r.source]
        ec, eb, _ = clip_boxes(sc, sb, (r.x0, r.y0, r.x1, r.y1), 0.5)
        tw, th = r.x1 - r.x0, r.y1 - r.y0
        lines = [l.split() for l in (tiles / "train" / "labels" / f"{r.tile}.txt").read_text().splitlines() if l.strip()]
        a = np.array(lines, float).reshape(-1, 5)
        if len(a) and (a[:, 0].min() < 0 or a[:, 0].max() > 4 or a[:, 1:].min() < 0 or a[:, 1:].max() > 1):
            raw_bad.append(r.tile)
        gc, gb = tile_labels(tiles, r.tile, tw, th)
        n_box += len(gc)
        if not labels_match(gc, gb, ec, eb):
            bad.append(r.tile)
    return dict(tiles_checked=int(idx.kept.sum()), boxes_checked=n_box, tiles_mismatched=len(bad),
                mismatched_examples=bad[:20], tiles_out_of_range=len(raw_bad), out_of_range_examples=raw_bad[:20])


def draw(img, boxes, classes, names, color=None, width=2, prefix=""):
    d = ImageDraw.Draw(img)
    for b, c in zip(boxes, classes):
        col = color or COLORS[int(c)]
        d.rectangle(list(map(float, b)), outline=col, width=width)
        d.text((float(b[0]), max(0.0, float(b[1]) - 11)), f"{prefix}{names[int(c)]}", fill=col)
    return img


def render(tiles, idx, sizes, names, out, seed=0):
    rng = random.Random(seed)
    k = idx[idx.kept]
    withb = k[k.n_boxes > 0]
    W = k.source.map(lambda s: sizes[s][0])
    H = k.source.map(lambda s: sizes[s][1])
    edge = k[(((k.x1 == W) & (k.x0 > 0)) | ((k.y1 == H) & (k.y0 > 0))) & (k.n_boxes > 0)]
    area = {s: sizes[s][0] * sizes[s][1] for s in k.source.unique()}
    big, small = max(area, key=area.get), min(area, key=area.get)
    pick = []
    for label, pool, n in (("random", withb, 4), ("edge", edge, 4), ("largest", k[k.source == big], 2),
                           ("smallest", k[k.source == small], 2)):
        pool = pool[~pool.tile.isin([p[1] for p in pick])]
        pool = pool.sort_values("n_boxes", ascending=False) if label in ("largest", "smallest") else pool
        names_ = list(pool.tile[:n]) if label in ("largest", "smallest") else rng.sample(list(pool.tile), min(n, len(pool)))
        pick += [(label, t) for t in names_]
    d = out / "label_tiles"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, t in pick:
        r = k[k.tile == t].iloc[0]
        tw, th = r.x1 - r.x0, r.y1 - r.y0
        im = Image.open(tiles / "train" / "images" / f"{t}.png").convert("RGB")
        gc, gb = tile_labels(tiles, t, tw, th)
        draw(im, gb, gc, names).save(d / f"{label}__{t}.png")
        rows.append(dict(kind=label, tile=t, source=r.source, x0=r.x0, y0=r.y0, w=tw, h=th, n_boxes=len(gc),
                         classes=",".join(names[int(c)] for c in sorted(set(gc)))))
    pd.DataFrame(rows).to_csv(d / "rendered_tiles.csv", index=False)
    return dict(largest_source=big, smallest_source=small, rendered=len(pick))


def pick_overfit(idx, tiles, names, n=16, seed=0):
    rng = random.Random(seed)
    k = idx[idx.kept & (idx.n_boxes > 0)]
    cls_of = {}
    for t in k.tile:
        lines = (tiles / "train" / "labels" / f"{t}.txt").read_text().split("\n")
        cls_of[t] = {int(l.split()[0]) for l in lines if l.strip()}
    counts = pd.Series([c for s in cls_of.values() for c in s]).value_counts()
    chosen = []
    for c in counts.sort_values().index:            # rarest class first
        if any(c in cls_of[t] for t in chosen):
            continue
        cand = sorted(t for t in cls_of if c in cls_of[t] and t not in chosen)
        chosen.append(rng.choice(cand))
    rest = sorted(t for t in cls_of if t not in chosen)
    chosen += rng.sample(rest, n - len(chosen))
    assert set().union(*(cls_of[t] for t in chosen)) == set(names)
    return chosen


def overfit(tiles, idx, names, out, epochs=300, device=None):
    o = out / "overfit"
    sel = pick_overfit(idx, tiles, names)
    ds = Path("/kaggle/tmp/overfit_data") if Path("/kaggle").exists() else o / "data"
    for sub in ("train", "val"):
        for kind in ("images", "labels"):
            (ds / sub / kind).mkdir(parents=True, exist_ok=True)
    for t in sel:
        for sub in ("train", "val"):
            shutil.copy2(tiles / "train" / "images" / f"{t}.png", ds / sub / "images" / f"{t}.png")
            shutil.copy2(tiles / "train" / "labels" / f"{t}.txt", ds / sub / "labels" / f"{t}.txt")
    (ds / "classmap.txt").write_text("".join(f"{c} {n}\n" for c, n in names.items()))
    (ds / "data.yaml").write_text(f"path: {ds}\ntrain: train/images\nval: val/images\nnc: {len(names)}\nnames:\n" +
                                  "".join(f"  {c}: {n}\n" for c, n in names.items()))
    o.mkdir(parents=True, exist_ok=True)
    (o / "tiles.txt").write_text("\n".join(sel) + "\n")
    from ultralytics import YOLO
    sys.path.insert(0, str(REPO))
    from train import add_checkpoint_callback
    m = YOLO("yolo11s.pt")
    add_checkpoint_callback(m, 50)
    args = dict(data=str(ds / "data.yaml"), imgsz=1024, epochs=epochs, batch=16, workers=2, seed=0, deterministic=True,
                optimizer="SGD", lr0=0.01, momentum=0.937, amp=True, val=False, plots=True, cache=False,
                mosaic=0.0, scale=0.0, translate=0.0, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                erasing=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, degrees=0.0, shear=0.0, perspective=0.0,
                close_mosaic=0, project=str(o), name="train", exist_ok=True, device=device)
    m.train(**args)
    wdir = o / "train" / "weights"
    shutil.copy2(o / "train" / "results.csv", o / "results.csv")
    evals = {}
    for e in CKPTS:
        w = wdir / (f"epoch{e:03d}.pt" if e != epochs else "last.pt")
        if not w.exists():
            w = wdir / f"epoch{e:03d}.pt"
        if not w.exists():
            continue
        ev = o / f"eval_ep{e:03d}"
        subprocess.run([sys.executable, str(REPO / "eval.py"), "--weights", str(w), "--mode", "full", "--imgsz", "1024",
                        "--data-root", str(ds), "--out", str(ev), "--run-dir", str(o), "--bootstrap", "0",
                        "--max-det", "1000"], check=True)
        pc = pd.read_csv(ev / "per_class.csv").set_index("class_name")
        evals[e] = {k: float(v) for k, v in pc.AP50_coco.items()}
    # failures at the final epoch, conf >= 0.25
    preds = pd.read_csv(o / f"eval_ep{epochs:03d}" / "predictions.csv")
    fails = []
    for t in sel:
        im = Image.open(ds / "val" / "images" / f"{t}.png").convert("RGB")
        gc, gb = read_yolo_labels(ds / "val" / "labels" / f"{t}.txt", *im.size)
        p = preds[(preds.image == f"{t}.png") & (preds.conf >= 0.25)]
        pb, pc_, ps = p[["x1", "y1", "x2", "y2"]].to_numpy(float), p.cls.to_numpy(int), p.conf.to_numpy(float)
        tp, gidx, _ = match_image(pb, pc_, ps, gb, gc, 0.5)
        matched = np.zeros(len(gc), bool)
        matched[gidx[tp]] = True
        iou = box_iou(gb, pb) if len(pb) else np.zeros((len(gb), 0))
        for j in np.flatnonzero(~matched):
            k = int(iou[j].argmax()) if iou.shape[1] else -1
            fails.append(dict(tile=t, gt_idx=int(j), gt_class=names[int(gc[j])],
                              size=float(np.sqrt((gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1]))),
                              best_iou=float(iou[j, k]) if k >= 0 else 0.0,
                              best_pred_class=names[int(pc_[k])] if k >= 0 else "", best_pred_conf=float(ps[k]) if k >= 0 else 0.0))
        im = draw(im, gb[matched], gc[matched], names, color=(0, 220, 0), width=2)
        im = draw(im, gb[~matched], gc[~matched], names, color=(255, 255, 0), width=3, prefix="MISSED ")
        im = draw(im, pb, pc_, names, color=(255, 40, 40), width=1, prefix="pred ")
        im.save(o / f"render_{t}.png")
    pd.DataFrame(fails, columns=["tile", "gt_idx", "gt_class", "size", "best_iou", "best_pred_class",
                                 "best_pred_conf"]).to_csv(o / "failures.csv", index=False)
    r = pd.read_csv(o / "results.csv")
    r.columns = [c.strip() for c in r.columns]
    last = r.iloc[-1]
    summary = dict(tiles=sel, n_gt=int(sum(len(read_yolo_labels(ds / "val" / "labels" / f"{t}.txt", 1, 1)[0]) for t in sel)),
                   ap50_by_epoch=evals, final_train_losses={c: float(last[c]) for c in r.columns if c.startswith("train/")},
                   n_failures_conf025=len(fails), train_args=args)
    (o / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--labels-only", action="store_true", help="label check and renders only (no overfit test; CPU)")
    a = ap.parse_args()
    data_root, tiles, out = Path(a.data_root), Path(a.tiles), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = load_classes(data_root)
    make_tiles(data_root, tiles)
    idx = pd.read_csv(tiles / "tiles_index.csv")
    im = pd.read_csv(EDA_TABLES / "images.csv").query("split == 'train'")
    sizes = {r.image: (int(r.width), int(r.height)) for r in im.itertuples()}
    res = dict(tiling_params=json.loads((tiles / "tiling_params.json").read_text())["counts"])
    res["label_check"] = check_labels(data_root, tiles, idx, sizes)
    print("[sanity] label check:", json.dumps(res["label_check"]), flush=True)
    res["render"] = render(tiles, idx, sizes, names, out)
    (out / "label_check.json").write_text(json.dumps(res, indent=2))
    if a.labels_only:
        return
    s = overfit(tiles, idx, names, out, a.epochs, a.device)
    print("[sanity] overfit AP50 by epoch:", json.dumps(s["ap50_by_epoch"]), flush=True)
    print("[sanity] final train losses:", json.dumps(s["final_train_losses"]), flush=True)
    print(f"[sanity] failures at conf>=0.25: {s['n_failures_conf025']}", flush=True)


if __name__ == "__main__":
    main()
