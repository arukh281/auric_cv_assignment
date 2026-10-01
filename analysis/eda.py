"""Phase 1 dataset analysis. Reads YOLO-format data/ and writes CSVs + figures.

Usage: python analysis/eda.py [--data data] [--seed 0]
Outputs: figures/eda/*.png, figures/eda/tables/*.csv, figures/eda/summary.json
"""
import argparse
import json
import random
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
SPLITS = ("train", "val")
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
# COCO area bins, in original-image pixels
SMALL, MEDIUM = 32**2, 96**2
TINY_PX = 4  # boxes with a side under this many px are flagged
DUP_IOU = 0.9
PHASH_MAX_DIST = 8  # Hamming distance on 64-bit pHash


def load_classes(data):
    names = {}
    for line in (data / "classmap.txt").read_text().splitlines():
        if line.strip():
            i, name = line.split(maxsplit=1)
            names[int(i)] = name.strip()
    return names


def load_split(data, split):
    imgs, boxes = [], []
    for p in sorted((data / split / "images").iterdir()):
        if p.name.startswith("."):
            continue
        with Image.open(p) as im:
            w, h = im.size
        imgs.append(dict(split=split, image=p.name, path=str(p), width=w, height=h))
        lbl = data / split / "labels" / (p.stem + ".txt")
        rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
        for k, r in enumerate(rows):
            c, xc, yc, bw, bh = int(r[0]), *map(float, r[1:5])
            boxes.append(dict(split=split, image=p.name, box_idx=k, cls=c, xc=xc, yc=yc, bw=bw, bh=bh,
                              img_w=w, img_h=h, n_fields=len(r)))
    return pd.DataFrame(imgs), pd.DataFrame(boxes)


def iou_matrix(b):
    x1, y1, x2, y2 = b.T
    ix = np.clip(np.minimum(x2[:, None], x2) - np.maximum(x1[:, None], x1), 0, None)
    iy = np.clip(np.minimum(y2[:, None], y2) - np.maximum(y1[:, None], y1), 0, None)
    inter = ix * iy
    area = (x2 - x1) * (y2 - y1)
    return inter / np.maximum(area[:, None] + area - inter, 1e-9)


def draw(path, rows, names, max_side=900, width=6):
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    for r in rows.itertuples():
        d.rectangle([r.x1, r.y1, r.x2, r.y2], outline=COLORS[r.cls % 5], width=width)
    im.thumbnail((max_side, max_side))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures/eda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    data, out = Path(a.data), Path(a.out)
    tab = out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    names = load_classes(data)

    parts = [load_split(data, s) for s in SPLITS]
    imgs = pd.concat([p[0] for p in parts], ignore_index=True)
    bx = pd.concat([p[1] for p in parts], ignore_index=True)
    bx["class_name"] = bx.cls.map(names)
    bx["w_px"], bx["h_px"] = bx.bw * bx.img_w, bx.bh * bx.img_h
    bx["x1"], bx["x2"] = (bx.xc - bx.bw / 2) * bx.img_w, (bx.xc + bx.bw / 2) * bx.img_w
    bx["y1"], bx["y2"] = (bx.yc - bx.bh / 2) * bx.img_h, (bx.yc + bx.bh / 2) * bx.img_h
    bx["area_px"] = bx.w_px * bx.h_px
    bx["sqrt_area_px"] = np.sqrt(bx.area_px)
    bx["aspect_w_over_h"] = bx.w_px / bx.h_px.replace(0, np.nan)
    bx["size_bin"] = pd.cut(bx.area_px, [-1, SMALL, MEDIUM, np.inf], labels=["small", "medium", "large"])
    bx.drop(columns="n_fields").to_csv(tab / "boxes.csv", index=False)
    imgs.to_csv(tab / "images.csv", index=False)

    # 1. instances and images per class
    inst = bx.pivot_table(index="class_name", columns="split", values="cls", aggfunc="size", fill_value=0)
    nimg = bx.groupby(["class_name", "split"]).image.nunique().unstack(fill_value=0)
    counts = inst.add_prefix("instances_").join(nimg.add_prefix("images_"))
    for s in SPLITS:
        counts[f"instance_share_{s}"] = (counts[f"instances_{s}"] / counts[f"instances_{s}"].sum()).round(4)
    counts.to_csv(tab / "class_counts.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    inst.plot.bar(ax=ax[0], title="Instances per class")
    nimg.plot.bar(ax=ax[1], title="Images containing class")
    fig.tight_layout(); fig.savefig(out / "class_counts.png", dpi=120); plt.close(fig)

    # 2. box sizes, aspect ratios, boxes per image
    size_tab = bx.groupby(["split", "class_name", "size_bin"], observed=False).size().unstack(fill_value=0)
    size_tab.to_csv(tab / "size_bins.csv")
    desc = bx.groupby(["split", "class_name"])[["w_px", "h_px", "sqrt_area_px", "aspect_w_over_h"]].describe(
        percentiles=[.05, .25, .5, .75, .95]).round(2)
    desc.to_csv(tab / "box_size_stats.csv")
    for col, fname, bins in [("sqrt_area_px", "sqrt_area_hist.png", 50), ("aspect_w_over_h", "aspect_hist.png", 50)]:
        fig, axes = plt.subplots(1, 5, figsize=(22, 3.5), sharex=True)
        rng = np.nanpercentile(bx[col], [0.5, 99.5])
        for c, ax in zip(sorted(names), axes):
            for s in SPLITS:
                v = bx[(bx.cls == c) & (bx.split == s)][col].dropna()
                ax.hist(v, bins=bins, range=rng, alpha=.6, density=True, label=f"{s} (n={len(v)})")
            ax.set_title(names[c]); ax.legend(fontsize=7); ax.set_xlabel(col)
        fig.tight_layout(); fig.savefig(out / fname, dpi=110); plt.close(fig)
    bpi = imgs.merge(bx.groupby(["split", "image"]).size().rename("n_boxes").reset_index(),
                     on=["split", "image"], how="left").fillna({"n_boxes": 0})
    bpi[["split", "image", "n_boxes"]].to_csv(tab / "boxes_per_image.csv", index=False)
    bpi.groupby("split").n_boxes.describe().to_csv(tab / "boxes_per_image_stats.csv")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for s in SPLITS:
        ax.hist(bpi[bpi.split == s].n_boxes, bins=40, alpha=.6, density=True, label=s)
    ax.set_xlabel("boxes per image"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "boxes_per_image.png", dpi=120); plt.close(fig)

    # 3. image resolution
    res = imgs.groupby(["split", "width", "height"]).size().rename("n_images").reset_index()
    res.sort_values("n_images", ascending=False).to_csv(tab / "resolutions.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    for s in SPLITS:
        d = imgs[imgs.split == s]
        ax.scatter(d.width, d.height, alpha=.5, label=s, s=14)
    ax.set_xlabel("width px"); ax.set_ylabel("height px"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "resolutions.png", dpi=120); plt.close(fig)

    # 4. leakage via perceptual hash (also within-split, for context)
    hashes = {}
    for r in imgs.itertuples():
        with Image.open(r.path) as im:
            g = im.convert("L")
            hashes[(r.split, r.image)] = (imagehash.phash(g), imagehash.dhash(g))
    keys = list(hashes)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            p1, d1 = hashes[keys[i]]; p2, d2 = hashes[keys[j]]
            dp = p1 - p2
            if dp <= PHASH_MAX_DIST:
                kind = "cross" if keys[i][0] != keys[j][0] else f"within_{keys[i][0]}"
                pairs.append(dict(kind=kind, split_a=keys[i][0], image_a=keys[i][1], split_b=keys[j][0],
                                  image_b=keys[j][1], phash_dist=dp, dhash_dist=d1 - d2))
    pairs = pd.DataFrame(pairs, columns=["kind", "split_a", "image_a", "split_b", "image_b", "phash_dist", "dhash_dist"])
    pairs.sort_values(["kind", "phash_dist"]).to_csv(tab / "near_duplicate_pairs.csv", index=False)
    # nearest train neighbour for every val image, so the threshold choice is inspectable
    nn = []
    for k in [k for k in keys if k[0] == "val"]:
        d = sorted((hashes[k][0] - hashes[t][0], t[1]) for t in keys if t[0] == "train")
        nn.append(dict(val_image=k[1], nearest_train=d[0][1], phash_dist=d[0][0]))
    pd.DataFrame(nn).sort_values("phash_dist").to_csv(tab / "val_nearest_train.csv", index=False)
    cross = pairs[pairs.kind == "cross"].sort_values("phash_dist").head(8)
    if len(cross):
        fig, axes = plt.subplots(len(cross), 2, figsize=(8, 4 * len(cross)), squeeze=False)
        for row, r in zip(axes, cross.itertuples()):
            for ax, s, n in [(row[0], r.split_a, r.image_a), (row[1], r.split_b, r.image_b)]:
                im = Image.open(data / s / "images" / n); im.thumbnail((600, 600))
                ax.imshow(im); ax.set_title(f"{s}/{n} (pHash d={r.phash_dist})", fontsize=8); ax.axis("off")
        fig.tight_layout(); fig.savefig(out / "leakage_pairs.png", dpi=100); plt.close(fig)

    # 5. label sanity
    issues = []
    eps = 1e-6
    for r in bx.itertuples():
        if r.w_px <= 0 or r.h_px <= 0:
            issues.append((r.split, r.image, r.box_idx, r.cls, "degenerate_zero_area", f"{r.w_px:.2f}x{r.h_px:.2f}"))
        elif min(r.w_px, r.h_px) < TINY_PX:
            issues.append((r.split, r.image, r.box_idx, r.cls, f"tiny_side_lt_{TINY_PX}px", f"{r.w_px:.2f}x{r.h_px:.2f}"))
        if r.x1 < -eps or r.y1 < -eps or r.x2 > r.img_w + eps or r.y2 > r.img_h + eps:
            issues.append((r.split, r.image, r.box_idx, r.cls, "outside_image",
                           f"x1={r.x1:.1f} y1={r.y1:.1f} x2={r.x2:.1f} y2={r.y2:.1f}"))
        if r.n_fields != 5 or r.cls not in names:
            issues.append((r.split, r.image, r.box_idx, r.cls, "malformed_row", f"fields={r.n_fields}"))
    for (s, n), g in bx.groupby(["split", "image"]):
        if len(g) < 2:
            continue
        m = iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy())
        ii, jj = np.where(np.triu(m, 1) > DUP_IOU)
        gi = g.reset_index(drop=True)
        for i, j in zip(ii, jj):
            same = gi.cls[i] == gi.cls[j]
            exact = np.allclose(gi.loc[i, ["xc", "yc", "bw", "bh"]].astype(float),
                                gi.loc[j, ["xc", "yc", "bw", "bh"]].astype(float), atol=1e-7)
            kind = ("exact" if exact else "near") + "_duplicate_" + ("same_class" if same else "diff_class")
            issues.append((s, n, int(gi.box_idx[i]), int(gi.cls[i]), kind,
                           f"with box {int(gi.box_idx[j])} cls={int(gi.cls[j])} IoU={m[i, j]:.3f}"))
    for r in bpi[bpi.n_boxes == 0].itertuples():
        issues.append((r.split, r.image, -1, -1, "empty_image", ""))
    iss = pd.DataFrame(issues, columns=["split", "image", "box_idx", "cls", "issue", "detail"])
    iss.to_csv(tab / "label_issues.csv", index=False)
    iss_summary = iss.groupby(["split", "issue"]).size().rename("count").reset_index()
    iss_summary.to_csv(tab / "label_issues_summary.csv", index=False)

    # 6. visual grids: 16 random images per class (full image, downscaled) + 16 random instance crops per class
    for c in sorted(names):
        for s in SPLITS:
            sel = bx[(bx.cls == c) & (bx.split == s)]
            ims = sorted(sel.image.unique())
            pick = random.sample(ims, min(16, len(ims)))
            if pick:
                fig, axes = plt.subplots(4, 4, figsize=(16, 13))
                for ax in axes.flat:
                    ax.axis("off")
                for ax, n in zip(axes.flat, pick):
                    rows = bx[(bx.split == s) & (bx.image == n)]
                    ax.imshow(draw(data / s / "images" / n, rows, names)); ax.set_title(n, fontsize=8)
                fig.suptitle(f"{names[c]} — {s} — all classes drawn (colour = class)")
                fig.tight_layout(); fig.savefig(out / f"grid_images_{s}_cls{c}.png", dpi=90); plt.close(fig)
            inst_pick = sel.sample(min(16, len(sel)), random_state=a.seed)
            if len(inst_pick):
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                for ax in axes.flat:
                    ax.axis("off")
                for ax, r in zip(axes.flat, inst_pick.itertuples()):
                    im = Image.open(data / s / "images" / r.image).convert("RGB")
                    pad = max(40, 2 * max(r.w_px, r.h_px))
                    box = (int(max(r.x1 - pad, 0)), int(max(r.y1 - pad, 0)),
                           int(min(r.x2 + pad, r.img_w)), int(min(r.y2 + pad, r.img_h)))
                    crop = im.crop(box)
                    rows = bx[(bx.split == s) & (bx.image == r.image)].copy()
                    for k in ("x1", "x2"):
                        rows[k] -= box[0]
                    for k in ("y1", "y2"):
                        rows[k] -= box[1]
                    d = ImageDraw.Draw(crop)
                    for q in rows.itertuples():
                        d.rectangle([q.x1, q.y1, q.x2, q.y2], outline=COLORS[q.cls % 5], width=1)
                    ax.imshow(crop.resize((256, int(256 * crop.height / max(crop.width, 1))), Image.NEAREST))
                    ax.set_title(f"{r.image} {r.w_px:.0f}x{r.h_px:.0f}px", fontsize=8)
                fig.suptitle(f"{names[c]} — {s} — instance crops (colour = class)")
                fig.tight_layout(); fig.savefig(out / f"grid_crops_{s}_cls{c}.png", dpi=90); plt.close(fig)

    summary = dict(
        classes=names,
        n_images={s: int((imgs.split == s).sum()) for s in SPLITS},
        n_boxes={s: int((bx.split == s).sum()) for s in SPLITS},
        cross_split_near_duplicate_pairs=int((pairs.kind == "cross").sum()),
        within_split_near_duplicate_pairs={s: int((pairs.kind == f"within_{s}").sum()) for s in SPLITS},
        phash_threshold=PHASH_MAX_DIST,
        label_issues=iss.groupby("issue").size().to_dict(),
        median_sqrt_area_px=bx.groupby("split").sqrt_area_px.median().round(2).to_dict(),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
