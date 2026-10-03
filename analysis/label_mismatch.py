"""Re-check the tiles flagged by analysis/sanity_check.py's label check, with per-box differences (CPU only).

Usage (Kaggle CPU kernel, after scripts/kaggle_setup.sh):
  python analysis/label_mismatch.py --data-root /kaggle/tmp/data --tiles /kaggle/tmp/work/tiles_b1h --out <dir>

Tiles: B1h's tiles, regenerated with tools/make_tiles.py and B1h's settings if missing (as sanity_check.py).
Every written tile is compared with labels re-derived from its source image (detlib.tiling.clip_boxes, min_vis 0.5),
using sanity_check.check_labels' rule (same multiset of class ids, coordinates within 0.5 px after sorting by
coordinates). For each flagged tile, the two label sets are paired one-to-one by IoU (class-agnostic, greedy, highest
first, IoU >= 0.1), then every pair or leftover is typed:
  same      paired, same class, every coordinate within 0.5 px
  shifted   paired, same class, max coordinate difference > 0.5 px (dx1, dy1, dx2, dy2 and max |d| saved)
  reclass   paired, different class
  missing   in the re-derived labels but not in the tile file
  extra     in the tile file but not in the re-derived labels
For each source image with a flagged tile: pairs of its GT boxes with IoU >= 0.5 (and >= 0.9), and whether those
pairs share a class. Flagged tiles are rendered (green = re-derived, red = tile file, class names).

Writes <out>/flagged_tiles.csv (all flagged tiles with counts per type), box_diffs.csv (one row per box),
source_overlaps.csv (one row per source image), renders/*.png, summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from detlib.data import EDA_TABLES, load_classes, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402
from detlib.tiling import clip_boxes  # noqa: E402
from sanity_check import make_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def flagged(gc, gb, ec, eb):
    """sanity_check.check_labels' comparison: True if the tile is flagged."""
    same = len(gc) == len(ec) and np.array_equal(np.sort(gc), np.sort(ec))
    if same and len(gc):
        o1, o2 = np.lexsort(gb.T[::-1]), np.lexsort(eb.T[::-1])
        same = np.array_equal(gc[o1], ec[o2]) and np.abs(gb[o1] - eb[o2]).max() <= 0.5
    return not same


def pair_boxes(gc, gb, ec, eb, min_iou=0.1):
    """Greedy one-to-one pairing by class-agnostic IoU. Returns list of rows describing every box."""
    rows, used_t, used_e = [], set(), set()
    iou = box_iou(eb, gb) if len(eb) and len(gb) else np.zeros((len(eb), len(gb)))
    for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)) if iou.size else []:
        if iou[i, j] < min_iou:
            break
        if i in used_e or j in used_t:
            continue
        used_e.add(i); used_t.add(j)
        d = gb[j] - eb[i]
        md = float(np.abs(d).max())
        kind = "reclass" if gc[j] != ec[i] else ("same" if md <= 0.5 else "shifted")
        rows.append(dict(type=kind, expected_cls=int(ec[i]), tile_cls=int(gc[j]), iou=float(iou[i, j]),
                         dx1=float(d[0]), dy1=float(d[1]), dx2=float(d[2]), dy2=float(d[3]), max_abs_d=md,
                         ex1=eb[i, 0], ey1=eb[i, 1], ex2=eb[i, 2], ey2=eb[i, 3]))
    for i in range(len(eb)):
        if i not in used_e:
            rows.append(dict(type="missing", expected_cls=int(ec[i]), tile_cls=-1, ex1=eb[i, 0], ey1=eb[i, 1],
                             ex2=eb[i, 2], ey2=eb[i, 3]))
    for j in range(len(gb)):
        if j not in used_t:
            rows.append(dict(type="extra", expected_cls=-1, tile_cls=int(gc[j]), ex1=gb[j, 0], ey1=gb[j, 1],
                             ex2=gb[j, 2], ey2=gb[j, 3]))
    return rows


def source_overlaps(sc, sb):
    iou = box_iou(sb, sb)
    np.fill_diagonal(iou, 0)
    iu = np.triu_indices(len(sb), 1)
    v = iou[iu]
    samec = sc[iu[0]] == sc[iu[1]]
    return dict(n_boxes=len(sb), pairs_iou_ge_0_5=int((v >= 0.5).sum()), pairs_iou_ge_0_9=int((v >= 0.9).sum()),
                pairs_iou_ge_0_9_same_class=int(((v >= 0.9) & samec).sum()),
                pairs_iou_ge_0_9_diff_class=int(((v >= 0.9) & ~samec).sum()),
                exact_duplicate_pairs=int((np.abs(sb[iu[0]] - sb[iu[1]]).max(1) <= 0.5).sum()) if len(v) else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, tiles, out = Path(a.data_root), Path(a.tiles), Path(a.out)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    names = load_classes(root)
    make_tiles(root, tiles)
    idx = pd.read_csv(tiles / "tiles_index.csv")
    im = pd.read_csv(EDA_TABLES / "images.csv").query("split == 'train'")
    sizes = {r.image: (int(r.width), int(r.height)) for r in im.itertuples()}
    flag_rows, box_rows, src_rows, srcs = [], [], [], {}
    for r in idx[idx.kept].itertuples():
        if r.source not in srcs:
            W, H = sizes[r.source]
            srcs = {r.source: read_yolo_labels(root / "train" / "labels" / f"{Path(r.source).stem}.txt", W, H)}
        sc, sb = srcs[r.source]
        ec, eb, _ = clip_boxes(sc, sb, (r.x0, r.y0, r.x1, r.y1), 0.5)
        tw, th = r.x1 - r.x0, r.y1 - r.y0
        gc, gb = read_yolo_labels(tiles / "train" / "labels" / f"{r.tile}.txt", tw, th)
        if not flagged(gc, gb, ec, eb):
            continue
        rows = pair_boxes(gc, gb, ec, eb)
        for x in rows:
            x.update(tile=r.tile, source=r.source)
        box_rows += rows
        t = pd.Series([x["type"] for x in rows]).value_counts()
        flag_rows.append(dict(tile=r.tile, source=r.source, n_tile_boxes=len(gc), n_expected=len(ec),
                              **{k: int(t.get(k, 0)) for k in ("same", "shifted", "reclass", "missing", "extra")}))
        img = Image.open(tiles / "train" / "images" / f"{r.tile}.png").convert("RGB")
        d = ImageDraw.Draw(img)
        for c, b in zip(ec, eb):
            d.rectangle(list(map(float, b)), outline=(0, 220, 0), width=3)
            d.text((float(b[0]), float(b[3]) + 1), f"exp {names[int(c)]}", fill=(0, 220, 0))
        for c, b in zip(gc, gb):
            d.rectangle(list(map(float, b)), outline=(255, 40, 40), width=1)
            d.text((float(b[0]), max(0.0, float(b[1]) - 11)), f"tile {names[int(c)]}", fill=(255, 40, 40))
        img.save(out / "renders" / f"{r.tile}.png")
    fl = pd.DataFrame(flag_rows)
    for s in sorted(set(fl.source)) if len(fl) else []:
        W, H = sizes[s]
        sc, sb = read_yolo_labels(root / "train" / "labels" / f"{Path(s).stem}.txt", W, H)
        src_rows.append(dict(source=s, n_flagged_tiles=int((fl.source == s).sum()), **source_overlaps(sc, sb)))
    fl.to_csv(out / "flagged_tiles.csv", index=False)
    bd = pd.DataFrame(box_rows)
    bd.to_csv(out / "box_diffs.csv", index=False)
    so = pd.DataFrame(src_rows)
    so.to_csv(out / "source_overlaps.csv", index=False)
    summary = dict(n_flagged_tiles=len(fl), n_sources=int(so.shape[0]),
                   box_types=bd.type.value_counts().to_dict() if len(bd) else {},
                   shifted_max_abs_d=float(bd[bd.type == "shifted"].max_abs_d.max()) if len(bd) and (bd.type == "shifted").any() else None,
                   sources_with_iou09_pairs=int((so.pairs_iou_ge_0_9 > 0).sum()) if len(so) else 0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[mismatch]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
