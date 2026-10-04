"""E10 training data: our 403 training images (train minus holdout40) plus every labelled xView train image that is
not in our dataset, with 13 classes. Val and holdout40 are hard-excluded by image ID.

  python tools/make_xview_dataset.py --xview <mirror root> --data-root /kaggle/tmp/data --out /kaggle/tmp/xv_data

Classes 0-4 are ours (Cargo Truck, Truck w/Box, Truck w/Flatbed, Truck Tractor, Truck w/Liquid). Classes 5-12 are
xView truck types that our dataset excludes: Pickup Truck (20), Utility Truck (21), Truck (23), Trailer (27), Crane
Truck (32), Dump Truck (60), Haul Truck (61), Cement Mixer (65). Evaluation scores only classes 0-4 (eval.py drops
predicted classes beyond the data root's classmap).
- Our 403 images: our labels for classes 0-4 (unchanged) + xView boxes of the 8 excluded types (same pixel grid;
  sizes verified equal).
- Extra xView images: xView boxes of the 5 types mapped to 0-4 + the 8 excluded types. Boxes are clipped to the
  image; boxes under 2 px on a side are dropped.
- Exclusion: splits/e10_exclude.txt (22 val + 40 holdout40 stems); every extra image's stem is also checked against
  all 465 of our image stems. The script aborts if any excluded stem reaches the training list.
Writes <out>/classmap.txt, data.yaml, train/images (symlinks), train/labels, val -> our val, e10_train_list.txt,
e10_dataset_summary.json.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, read_yolo_labels  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OURS = {24: 0, 25: 1, 28: 2, 26: 3, 29: 4}
EXTRA = {20: 5, 21: 6, 23: 7, 27: 8, 32: 9, 60: 10, 61: 11, 65: 12}
NAMES = ["Cargo Truck", "Truck w/Box", "Truck w/Flatbed", "Truck Tractor", "Truck w/Liquid", "Pickup Truck",
         "Utility Truck", "Truck", "Trailer", "Crane Truck", "Dump Truck", "Haul Truck", "Cement Mixer"]


def select_extra(xview_stems, our_stems, exclude):
    """xView stems to add: not ours, not excluded. Raises if anything excluded slips through."""
    sel = sorted(s for s in xview_stems if s not in our_stems and s not in exclude)
    assert not (set(sel) & set(exclude)), "excluded image in training list"
    return sel


def yolo_lines(boxes, W, H):
    out = []
    for c, (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xview", required=True); ap.add_argument("--data-root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xv, root, out = Path(a.xview), Path(a.data_root), Path(a.out)
    (out / "train" / "images").mkdir(parents=True, exist_ok=True); (out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    exclude = set(Path(l.strip()).stem for l in (REPO / "splits" / "e10_exclude.txt").read_text().splitlines() if l.strip())
    im = pd_read = __import__("pandas").read_csv(EDA_TABLES / "images.csv")
    our_stems = set(Path(i).stem for i in im.image)
    hold = set(Path(l.strip()).stem for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip())
    vals = set(Path(i).stem for i in im[im.split == "val"].image)
    assert hold | vals <= exclude, "exclusion list incomplete"
    feats = json.load(open(xv / "train_labels" / "xView_train.geojson"))["features"]
    xb = defaultdict(list)
    for f in feats:
        p = f["properties"]; t = int(p["type_id"])
        if t in OURS or t in EXTRA:
            try:
                xb[Path(p["image_id"]).stem].append((t, [float(v) for v in p["bounds_imcoords"].split(",")]))
            except Exception:
                pass
    all_xv = set(p.stem for p in (xv / "train_images" / "train_images").glob("*.tif"))
    del feats
    counts, n_img, listing = Counter(), Counter(), []
    for r in im[(im.split == "train")].itertuples():
        s = Path(r.image).stem
        if s in exclude:
            continue
        with Image.open(xv / "train_images" / "train_images" / f"{s}.tif") as t:
            assert t.size == (r.width, r.height), f"size mismatch {s}"
        gc, gb = read_yolo_labels(root / "train" / "labels" / f"{s}.txt", r.width, r.height)
        lines = [f"{int(c)} {(b[0] + b[2]) / 2 / r.width:.6f} {(b[1] + b[3]) / 2 / r.height:.6f} {(b[2] - b[0]) / r.width:.6f} {(b[3] - b[1]) / r.height:.6f}" for c, b in zip(gc, gb)]
        lines += yolo_lines([(EXTRA[t], b) for t, b in xb.get(s, []) if t in EXTRA], r.width, r.height)
        os.symlink(root / "train" / "images" / r.image, out / "train" / "images" / r.image)
        (out / "train" / "labels" / f"{s}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        listing.append(r.image); n_img["ours"] += 1
        for l in lines:
            counts[("ours", int(l.split()[0]))] += 1
    for s in select_extra(all_xv & set(xb), our_stems, exclude):
        p = xv / "train_images" / "train_images" / f"{s}.tif"
        with Image.open(p) as t:
            W, H = t.size
        lines = yolo_lines([((OURS.get(t, EXTRA.get(t))), b) for t, b in xb[s]], W, H)
        os.symlink(p, out / "train" / "images" / f"{s}.tif")
        (out / "train" / "labels" / f"{s}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        listing.append(f"{s}.tif"); n_img["extra"] += 1
        for l in lines:
            counts[("extra", int(l.split()[0]))] += 1
    bad = [x for x in listing if Path(x).stem in exclude]
    assert not bad, f"excluded images in training list: {bad[:5]}"
    os.symlink(root / "val", out / "val") if not (out / "val").exists() else None
    (out / "classmap.txt").write_text("\n".join(f"{i} {n}" for i, n in enumerate(NAMES)) + "\n")
    (out / "e10_train_list.txt").write_text("\n".join(listing) + "\n")
    summ = dict(images=dict(n_img), classes=NAMES,
                instances={f"{src} {NAMES[c]}": n for (src, c), n in sorted(counts.items())},
                excluded_ids=len(exclude), excluded_found_in_list=0)
    (out / "e10_dataset_summary.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
