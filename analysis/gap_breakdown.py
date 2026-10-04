"""Diagnostic re-scoring of B1h on val (CPU). Not a claimed result: xView labels are used only here, to explain the
gap, never for training or for choosing anything.

  python analysis/gap_breakdown.py --data-root "$DATA_ROOT" --b1h <B1h run dir> --xview <mirror root> --out <dir>

Rows (each with mAP50, per-class AP50 and class-agnostic AP50; COCO 101-point; conf >= 0.001):
1 plain             B1h's saved val predictions (reproduces 0.1065)
2 excluded-ignore   predictions that overlap (IoU >= 0.3) an xView box of an excluded truck type (Pickup 20, Utility 21,
                    Truck 23, Trailer 27, Crane 32, Dump 60, Haul 61, Cement Mixer 65) AND have IoU < 0.5 with every
                    one of our GT boxes are removed (neither TP nor FP). xView boxes are scaled to our image size for
                    the 4 rescaled images.
3 native-scale      the 4 rescaled val images (2308, 2384, 2391, 2460) are resized (bilinear) to their xView original
                    size, run through B1h with eval.py's pipeline (sliced, merge, max_det 902), and the boxes mapped
                    back; the other 18 images keep their saved predictions
4 both              2 applied to 3
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import list_images, read_yolo_labels  # noqa: E402
from detlib.scoring import ap_from_records, box_iou, score_images  # noqa: E402
from eval import collect, finalize, image_sizes, run_predictions  # noqa: E402

EXCL = {20, 21, 23, 27, 32, 60, 61, 65}
RESCALED = ["2308.png", "2384.png", "2391.png", "2460.png"]
NAMES = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]


def score(preds, imgs, ldir, sizes):
    per = collect(preds[preds.conf >= 0.001], imgs, ldir, sizes)
    aps, m, _ = ap_from_records(score_images(per, 5), 5, "coco")
    ag = [dict(d, p_cls=np.zeros(len(d["p_cls"]), int), g_cls=np.zeros(len(d["g_cls"]), int)) for d in per]
    _, mag, _ = ap_from_records(score_images(ag, 1), 1, "coco")
    return dict(mAP50=float(m), class_agnostic_AP50=float(mag), **{f"AP50 {n}": float(a) for n, a in zip(NAMES, aps)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True); ap.add_argument("--b1h", required=True)
    ap.add_argument("--xview", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, B, xv, out = Path(a.data_root), Path(a.b1h), Path(a.xview), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    imgs = list_images(root / "val" / "images"); sizes = image_sizes(imgs); L = root / "val" / "labels"
    plain = pd.read_csv(B / "eval" / "predictions.csv")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    xsz = {}
    for n in sizes:
        with Image.open(xv / "train_images" / "train_images" / f"{Path(n).stem}.tif") as t:
            xsz[n] = t.size
    feats = json.load(open(xv / "train_labels" / "xView_train.geojson"))["features"]
    stems = {Path(n).stem: n for n in sizes}; ex = defaultdict(list)
    for f in feats:
        p = f["properties"]; s = Path(p["image_id"]).stem
        if s in stems and int(p["type_id"]) in EXCL:
            try:
                ex[stems[s]].append([float(v) for v in p["bounds_imcoords"].split(",")])
            except Exception:
                pass
    del feats
    for n in ex:
        (W, H), (XW, XH) = sizes[n], xsz[n]
        ex[n] = np.array(ex[n]) * np.array([W / XW, H / XH, W / XW, H / XH])

    def ignore(preds):
        keep = []
        for n, g in preds.groupby("image"):
            pb = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
            _, gb = read_yolo_labels(L / f"{Path(n).stem}.txt", *sizes[n])
            e = ex.get(n, np.zeros((0, 4)))
            hit_ex = box_iou(pb, e).max(1) >= 0.3 if len(e) else np.zeros(len(pb), bool)
            hit_gt = box_iou(pb, gb).max(1) >= 0.5 if len(gb) else np.zeros(len(pb), bool)
            keep.append(g[~(hit_ex & ~hit_gt)])
        return pd.concat(keep)

    cfg = yaml.safe_load(open(REPO / "configs" / "b1h.yaml")); ev = cfg["eval"]
    A = SimpleNamespace(weights=str(B / "train" / "weights" / "last.pt"), device="cpu", batch=16, mode="sliced",
                        imgsz=ev["imgsz"], tile=ev["tile"], overlap=ev["overlap"], conf=ev["conf"], nms_iou=ev["nms_iou"],
                        max_det=ev["max_det"])
    tmp = Path("/kaggle/tmp/native") if Path("/kaggle").exists() else out / "tmp"; tmp.mkdir(parents=True, exist_ok=True)
    nat = []
    for n in RESCALED:
        im = cv2.imread(str(root / "val" / "images" / n), cv2.IMREAD_COLOR); XW, XH = xsz[n]
        cv2.imwrite(str(tmp / n), cv2.resize(im, (XW, XH), interpolation=cv2.INTER_LINEAR))
        raw = run_predictions(A, [tmp / n])
        p = finalize(raw, {n: (XW, XH)}, ev["merge"], ev["merge_thr"], ev["merge_metric"], ev["max_det"])
        W, H = sizes[n]
        for c, f in (("x1", W / XW), ("x2", W / XW), ("y1", H / XH), ("y2", H / XH)):
            p[c] = p[c] * f
        nat.append(p)
    native = pd.concat([plain[~plain.image.isin(RESCALED)]] + nat, ignore_index=True)
    rows = [dict(row="1 plain", **score(plain, imgs, L, sizes)),
            dict(row="2 excluded-type ignore", **score(ignore(plain), imgs, L, sizes)),
            dict(row="3 native scale (4 rescaled images)", **score(native, imgs, L, sizes)),
            dict(row="4 both", **score(ignore(native), imgs, L, sizes))]
    r = pd.DataFrame(rows); r.to_csv(out / "gap_breakdown.csv", index=False)
    removed = len(plain) - len(ignore(plain))
    (out / "gap_breakdown.json").write_text(json.dumps(dict(rows=rows, predictions_removed_by_ignore=removed,
                                                           excluded_boxes_on_val=int(sum(len(v) for v in ex.values()))), indent=2))
    print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}")); print("removed", removed)


if __name__ == "__main__":
    main()
