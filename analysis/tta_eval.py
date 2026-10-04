"""Test-time augmentation for a sliced model, without Ultralytics' augment=True (which also downscales). CPU or GPU.

  python analysis/tta_eval.py --config configs/b1h.yaml --weights <last.pt> --data-root "$DATA_ROOT" \
      --split holdout --out <dir> [--device cpu]

Every 1024 tile (eval.py geometry: tile 1024, overlap 256) is predicted in four variants, each mapped back to tile
coordinates:
  orig    the tile as is
  hflip   horizontally flipped                 x -> W - x
  vflip   vertically flipped                   y -> H - y
  up1.5   resized 1.5x (bilinear) and predicted at imgsz 1.5 x 1024; boxes / 1.5
Same conf (0.001), NMS IoU (0.7) and per-tile max_det as eval.py. Raw rows (RAW_COLS + variant) go to
<out>/predictions_raw_tta.csv. Then, for each variant set {orig}, {orig,hflip}, {orig,vflip}, {orig,up1.5}, {all four},
the union of the raw rows is written as predictions_raw.csv and scored by eval.py --from-preds, which merges it with
the config's merge (class-wise NMS on IoS 0.6, max_det 902) and the same scorer. The {orig} row must reproduce the
model's normal eval. Writes <out>/<set>/ (eval.py outputs) and <out>/tta_summary.csv.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import list_images  # noqa: E402
from detlib.tiling import tile_windows  # noqa: E402
from eval import RAW_COLS, boxes_of  # noqa: E402

SETS = {"orig": ["orig"], "orig+hflip": ["orig", "hflip"], "orig+vflip": ["orig", "vflip"],
        "orig+up1.5": ["orig", "up1.5"], "all4": ["orig", "hflip", "vflip", "up1.5"]}


def variants(tile, imgsz):
    h, w = tile.shape[:2]
    yield "orig", tile, imgsz, lambda b: b
    yield "hflip", tile[:, ::-1].copy(), imgsz, lambda b: np.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], 1)
    yield "vflip", tile[::-1].copy(), imgsz, lambda b: np.stack([b[:, 0], h - b[:, 3], b[:, 2], h - b[:, 1]], 1)
    up = cv2.resize(tile, (round(w * 1.5), round(h * 1.5)), interpolation=cv2.INTER_LINEAR)
    yield "up1.5", up, round(imgsz * 1.5), lambda b: b / 1.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["holdout", "val"], default="holdout")
    ap.add_argument("--device", default=None); ap.add_argument("--max-images", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); ev = cfg["eval"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data_root)
    if a.split == "holdout":
        names = [l.strip() for l in (REPO / cfg["holdout_list"]).read_text().splitlines() if l.strip()]
        images = [root / "train" / "images" / n for n in names]
    else:
        images = list_images(root / "val" / "images")
    images = images[: a.max_images]
    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows = []
    for i, p in enumerate(images):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        for x0, y0, x1, y1 in tile_windows(W, H, ev["tile"], ev["overlap"]):
            tile = img[y0:y1, x0:x1]
            for name, im, sz, back in variants(tile, ev["imgsz"]):
                r = model.predict(im, imgsz=sz, conf=ev["conf"], iou=ev["nms_iou"], max_det=ev["max_det"],
                                  device=a.device, verbose=False)[0]
                b, s, c = boxes_of(r)
                if len(c):
                    b = back(b) + np.array([x0, y0, x0, y0], float)
                    rows += [dict(image=p.name, tile_x0=x0, tile_y0=y0, cls=int(k), conf=float(v), x1=q[0], y1=q[1],
                                  x2=q[2], y2=q[3], variant=name) for q, v, k in zip(b, s, c)]
        print(f"[tta] {i + 1}/{len(images)} {p.name}", flush=True)
    raw = pd.DataFrame(rows, columns=RAW_COLS + ["variant"])
    raw.to_csv(out / "predictions_raw_tta.csv", index=False)
    summ = []
    for k, vs in SETS.items():
        d = out / k; d.mkdir(exist_ok=True)
        raw[raw.variant.isin(vs)][RAW_COLS].to_csv(d / "predictions_raw_in.csv", index=False)
        cmd = [sys.executable, str(REPO / "eval.py"), "--config", a.config, "--data-root", str(root), "--weights",
               a.weights, "--run-dir", str(out), "--split", a.split, "--from-preds", str(d / "predictions_raw_in.csv"),
               "--out", str(d)]
        if a.split == "val":
            cmd = [c for c in cmd if c not in ("--split", "val")]
        subprocess.run(cmd, check=True)
        m = pd.read_csv(d / "per_class.csv")
        allr = m[m.cls == -1].iloc[0]
        summ.append(dict(set=k, mAP50=allr.AP50_coco, ci_lo=allr.get("AP50_coco_ci95_lo"), ci_hi=allr.get("AP50_coco_ci95_hi"),
                         **{f"AP50 {r.class_name}": r.AP50_coco for r in m[m.cls >= 0].itertuples()}))
    s = pd.DataFrame(summ); s.to_csv(out / "tta_summary.csv", index=False); print(s.to_string(index=False))


if __name__ == "__main__":
    main()
