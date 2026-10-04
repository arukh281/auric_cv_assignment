"""Single entry point: weights + an image folder -> predictions (and metrics if labels are given).

  python predict.py --weights b1h_last.pt --images path/to/images --out out/            # predictions only
  python predict.py --weights b1h_last.pt --images data/val/images --labels data/val/labels --out out/

Inference and post-processing are exactly eval.py's (eval.run_predictions + eval.finalize) with the `eval:` settings
of --config (default configs/b1h.yaml: sliced 1024 tiles, overlap 256, conf 0.001, NMS IoU 0.7, class-wise NMS merge
on IoS 0.6, max_det 902). Scoring is detlib/scoring.py (COCO 101-point AP50), as in eval.py, at conf >= 0.001.
Writes <out>/predictions.csv (image, cls, conf, x1, y1, x2, y2 in full-image pixels; also predictions_raw.csv,
pre-merge) and, with --labels, metrics.json and per_class.csv. Class names: --classmap, else <labels>/../../classmap.txt,
else the five names of this assignment. Use --device cpu on a machine without a GPU.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from detlib.data import list_images, load_classes  # noqa: E402
from detlib.scoring import AP_METHODS, ap_from_records, score_images  # noqa: E402
from eval import collect, finalize, image_sizes, run_predictions, sha256  # noqa: E402

NAMES = {0: "Cargo Truck", 1: "Truck w/Box", 2: "Truck w/Flatbed", 3: "Truck Tractor", 4: "Truck w/Liquid"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--labels", help="folder of YOLO .txt labels (same stems); enables metrics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(REPO / "configs" / "b1h.yaml"), help="its `eval:` section is used")
    ap.add_argument("--classmap", help="classmap.txt (default: <labels>/../../classmap.txt, else built-in names)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16, help="tiles per predict call")
    ap.add_argument("--max-images", type=int, help="first N images (sorted by name) only")
    a = ap.parse_args()
    ev = yaml.safe_load(open(a.config))["eval"]
    s = SimpleNamespace(weights=a.weights, device=a.device, batch=a.batch, mode=ev["mode"], imgsz=ev["imgsz"],
                        tile=ev.get("tile"), overlap=ev.get("overlap"), conf=ev["conf"], nms_iou=ev["nms_iou"],
                        max_det=ev["max_det"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    images = list_images(a.images)[: a.max_images]
    sizes = image_sizes(images)
    raw = run_predictions(s, images)
    raw.to_csv(out / "predictions_raw.csv", index=False)
    method = ev["merge"] if ev["mode"] == "sliced" else "none"
    preds = finalize(raw, sizes, method, ev.get("merge_thr"), ev.get("merge_metric"), ev["max_det"])
    preds.to_csv(out / "predictions.csv", index=False)
    print(f"[predict] {len(preds)} predictions on {len(images)} images -> {out / 'predictions.csv'}")
    info = dict(weights=str(a.weights), weights_sha256=sha256(a.weights), images=[p.name for p in images],
                config=str(a.config), eval_settings=ev)
    if a.labels:
        cm = Path(a.classmap) if a.classmap else Path(a.labels).parent.parent / "classmap.txt"
        names = load_classes(cm.parent) if cm.exists() else NAMES
        per = collect(preds[preds.conf >= ev["conf"]], images, Path(a.labels), sizes)
        recs = score_images(per, len(names))
        res = {m: ap_from_records(recs, len(names), m) for m in AP_METHODS}
        n_gt = res["coco"][2]
        rows = [dict(cls=c, class_name=n, n_gt=int(n_gt[c]), n_pred=int((preds.cls == c).sum()),
                     **{f"AP50_{m}": res[m][0][c] for m in AP_METHODS}) for c, n in names.items()]
        rows.append(dict(cls=-1, class_name="all (mAP50)", n_gt=int(n_gt.sum()), n_pred=len(preds),
                         **{f"AP50_{m}": res[m][1] for m in AP_METHODS}))
        pd.DataFrame(rows).to_csv(out / "per_class.csv", index=False)
        info.update(mAP50=res["coco"][1], mAP50_ultralytics_interp=res["ultralytics"][1],
                    per_class_AP50={names[c]: res["coco"][0][c] for c in names})
        print(f"[predict] mAP50 {res['coco'][1]:.4f} (COCO 101-point, conf >= {ev['conf']})")
    (out / "metrics.json").write_text(json.dumps(info, indent=2, default=float))


if __name__ == "__main__":
    main()
