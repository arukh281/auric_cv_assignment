"""Validation images -> predictions -> metrics, identical scoring for full-image (B0) and sliced (B1) models.

Examples:
  python eval.py --config configs/b0.yaml                       # weights = runs/<name>/train/weights/last.pt
  python eval.py --config configs/b1.yaml --runs-root /content/drive/MyDrive/auric_runs
  python eval.py --config configs/b1.yaml --from-preds runs/b1_tile1024/eval/predictions.csv   # re-score only

Writes <run>/eval/ (or --out): predictions.csv, metrics.json, per_class.csv, bootstrap_ci.csv,
pr_curves.png, eval_args.json, plus env/eval_* and a line in <run>/command.txt.

mAP50 here = mean over classes of AP at IoU 0.5 (COCO 101-point interpolation) at conf >= --conf.
The Ultralytics-interpolated value is also reported; see detlib/scoring.py for the difference.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detlib.data import (eda_max_boxes_per_image, list_images, load_classes, read_yolo_labels,  # noqa: E402
                         write_resolved_data_yaml)
from detlib.envlog import log_env  # noqa: E402
from detlib.merge import merge  # noqa: E402
from detlib.scoring import AP_METHODS, ap_from_records, bootstrap, pr_curve, score_images  # noqa: E402
from detlib.tiling import tile_windows  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PRED_COLS = ["image", "cls", "conf", "x1", "y1", "x2", "y2"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def boxes_of(result, dx=0, dy=0):
    b = result.boxes
    xyxy = b.xyxy.cpu().numpy().astype(float) + np.array([dx, dy, dx, dy], float)
    return xyxy, b.conf.cpu().numpy().astype(float), b.cls.cpu().numpy().astype(int)


def predict_full(model, img, a):
    r = model.predict(img, imgsz=a.imgsz, conf=a.conf, iou=a.nms_iou, max_det=a.max_det, device=a.device,
                      verbose=False)[0]
    return boxes_of(r)


def predict_sliced(model, img, a):
    h, w = img.shape[:2]
    wins = tile_windows(w, h, a.tile, a.overlap)
    parts = []
    for s in range(0, len(wins), a.batch):
        chunk = wins[s:s + a.batch]
        res = model.predict([img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk], imgsz=a.imgsz, conf=a.conf,
                            iou=a.nms_iou, max_det=a.max_det, device=a.device, verbose=False)
        parts += [boxes_of(r, x0, y0) for r, (x0, y0, _, _) in zip(res, chunk)]
    xyxy = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, 4))
    conf = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
    cls = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, int)
    xyxy, conf, cls = merge(xyxy, conf, cls, w, h, a.merge, a.merge_thr, a.merge_metric)
    order = np.argsort(-conf, kind="mergesort")[: a.max_det]
    return xyxy[order], conf[order], cls[order]


def run_predictions(a, images):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows, t0 = [], time.time()
    for i, p in enumerate(images):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        fn = predict_sliced if a.mode == "sliced" else predict_full
        xyxy, conf, cls = fn(model, img, a)
        rows += [dict(image=p.name, cls=int(c), conf=float(s), x1=b[0], y1=b[1], x2=b[2], y2=b[3])
                 for b, s, c in zip(xyxy, conf, cls)]
        print(f"[eval] {i + 1}/{len(images)} {p.name}: {len(conf)} preds ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows, columns=PRED_COLS)


def collect(preds, images, label_dir):
    per = []
    g = {k: v for k, v in preds.groupby("image")}
    for p in images:
        with Image.open(p) as im:  # header only
            w, h = im.size
        gc, gb = read_yolo_labels(label_dir / f"{p.stem}.txt", w, h)
        d = g.get(p.name, pd.DataFrame(columns=PRED_COLS))
        per.append(dict(image=p.name, p_xyxy=d[["x1", "y1", "x2", "y2"]].to_numpy(float),
                        p_cls=d.cls.to_numpy(int), p_conf=d.conf.to_numpy(float), g_xyxy=gb, g_cls=gc))
    return per


def plot_pr(recs, names, path):
    nc = len(names)
    cls = np.concatenate([r["cls"] for r in recs])
    conf = np.concatenate([r["conf"] for r in recs])
    tp = np.concatenate([r["tp"] for r in recs])
    n_gt = np.sum([r["n_gt"] for r in recs], axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(nc):
        m = cls == c
        if n_gt[c] == 0 or not m.any():
            continue
        rec, prec = pr_curve(conf[m], tp[m], n_gt[c])
        ax.plot(rec, np.maximum.accumulate(prec[::-1])[::-1], label=f"{names[c]} (n_gt={n_gt[c]})")
    ax.set(xlabel="recall", ylabel="precision (envelope)", xlim=(0, 1), ylim=(0, 1.02), title="PR @ IoU 0.5")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def ultra_crosscheck(a, root, names, out):
    """Ultralytics' own validator on the full val images, to sanity-check our scorer (full mode only)."""
    from ultralytics import YOLO
    yml = write_resolved_data_yaml(out / "crosscheck_data.yaml", root, "val/images", "val/images", names)
    m = YOLO(a.weights).val(data=str(yml), imgsz=a.imgsz, conf=a.conf, iou=a.nms_iou, max_det=a.max_det,
                            batch=1, device=a.device, plots=False, project=str(out.resolve()), name="ultra_val",
                            exist_ok=True, verbose=False)
    per = {names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)}
    res = dict(note="Ultralytics val: rect letterbox batches, its own matching; expect small differences",
               mAP50=float(m.box.map50), per_class_AP50=per)
    (out / "ultralytics_crosscheck.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="configs/*.yaml; its `eval` section gives defaults")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run-dir", help="default: <runs-root>/<config name>")
    ap.add_argument("--weights", help="default: <run-dir>/train/weights/last.pt")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", help="default: <run-dir>/eval")
    ap.add_argument("--mode", choices=["full", "sliced"])
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--tile", type=int)
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--merge", choices=["nms", "wbf", "none"])
    ap.add_argument("--merge-thr", type=float)
    ap.add_argument("--merge-metric", choices=["iou", "ios"])
    ap.add_argument("--conf", type=float)
    ap.add_argument("--nms-iou", type=float, help="per-image/per-tile NMS IoU inside Ultralytics")
    ap.add_argument("--max-det", type=int, help="default: max(300, 2 x max GT boxes in a val image)")
    ap.add_argument("--batch", type=int, default=16, help="tiles per predict call")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-images", type=int, help="testing only")
    ap.add_argument("--from-preds", help="skip inference; score this predictions.csv")
    ap.add_argument("--ultra-crosscheck", action="store_true", help="also run Ultralytics val (full mode)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config)) if a.config else {}
    defaults = dict(mode="full", imgsz=640, tile=1024, overlap=256, merge="nms", merge_thr=0.6,
                    merge_metric="ios", conf=0.001, nms_iou=0.7)
    for k, v in {**defaults, **(cfg.get("eval") or {})}.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    run_dir = Path(a.run_dir) if a.run_dir else Path(a.runs_root) / cfg.get("name", "adhoc")
    a.weights = a.weights or str(run_dir / "train" / "weights" / "last.pt")
    out = Path(a.out) if a.out else run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)

    max_gt = eda_max_boxes_per_image("val")
    if a.max_det is None:
        a.max_det = max(300, 2 * max_gt)
    print(f"[eval] max GT boxes in one val image (EDA table): {max_gt}; max_det = {a.max_det}")
    if a.max_det <= max_gt:
        sys.exit("max_det must exceed the max number of GT boxes per val image")

    root = Path(a.data_root)
    names = load_classes(root)
    images = list_images(root / "val" / "images")[: a.max_images]
    log_env(run_dir, "eval")

    if a.from_preds:
        preds = pd.read_csv(a.from_preds)
        if Path(a.from_preds).resolve() != (out / "predictions.csv").resolve():
            preds.to_csv(out / "predictions.csv", index=False)
    else:
        preds = run_predictions(a, images)
        preds.to_csv(out / "predictions.csv", index=False)
    preds = preds[preds.conf >= a.conf]

    recs = score_images(collect(preds, images, root / "val" / "labels"), len(names))
    res = {m: ap_from_records(recs, len(names), m) for m in AP_METHODS}
    boot = {m: bootstrap(recs, len(names), a.bootstrap, a.seed, m) for m in AP_METHODS} if a.bootstrap else {}

    n_pred = preds.cls.value_counts().reindex(range(len(names)), fill_value=0)
    n_gt = res["coco"][2]
    rows = []
    for c, name in names.items():
        r = dict(cls=c, class_name=name, n_gt=int(n_gt[c]), n_pred=int(n_pred[c]))
        for m in AP_METHODS:
            r[f"AP50_{m}"] = res[m][0][c]
            if boot:
                lo, hi = np.nanpercentile(boot[m][:, c], [2.5, 97.5])
                r[f"AP50_{m}_ci95_lo"], r[f"AP50_{m}_ci95_hi"] = lo, hi
        rows.append(r)
    allrow = dict(cls=-1, class_name="all (mAP50)", n_gt=int(n_gt.sum()), n_pred=len(preds))
    for m in AP_METHODS:
        allrow[f"AP50_{m}"] = res[m][1]
        if boot:
            allrow[f"AP50_{m}_ci95_lo"], allrow[f"AP50_{m}_ci95_hi"] = np.nanpercentile(boot[m][:, -1], [2.5, 97.5])
    table = pd.DataFrame(rows + [allrow])
    table.to_csv(out / "per_class.csv", index=False)
    if boot:
        ci = table[["class_name"] + [c for c in table.columns if "ci95" in c or c.startswith("AP50_")]]
        ci.to_csv(out / "bootstrap_ci.csv", index=False)
        pd.DataFrame(boot["coco"], columns=[names[c] for c in names] + ["mAP50"]).to_csv(
            out / "bootstrap_samples_coco.csv", index=False)

    plot_pr(recs, names, out / "pr_curves.png")
    w = Path(a.weights)
    metrics = dict(
        weights=str(w), weights_sha256=sha256(w) if w.exists() else None, n_images=len(images),
        mAP50=res["coco"][1], mAP50_ultralytics_interp=res["ultralytics"][1],
        per_class_AP50={names[c]: res["coco"][0][c] for c in names},
        per_class_AP50_ultralytics_interp={names[c]: res["ultralytics"][0][c] for c in names},
        n_gt={names[c]: int(n_gt[c]) for c in names},
        bootstrap=dict(n=a.bootstrap, seed=a.seed, unit="val image", ci="percentile 95%") if boot else None,
    )
    args = {k: v for k, v in vars(a).items()}
    (out / "eval_args.json").write_text(json.dumps(args, indent=2, default=str))
    if a.ultra_crosscheck and a.mode == "full" and not a.from_preds:
        metrics["ultralytics_val"] = ultra_crosscheck(a, root, names, out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
