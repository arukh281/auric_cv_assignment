"""Validation images -> predictions -> metrics, identical scoring for full-image (B0) and sliced (B1) models.

Examples:
  python eval.py --config configs/b0.yaml                       # weights = runs/<name>/train/weights/last.pt
  python eval.py --config configs/b1.yaml --runs-root /content/drive/MyDrive/auric_runs
  python eval.py --config configs/b1.yaml --from-preds runs/b1_tile1024/eval/predictions.csv   # re-score only

Writes <run>/eval/ (or --out): predictions_raw.csv (pre-merge, with tile offsets), predictions.csv (merged,
capped at max_det), metrics.json, per_class.csv, bootstrap_ci.csv, pr_curves.png, eval_args.json,
scorer_comparison.csv + coco_gt.json/coco_dets.json (pycocotools cross-check), plus env/eval_* and a line in
<run>/command.txt. --from-preds accepts either predictions_raw.csv (re-merged with the current settings) or
predictions.csv.

Train-set sanity check (--split train): scores FULL-RESOLUTION original train images (not tiles) with the same
inference, merge and scorer as val. --max-images N then samples N images with --sample-seed (1938.png, the empty
train image, is never sampled); the sampled list goes into metrics.json. --out is required and may not be the
run's eval/ folder. Defaults (--split val) are unchanged.
  python eval.py --config configs/b1.yaml --split train --max-images 40 --out runs/b1_tile1024/eval_train40
Held-out train images (--split holdout): every image named in the config's holdout_list (or --holdout-list), at full
resolution from train/images + train/labels, same inference, merge and scorer; --out required. When the config
has a holdout_list, --split train never samples those images. Both record the image list in metrics.json.

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
RAW_COLS = ["image", "tile_x0", "tile_y0", "cls", "conf", "x1", "y1", "x2", "y2"]
TRAIN_EMPTY_IMAGE = "1938.png"  # the one train image without boxes (Phase 1 EDA); not sampled by --split train
SCORER_TOL = 0.005  # flag if our COCO-style mAP50 and pycocotools differ by more than this


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
    """Raw (pre-merge) tile predictions in full-image coordinates, with the tile offset of each box."""
    h, w = img.shape[:2]
    wins = tile_windows(w, h, a.tile, a.overlap)
    parts = []
    for s in range(0, len(wins), a.batch):
        chunk = wins[s:s + a.batch]
        res = model.predict([img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk], imgsz=a.imgsz, conf=a.conf,
                            iou=a.nms_iou, max_det=a.max_det, device=a.device, verbose=False)
        for r, (x0, y0, _, _) in zip(res, chunk):
            b, s_, c = boxes_of(r, x0, y0)
            parts.append((b, s_, c, np.full(len(c), x0), np.full(len(c), y0)))
    if not parts:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(5))


def finalize(raw, sizes, method, thr, metric, max_det):
    """Merge raw tile predictions per image (method 'none' for full-image mode) and cap at max_det.

    raw: DataFrame with RAW_COLS; sizes: {image name: (w, h)}. Returns a DataFrame with PRED_COLS.
    Shared by eval.py and analysis/merge_sensitivity.py so both apply exactly the same post-processing.
    """
    rows = []
    for name, d in raw.groupby("image", sort=False):
        w, h = sizes[name]
        xyxy, conf, cls = merge(d[["x1", "y1", "x2", "y2"]].to_numpy(float), d.conf.to_numpy(float),
                                d.cls.to_numpy(int), w, h, method, thr, metric)
        order = np.argsort(-conf, kind="mergesort")[:max_det]
        rows += [dict(image=name, cls=int(cls[i]), conf=float(conf[i]), x1=xyxy[i, 0], y1=xyxy[i, 1],
                      x2=xyxy[i, 2], y2=xyxy[i, 3]) for i in order]
    return pd.DataFrame(rows, columns=PRED_COLS)


def run_predictions(a, images):
    """Returns raw predictions (RAW_COLS). Full mode: one 'tile' at (0, 0) per image."""
    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows, t0 = [], time.time()
    for i, p in enumerate(images):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if a.mode == "sliced":
            xyxy, conf, cls, tx, ty = predict_sliced(model, img, a)
        else:
            xyxy, conf, cls = predict_full(model, img, a)
            tx = ty = np.zeros(len(cls), int)
        rows += [dict(image=p.name, tile_x0=int(x0), tile_y0=int(y0), cls=int(c), conf=float(s),
                      x1=b[0], y1=b[1], x2=b[2], y2=b[3]) for b, s, c, x0, y0 in zip(xyxy, conf, cls, tx, ty)]
        print(f"[eval] {i + 1}/{len(images)} {p.name}: {len(conf)} raw preds ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows, columns=RAW_COLS)


def image_sizes(images):
    out = {}
    for p in images:
        with Image.open(p) as im:  # header only
            out[p.name] = im.size
    return out


def collect(preds, images, label_dir, sizes=None):
    sizes = sizes or image_sizes(images)
    per = []
    g = {k: v for k, v in preds.groupby("image")}
    for p in images:
        w, h = sizes[p.name]
        gc, gb = read_yolo_labels(label_dir / f"{p.stem}.txt", w, h)
        d = g.get(p.name, pd.DataFrame(columns=PRED_COLS))
        per.append(dict(image=p.name, w=w, h=h, p_xyxy=d[["x1", "y1", "x2", "y2"]].to_numpy(float),
                        p_cls=d.cls.to_numpy(int), p_conf=d.conf.to_numpy(float), g_xyxy=gb, g_cls=gc))
    return per


def coco_crosscheck(per, names, max_det, out):
    """Score the same predictions with pycocotools (AP at IoU 0.5, area 'all', maxDets = max_det)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    imgs, anns, dets = [], [], []
    for i, d in enumerate(per, 1):
        imgs.append(dict(id=i, file_name=d["image"], width=int(d["w"]), height=int(d["h"])))
        for c, b in zip(d["g_cls"], d["g_xyxy"]):
            wb, hb = b[2] - b[0], b[3] - b[1]
            anns.append(dict(id=len(anns) + 1, image_id=i, category_id=int(c) + 1, iscrowd=0, area=float(wb * hb),
                             bbox=[float(b[0]), float(b[1]), float(wb), float(hb)]))
        for c, s, b in zip(d["p_cls"], d["p_conf"], d["p_xyxy"]):
            dets.append(dict(image_id=i, category_id=int(c) + 1, score=float(s),
                             bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]))
    gt = dict(images=imgs, annotations=anns, categories=[dict(id=c + 1, name=n) for c, n in names.items()])
    (out / "coco_gt.json").write_text(json.dumps(gt))
    (out / "coco_dets.json").write_text(json.dumps(dets))
    if not dets:  # loadRes cannot take an empty list
        return dict(mAP50=0.0, per_class_AP50={n: 0.0 for n in names.values()}, params="no detections")
    cg = COCO(str(out / "coco_gt.json"))
    ev = COCOeval(cg, cg.loadRes(str(out / "coco_dets.json")), "bbox")
    ev.params.iouThrs = np.array([0.5])
    ev.params.maxDets = [1, 10, int(max_det)]
    ev.evaluate(); ev.accumulate()
    prec = ev.eval["precision"][0, :, :, 0, -1]  # [T=0.5, R, K, area=all, maxDets=max_det]
    per_class = {}
    for k, c in enumerate(ev.params.catIds):
        p = prec[:, k]
        per_class[names[c - 1]] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    vals = [v for v in per_class.values() if np.isfinite(v)]
    return dict(mAP50=float(np.mean(vals)) if vals else float("nan"), per_class_AP50=per_class,
                params="iouThrs=[0.5], areaRng=all, maxDets=max_det, 101 recall points")


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
    ap.add_argument("--max-images", type=int,
                    help="val: first N images (testing only); train: N images sampled with --sample-seed")
    ap.add_argument("--holdout-list", help="--split holdout: image list (default: the config's holdout_list)")
    ap.add_argument("--split", choices=["val", "train", "holdout"], default="val",
                    help="train = sanity check on full-resolution train images; requires --out")
    ap.add_argument("--sample-seed", type=int, default=0, help="--split train: seed for sampling --max-images")
    ap.add_argument("--from-preds", help="skip inference; score this predictions.csv")
    ap.add_argument("--ultra-crosscheck", action="store_true", help="also run Ultralytics val (full mode)")
    ap.add_argument("--no-coco-crosscheck", action="store_true", help="skip the pycocotools cross-check")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config)) if a.config else {}
    defaults = dict(mode="full", imgsz=640, tile=1024, overlap=256, merge="nms", merge_thr=0.6,
                    merge_metric="ios", conf=0.001, nms_iou=0.7)
    for k, v in {**defaults, **(cfg.get("eval") or {})}.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    run_dir = Path(a.run_dir) if a.run_dir else Path(a.runs_root) / cfg.get("name", "adhoc")
    a.weights = a.weights or str(run_dir / "train" / "weights" / "last.pt")
    if a.split != "val":
        if not a.out:
            sys.exit(f"--split {a.split} requires an explicit --out (never the run's eval/ folder)")
        if Path(a.out).resolve() == (run_dir / "eval").resolve():
            sys.exit(f"--split {a.split} must not write into the run's eval/ folder")
    holdout_file = a.holdout_list or cfg.get("holdout_list")
    holdout = [l.strip() for l in Path(holdout_file).read_text().splitlines() if l.strip()] if holdout_file else []
    if a.split == "holdout" and not holdout:
        sys.exit("--split holdout needs a holdout list (config holdout_list or --holdout-list)")
    out = Path(a.out) if a.out else run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)

    max_gt = eda_max_boxes_per_image("val")
    if a.max_det is None:
        a.max_det = max(300, 2 * max_gt)
    if a.split != "val":  # keep val's max_det unless a train image has more GT than it allows
        max_gt = max(max_gt, eda_max_boxes_per_image("train"))
        a.max_det = max(a.max_det, 2 * max_gt)
    print(f"[eval] max GT boxes in one val image (EDA table): {max_gt}; max_det = {a.max_det}")
    if a.max_det <= max_gt:
        sys.exit("max_det must exceed the max number of GT boxes per val image")

    root = Path(a.data_root)
    names = load_classes(root)
    sampled = None
    if a.split == "holdout":
        images = [root / "train" / "images" / n for n in holdout]
        missing = [p.name for p in images if not p.exists()]
        if missing:
            sys.exit(f"holdout images not found in {root / 'train' / 'images'}: {missing[:5]}")
        sampled = list(holdout)
    elif a.split == "train":
        pool = [p for p in list_images(root / "train" / "images")
                if p.name != TRAIN_EMPTY_IMAGE and p.name not in set(holdout)]
        if a.max_images is not None and a.max_images < len(pool):
            idx = np.sort(np.random.default_rng(a.sample_seed).choice(len(pool), a.max_images, replace=False))
            pool = [pool[i] for i in idx]
        images, sampled = pool, [p.name for p in pool]
    else:
        images = list_images(root / "val" / "images")[: a.max_images]
    log_env(run_dir, "eval" if a.split == "val" else f"eval_{a.split}")

    sizes = image_sizes(images)
    if a.from_preds:  # a predictions_raw.csv (re-merged here) or an already-merged predictions.csv
        src = pd.read_csv(a.from_preds)
        if "tile_x0" in src.columns:
            raw = src
        else:
            preds = src
    else:
        raw = run_predictions(a, images)
        raw.to_csv(out / "predictions_raw.csv", index=False)  # pre-merge; analysis/merge_sensitivity.py re-scores it
    if not a.from_preds or "tile_x0" in src.columns:
        method = a.merge if a.mode == "sliced" else "none"
        preds = finalize(raw, sizes, method, a.merge_thr, a.merge_metric, a.max_det)
    preds.to_csv(out / "predictions.csv", index=False)
    preds = preds[preds.conf >= a.conf]

    per = collect(preds, images, root / ("val" if a.split == "val" else "train") / "labels", sizes)
    recs = score_images(per, len(names))
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
    if a.split == "train":
        metrics.update(split="train", sample_seed=a.sample_seed, max_images=a.max_images, sampled_images=sampled,
                       excluded_images=[TRAIN_EMPTY_IMAGE] + list(holdout), bootstrap_unit_note="train image")
    elif a.split == "holdout":
        metrics.update(split="holdout", holdout_list=str(holdout_file), sampled_images=sampled,
                       bootstrap_unit_note="held-out train image")
    if a.split != "val" or holdout_file:
        metrics["max_det"] = a.max_det
    args = {k: v for k, v in vars(a).items()}
    (out / "eval_args.json").write_text(json.dumps(args, indent=2, default=str))
    if a.ultra_crosscheck and a.mode == "full" and not a.from_preds and a.split == "val":
        metrics["ultralytics_val"] = ultra_crosscheck(a, root, names, out)
    if not a.no_coco_crosscheck:
        coco = coco_crosscheck(per, names, a.max_det, out)
        diff = abs(coco["mAP50"] - metrics["mAP50"])
        metrics["pycocotools"] = coco
        metrics["scorer_comparison"] = dict(
            ours_coco101=metrics["mAP50"], pycocotools=coco["mAP50"],
            ours_ultralytics_interp=metrics["mAP50_ultralytics_interp"],
            ultralytics_val=metrics.get("ultralytics_val", {}).get("mAP50",
                                                                 "n/a (sliced mode or --ultra-crosscheck not set)"),
            abs_diff_ours_vs_pycocotools=diff, tolerance=SCORER_TOL, flag=bool(diff > SCORER_TOL))
        pd.DataFrame([metrics["scorer_comparison"]]).to_csv(out / "scorer_comparison.csv", index=False)
        if diff > SCORER_TOL:
            print(f"[eval] WARNING: our mAP50 {metrics['mAP50']:.4f} vs pycocotools {coco['mAP50']:.4f} "
                  f"differ by {diff:.4f} > {SCORER_TOL}")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "scorer_comparison" in metrics:
        print("[eval] scorer comparison:", json.dumps(metrics["scorer_comparison"], default=float))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
