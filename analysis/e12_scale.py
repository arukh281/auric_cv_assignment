"""E12: scale-robust inference with B1h's weights (no training). Rules use only the image itself.

  python analysis/e12_scale.py test --config configs/b1h.yaml --weights <last.pt> --data-root "$DATA_ROOT" --out <dir>
  python analysis/e12_scale.py val  --config configs/b1h.yaml --weights <last.pt> --data-root "$DATA_ROOT" --rule <name> --out <dir>

Inference at scale s: the image is resized by s (bilinear), predicted with eval.py's sliced inference (same tile,
overlap, conf, NMS IoU, max_det), and boxes are divided by s. Merge: eval.finalize (class-wise NMS on IoS 0.6,
max_det 902). Scoring: eval.collect + detlib.scoring (COCO 101-point mAP50).
Rules:
  plain     s = 1
  multi     union of raw predictions at s = 0.5, 1, 2, then the usual merge
  auto      predict at s = 1; m = median sqrt(area) of predictions with conf >= 0.25 (none: keep s = 1); choose
            s in {0.5, 1, 2} minimising |log(m * s / 22)| (22 px = median training truck size, figures/eda/summary.json);
            use the predictions at that s
test: holdout40 at three versions, original and bilinear 0.5x / 2x copies (labels are normalised, so unchanged);
      every rule on every version. Writes test_scores.csv.
val:  the chosen rule on val, once; per-image chosen scale; mAP50 overall, on the 18 unrescaled and on the 4
      rescaled images (2308, 2384, 2391, 2460; the list is used only for reporting, never by the rule).
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import list_images  # noqa: E402
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from eval import collect, finalize, image_sizes, run_predictions  # noqa: E402

SCALES = (0.5, 1.0, 2.0)
RESCALED_VAL = {"2308.png", "2384.png", "2391.png", "2460.png"}


def resized_copy(src_imgs, s, tmp):
    tmp.mkdir(parents=True, exist_ok=True); out = []
    for p in src_imgs:
        q = tmp / p.name
        if not q.exists():
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if s != 1.0:
                im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(q), im)
        out.append(q)
    return out


def raw_at(a, imgs, s, tmp):
    """Raw predictions for images `imgs` predicted at scale s, in the coordinates of `imgs`."""
    src = resized_copy(imgs, s, tmp) if s != 1.0 else imgs
    r = run_predictions(a, src)
    for c in ("x1", "y1", "x2", "y2"):
        r[c] = r[c] / s
    r["tile_x0"] = (r.tile_x0 / s).round().astype(int); r["tile_y0"] = (r.tile_y0 / s).round().astype(int)
    return r


def choose_auto(raw1, sizes, ev):
    p = finalize(raw1, sizes, ev["merge"], ev["merge_thr"], ev["merge_metric"], ev["max_det"])
    out = {}
    for n in sizes:
        q = p[(p.image == n) & (p.conf >= 0.25)]
        if not len(q):
            out[n] = 1.0; continue
        m = float(np.median(np.sqrt((q.x2 - q.x1) * (q.y2 - q.y1))))
        out[n] = min(SCALES, key=lambda s: abs(np.log(m * s / 22.0)))
    return out


def apply_rule(rule, raws, sizes, ev):
    if rule == "plain":
        r = raws[1.0]; chosen = {n: 1.0 for n in sizes}
    elif rule == "multi":
        r = pd.concat([raws[s] for s in SCALES]); chosen = {n: "0.5+1+2" for n in sizes}
    else:
        chosen = choose_auto(raws[1.0], sizes, ev)
        r = pd.concat([raws[chosen[n]][raws[chosen[n]].image == n] for n in sizes])
    return finalize(r, sizes, ev["merge"], ev["merge_thr"], ev["merge_metric"], ev["max_det"]), chosen


def score(preds, imgs, label_dir, sizes, subset=None):
    keep = [p for p in imgs if subset is None or p.name in subset]
    per = collect(preds[preds.image.isin([p.name for p in keep])], keep, label_dir, {p.name: sizes[p.name] for p in keep})
    return ap_from_records(score_images(per, 5), 5, "coco")[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["test", "val"]); ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rule", choices=["plain", "multi", "auto"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); ev = cfg["eval"]; root = Path(a.data_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A = SimpleNamespace(weights=a.weights, device=a.device, batch=16, mode="sliced", imgsz=ev["imgsz"], tile=ev["tile"],
                        overlap=ev["overlap"], conf=ev["conf"], nms_iou=ev["nms_iou"], max_det=ev["max_det"])
    tmp = Path("/kaggle/tmp/e12") if Path("/kaggle").exists() else out / "tmp"
    if a.mode == "test":
        hold = [l.strip() for l in (REPO / cfg["holdout_list"]).read_text().splitlines() if l.strip()]
        base = [root / "train" / "images" / n for n in hold]; rows = []
        for vs in SCALES:
            imgs = resized_copy(base, vs, tmp / f"version_{vs}") if vs != 1.0 else base
            sizes = image_sizes(imgs)
            raws = {s: raw_at(A, imgs, s, tmp / f"version_{vs}_s{s}") for s in SCALES}
            for rule in ("plain", "multi", "auto"):
                p, chosen = apply_rule(rule, raws, sizes, ev)
                p.to_csv(out / f"preds_v{vs}_{rule}.csv", index=False)
                rows.append(dict(holdout_version=f"{vs}x", rule=rule, mAP50=score(p, imgs, root / "train" / "labels", sizes),
                                 chosen_scales=pd.Series(chosen).astype(str).value_counts().to_dict()))
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out / "test_scores.csv", index=False)
    else:
        imgs = list_images(root / "val" / "images"); sizes = image_sizes(imgs)
        need = SCALES if a.rule != "plain" else (1.0,)
        raws = {s: raw_at(A, imgs, s, tmp / f"val_s{s}") for s in need}
        p, chosen = apply_rule(a.rule, raws, sizes, ev); p.to_csv(out / f"val_preds_{a.rule}.csv", index=False)
        L = root / "val" / "labels"
        res = dict(rule=a.rule, val_mAP50=score(p, imgs, L, sizes),
                   val_mAP50_18_unrescaled=score(p, imgs, L, sizes, {i.name for i in imgs} - RESCALED_VAL),
                   val_mAP50_4_rescaled=score(p, imgs, L, sizes, RESCALED_VAL))
        pd.DataFrame([res]).to_csv(out / "val_score.csv", index=False)
        pd.Series(chosen, name="scale").to_csv(out / "val_chosen_scales.csv"); print(res)


if __name__ == "__main__":
    main()
