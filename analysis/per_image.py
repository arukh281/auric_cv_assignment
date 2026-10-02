"""Per-image breakdown of saved predictions (no inference, CPU).

Usage:
  python analysis/per_image.py                       # B1 val (with visual-inspection flags) + B1 train40
  python analysis/per_image.py --eval results/b1_tile1024/eval --name val --flags

GT comes from the coco_gt.json that eval.py wrote next to the predictions: the exact GT it scored against
(class = category_id - 1; bbox xywh -> xyxy). This also covers --split train evals whose images are not on this
machine. tests/test_per_image.py checks it equals the val YOLO labels. Predictions below eval_args.json's conf are
dropped, as in eval.py. Scorer: detlib/scoring.py (IoU 0.5, COCO 101-point AP).

Per image:
  n_gt, n_pred
  AP50_agnostic       AP50 with every class set to one class
  mAP50_per_class     mean AP50 over the classes that have GT in this image
  recall_conf0.25 / recall_conf0.001             class-aware: TP / n_gt at conf >= threshold
  recall_agnostic_conf0.25 / _conf0.001          class-agnostic
  n_bkg_fp_conf0.25   predictions at conf >= 0.25 typed "bkg" by analysis/errors.tide_image (IoU <= 0.1 with every GT)
  flag                from analysis/notes/visual_inspection.md (--flags), ";"-joined, "none" if none

Writes figures/<run>/per_image/: per_image_<name>.csv (sorted by AP50_agnostic), and for flagged sets
flag_summary_<name>.csv (means of the per-image columns, flagged vs unflagged images) and
subset_map_<name>.csv (mAP50 over all / unflagged / flagged images, with 95% percentile bootstrap CIs over images,
1000 resamples, seed 0); plus per_image.png (one bar per image).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from detlib.scoring import ap_from_records, bootstrap, score_images  # noqa: E402
from errors import tide_image  # noqa: E402

NC = 5
OP_CONFS = (0.25, 0.001)
# analysis/notes/visual_inspection.md (2026-10-02); an image can carry several flags
FLAGS = dict(haze=["2460", "2470", "2472"], blur=["2292", "2308", "2543"], scale=["2391", "2308"],
             water=["1399", "1447", "1456"], label_issue=["2470", "2472", "31", "1211"])


def flags_of(image):
    stem = Path(image).stem
    f = [k for k, v in FLAGS.items() if stem in v]
    return ";".join(f) if f else "none"


def load_eval(eval_dir):
    """Per-image dicts (eval.collect format) from predictions.csv + coco_gt.json of one eval folder."""
    eval_dir = Path(eval_dir)
    gt = json.loads((eval_dir / "coco_gt.json").read_text())
    args = json.loads((eval_dir / "eval_args.json").read_text())
    preds = pd.read_csv(eval_dir / "predictions.csv")
    preds = preds[preds.conf >= args.get("conf", 0.001)]
    anns = pd.DataFrame(gt["annotations"])
    by_img = {k: d for k, d in anns.groupby("image_id")} if len(anns) else {}
    pg = {k: d for k, d in preds.groupby("image")}
    per = []
    for im in gt["images"]:
        a = by_img.get(im["id"])
        if a is not None:
            b = np.array(a.bbox.tolist(), float)
            g_xyxy, g_cls = np.c_[b[:, :2], b[:, :2] + b[:, 2:]], a.category_id.to_numpy(int) - 1
        else:
            g_xyxy, g_cls = np.zeros((0, 4)), np.zeros(0, int)
        d = pg.get(im["file_name"], pd.DataFrame(columns=["cls", "conf", "x1", "y1", "x2", "y2"]))
        per.append(dict(image=im["file_name"], p_xyxy=d[["x1", "y1", "x2", "y2"]].to_numpy(float),
                        p_cls=d.cls.to_numpy(int), p_conf=d.conf.to_numpy(float), g_xyxy=g_xyxy, g_cls=g_cls))
    return per


def at_conf(d, c):
    k = d["p_conf"] >= c
    return dict(d, p_xyxy=d["p_xyxy"][k], p_cls=d["p_cls"][k], p_conf=d["p_conf"][k])


def agnostic(d):
    return dict(d, p_cls=np.zeros(len(d["p_cls"]), int), g_cls=np.zeros(len(d["g_cls"]), int))


def recall(d, c, agn=False):
    x = at_conf(agnostic(d) if agn else d, c)
    n = len(x["g_cls"])
    tp = score_images([x], 1 if agn else NC)[0]["tp"].sum()
    return float(tp / n) if n else float("nan")


def image_row(d):
    _, m, _ = ap_from_records(score_images([d], NC), NC, "coco")
    _, m_ag, _ = ap_from_records(score_images([agnostic(d)], 1), 1, "coco")
    op = at_conf(d, 0.25)
    kind = tide_image(op["p_xyxy"], op["p_cls"], op["p_conf"], op["g_xyxy"], op["g_cls"])[0]
    r = dict(image=d["image"], n_gt=len(d["g_cls"]), n_pred=len(d["p_cls"]), AP50_agnostic=m_ag, mAP50_per_class=m)
    for c in OP_CONFS:
        r[f"recall_conf{c:g}"] = recall(d, c)
    for c in OP_CONFS:
        r[f"recall_agnostic_conf{c:g}"] = recall(d, c, True)
    r["n_bkg_fp_conf0.25"] = int((kind == "bkg").sum())
    return r


def per_image_table(per, with_flags):
    t = pd.DataFrame([image_row(d) for d in per])
    if with_flags:
        t["flag"] = t.image.map(flags_of)
    return t.sort_values(["AP50_agnostic", "image"], ascending=[False, True]).reset_index(drop=True)


def subset_map(per, keep, n_boot=1000, seed=0):
    recs = score_images([d for d, k in zip(per, keep) if k], NC)
    if not recs:
        return dict(n_images=0, mAP50=float("nan"), ci95_lo=float("nan"), ci95_hi=float("nan"))
    aps, m, n_gt = ap_from_records(recs, NC, "coco")
    b = bootstrap(recs, NC, n_boot, seed, "coco")[:, -1]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(n_images=len(recs), n_gt=int(n_gt.sum()), mAP50=m, ci95_lo=float(lo), ci95_hi=float(hi),
                **{f"AP50_cls{c}": aps[c] for c in range(NC)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="b1_tile1024")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-root", default="figures")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out_root) / a.run / "per_image"
    out.mkdir(parents=True, exist_ok=True)
    sets = [("val", Path(a.results_root) / a.run / "eval", True),
            ("train40", Path(a.results_root) / a.run / "eval_train40", False)]
    tables = {}
    for name, ev, with_flags in sets:
        if not (ev / "predictions.csv").exists():
            print(f"[per_image] skip {name}: {ev} has no predictions.csv")
            continue
        per = load_eval(ev)
        t = per_image_table(per, with_flags)
        t.to_csv(out / f"per_image_{name}.csv", index=False)
        tables[name] = t
        mj = json.loads((ev / "metrics.json").read_text())
        rows = [dict(subset="all", **subset_map(per, [True] * len(per), a.bootstrap), eval_metrics_mAP50=mj["mAP50"])]
        if with_flags:
            fl = np.array([flags_of(d["image"]) != "none" for d in per])
            rows += [dict(subset="unflagged", **subset_map(per, ~fl, a.bootstrap)),
                     dict(subset="flagged", **subset_map(per, fl, a.bootstrap))]
            cols = [c for c in t.columns if c not in ("image", "flag")]
            g = t.assign(group=np.where(t.flag == "none", "unflagged", "flagged")).groupby("group")[cols]
            s = g.mean().assign(n_images=g.size()).reset_index()
            s.to_csv(out / f"flag_summary_{name}.csv", index=False)
            print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        sm = pd.DataFrame(rows)
        sm.to_csv(out / f"subset_map_{name}.csv", index=False)
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(sm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(len(tables), 1, figsize=(14, 4.2 * len(tables)), squeeze=False)
    for ax, (name, t) in zip(axes[:, 0], tables.items()):
        col = ["#999" if t.get("flag", pd.Series(["none"] * len(t)))[i] == "none" else "#c44" for i in range(len(t))]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.AP50_agnostic, 0.4, color=col, label="AP50 class-agnostic (red = flagged)")
        ax.bar(x + 0.2, t.mAP50_per_class, 0.4, color="#48c", label="mAP50 per-class mean")
        lab = [f"{i} ({f})" if f != "none" else i for i, f in zip(t.image, t.get("flag", ["none"] * len(t)))]
        ax.set_xticks(x, lab, rotation=70, ha="right", fontsize=7)
        ax.set(ylim=(0, 1), ylabel="AP50", title=f"B1 per image: {name} ({len(t)} images), sorted by class-agnostic AP50")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "per_image.png", dpi=110); plt.close(fig)
    print(f"[per_image] wrote {out}")


if __name__ == "__main__":
    main()
