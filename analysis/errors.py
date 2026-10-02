"""Phase 3 error analysis of one run's val predictions. Reads the predictions eval.py saved; no re-inference.

Usage:
  python analysis/errors.py --preds runs/b1_tile1024/eval/predictions.csv --name b1_tile1024 [--conf-op 0.25]

--preds may be predictions.csv (merged, as scored by eval.py) or predictions_raw.csv (re-merged here with
eval.finalize and the merge settings in eval_args.json next to it, else --merge/--merge-thr/--merge-metric).

1. TIDE-style error bins (Bolya et al., ECCV 2020), at every confidence >= --min-conf (the mAP50 setting).
   Every prediction that is not a TP under the scorer's matching (detlib/scoring.match_image, IoU 0.5) gets
   exactly one type, checked in this order (fg = 0.5, bg = 0.1, IoU against all GT of the image):
     cls    IoU >= fg with a GT of another class
     dupe   IoU >= fg with a same-class GT that a higher-confidence prediction already matched
     loc    bg <= best same-class IoU < fg
     both   bg <= best other-class IoU < fg (wrong class and poorly localized)
     bkg    IoU < bg with every GT
   missed  a GT matched by no TP and not the target of any cls or loc error.
   Oracle fix of one type (all others left as they are), then re-scored with the same scorer:
     cls  -> relabel to the GT class; loc -> snap the box to the GT box. Either is dropped instead if that GT
             is already matched (by a TP or a higher-confidence fixed prediction): it would only be a duplicate.
     dupe, both, bkg -> prediction removed.   missed -> those GT removed (TIDE's definition).
   dAP50 = mAP50(fixed) - mAP50(base). "all FP" fixes cls+loc+both+dupe+bkg together, "all FN" = missed.
   Fixes interact, so the dAP50 values are not additive.

2. At the operating threshold --conf-op (default 0.25, same as pred_review.py): FN rate (1 - recall) and FP rate
   (FP / predictions) sliced by class, box size (sqrt of area, px), GT objects per image and box brightness
   (mean gray value inside the box), plus size x class tables. GT boxes are binned for FN, predicted boxes for FP.
   Objects-per-image bins are quartiles over val images; brightness bins are quartiles over GT boxes.
3. Confusion matrix at --conf-op with a background row/column (pred_review.confusion).
4. Example crops at --conf-op: one grid per error type, one row per class, --n-crops seeded random examples.
   Red = prediction, orange = the GT it was assigned to (cls/loc/both/dupe), cyan = missed GT.

Writes figures/<name>/errors/: tide_dAP.csv/.png, tide_counts.csv (type x class, both thresholds),
slices.csv, slices.png, size_x_class_fn.csv, size_x_class_fp.csv, confusion_matrix.csv/.png,
crops_<type>.png, op_predictions.csv and op_gt.csv (one row per box with its type and bins), errors_args.json.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from detlib.data import eda_max_boxes_per_image, list_images, load_classes  # noqa: E402
from detlib.scoring import ap_from_records, box_iou, match_image, score_images  # noqa: E402
from eval import collect, finalize, image_sizes  # noqa: E402
from pred_review import COL, confusion, crop  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
FG, BG = 0.5, 0.1
FP_TYPES = ("cls", "loc", "both", "dupe", "bkg")
TYPES = FP_TYPES + ("missed",)
FIXES = [(t, {t}) for t in TYPES] + [("all FP", set(FP_TYPES)), ("all FN", {"missed"})]
SIZE_EDGES = [0, 8, 16, 24, 32, 48, 96, np.inf]


def tide_image(pb, pc, ps, gb, gc, fg=FG, bg=BG):
    """Error type of every prediction ('TP' or one of FP_TYPES), the GT index it refers to (-1 if none),
    and per GT: matched by a TP, missed (TIDE sense)."""
    tp, gidx, _ = match_image(pb, pc, ps, gb, gc, fg)
    iou = box_iou(pb, gb)
    kind = np.array(["TP"] * len(pb), dtype=object)
    ref = np.where(tp, gidx, -1)
    for i in np.where(~tp)[0]:
        if len(gb) == 0:
            kind[i] = "bkg"
            continue
        same = np.where(gc == pc[i], iou[i], 0.0)
        other = np.where(gc != pc[i], iou[i], 0.0)
        if other.max() >= fg:
            kind[i], ref[i] = "cls", int(other.argmax())
        elif same.max() >= fg:
            kind[i], ref[i] = "dupe", int(same.argmax())
        elif same.max() >= bg:
            kind[i], ref[i] = "loc", int(same.argmax())
        elif other.max() >= bg:
            kind[i], ref[i] = "both", int(other.argmax())
        else:
            kind[i] = "bkg"
    matched = np.zeros(len(gb), bool)
    matched[gidx[tp]] = True
    covered = np.zeros(len(gb), bool)
    covered[ref[np.isin(kind, ["cls", "loc"])].astype(int)] = True
    return kind, ref, matched, ~matched & ~covered


def annotate(per):
    """Adds kind/ref/matched/missed to every per-image dict (as built by eval.collect)."""
    for d in per:
        d["kind"], d["ref"], d["matched"], d["missed"] = tide_image(d["p_xyxy"], d["p_cls"], d["p_conf"],
                                                                    d["g_xyxy"], d["g_cls"])
    return per


def apply_fix(d, types):
    """Oracle-fixed copy of one annotated image dict (see module docstring)."""
    claimed = set(np.where(d["matched"])[0].tolist())
    keep_b, keep_c, keep_s = [], [], []
    for i in np.argsort(-d["p_conf"], kind="mergesort"):
        k, b, c = d["kind"][i], d["p_xyxy"][i], d["p_cls"][i]
        if k in types:
            if k not in ("cls", "loc"):
                continue
            j = int(d["ref"][i])
            if j in claimed:
                continue
            claimed.add(j)
            if k == "cls":
                c = d["g_cls"][j]
            else:
                b = d["g_xyxy"][j]
        keep_b.append(b); keep_c.append(c); keep_s.append(d["p_conf"][i])
    g = ~d["missed"] if "missed" in types else np.ones(len(d["g_cls"]), bool)
    return dict(p_xyxy=np.array(keep_b, float).reshape(-1, 4), p_cls=np.array(keep_c, int),
                p_conf=np.array(keep_s, float), g_xyxy=d["g_xyxy"][g], g_cls=d["g_cls"][g])


def tide_table(per, names):
    nc = len(names)
    base_aps, base, _ = ap_from_records(score_images(per, nc), nc, "coco")
    n_err = {t: int(sum((d["kind"] == t).sum() for d in per)) for t in FP_TYPES}
    n_err["missed"] = int(sum(d["missed"].sum() for d in per))
    n_err["all FP"] = sum(n_err[t] for t in FP_TYPES)
    n_err["all FN"] = n_err["missed"]
    rows = [dict(fix="none (base)", n_errors=0, mAP50=base, dAP50=0.0,
                 **{f"AP50 {names[c]}": base_aps[c] for c in names})]
    for label, types in FIXES:
        aps, m, _ = ap_from_records(score_images([apply_fix(d, types) for d in per], nc), nc, "coco")
        rows.append(dict(fix=label, n_errors=n_err[label], mAP50=m, dAP50=m - base,
                         **{f"AP50 {names[c]}": aps[c] for c in names}))
    return pd.DataFrame(rows)


def counts_by_class(per, names, level):
    rows = []
    for t in TYPES:
        r = dict(level=level, type=t)
        for c, n in names.items():
            if t == "missed":
                r[n] = int(sum((d["missed"] & (d["g_cls"] == c)).sum() for d in per))
            else:
                r[n] = int(sum(((d["kind"] == t) & (d["p_cls"] == c)).sum() for d in per))
        r["total"] = sum(r[n] for n in names.values())
        rows.append(r)
    return rows


def box_tables(per, names):
    """One row per prediction and per GT at the operating threshold, with type, size and image object count."""
    P, G = [], []
    for d in per:
        n_obj = len(d["g_cls"])
        for i, b in enumerate(d["p_xyxy"]):
            P.append(dict(image=d["image"], idx=i, cls=names[int(d["p_cls"][i])], conf=d["p_conf"][i],
                          type=d["kind"][i], ref_gt=int(d["ref"][i]), size=np.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0)),
                          n_obj=n_obj, x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
        for j, b in enumerate(d["g_xyxy"]):
            G.append(dict(image=d["image"], idx=j, cls=names[int(d["g_cls"][j])], matched=bool(d["matched"][j]),
                          missed=bool(d["missed"][j]), size=np.sqrt((b[2] - b[0]) * (b[3] - b[1])), n_obj=n_obj,
                          x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
    cols_p = ["image", "idx", "cls", "conf", "type", "ref_gt", "size", "n_obj", "x1", "y1", "x2", "y2"]
    cols_g = ["image", "idx", "cls", "matched", "missed", "size", "n_obj", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(P, columns=cols_p), pd.DataFrame(G, columns=cols_g)


def mean_gray(gray, b):
    h, w = gray.shape
    x1, y1 = int(np.clip(np.floor(b[0]), 0, w - 1)), int(np.clip(np.floor(b[1]), 0, h - 1))
    x2, y2 = int(np.clip(np.ceil(b[2]), x1 + 1, w)), int(np.clip(np.ceil(b[3]), y1 + 1, h))
    return float(gray[y1:y2, x1:x2].mean())


def ordered(b):
    """(x1, y1, x2, y2) with x1 <= x2 and y1 <= y2, for drawing."""
    return [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]


def bin_edges(values, q=4):
    e = np.unique(np.quantile(values, np.linspace(0, 1, q + 1))) if len(values) else np.array([0.0, 1.0])
    e = e.astype(float) if len(e) > 1 else np.array([0.0, 1.0])
    e[0], e[-1] = -np.inf, np.inf
    return e


def label_bins(values, edges, fmt="{:.0f}"):
    labels = [f"[{'-inf' if np.isinf(lo) else fmt.format(lo)}, {'inf' if np.isinf(hi) else fmt.format(hi)})"
              for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.Categorical(pd.cut(values, edges, right=False, labels=labels), categories=labels, ordered=True)


def slice_rates(P, G, names):
    """FN rate per GT bin and FP rate per prediction bin, for each slicing variable."""
    rows = []
    for var in ("cls", "size_bin", "objects_bin", "brightness_bin"):
        cats = list(names.values()) if var == "cls" else list(G[var].cat.categories)
        for b in cats:
            g, p = G[G[var] == b], P[P[var] == b]
            n_fp = int((p.type != "TP").sum())
            rows.append(dict(slice=var, bin=b, n_gt=len(g), n_fn=int((~g.matched).sum()),
                             fn_rate=float((~g.matched).mean()) if len(g) else np.nan,
                             n_pred=len(p), n_fp=n_fp, fp_rate=n_fp / len(p) if len(p) else np.nan))
    return pd.DataFrame(rows)


def cross_table(df, flag, names):
    """size_bin x class: rate of `flag` with counts, long format."""
    rows = []
    for s in df.size_bin.cat.categories:
        for n in names.values():
            x = df[(df.size_bin == s) & (df.cls == n)]
            k = int(x[flag].sum())
            rows.append(dict(size_bin=s, cls=n, n=len(x), n_err=k, rate=k / len(x) if len(x) else np.nan))
    return pd.DataFrame(rows)


def plot_tide(t, path):
    x = t[t.fix != "none (base)"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x.fix, x.dAP50, color=["#c44"] * 5 + ["#48c"] + ["#888"] * 2)
    for i, (v, n) in enumerate(zip(x.dAP50, x.n_errors)):
        ax.text(i, v, f"{v:+.3f}\nn={n}", ha="center", va="bottom", fontsize=7)
    ax.set(ylabel="mAP50 gain if fixed", title=f"TIDE-style oracle fixes (base mAP50 {t.mAP50.iloc[0]:.4f})")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_slices(s, path):
    vars_ = list(dict.fromkeys(s.slice))
    fig, axes = plt.subplots(1, len(vars_), figsize=(4 * len(vars_), 3.8), squeeze=False)
    for ax, v in zip(axes[0], vars_):
        x = s[s.slice == v]
        idx = np.arange(len(x))
        ax.bar(idx - 0.2, x.fn_rate, 0.4, label="FN rate")
        ax.bar(idx + 0.2, x.fp_rate, 0.4, label="FP rate")
        ax.set_xticks(idx, x.bin, rotation=45, ha="right", fontsize=7)
        ax.set(ylim=(0, 1), title=v)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(m, labels, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set(xlabel="predicted", ylabel="ground truth")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def crop_grid(cells, names, n, title, path):
    fig, axes = plt.subplots(len(names), n, figsize=(1.9 * n, 2.1 * len(names)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for r, cname in enumerate(names.values()):
        axes[r][0].set_title(cname, fontsize=8, loc="left")
        for k, (im, cap) in enumerate(cells.get(cname, [])[:n]):
            axes[r][k].imshow(im)
            axes[r][k].set_title((cname + "\n" if k == 0 else "\n") + cap, fontsize=6, loc="left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def load_preds(a):
    src = pd.read_csv(a.preds)
    if "tile_x0" not in src.columns:
        return src, "predictions.csv as given"
    ea = Path(a.preds).parent / "eval_args.json"
    cfg = json.loads(ea.read_text()) if ea.exists() else {}
    method = a.merge or (cfg.get("merge", "nms") if cfg.get("mode", "sliced") == "sliced" else "none")
    thr = a.merge_thr if a.merge_thr is not None else cfg.get("merge_thr", 0.6)
    metric = a.merge_metric or cfg.get("merge_metric", "ios")
    max_det = cfg.get("max_det") or max(300, 2 * eda_max_boxes_per_image("val"))
    return src, (method, thr, metric, max_det)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="predictions.csv or predictions_raw.csv written by eval.py")
    ap.add_argument("--name", required=True, help="figures/<name>/errors/")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-root", default="figures")
    ap.add_argument("--min-conf", type=float, default=0.001, help="confidence floor for the TIDE/mAP part (eval.py)")
    ap.add_argument("--conf-op", type=float, default=0.25, help="operating threshold for rates, confusion, crops")
    ap.add_argument("--merge", choices=["nms", "wbf", "none"], help="raw input only; default from eval_args.json")
    ap.add_argument("--merge-thr", type=float)
    ap.add_argument("--merge-metric", choices=["iou", "ios"])
    ap.add_argument("--n-crops", type=int, default=6, help="examples per class per error type")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, out = Path(a.data_root), Path(a.out_root) / a.name / "errors"
    out.mkdir(parents=True, exist_ok=True)
    names = load_classes(root)
    nc = len(names)
    images = list_images(root / "val" / "images")
    sizes = image_sizes(images)
    preds, how = load_preds(a)
    if how != "predictions.csv as given":
        method, thr, metric, max_det = how
        preds = finalize(preds, sizes, method, thr, metric, max_det)
        how = dict(merged_from_raw=True, merge=method, thr=thr, metric=metric, max_det=max_det)
    preds = preds[preds.conf >= a.min_conf]

    # 1. TIDE at all confidences >= min_conf
    per = annotate(collect(preds, images, root / "val" / "labels", sizes))
    tide = tide_table(per, names)
    tide.to_csv(out / "tide_dAP.csv", index=False)
    plot_tide(tide, out / "tide_dAP.png")
    base = float(tide.mAP50.iloc[0])
    mj = Path(a.preds).parent / "metrics.json"
    check = None
    if mj.exists():
        ev = json.loads(mj.read_text()).get("mAP50")
        check = dict(eval_mAP50=ev, errors_base_mAP50=base, abs_diff=abs(ev - base) if ev is not None else None)
        if ev is not None and abs(ev - base) > 1e-9:
            print(f"[errors] WARNING: base mAP50 {base:.6f} differs from metrics.json {ev:.6f}")

    # 2-4. operating threshold
    per_op = annotate(collect(preds[preds.conf >= a.conf_op], images, root / "val" / "labels", sizes))
    counts = pd.DataFrame(counts_by_class(per, names, f"conf>={a.min_conf}") +
                          counts_by_class(per_op, names, f"conf>={a.conf_op}"))
    counts.to_csv(out / "tide_counts.csv", index=False)
    P, G = box_tables(per_op, names)
    cm = sum(confusion(pd.DataFrame(dict(x1=d["p_xyxy"][:, 0], y1=d["p_xyxy"][:, 1], x2=d["p_xyxy"][:, 2],
                                         y2=d["p_xyxy"][:, 3], cls=d["p_cls"])), d["g_cls"], d["g_xyxy"], nc)
             for d in per_op)
    labels = list(names.values()) + ["background"]
    pd.DataFrame(cm, index=[f"gt {n}" for n in labels], columns=[f"pred {n}" for n in labels]).to_csv(
        out / "confusion_matrix.csv")
    plot_confusion(cm, labels, out / "confusion_matrix.png")

    # crops: sample before reading pixels so each image is opened once
    rng = random.Random(a.seed)
    want = {}  # (image, 'p'|'g', idx) -> (type, class name, caption)
    events = [(t, r) for t in FP_TYPES for r in P[P.type == t].itertuples()] + \
             [("missed", r) for r in G[G.missed].itertuples()]
    by = {}
    for t, r in events:
        by.setdefault((t, r.cls), []).append(r)
    for (t, cname), rs in sorted(by.items()):
        for r in rng.sample(rs, min(a.n_crops, len(rs))):
            cap = f"{r.image} conf {r.conf:.2f}" if t != "missed" else f"{r.image}"
            want[(r.image, "g" if t == "missed" else "p", r.idx)] = (t, cname, cap)

    # brightness (and crops) need pixels: one pass over the images
    P["brightness"], G["brightness"] = np.nan, np.nan
    cells = {t: {} for t in TYPES}
    dmap = {d["image"]: d for d in per_op}
    for p in images:
        with Image.open(p) as im:
            im.load()
            gray = np.asarray(im.convert("L"), dtype=np.float32)
            d = dmap[p.name]
            pi, gi = P.index[P.image == p.name], G.index[G.image == p.name]
            P.loc[pi, "brightness"] = [mean_gray(gray, b) for b in d["p_xyxy"]]
            G.loc[gi, "brightness"] = [mean_gray(gray, b) for b in d["g_xyxy"]]
            for (name, side, idx), (t, cname, cap) in want.items():
                if name != p.name:
                    continue
                if side == "g":
                    box, marks = d["g_xyxy"][idx], [(d["g_xyxy"][idx], COL["FN"])]
                else:
                    box, marks = d["p_xyxy"][idx], [(d["p_xyxy"][idx], COL["FP"])]
                    j = int(d["ref"][idx])
                    if j >= 0:
                        marks.insert(0, (d["g_xyxy"][j], COL["REF"]))
                        cap += f"\nGT {names[int(d['g_cls'][j])]}"
                marks = [(ordered(b), col) for b, col in marks]
                cells[t].setdefault(cname, []).append((crop(im, ordered(box), marks, 160), cap))
    for t in TYPES:
        crop_grid(cells[t], names, a.n_crops, f"{t} errors at conf >= {a.conf_op}", out / f"crops_{t}.png")

    G["size_bin"], P["size_bin"] = label_bins(G["size"], SIZE_EDGES), label_bins(P["size"], SIZE_EDGES)
    img_counts = np.array([len(d["g_cls"]) for d in per_op], float)
    e_obj = bin_edges(img_counts)
    G["objects_bin"], P["objects_bin"] = label_bins(G.n_obj, e_obj), label_bins(P.n_obj, e_obj)
    e_br = bin_edges(G.brightness.to_numpy())
    G["brightness_bin"], P["brightness_bin"] = label_bins(G.brightness, e_br), label_bins(P.brightness, e_br)
    P["fp"], G["fn"] = P.type != "TP", ~G.matched
    sl = slice_rates(P, G, names)
    sl.to_csv(out / "slices.csv", index=False)
    plot_slices(sl, out / "slices.png")
    cross_table(G, "fn", names).to_csv(out / "size_x_class_fn.csv", index=False)
    cross_table(P, "fp", names).to_csv(out / "size_x_class_fp.csv", index=False)
    P.to_csv(out / "op_predictions.csv", index=False)
    G.to_csv(out / "op_gt.csv", index=False)

    args = dict(vars(a), input=how, fg=FG, bg=BG, size_edges=[str(e) for e in SIZE_EDGES],
                objects_edges=e_obj.tolist(), brightness_edges=e_br.tolist(), base_check=check)
    (out / "errors_args.json").write_text(json.dumps(args, indent=2, default=str))
    print(tide[["fix", "n_errors", "mAP50", "dAP50"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[errors] wrote {out}")


if __name__ == "__main__":
    main()
