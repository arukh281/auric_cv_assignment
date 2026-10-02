"""§5.2 Which training examples resist learning: per-GT-box outcome across B1h's kept checkpoints (GPU for inference).

Usage (Kaggle; checkpoints from the B1h kernel output):
  python analysis/training_dynamics.py --weights-dir <.../weights> --data-root /kaggle/tmp/data --out <dir>
      [--max-images 5]   # smoke test

Images: the 403 B1h training images (all train images minus splits/holdout40_seed0.txt), each tagged with its half
from splits/s52_inspect_seed0.txt / s52_test_seed0.txt (seed 0, committed before any computation).

1. Inference per checkpoint (epoch010/020/030/040/050.pt): exactly eval.py's sliced inference and merge
   (eval.run_predictions + eval.finalize with the B1h config's eval settings: tile 1024, overlap 256, conf 0.001,
   NMS IoU 0.7, class-wise NMS on IoS 0.6, max_det 902), on the full-resolution train images.
   -> <out>/preds/predictions_ep{NNN}.csv.gz
2. Per GT box and checkpoint (IoU against the merged predictions):
   detected      some prediction of any class with IoU >= 0.5 and conf >= 0.25
   correct       the highest-confidence prediction among those (IoU >= 0.5, conf >= 0.25) has the GT's class
   best_iou      highest IoU with any prediction at conf >= 0.25 (0 if none)
   true_score    highest confidence of a GT-class prediction with IoU >= 0.5 (any conf >= 0.001; 0 if none)
   wrong_cls / wrong_score   class and confidence of the highest-confidence other-class prediction with IoU >= 0.5
3. Category over the 5 checkpoints (first rule that applies):
   never-detected  not detected at any checkpoint
   never-learned   detected at some checkpoint but never correct
   forgotten       correct at some checkpoint and not correct at a later one
   learned-early   correct at every checkpoint from epoch 10 on
   learned-late    first correct after epoch 10 and correct at every later checkpoint
4. Outputs:
   <out>/per_box_inspect.csv, <out>/per_box_test.csv  (one row per GT box, all columns; both halves)
   <out>/inspect/ (inspect half ONLY; the test half is never summarised or plotted):
     counts_by_class.csv, counts_by_size.csv, counts_by_objects.csv, counts_by_image.csv, categories.png,
     crops_<category>__<class>.png  (up to 24 seeded crops; green = GT, red = highest-IoU prediction at the last
     checkpoint, captioned with its class and confidence)
   <out>/training_dynamics_args.json (checkpoints with sha256, settings, counts of images/boxes per half)
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from detlib.data import list_images, load_classes, read_yolo_labels  # noqa: E402
from detlib.scoring import box_iou  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
EPOCHS = (10, 20, 30, 40, 50)
CATEGORIES = ("learned-early", "learned-late", "forgotten", "never-learned", "never-detected")
CONF_OP, IOU = 0.25, 0.5
SIZE_EDGES = [0, 8, 16, 24, 32, 48, 96, np.inf]


def box_metrics(gb, gc, pb, pc, ps):
    """Per GT box: detected, correct, best_iou, true_score, wrong_cls, wrong_score (see module docstring)."""
    n = len(gb)
    out = dict(detected=np.zeros(n, bool), correct=np.zeros(n, bool), best_iou=np.zeros(n),
               true_score=np.zeros(n), wrong_cls=np.full(n, -1), wrong_score=np.zeros(n))
    if n == 0 or len(pb) == 0:
        return out
    iou = box_iou(gb, pb)                                 # (G, P)
    op = ps >= CONF_OP
    out["best_iou"] = np.where(op[None, :], iou, 0).max(1)
    hit = iou >= IOU
    for g in range(n):
        h = hit[g]
        hop = h & op
        if hop.any():
            out["detected"][g] = True
            j = np.flatnonzero(hop)[np.argmax(ps[hop])]
            out["correct"][g] = pc[j] == gc[g]
        same = h & (pc == gc[g])
        if same.any():
            out["true_score"][g] = ps[same].max()
        other = h & (pc != gc[g])
        if other.any():
            j = np.flatnonzero(other)[np.argmax(ps[other])]
            out["wrong_cls"][g], out["wrong_score"][g] = pc[j], ps[j]
    return out


def categorize(det, cor):
    """det, cor: bool sequences over the checkpoints (epoch order)."""
    det, cor = np.asarray(det, bool), np.asarray(cor, bool)
    if not det.any():
        return "never-detected"
    if not cor.any():
        return "never-learned"
    first = int(np.argmax(cor))
    if not cor[first:].all():
        return "forgotten"
    return "learned-early" if first == 0 else "learned-late"


def find_checkpoints(weights_dir):
    w = Path(weights_dir)
    ck = {e: w / f"epoch{e:03d}.pt" for e in EPOCHS}
    missing = [str(p) for p in ck.values() if not p.exists()]
    if missing:
        sys.exit(f"missing checkpoints: {missing}")
    return ck


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def infer(ck, images, ev, out, device):
    import eval as ev_mod
    sizes = ev_mod.image_sizes(images)
    for e, w in ck.items():
        f = out / "preds" / f"predictions_ep{e:03d}.csv.gz"
        if f.exists():
            print(f"[dyn] ep{e}: exists, skipping inference", flush=True)
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        a = SimpleNamespace(weights=str(w), mode="sliced", imgsz=ev["imgsz"], tile=ev["tile"], overlap=ev["overlap"],
                            conf=ev["conf"], nms_iou=ev["nms_iou"], max_det=ev["max_det"], device=device, batch=16)
        t0 = time.time()
        raw = ev_mod.run_predictions(a, images)
        preds = ev_mod.finalize(raw, sizes, ev["merge"], ev["merge_thr"], ev["merge_metric"], ev["max_det"])
        preds.to_csv(f, index=False)
        print(f"[dyn] ep{e}: {len(preds)} predictions on {len(images)} images ({time.time() - t0:.0f}s)", flush=True)
    return sizes


def per_box_table(images, sizes, root, out, names, half_of):
    preds = {e: pd.read_csv(out / "preds" / f"predictions_ep{e:03d}.csv.gz") for e in EPOCHS}
    pg = {e: {k: d for k, d in p.groupby("image")} for e, p in preds.items()}
    rows = []
    for p in images:
        w, h = sizes[p.name]
        gc, gb = read_yolo_labels(root / "train" / "labels" / f"{p.stem}.txt", w, h)
        if not len(gc):
            continue
        per_ep = {}
        for e in EPOCHS:
            d = pg[e].get(p.name)
            pb = d[["x1", "y1", "x2", "y2"]].to_numpy(float) if d is not None else np.zeros((0, 4))
            pc = d.cls.to_numpy(int) if d is not None else np.zeros(0, int)
            ps = d.conf.to_numpy(float) if d is not None else np.zeros(0)
            per_ep[e] = box_metrics(gb, gc, pb, pc, ps)
        for j in range(len(gc)):
            r = dict(image=p.name, half=half_of[p.name], gt_idx=j, cls=int(gc[j]), class_name=names[int(gc[j])],
                     size=float(np.sqrt((gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1]))), n_obj=len(gc),
                     x1=gb[j, 0], y1=gb[j, 1], x2=gb[j, 2], y2=gb[j, 3])
            for e in EPOCHS:
                m = per_ep[e]
                r.update({f"det_ep{e}": bool(m["detected"][j]), f"correct_ep{e}": bool(m["correct"][j]),
                          f"best_iou_ep{e}": float(m["best_iou"][j]), f"true_score_ep{e}": float(m["true_score"][j]),
                          f"wrong_cls_ep{e}": names.get(int(m["wrong_cls"][j]), "") if m["wrong_cls"][j] >= 0 else "",
                          f"wrong_score_ep{e}": float(m["wrong_score"][j])})
            r["category"] = categorize([r[f"det_ep{e}"] for e in EPOCHS], [r[f"correct_ep{e}"] for e in EPOCHS])
            rows.append(r)
    return pd.DataFrame(rows)


def summarize_inspect(t, root, out, names, preds_last, n_crops=24, seed=0):
    """Inspect half only."""
    t = t[t.half == "inspect"].copy()
    assert (t.half == "inspect").all()
    d = out / "inspect"
    d.mkdir(parents=True, exist_ok=True)
    if t.empty:
        print("[dyn] no inspect-half boxes in this image set; nothing summarised")
        return pd.Series(0, index=list(CATEGORIES))
    labels = [f"[{lo:g}, {hi:g})" for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:])]
    t["size_bin"] = pd.cut(t["size"], SIZE_EDGES, right=False, labels=labels)
    img_n = t.groupby("image").n_obj.first()
    q = np.unique(np.quantile(img_n, [0, .25, .5, .75, 1]))
    q[0], q[-1] = -np.inf, np.inf
    t["objects_bin"] = pd.cut(t.n_obj, q, right=False)
    for col, f in (("class_name", "counts_by_class.csv"), ("size_bin", "counts_by_size.csv"),
                   ("objects_bin", "counts_by_objects.csv"), ("image", "counts_by_image.csv")):
        c = pd.crosstab(t[col].astype(str), t.category).reindex(columns=list(CATEGORIES), fill_value=0)
        c["total"] = c.sum(1)
        c.to_csv(d / f)
    tot = t.category.value_counts().reindex(list(CATEGORIES), fill_value=0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(tot.index, tot.values, color="#48c")
    ax[0].set_title(f"inspect half: {len(t)} GT boxes, {t.image.nunique()} images", fontsize=9)
    ax[0].tick_params(axis="x", rotation=20, labelsize=8)
    c = pd.crosstab(t.class_name, t.category).reindex(columns=list(CATEGORIES), fill_value=0)
    (c.T / c.sum(1)).T.plot(kind="bar", stacked=True, ax=ax[1], fontsize=8)
    ax[1].set_title("share of each category per class", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(d / "categories.png", dpi=110); plt.close(fig)

    rng = random.Random(seed)
    want = {}
    for (cat, cname), g in t.groupby(["category", "class_name"]):
        idx = sorted(g.index)
        for i in rng.sample(idx, min(n_crops, len(idx))):
            want.setdefault(t.at[i, "image"], []).append((cat, cname, i))
    cells = {}
    pl = {k: v for k, v in preds_last.groupby("image")}
    for name in sorted(want):
        with Image.open(root / "train" / "images" / name) as im:
            im = im.convert("RGB")
            pd_ = pl.get(name)
            for cat, cname, i in want[name]:
                r = t.loc[i]
                gb = np.array([[r.x1, r.y1, r.x2, r.y2]])
                cap = f"{name}"
                best = None
                if pd_ is not None and len(pd_):
                    pb = pd_[["x1", "y1", "x2", "y2"]].to_numpy(float)
                    iou = box_iou(gb, pb)[0]
                    k = int(np.argmax(iou))
                    if iou[k] > 0:
                        best = pb[k]
                        cap += f"\n{names[int(pd_.cls.iloc[k])]} {pd_.conf.iloc[k]:.2f} IoU {iou[k]:.2f}"
                pad = max(48, 1.5 * max(r.x2 - r.x1, r.y2 - r.y1))
                win = (int(max(r.x1 - pad, 0)), int(max(r.y1 - pad, 0)), int(min(r.x2 + pad, im.width)),
                       int(min(r.y2 + pad, im.height)))
                c = im.crop(win)
                dr = ImageDraw.Draw(c)
                for b, col in ((best, (230, 0, 0)), (gb[0], (0, 220, 0))):
                    if b is not None:
                        dr.rectangle([b[0] - win[0], b[1] - win[1], b[2] - win[0], b[3] - win[1]], outline=col, width=1)
                s = 160 / max(c.size)
                c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))), Image.NEAREST)
                cells.setdefault((cat, cname), []).append((c, cap))
    for (cat, cname), items in cells.items():
        fig, axes = plt.subplots(4, 6, figsize=(13, 10))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (c, cap) in zip(axes.flat, items):
            ax.imshow(c)
            ax.set_title(cap, fontsize=6)
        fig.suptitle(f"{cat} / {cname}: {len(items)} seeded examples (inspect half); green = GT, red = highest-IoU "
                     f"prediction at epoch 50", fontsize=9)
        fig.tight_layout()
        fig.savefig(d / f"crops_{cat}__{cname.replace('/', '_').replace(' ', '_')}.png", dpi=90)
        plt.close(fig)
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--config", default="configs/b1h.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, help="smoke test: first N images of the pool")
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(REPO / a.config))
    ev = {**dict(merge="nms", merge_thr=0.6, merge_metric="ios"), **cfg["eval"]}
    names = load_classes(root)
    hold = set((REPO / cfg["holdout_list"]).read_text().split())
    half_of = {n: "inspect" for n in (REPO / "splits/s52_inspect_seed0.txt").read_text().split()}
    half_of.update({n: "test" for n in (REPO / "splits/s52_test_seed0.txt").read_text().split()})
    images = [p for p in list_images(root / "train" / "images") if p.name not in hold][: a.max_images]
    assert all(p.name in half_of for p in images), "image not in either half"
    ck = find_checkpoints(a.weights_dir)
    device = a.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    sizes = infer(ck, images, ev, out, device)
    t = per_box_table(images, sizes, root, out, names, half_of)
    for half in ("inspect", "test"):
        t[t.half == half].to_csv(out / f"per_box_{half}.csv", index=False)
    preds_last = pd.read_csv(out / "preds" / f"predictions_ep{EPOCHS[-1]:03d}.csv.gz")
    tot = summarize_inspect(t, root, out, names, preds_last)
    args = dict(vars(a), eval=ev, conf_op=CONF_OP, iou=IOU, checkpoints={e: dict(path=str(p), sha256=sha256(p))
                                                                          for e, p in ck.items()},
                n_images={h: int(sum(half_of[p.name] == h for p in images)) for h in ("inspect", "test")},
                n_boxes={h: int((t.half == h).sum()) for h in ("inspect", "test")},
                inspect_category_counts={k: int(v) for k, v in tot.items()})
    (out / "training_dynamics_args.json").write_text(json.dumps(args, indent=2, default=str))
    print("[dyn] inspect half categories:", json.dumps(args["inspect_category_counts"]))
    print(f"[dyn] wrote {out}")


if __name__ == "__main__":
    main()
