"""Numbers behind the independent visual review of figures/inspect_val, plus follow-ups (a)-(c). CPU only.

  python analysis/val_review.py --data-root /kaggle/tmp/data --b1h <B1h run dir> --xview <mirror root> --out <dir>

R1 label shift: per val image, global (dx, dy) in [-60, 60] step 3 applied to GT; GT matched (one-to-one, class-
   agnostic, IoU >= 0.5) by B1h predictions with conf >= 0.05. Reports best shift and gain over (0, 0).
R2 re-merge of holdout40 raw tile predictions (eval_holdout40/predictions_raw.csv), scored like eval.py:
   baseline (class-wise NMS, IoS 0.6), NMS IoU 0.5, NMS IoU 0.7, drop predictions touching an interior tile edge
   (within 2 px) then baseline, cross-tile-only NMS (IoS 0.6; boxes from the same tile never suppress each other).
R3 per-image GT class counts on val. R5 image sizes, val vs train.
A  image quality for every train (incl. holdout40) and val image, on the grey image: mean brightness, RMS contrast
   (std), blur = variance of the Laplacian, noise = Immerkaer (1996) sigma estimate. Per-image recall (class-
   agnostic, IoU 0.5, conf >= 0.001) for val (eval/) and holdout40 (eval_holdout40/); Spearman correlations;
   holdout40 recall re-weighted to val's quartile distribution of each metric (pooled quartile bins), as
   analysis/recall_gap.py does for size.
B  2391.png: every GT box with no prediction at IoU >= 0.5 (any conf): oracle pool scores
   (analysis/inputs/inspect_val/val_gt_scores.csv), raw tile predictions with IoU >= 0.1 (any conf), tiles that
   contain it whole; the tile with most such boxes rendered at full resolution with all raw predictions conf >= 0.001.
C  geographic footprints from the GeoTIFF tags of the mirror's tifs (ModelTiepoint + ModelPixelScale); pairs of our
   images whose footprints intersect, train vs val and holdout40 vs val.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, list_images, read_yolo_labels  # noqa: E402
from detlib.merge import _overlap, nms  # noqa: E402
from detlib.scoring import ap_from_records, box_iou, score_images  # noqa: E402
from detlib.tiling import tile_windows  # noqa: E402
from eval import collect, finalize  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SHORT = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]
COL = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180)]


def match_count(gb, pb, ps, thr=0.5):
    hit = np.zeros(len(gb), bool)
    if len(gb) and len(pb):
        iou = box_iou(pb, gb)
        for i in np.argsort(-ps):
            row = np.where(hit, -1, iou[i]); j = int(np.argmax(row))
            if row[j] >= thr:
                hit[j] = True
    return hit


def score(preds, images, label_dir, sizes):
    per = collect(preds[preds.conf >= 0.001], images, label_dir, sizes)
    return ap_from_records(score_images(per, 5), 5, "coco")[1]


def immerkaer(g):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = g.shape
    s = np.abs(cv2.filter2D(g.astype(np.float64), -1, k))[1:-1, 1:-1].sum()
    return float(s * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True); ap.add_argument("--b1h", required=True)
    ap.add_argument("--xview", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, B, out = Path(a.data_root), Path(a.b1h), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    im = pd.read_csv(EDA_TABLES / "images.csv")
    size = {(r.split, r.image): (int(r.width), int(r.height)) for r in im.itertuples()}
    hold = [l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip()]
    vals = sorted(im[im.split == "val"].image)
    pv, ph = pd.read_csv(B / "eval" / "predictions.csv"), pd.read_csv(B / "eval_holdout40" / "predictions.csv")
    S = {}

    # R1
    rows = []
    for n in vals:
        W, H = size[("val", n)]; gc, gb = read_yolo_labels(root / "val" / "labels" / f"{Path(n).stem}.txt", W, H)
        q = pv[(pv.image == n) & (pv.conf >= 0.05)]; pb, ps = q[["x1", "y1", "x2", "y2"]].to_numpy(float), q.conf.to_numpy(float)
        base = int(match_count(gb, pb, ps).sum()); best = (base, 0, 0)
        for dx in range(-60, 61, 3):
            for dy in range(-60, 61, 3):
                m = int(match_count(gb + np.array([dx, dy, dx, dy]), pb, ps).sum())
                if m > best[0]:
                    best = (m, dx, dy)
        rows.append(dict(image=n, n_gt=len(gb), matched_at_zero=base, best_matched=best[0], best_dx=best[1],
                         best_dy=best[2], gain=best[0] - base))
    r1 = pd.DataFrame(rows); r1.to_csv(out / "r1_label_shift.csv", index=False)
    S["r1"] = dict(images=len(r1), best_shift_zero=int(((r1.best_dx == 0) & (r1.best_dy == 0)).sum()), max_gain=int(r1.gain.max()))

    # R2
    raw = pd.read_csv(B / "eval_holdout40" / "predictions_raw.csv")
    hsz = {n: size[("train", n)] for n in hold}; himgs = [root / "train" / "images" / n for n in hold]
    ldir = root / "train" / "labels"
    var = {"baseline nms ios 0.6": finalize(raw, hsz, "nms", 0.6, "ios", 902),
           "nms iou 0.5": finalize(raw, hsz, "nms", 0.5, "iou", 902),
           "nms iou 0.7": finalize(raw, hsz, "nms", 0.7, "iou", 902)}
    keep = []
    for n, g in raw.groupby("image"):
        W, H = hsz[n]; tx, ty = g.tile_x0.to_numpy(), g.tile_y0.to_numpy()
        x1, y1, x2, y2 = (g[c].to_numpy() for c in ("x1", "y1", "x2", "y2"))
        right = np.minimum(tx + 1024, W); bottom = np.minimum(ty + 1024, H)
        touch = ((tx > 0) & (x1 <= tx + 2)) | ((ty > 0) & (y1 <= ty + 2)) | ((right < W) & (x2 >= right - 2)) | ((bottom < H) & (y2 >= bottom - 2))
        keep.append(g[~touch])
    var["drop interior-edge boxes, then baseline"] = finalize(pd.concat(keep), hsz, "nms", 0.6, "ios", 902)
    rows = []
    for n, g in raw.groupby("image"):
        xy, cf, cl = g[["x1", "y1", "x2", "y2"]].to_numpy(float), g.conf.to_numpy(float), g.cls.to_numpy(int)
        tile = (g.tile_x0.astype(str) + "_" + g.tile_y0.astype(str)).to_numpy()
        order = np.argsort(-cf, kind="mergesort"); alive = np.ones(len(g), bool); kept = []
        for i in order:
            if not alive[i]:
                continue
            kept.append(i)
            o = (cl == cl[i]) & (tile != tile[i]) & alive
            o[i] = False
            idx = np.where(o)[0]
            if len(idx):
                alive[idx[_overlap(xy[i], xy[idx], "ios") >= 0.6]] = False
        kept = np.array(kept[:902], int)
        rows += [dict(image=n, cls=int(cl[k]), conf=float(cf[k]), x1=xy[k, 0], y1=xy[k, 1], x2=xy[k, 2], y2=xy[k, 3]) for k in kept]
    var["cross-tile-only NMS ios 0.6"] = pd.DataFrame(rows)
    r2 = pd.DataFrame([dict(variant=k, holdout40_mAP50=score(v, himgs, ldir, hsz)) for k, v in var.items()])
    r2["delta_vs_baseline"] = r2.holdout40_mAP50 - r2.holdout40_mAP50.iloc[0]
    r2.to_csv(out / "r2_remerge.csv", index=False); S["r2"] = r2.to_dict("records")

    # R3, R5
    cc = []
    for n in vals:
        W, H = size[("val", n)]; gc, _ = read_yolo_labels(root / "val" / "labels" / f"{Path(n).stem}.txt", W, H)
        cc.append(dict(image=n, width=W, height=H, **{SHORT[c]: int((gc == c).sum()) for c in range(5)}, total=len(gc)))
    r3 = pd.DataFrame(cc); r3.to_csv(out / "r3_val_class_counts.csv", index=False)
    tr = im[im.split == "train"]; tmax = np.maximum(tr.width, tr.height); vmax = np.maximum(r3.width, r3.height)
    S["r5"] = dict(train_max_side_min=int(tmax.min()), train_max_side_max=int(tmax.max()),
                   val_smaller_than_all_train=r3[vmax < tmax.min()].image.tolist(),
                   val_larger_than_all_train=r3[vmax > tmax.max()].image.tolist(),
                   val_max_sides=dict(zip(r3.image, vmax.astype(int).tolist())))
    S["r3_box_share_2470_2472"] = int(r3[r3.image.isin(["2470.png", "2472.png"])].Box.sum())
    S["r3_box_total"] = int(r3.Box.sum())

    # A
    q = []
    for r in im.itertuples():
        g = cv2.imread(str(root / r.split / "images" / r.image), cv2.IMREAD_GRAYSCALE)
        q.append(dict(split="val" if r.split == "val" else ("holdout40" if r.image in hold else "train"), image=r.image,
                      brightness=float(g.mean()), rms_contrast=float(g.std()),
                      blur_laplacian_var=float(cv2.Laplacian(g, cv2.CV_64F).var()), noise_sigma=immerkaer(g)))
    Q = pd.DataFrame(q)
    rec = []
    for split, P, names, lab in (("val", pv, vals, "val"), ("holdout40", ph, hold, "train")):
        for n in names:
            W, H = size[(lab, n)]; _, gb = read_yolo_labels(root / lab / "labels" / f"{Path(n).stem}.txt", W, H)
            qq = P[P.image == n]; h = match_count(gb, qq[["x1", "y1", "x2", "y2"]].to_numpy(float), qq.conf.to_numpy(float))
            rec.append(dict(image=n, split=split, n_gt=len(gb), matched=int(h.sum())))
    Rc = pd.DataFrame(rec); Q = Q.merge(Rc, on=["image", "split"], how="left"); Q.to_csv(out / "a_image_quality.csv", index=False)
    mets = ["brightness", "rms_contrast", "blur_laplacian_var", "noise_sigma"]
    corr, stdz = [], []
    ev = Q[Q.split.isin(["val", "holdout40"])].copy(); ev["recall"] = ev.matched / ev.n_gt
    for m in mets:
        for sp in ("val", "holdout40", "both"):
            d = ev if sp == "both" else ev[ev.split == sp]
            corr.append(dict(metric=m, split=sp, n=len(d), spearman_recall=float(d[m].corr(d.recall, method="spearman"))))
        bins = np.quantile(ev[m], [0, .25, .5, .75, 1]); bins[0] -= 1e-9; bins[-1] += 1e-9
        ev["bin"] = pd.cut(ev[m], bins)
        v, h = ev[ev.split == "val"], ev[ev.split == "holdout40"]
        wv = v.groupby("bin", observed=False).n_gt.sum() / v.n_gt.sum()
        hn = h.groupby("bin", observed=False).n_gt.sum()
        rh = h.groupby("bin", observed=False).matched.sum() / hn.clip(lower=1)
        ok = hn > 0
        rv_all = v.matched.sum() / v.n_gt.sum(); rh_all = h.matched.sum() / h.n_gt.sum()
        st = float((rh[ok] * wv[ok]).sum() / wv[ok].sum())
        stdz.append(dict(metric=m, val_recall=rv_all, holdout_recall=rh_all, holdout_reweighted_to_val=st,
                         val_box_share_covered=float(wv[ok].sum()),
                         share_of_gap_explained=(rh_all - st) / (rh_all - rv_all)))
    pd.DataFrame(corr).to_csv(out / "a_correlations.csv", index=False); pd.DataFrame(stdz).to_csv(out / "a_standardised.csv", index=False)
    S["a_corr"] = corr; S["a_std"] = stdz
    S["a_medians"] = Q.groupby("split")[mets].median().round(3).to_dict("index")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, m in zip(axs, mets):
        for sp, c in (("train", "grey"), ("holdout40", "tab:green"), ("val", "tab:orange")):
            ax.hist(Q[Q.split == sp][m], bins=30, alpha=.55, density=True, label=sp, color=c, log=False)
        ax.set_title(m, fontsize=9)
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "a_quality_distributions.png", dpi=90); plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, m in zip(axs, mets):
        for sp, c in (("holdout40", "tab:green"), ("val", "tab:orange")):
            d = ev[ev.split == sp]; ax.scatter(d[m], d.recall, s=14, color=c, label=sp)
        ax.set(title=m, ylabel="recall"); 
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "a_recall_vs_quality.png", dpi=90); plt.close(fig)

    # B
    n = "2391.png"; W, H = size[("val", n)]; gc, gb = read_yolo_labels(root / "val" / "labels" / "2391.txt", W, H)
    allp = pv[pv.image == n]; pb = allp[["x1", "y1", "x2", "y2"]].to_numpy(float)
    best = box_iou(gb, pb).max(1) if len(pb) else np.zeros(len(gb))
    rawv = pd.read_csv(B / "eval" / "predictions_raw.csv"); rv = rawv[rawv.image == n]
    rb = rv[["x1", "y1", "x2", "y2"]].to_numpy(float)
    gs = pd.read_csv(REPO / "analysis" / "inputs" / "inspect_val" / "val_gt_scores.csv"); gs = gs[gs.image == n]
    wins = tile_windows(W, H, 1024, 256); rows = []
    for j in np.where(best < 0.5)[0]:
        b = gb[j]; o = gs[gs.gt_idx == j].iloc[0]
        ri = box_iou(b[None], rb)[0] if len(rb) else np.zeros(0)
        near = rv[ri >= 0.1].assign(iou=ri[ri >= 0.1]).sort_values("conf", ascending=False)
        inside = [f"{x0},{y0}" for x0, y0, x1, y1 in wins if b[0] >= x0 and b[2] <= x1 and b[1] >= y0 and b[3] <= y1]
        rows.append(dict(gt_idx=int(j), gt_cls=SHORT[int(gc[j])], x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                         box_w=b[2] - b[0], box_h=b[3] - b[1], best_merged_iou=float(best[j]),
                         oracle_pred=SHORT[int(o.pred_pool)], oracle_no_anchor_at_iou=bool(o.no_anchor_at_iou),
                         **{f"oracle_{s}": float(o[f"pool_{nm}"]) for s, nm in zip(SHORT, ["Cargo Truck", "Truck w/Box", "Truck w/Flatbed", "Truck Tractor", "Truck w/Liquid"])},
                         n_raw_near=len(near), raw_near_top=";".join(f"{SHORT[int(r.cls)]} {r.conf:.3f} iou {r.iou:.2f} tile {r.tile_x0},{r.tile_y0}" for r in near.head(5).itertuples()),
                         tiles_containing=" | ".join(inside)))
    rb_df = pd.DataFrame(rows); rb_df.to_csv(out / "b_2391_missed.csv", index=False)
    S["b"] = dict(n_gt=len(gb), n_missed=len(rb_df), missed_by_class=rb_df.gt_cls.value_counts().to_dict() if len(rb_df) else {})
    if len(rb_df):
        cnt = {w: 0 for w in wins}
        for r in rb_df.itertuples():
            for w in wins:
                if r.x1 >= w[0] and r.x2 <= w[2] and r.y1 >= w[1] and r.y2 <= w[3]:
                    cnt[w] += 1
        w = max(cnt, key=cnt.get); x0, y0, x1, y1 = w
        t = Image.open(root / "val" / "images" / n).convert("RGB").crop(w); d = ImageDraw.Draw(t)
        for c, b in zip(gc, gb):
            d.rectangle([b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0], outline=COL[c], width=1)
        tp = rv[(rv.tile_x0 == x0) & (rv.tile_y0 == y0)]
        for r in tp.itertuples():
            bb = [r.x1 - x0, r.y1 - y0, r.x2 - x0, r.y2 - y0]
            for k in range(0, int(bb[2] - bb[0]), 6):
                d.line([bb[0] + k, bb[1], min(bb[0] + k + 3, bb[2]), bb[1]], fill=COL[int(r.cls)])
                d.line([bb[0] + k, bb[3], min(bb[0] + k + 3, bb[2]), bb[3]], fill=COL[int(r.cls)])
            d.text((bb[0], bb[3] + 1), f"{SHORT[int(r.cls)]} {r.conf:.3f}", fill=COL[int(r.cls)])
        hdr = Image.new("RGB", (t.width, 34), "white")
        ImageDraw.Draw(hdr).text((4, 4), f"2391.png tile x{x0} y{y0} (full resolution). GT solid; ALL raw tile predictions conf>=0.001 dashed with class and conf.\n"
                                         f"{cnt[w]} of {len(rb_df)} missed GT boxes are inside this tile. Colours: Cargo red, Box blue, Flatbed green, Tractor orange, Liquid purple.", fill=(0, 0, 0))
        o2 = Image.new("RGB", (t.width, t.height + 34), "white"); o2.paste(hdr, (0, 0)); o2.paste(t, (0, 34))
        o2.save(out / "b_2391_tile.jpg", quality=88); S["b"]["rendered_tile"] = f"{x0},{y0}"

    # C
    from PIL.TiffTags import TAGS  # noqa: F401
    foot = []
    for r in im.itertuples():
        p = Path(a.xview) / "train_images" / "train_images" / (Path(r.image).stem + ".tif")
        if not p.exists():
            p = Path(a.xview) / "val_images" / "val_images" / (Path(r.image).stem + ".tif")
        if not p.exists():
            continue
        with Image.open(p) as t:
            tags = t.tag_v2; tp, sc = tags.get(33922), tags.get(33550); w, h = t.size
        if not tp or not sc or (w, h) != (r.width, r.height):
            continue
        lon0, lat0 = tp[3], tp[4]; foot.append(dict(image=r.image, split="val" if r.split == "val" else ("holdout40" if r.image in hold else "train"),
                                                   lon0=lon0, lat1=lat0, lon1=lon0 + sc[0] * w, lat0=lat0 - sc[1] * h))
    F = pd.DataFrame(foot, columns=["image", "split", "lon0", "lat1", "lon1", "lat0"]); F.to_csv(out / "c_footprints.csv", index=False)
    pairs = []
    vv = F[F.split == "val"]
    for r in F[F.split != "val"].itertuples():
        for s in vv.itertuples():
            ix = min(r.lon1, s.lon1) - max(r.lon0, s.lon0); iy = min(r.lat1, s.lat1) - max(r.lat0, s.lat0)
            if ix > 0 and iy > 0:
                ar = (r.lon1 - r.lon0) * (r.lat1 - r.lat0); as_ = (s.lon1 - s.lon0) * (s.lat1 - s.lat0)
                pairs.append(dict(train_image=r.image, train_split=r.split, val_image=s.image,
                                  overlap_share_of_val=ix * iy / as_, overlap_share_of_train=ix * iy / ar))
    P2 = pd.DataFrame(pairs, columns=["train_image", "train_split", "val_image", "overlap_share_of_val", "overlap_share_of_train"])
    P2.sort_values("overlap_share_of_val", ascending=False).to_csv(out / "c_geo_overlaps.csv", index=False)
    S["c"] = dict(footprints=len(F), by_split=F.split.value_counts().to_dict(), overlapping_pairs=len(P2),
                  val_images_with_overlap=sorted(P2.val_image.unique().tolist()),
                  pairs_with_holdout40=int((P2.train_split == "holdout40").sum()),
                  pair_2459_2470_2472=P2[(P2.train_image == "2459.png")].to_dict("records"))
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps(S, indent=2, default=float))


if __name__ == "__main__":
    main()
