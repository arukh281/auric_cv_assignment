"""Train-vs-val domain-shift diagnostics on full-resolution images. CPU, seed 0, no detector involved.

Usage:
  python analysis/domain_shift.py [--data-root data_small] [--out figures/domain_shift]

(a) Image-level statistics per image: mean / std of R, G, B (0-255), mean HSV saturation (0-1), sharpness
    (variance of the Laplacian of the gray image), high-frequency energy (share of the gray image's spectral power,
    DC removed, at normalized radial frequency > 0.25 of Nyquist), width, height, megapixels.
    -> tables/image_stats.csv, tables/image_stats_summary.csv (per stat: train / val median, Cohen's d,
       Mann-Whitney U p), image_stats_boxplots.png
(b) Truck-crop montages: per class, up to 12 train and 12 val GT crops (seeded sample, at most 2 per image unless
    fewer images are available), each a fixed 160 x 160 px
    native-resolution window centred on the box (no per-crop resizing; padded with gray at image borders), shown
    side by side at the same scale. -> montage_<class>.png
(c) Domain classifiers (train = 0, val = 1), ROC AUC, 95% percentile bootstrap CI over images (1000 resamples):
    - image stats from (a), standardized, logistic regression, leave-one-image-out
    - GT truck crops (box + 10% margin, resized to 64 x 64) embedded with ImageNet ResNet18 (torchvision,
      global-average-pooled 512-d, CPU), standardized, logistic regression, 5-fold GroupKFold by image.
      At most --max-crops-per-image crops per image (seeded) so dense images do not dominate. AUC is reported
      per crop and per image (mean crop probability).
    AUC ~0.5 = indistinguishable, ~1.0 = separable. With 20 vs 22 images the CIs are wide. Out-of-fold AUC can also
    fall below 0.5 with no signal: holding a sample out removes it from its own class, which shifts the training
    class balance against it (strongest for leave-one-out on small, balanced samples).
    -> tables/domain_auc.csv, tables/domain_auc.json, domain_auc_roc.png
(d) Box sizes per class, train vs val, from the full-dataset EDA table figures/eda/tables/boxes.csv (not recomputed
    from data_small). -> tables/box_sizes_full_dataset.csv
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import EDA_TABLES, list_images, load_classes, read_yolo_labels  # noqa: E402

SEED = 0
STAT_COLS = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "saturation", "sharpness_lapvar",
             "hf_energy", "width", "height", "megapixels"]
CROP = 160          # montage window, px (largest train box side is 161 px)
EMB_SIZE = 64       # crop size fed to ResNet18
HF_CUTOFF = 0.25    # fraction of Nyquist


def image_stats(rgb):
    """rgb: H x W x 3 uint8. Returns dict of STAT_COLS."""
    x = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = gray.shape
    return dict(mean_R=float(x[..., 0].mean()), mean_G=float(x[..., 1].mean()), mean_B=float(x[..., 2].mean()),
                std_R=float(x[..., 0].std()), std_G=float(x[..., 1].std()), std_B=float(x[..., 2].std()),
                saturation=float(hsv[..., 1].mean() / 255.0),
                sharpness_lapvar=float(cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var()),
                hf_energy=hf_energy(gray), width=w, height=h, megapixels=w * h / 1e6)


def hf_energy(gray, cutoff=HF_CUTOFF):
    """Share of spectral power (DC removed) at normalized radial frequency > cutoff (1.0 = Nyquist)."""
    p = np.abs(np.fft.rfft2(gray - gray.mean())) ** 2
    fy = np.fft.fftfreq(gray.shape[0])[:, None] * 2   # in units of Nyquist
    fx = np.fft.rfftfreq(gray.shape[1])[None, :] * 2
    r = np.sqrt(fy ** 2 + fx ** 2)
    tot = p.sum()
    return float(p[r > cutoff].sum() / tot) if tot > 0 else 0.0


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def bootstrap_auc(y, s, groups, n=1000, seed=SEED):
    """Percentile 95% CI, resampling images (groups) with replacement."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ug}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(ug, len(ug), replace=True)])
        if len(set(np.asarray(y)[pick])) == 2:
            vals.append(roc_auc(np.asarray(y)[pick], np.asarray(s)[pick]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def cv_scores(X, y, groups, n_splits=None):
    """Out-of-fold P(val) from standardized logistic regression; leave-one-group-out if n_splits is None,
    else GroupKFold(n_splits)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits=n_splits)
    s = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        if len(set(y[tr])) < 2:
            s[te] = 0.5
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))
        m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def domain_auc(X, y, groups, n_splits=None, n_boot=1000):
    s = cv_scores(np.asarray(X, float), np.asarray(y), np.asarray(groups), n_splits)
    lo, hi = bootstrap_auc(y, s, groups, n_boot)
    return dict(auc=roc_auc(y, s), ci95_lo=lo, ci95_hi=hi, n=len(y), n_groups=int(len(np.unique(groups)))), s


def window(rgb, cx, cy, size=CROP):
    """size x size native window centred on (cx, cy); gray (114) padding outside the image."""
    h, w = rgb.shape[:2]
    out = np.full((size, size, 3), 114, np.uint8)
    x0, y0 = int(round(cx)) - size // 2, int(round(cy)) - size // 2
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + size, w), min(y0 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    return out


def box_crop(rgb, b, margin=0.1, size=EMB_SIZE):
    h, w = rgb.shape[:2]
    bw, bh = b[2] - b[0], b[3] - b[1]
    x0, y0 = int(max(b[0] - margin * bw, 0)), int(max(b[1] - margin * bh, 0))
    x1, y1 = int(min(np.ceil(b[2] + margin * bw), w)), int(min(np.ceil(b[3] + margin * bh), h))
    c = rgb[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def embed(crops, batch=128):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    torch.manual_seed(SEED)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    net.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(np.stack(crops[i:i + batch])).permute(0, 3, 1, 2).float() / 255.0
            out.append(net((x - mean) / std).numpy())
    return np.concatenate(out) if out else np.zeros((0, 512))


def montage_sample(cands, n, per_image=2, seed=SEED):
    """Seeded sample of n (image, box) pairs, at most `per_image` per image first, so one dense image cannot fill
    the montage; tops up from the remaining boxes only if fewer than n images are available."""
    order = [cands[i] for i in np.random.default_rng(seed).permutation(len(cands))]
    cnt, first, rest = {}, [], []
    for name, j in order:
        (first if cnt.get(name, 0) < per_image else rest).append((name, j))
        cnt[name] = cnt.get(name, 0) + 1
    return (first + rest)[:n]


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(b) - np.mean(a)) / sp) if sp > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data_small")
    ap.add_argument("--out", default="figures/domain_shift")
    ap.add_argument("--max-crops-per-image", type=int, default=60)
    ap.add_argument("--n-montage", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    root, out = Path(a.data_root), Path(a.out)
    tab = out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    names = load_classes(root)
    rng = np.random.default_rng(SEED)

    rows, crops, crop_meta, mont = [], [], [], {(s, c): [] for s in ("train", "val") for c in names}
    for split in ("train", "val"):
        for p in list_images(root / split / "images"):
            rgb = np.asarray(Image.open(p).convert("RGB"))
            h, w = rgb.shape[:2]
            rows.append(dict(split=split, image=p.name, **image_stats(rgb)))
            gc, gb = read_yolo_labels(root / split / "labels" / f"{p.stem}.txt", w, h)
            keep = np.sort(rng.permutation(len(gc))[: a.max_crops_per_image])
            for j in keep:
                crops.append(box_crop(rgb, gb[j]))
                crop_meta.append(dict(split=split, image=p.name, cls=int(gc[j])))
            for j in range(len(gc)):  # montage candidates: keep windows only for a seeded subset later
                mont[(split, int(gc[j]))].append((p.name, j))
            print(f"[domain] {split} {p.name}: {len(gc)} GT", flush=True)

    # (a) stats
    st = pd.DataFrame(rows)
    st.to_csv(tab / "image_stats.csv", index=False)
    summ = []
    tr, va = st[st.split == "train"], st[st.split == "val"]
    for c in STAT_COLS:
        p = mannwhitneyu(tr[c], va[c]).pvalue
        summ.append(dict(stat=c, train_median=tr[c].median(), val_median=va[c].median(), train_mean=tr[c].mean(),
                         val_mean=va[c].mean(), cohens_d_val_minus_train=cohens_d(tr[c].to_numpy(), va[c].to_numpy()),
                         mannwhitney_p=p, n_train=len(tr), n_val=len(va)))
    summ = pd.DataFrame(summ).sort_values("cohens_d_val_minus_train", key=np.abs, ascending=False)
    summ.to_csv(tab / "image_stats_summary.csv", index=False)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    for ax, c in zip(axes.flat, STAT_COLS):
        ax.boxplot([tr[c], va[c]], tick_labels=["train", "val"])
        for k, d in enumerate((tr[c], va[c]), 1):
            ax.scatter(np.full(len(d), k) + rng.uniform(-0.08, 0.08, len(d)), d, s=8, alpha=0.6)
        ax.set_title(c, fontsize=9)
    fig.suptitle(f"Image statistics, {root.name}: {len(tr)} train vs {len(va)} val images (full resolution)")
    fig.tight_layout(); fig.savefig(out / "image_stats_boxplots.png", dpi=110); plt.close(fig)

    # (b) montages: seeded sample, then a second pass reads only the needed images
    pick = {k: montage_sample(v, a.n_montage) for k, v in mont.items()}
    need = {}
    for (split, c), lst in pick.items():
        for name, j in lst:
            need.setdefault((split, name), []).append((c, j))
    tiles = {k: [] for k in pick}
    for (split, name), lst in sorted(need.items()):
        rgb = np.asarray(Image.open(root / split / "images" / name).convert("RGB"))
        h, w = rgb.shape[:2]
        gc, gb = read_yolo_labels(root / split / "labels" / f"{Path(name).stem}.txt", w, h)
        for c, j in lst:
            b = gb[j]
            tiles[(split, c)].append((window(rgb, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2), name,
                                      f"{b[2] - b[0]:.0f}x{b[3] - b[1]:.0f}"))
    for c, cname in names.items():
        fig, axes = plt.subplots(3, 9, figsize=(18, 6.6))
        for ax in axes.flat:
            ax.axis("off")
        for side, split in ((0, "train"), (1, "val")):
            for k, (im, name, wh) in enumerate(tiles[(split, c)][: a.n_montage]):
                ax = axes[k // 4][side * 5 + k % 4]
                ax.imshow(im, interpolation="nearest")
                ax.set_title(f"{split} {name} {wh}", fontsize=6)
        for r in range(3):
            axes[r][4].axis("off")
        fig.suptitle(f"{cname}: left = train, right = val (n = {len(tiles[('train', c)])} / {len(tiles[('val', c)])}); "
                     f"each tile a {CROP} x {CROP} px native window centred on the GT box, same scale, no resizing",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"montage_{cname.replace('/', '_').replace(' ', '_')}.png", dpi=110)
        plt.close(fig)

    # (c) domain classifiers
    y_img = (st.split == "val").astype(int).to_numpy()
    res_stats, s_img = domain_auc(st[STAT_COLS].to_numpy(), y_img, st.image.to_numpy(), None, a.bootstrap)
    cm = pd.DataFrame(crop_meta)
    E = embed(crops)
    y_c = (cm.split == "val").astype(int).to_numpy()
    groups = (cm.split + "/" + cm.image).to_numpy()
    res_crop, s_c = domain_auc(E, y_c, groups, 5, a.bootstrap)
    per_img = pd.DataFrame(dict(g=groups, y=y_c, s=s_c)).groupby("g").agg(y=("y", "first"), s=("s", "mean"))
    lo, hi = bootstrap_auc(per_img.y.to_numpy(), per_img.s.to_numpy(), per_img.index.to_numpy(), a.bootstrap)
    res_crop_img = dict(auc=roc_auc(per_img.y, per_img.s), ci95_lo=lo, ci95_hi=hi, n=len(per_img),
                        n_groups=len(per_img))
    auc = pd.DataFrame([dict(classifier="image stats (a), logistic regression, leave-one-image-out", unit="image",
                             **res_stats),
                        dict(classifier="ResNet18 crop embeddings, logistic regression, GroupKFold(5) by image",
                             unit="crop", **res_crop),
                        dict(classifier="ResNet18 crop embeddings, mean crop probability per image", unit="image",
                             **res_crop_img)])
    auc.to_csv(tab / "domain_auc.csv", index=False)
    caveat = (f"{len(tr)} train vs {len(va)} val images from {root}; the train images are a 20-image subset of 443. "
              "Bootstrap resamples images, so CIs reflect image-level variability only. Out-of-fold AUC can fall below "
              "0.5 with no signal: holding a sample out removes it from its own class, shifting the training class "
              "balance against it (strongest for leave-one-out on small balanced samples).")
    (tab / "domain_auc.json").write_text(json.dumps(dict(results=auc.to_dict("records"), caveat=caveat, seed=SEED,
                                                         crops_per_image_max=a.max_crops_per_image,
                                                         n_crops_train=int((y_c == 0).sum()),
                                                         n_crops_val=int((y_c == 1).sum())), indent=2))
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, yy, ss in (("image stats", y_img, s_img), ("crop emb. (per crop)", y_c, s_c),
                        ("crop emb. (per image)", per_img.y.to_numpy(), per_img.s.to_numpy())):
        fpr, tpr, _ = roc_curve(yy, ss)
        ax.plot(fpr, tpr, label=f"{lab}: AUC {roc_auc(yy, ss):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="FPR", ylabel="TPR", title="train-vs-val domain classifiers (out-of-fold)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "domain_auc_roc.png", dpi=110); plt.close(fig)

    # (d) box sizes from the full-dataset EDA table
    bx = pd.read_csv(EDA_TABLES / "boxes.csv")
    rows = []
    for cname in names.values():
        r = dict(class_name=cname)
        for split in ("train", "val"):
            b = bx[(bx.split == split) & (bx.class_name == cname)]
            r[f"{split}_n"] = len(b)
            for col in ("w_px", "h_px", "sqrt_area_px"):
                q = b[col].quantile([0.25, 0.5, 0.75]) if len(b) else pd.Series([np.nan] * 3)
                r[f"{split}_{col}_q25"], r[f"{split}_{col}_median"], r[f"{split}_{col}_q75"] = q.to_numpy()
        r["val_over_train_median_sqrt_area"] = r["val_sqrt_area_px_median"] / r["train_sqrt_area_px_median"]
        rows.append(r)
    pd.DataFrame(rows).to_csv(tab / "box_sizes_full_dataset.csv", index=False)

    print(summ[["stat", "train_median", "val_median", "cohens_d_val_minus_train", "mannwhitney_p"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(auc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"[domain] wrote {out}")


if __name__ == "__main__":
    main()
