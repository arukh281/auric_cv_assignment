"""§5.4: what is in the smart and random subsets? CPU only, from committed files (no images, no inference).

  python analysis/subset_characterise.py [--out figures/subset_compare]

Subsets: splits/train_{f50,smart50,f75,smart75}_seed0.txt and the full pool (403 non-holdout train images). The
random runs with seed 1 used the same image lists as seed 0 (only the training seed differs).
Per subset: images, boxes, boxes and images per class, box-size distribution (sqrt area px: median and share in
<16 / 16-32 / >=32), boxes per image (mean, median, share of images with >= 30 boxes), empty images, and scene
coverage. Scene proxy: k-means (k = 20, seed 0) on the DINOv2-small image embeddings used for smart selection
(results/s54/embeddings.npz; 402 images, 1938.png not embedded); coverage = clusters with >= 1 image in the subset.
Box data: figures/eda/tables/boxes.csv. Writes <out>/subset_characterisation.csv and subset_class_coverage.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SUBSETS = {"random50": "train_f50_seed0.txt", "smart50": "train_smart50_seed0.txt",
           "random75": "train_f75_seed0.txt", "smart75": "train_smart75_seed0.txt"}


def kmeans(x, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "figures" / "subset_compare"))
    ap.add_argument("--embeddings", default=str(REPO / "results" / "s54" / "embeddings.npz"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rd = lambda f: [l.strip() for l in (REPO / "splits" / f).read_text().splitlines() if l.strip()]
    hold = set(rd("holdout40_seed0.txt"))
    b = pd.read_csv(REPO / "figures" / "eda" / "tables" / "boxes.csv").query("split == 'train'")
    allimgs = sorted(pd.read_csv(REPO / "figures" / "eda" / "tables" / "images.csv").query("split == 'train'").image)
    pool = [i for i in allimgs if i not in hold]
    sets = {"full403": pool, **{k: rd(v) for k, v in SUBSETS.items()}}
    z = np.load(a.embeddings, allow_pickle=True)
    names, emb = [str(n) for n in z["names"]], z["emb"].astype(float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cl = dict(zip(names, kmeans(emb, 20)))
    classes = sorted(b.class_name.unique())
    rows, cov = [], []
    for k, imgs in sets.items():
        s = set(imgs)
        bb = b[b.image.isin(s)]
        per_img = bb.groupby("image").size().reindex(sorted(s), fill_value=0)
        sz = bb.sqrt_area_px
        rows.append(dict(subset=k, images=len(s), boxes=len(bb), empty_images=int((per_img == 0).sum()),
                         boxes_per_image_mean=per_img.mean(), boxes_per_image_median=per_img.median(),
                         share_images_ge30_boxes=(per_img >= 30).mean(),
                         size_median_px=sz.median(), share_lt16=(sz < 16).mean(),
                         share_16_32=((sz >= 16) & (sz < 32)).mean(), share_ge32=(sz >= 32).mean(),
                         scene_clusters_covered=len({cl[i] for i in s if i in cl}), scene_clusters_total=20))
        for c in classes:
            bc = bb[bb.class_name == c]
            fb = b[(b.image.isin(set(pool))) & (b.class_name == c)]
            cov.append(dict(subset=k, class_name=c, boxes=len(bc), images=bc.image.nunique(),
                            share_of_pool_boxes=len(bc) / max(len(fb), 1), share_of_subset_boxes=len(bc) / max(len(bb), 1)))
    r = pd.DataFrame(rows); c = pd.DataFrame(cov)
    r.to_csv(out / "subset_characterisation.csv", index=False); c.to_csv(out / "subset_class_coverage.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}")); print(c.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
