"""Operating points: mAP50, precision, recall and F1 (class-aware, IoU 0.5, one-to-one as detlib.scoring) of a set of
saved predictions at fixed confidence thresholds. CPU only.

  python analysis/operating_points.py --preds <predictions.csv> --data-root "$DATA_ROOT" --split val|holdout \
      --name <tag> --out <dir>

Thresholds: 0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8. mAP50 at threshold t = COCO 101-point AP
computed only from predictions with conf >= t (so it can only fall as t rises). P = TP / predictions, R = TP / GT,
F1 = 2PR / (P + R). Writes <out>/operating_points_<tag>.csv and .png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import list_images  # noqa: E402
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from eval import collect, image_sizes  # noqa: E402

TH = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", choices=["val", "holdout"], required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.split == "val":
        imgs, L = list_images(root / "val" / "images"), root / "val" / "labels"
    else:
        hold = [l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip()]
        imgs, L = [root / "train" / "images" / n for n in hold], root / "train" / "labels"
    sizes = image_sizes(imgs); P = pd.read_csv(a.preds); P = P[P.cls < 5]
    rows = []
    for t in TH:
        recs = score_images(collect(P[P.conf >= t], imgs, L, sizes), 5)
        _, m, _ = ap_from_records(recs, 5, "coco")
        tp = int(sum(r["tp"].sum() for r in recs)); npred = int(sum(len(r["tp"]) for r in recs))
        ngt = int(sum(np.sum(r["n_gt"]) for r in recs))
        p = tp / npred if npred else float("nan"); rc = tp / ngt; f1 = 2 * p * rc / (p + rc) if npred and (p + rc) else 0.0
        rows.append(dict(threshold=t, mAP50=float(m), precision=p, recall=rc, F1=f1, TP=tp, predictions=npred, GT=ngt))
    T = pd.DataFrame(rows); T.to_csv(out / f"operating_points_{a.name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for c in ("mAP50", "precision", "recall", "F1"):
        ax.plot(T.threshold, T[c], marker="o", label=c)
    b = T.loc[T.F1.idxmax()]
    ax.axvline(b.threshold, ls=":", color="grey"); ax.set(xlabel="confidence threshold", ylim=(0, 1),
                                                         title=f"{a.name}: F1-optimal threshold {b.threshold:g} (F1 {b.F1:.3f})")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / f"operating_points_{a.name}.png", dpi=110); plt.close(fig)
    print(T.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
