"""§5.3 per class: holdout40 AP of class c vs the number of training instances of c, and a +500-instance projection.
CPU only, from committed files.

  python analysis/lc_per_class.py [--out figures/learning_curve]

Points: the random learning-curve runs b1h_f25, b1h_f50 (+seed 1), b1h_f75 (+seed 1), b1h_tile1024_holdout40 and
b1h_seed1 (7 points; the smart subsets are excluded because their class mix was selected). Holdout AP per class:
analysis/inputs/lc_holdout/<run>.csv (copies of results/<run>/eval_holdout40/per_class.csv). Training instances of
c: boxes of c in the run's train images (figures/eda/tables/boxes.csv; splits/train_f*_seed0.txt, or all train images
minus splits/holdout40_seed0.txt).
Fit per class: AP = a * n^b by least squares on log AP vs log n, using points with AP > 0 (zeros are reported and
excluded). Projection: AP at n_full + 500, where n_full is the full-data count; dAP vs the fit at n_full; d mAP50 =
dAP / 5. Extrapolation ratio = (n_full + 500) / largest measured n; > 2 is flagged "unreliable extrapolation".
Also flagged: fewer than 4 positive points, or b <= 0 (AP not rising with n). Writes <out>/lc_per_class.csv and
lc_per_class_points.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RUNS = {"b1h_f25": "train_f25_seed0.txt", "b1h_f50": "train_f50_seed0.txt", "b1h_f50_seed1": "train_f50_seed0.txt",
        "b1h_f75": "train_f75_seed0.txt", "b1h_f75_seed1": "train_f75_seed0.txt",
        "b1h_tile1024_holdout40": None, "b1h_seed1": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "figures" / "learning_curve"))
    ap.add_argument("--extra", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rd = lambda f: [l.strip() for l in (REPO / "splits" / f).read_text().splitlines() if l.strip()]
    hold = set(rd("holdout40_seed0.txt"))
    b = pd.read_csv(REPO / "figures" / "eda" / "tables" / "boxes.csv").query("split == 'train'")
    pool = set(b.image) - hold
    pts = []
    for run, lst in RUNS.items():
        imgs = set(rd(lst)) if lst else pool
        cnt = b[b.image.isin(imgs)].class_name.value_counts()
        pc = pd.read_csv(REPO / "analysis" / "inputs" / "lc_holdout" / f"{run}.csv")
        for r in pc[pc.cls >= 0].itertuples():
            pts.append(dict(run=run, class_name=r.class_name, n_train=int(cnt.get(r.class_name, 0)),
                            holdout_n_gt=int(r.n_gt), AP50=float(r.AP50_coco)))
    p = pd.DataFrame(pts); p.to_csv(out / "lc_per_class_points.csv", index=False)
    rows = []
    for c, g in p.groupby("class_name"):
        pos = g[g.AP50 > 0]
        n_full = int(g[g.run == "b1h_tile1024_holdout40"].n_train.iloc[0])
        nmax = int(g.n_train.max())
        row = dict(class_name=c, n_train_full=n_full, holdout_n_gt=int(g.holdout_n_gt.iloc[0]), n_points=len(g),
                   n_points_positive=len(pos), n_range=f"{int(g.n_train.min())}-{nmax}",
                   AP_full_seed0=float(g[g.run == "b1h_tile1024_holdout40"].AP50.iloc[0]),
                   AP_full_seed1=float(g[g.run == "b1h_seed1"].AP50.iloc[0]),
                   extrap_ratio=(n_full + a.extra) / nmax)
        flags = []
        if len(pos) >= 2:
            bb, la = np.polyfit(np.log(pos.n_train), np.log(pos.AP50), 1)
            aa = float(np.exp(la))
            fit_full = aa * n_full ** bb; proj = aa * (n_full + a.extra) ** bb
            row.update(a=aa, b=float(bb), fit_at_full=fit_full, proj_at_full_plus=proj, dAP=proj - fit_full,
                       d_mAP50=(proj - fit_full) / 5)
            if bb <= 0:
                flags.append("AP not rising with n (b <= 0)")
        else:
            flags.append("fit impossible (< 2 positive points)")
        if len(pos) < 4:
            flags.append(f"only {len(pos)} positive points")
        if row["extrap_ratio"] > 2:
            flags.append("unreliable extrapolation (> 2x largest measured n)")
        row["flags"] = "; ".join(flags)
        rows.append(row)
    r = pd.DataFrame(rows).sort_values("d_mAP50", ascending=False)
    r.to_csv(out / "lc_per_class.csv", index=False)
    pd.set_option("display.width", 250); print(p.to_string(index=False)); print(r.to_string(index=False))


if __name__ == "__main__":
    main()
