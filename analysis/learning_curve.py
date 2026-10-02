"""Learning curves (§5.3) from saved predictions: metric vs number of training images, val and holdout40. CPU only.

Usage:
  python analysis/learning_curve.py [--results-root results] [--out figures/learning_curve]

Runs (missing ones are skipped): b1h_f25 (101 images), b1h_f50 (202), b1h_f75 (302), b1h_tile1024_holdout40 (403,
seed 0) and b1h_seed1 (403, seed 1). All share the holdout list, eval settings (sliced, max_det 902) and total
iterations (splits/learning_curve_seed0.json). Per run and split ("val" = eval/, "holdout" = eval_holdout40/), from
predictions.csv + the coco_gt.json eval.py wrote (analysis/per_image.load_eval), with detlib/scoring.py:
  mAP50, AP50 per class, AP50_agnostic (all classes merged into one).
95% CIs: percentile bootstrap over images (--bootstrap resamples, seed 0), as eval.py.

Power-law fit per metric and split: y(n) = a * n^b, least squares in log-log space on the curve points, with the 403
point = mean of the two seeds when both exist. Extrapolation to n = 403 + 500 = 903. Its uncertainty is the 2.5-97.5
percentile of refits on the bootstrap replicates (each point's bootstrap draws; the 403 point uses the mean of the two
seeds' draws), so it carries the image-sampling noise of every point but NOT seed noise, which is reported
separately as |seed0 - seed1| at 403. The fit is flagged unreliable, with the reasons listed, if any of:
  fewer than 3 points, a point <= 0 (no log), the curve is not monotone non-decreasing, the extrapolated 95% interval
  is wider than the extrapolated value, or the target n is more than 2x the largest observed n.
(903 / 403 = 2.24, so every +500 extrapolation here carries the last flag.)

Writes figures/learning_curve/: learning_curve.csv (run, n_images, seed, split, metric, value, ci95_lo, ci95_hi),
power_law_fit.csv (per split and metric: a, b, fitted y at 403, extrapolated y at 903 with its bootstrap interval,
gain 403 -> 903, seed spread at 403, reliability flags), learning_curve.png.
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
from per_image import agnostic, load_eval  # noqa: E402

NAMES = ["Cargo Truck", "Truck w/Box", "Truck w/Flatbed", "Truck Tractor", "Truck w/Liquid"]
NC = len(NAMES)
RUNS = [("b1h_f25", 101, 0), ("b1h_f50", 202, 0), ("b1h_f75", 302, 0), ("b1h_tile1024_holdout40", 403, 0),
        ("b1h_seed1", 403, 1)]
SPLITS = [("val", "eval"), ("holdout", "eval_holdout40")]
METRICS = ["mAP50", "AP50_agnostic"] + [f"AP50 {n}" for n in NAMES]
N_FULL, N_TARGET = 403, 903


def run_metrics(per, n_boot, seed=0):
    """{metric: (value, bootstrap draws)} for one run/split."""
    recs = score_images(per, NC)
    aps, m, _ = ap_from_records(recs, NC, "coco")
    b = bootstrap(recs, NC, n_boot, seed, "coco")
    recs_a = score_images([agnostic(d) for d in per], 1)
    _, ma, _ = ap_from_records(recs_a, 1, "coco")
    ba = bootstrap(recs_a, 1, n_boot, seed, "coco")
    out = {"mAP50": (m, b[:, -1]), "AP50_agnostic": (ma, ba[:, -1])}
    out.update({f"AP50 {NAMES[c]}": (aps[c], b[:, c]) for c in range(NC)})
    return out


def fit_power(n, y):
    """Least squares of log y = log a + b log n. Returns (a, b) or None if any y <= 0 or fewer than 2 points."""
    n, y = np.asarray(n, float), np.asarray(y, float)
    if len(n) < 2 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        return None
    b, la = np.polyfit(np.log(n), np.log(y), 1)
    return float(np.exp(la)), float(b)


def fit_with_uncertainty(ns, ys, draws, target=N_TARGET, full=N_FULL):
    """ns, ys: curve points; draws: list of bootstrap arrays (same length) per point. Returns dict."""
    flags = []
    if len(ns) < 3:
        flags.append(f"only {len(ns)} points")
    if any(not np.isfinite(v) or v <= 0 for v in ys):
        flags.append("a point is <= 0 (power law needs positive values)")
    if np.any(np.diff(np.asarray(ys)[np.argsort(ns)]) < 0):
        flags.append("curve not monotone non-decreasing")
    if target > 2 * max(ns):
        flags.append(f"target n={target} is {target / max(ns):.2f}x the largest observed n={max(ns)}")
    p = fit_power(ns, ys)
    res = dict(n_points=len(ns), a=np.nan, b=np.nan, fitted_at_403=np.nan, extrapolated_at_903=np.nan,
               extrap_ci95_lo=np.nan, extrap_ci95_hi=np.nan, gain_403_to_903=np.nan)
    if p is None:
        res.update(reliable=False, flags="; ".join(flags + ["no fit"]))
        return res
    a, b = p
    ext = a * target ** b
    reps = []
    for k in range(len(draws[0])):
        q = fit_power(ns, [d[k] for d in draws])
        if q is not None:
            reps.append(q[0] * target ** q[1])
    lo, hi = (np.percentile(reps, [2.5, 97.5]) if len(reps) >= 10 else (np.nan, np.nan))
    if len(reps) < 0.9 * len(draws[0]):
        flags.append(f"only {len(reps)}/{len(draws[0])} bootstrap refits possible")
    if np.isfinite(hi - lo) and hi - lo > ext:
        flags.append("extrapolated 95% interval wider than the extrapolated value")
    res.update(a=a, b=b, fitted_at_403=a * full ** b, extrapolated_at_903=ext, extrap_ci95_lo=float(lo),
               extrap_ci95_hi=float(hi), gain_403_to_903=ext - a * full ** b,
               reliable=not flags, flags="; ".join(flags) if flags else "")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="figures/learning_curve")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, draws = [], {}
    for run, n, seed in RUNS:
        for split, d in SPLITS:
            ev = Path(a.results_root) / run / d
            if not (ev / "predictions.csv").exists():
                print(f"[lc] skip {run}/{d}: no predictions.csv")
                continue
            mj = json.loads((ev / "metrics.json").read_text())
            for metric, (v, bd) in run_metrics(load_eval(ev), a.bootstrap).items():
                lo, hi = np.nanpercentile(bd, [2.5, 97.5])
                rows.append(dict(run=run, n_images=n, seed=seed, split=split, metric=metric, value=v,
                                 ci95_lo=lo, ci95_hi=hi))
                draws[(run, split, metric)] = bd
            got = [r["value"] for r in rows if r["run"] == run and r["split"] == split and r["metric"] == "mAP50"][0]
            if abs(got - mj["mAP50"]) > 1e-9:
                print(f"[lc] WARNING {run}/{d}: mAP50 {got} differs from metrics.json {mj['mAP50']}")
    t = pd.DataFrame(rows)
    t.to_csv(out / "learning_curve.csv", index=False)

    fits = []
    for split, _ in SPLITS:
        for metric in METRICS:
            s = t[(t.split == split) & (t.metric == metric)] if len(t) else t
            if s.empty:
                continue
            ns, ys, ds = [], [], []
            for n in sorted(s.n_images.unique()):
                g = s[s.n_images == n]
                ns.append(int(n))
                ys.append(float(g.value.mean()))
                ds.append(np.nanmean([draws[(r, split, metric)] for r in g.run], axis=0))
            f = fit_with_uncertainty(ns, ys, ds)
            full = s[s.n_images == N_FULL].set_index("seed").value
            f.update(split=split, metric=metric, points=json.dumps(dict(zip(ns, [round(y, 4) for y in ys]))),
                     seed0_at_403=full.get(0, np.nan), seed1_at_403=full.get(1, np.nan),
                     seed_spread_at_403=abs(full.get(0, np.nan) - full.get(1, np.nan)))
            fits.append(f)
    fits = pd.DataFrame(fits)
    cols = ["split", "metric", "points", "n_points", "a", "b", "fitted_at_403", "extrapolated_at_903", "extrap_ci95_lo",
            "extrap_ci95_hi", "gain_403_to_903", "seed0_at_403", "seed1_at_403", "seed_spread_at_403", "reliable", "flags"]
    fits = fits[cols] if len(fits) else fits
    fits.to_csv(out / "power_law_fit.csv", index=False)

    fig, axes = plt.subplots(len(SPLITS), len(METRICS), figsize=(3.1 * len(METRICS), 3.3 * len(SPLITS)),
                             squeeze=False, sharex=True)
    for i, (split, _) in enumerate(SPLITS):
        for j, metric in enumerate(METRICS):
            ax = axes[i][j]
            s = t[(t.split == split) & (t.metric == metric)] if len(t) else t
            for seed, mk, col in ((0, "o", "#2a6fbb"), (1, "s", "#d1495b")):
                g = s[s.seed == seed] if len(s) else s
                if len(g):
                    x = g.n_images + (6 if seed else 0)
                    ax.errorbar(x, g.value, yerr=[g.value - g.ci95_lo, g.ci95_hi - g.value], fmt=mk + ("-" if not seed else ""),
                                color=col, ms=4, capsize=2, lw=1, label=f"seed {seed}")
            f = fits[(fits.split == split) & (fits.metric == metric)] if len(fits) else fits
            if len(f) and np.isfinite(f.a.iloc[0]):
                xx = np.linspace(80, N_TARGET, 100)
                ax.plot(xx, f.a.iloc[0] * xx ** f.b.iloc[0], "--", color="#888", lw=1,
                        label="power law" + ("" if f.reliable.iloc[0] else " (unreliable)"))
                ax.errorbar([N_TARGET], [f.extrapolated_at_903.iloc[0]],
                            yerr=[[f.extrapolated_at_903.iloc[0] - f.extrap_ci95_lo.iloc[0]],
                                  [f.extrap_ci95_hi.iloc[0] - f.extrapolated_at_903.iloc[0]]],
                            fmt="D", color="#888", ms=4, capsize=2)
            ax.set_title(f"{split}: {metric}", fontsize=8)
            ax.set_ylim(0, 1)
            if i == len(SPLITS) - 1:
                ax.set_xlabel("training images", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Learning curves (B1h family, equal total iterations); 95% bootstrap CIs over images; "
                 "grey diamond = power-law extrapolation to 903 images", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=110)
    plt.close(fig)
    if len(t):
        print(t[t.metric.isin(["mAP50", "AP50_agnostic"])].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(fits):
        print(fits[["split", "metric", "points", "extrapolated_at_903", "extrap_ci95_lo", "extrap_ci95_hi",
                    "gain_403_to_903", "seed_spread_at_403", "reliable", "flags"]].to_string(index=False,
                                                                                            float_format=lambda v: f"{v:.4f}"))
    print(f"[lc] wrote {out}")


if __name__ == "__main__":
    main()
