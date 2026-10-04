"""§5.4 smart vs random subsets (S54 pre-registration in DETAILED_EXPERIMENTS.md). CPU only, from saved predictions.

Usage:
  python analysis/subset_compare.py [--results-root results] [--out figures/subset_compare]

Runs (missing ones are skipped): random b1h_f50 / b1h_f50_seed1 / b1h_f75 / b1h_f75_seed1, smart b1h_smart50 /
b1h_smart75, and the full-data reference b1h_tile1024_holdout40 (seed 0) / b1h_seed1. Metrics per run and split
(val = eval/, holdout = eval_holdout40/) exactly as analysis/learning_curve.py: mAP50 and class-agnostic AP50 with
95% percentile bootstrap CIs over images (1000 resamples, seed 0), plus held-out mAP50 as % of the full-data held-out
mAP50 (0.1420 = mean of the two full-data seeds, the S54 reference).

Pre-registered rule, applied mechanically per size (reported as numbers, no interpretation):
  seed spread       |random seed 0 - random seed 1| on held-out class-agnostic AP50 (primary) and mAP50 (secondary)
  margin            smart - max(random seed 0, random seed 1)
  exceeds           margin > seed spread
  recovers_90pct    held-out mAP50 >= 0.90 x 0.1420 = 0.128 (every run)

Writes figures/subset_compare/: subset_compare.csv (one row per run), decision_rule.csv (one row per size and
metric), subset_compare.png.
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
from learning_curve import run_metrics  # noqa: E402
from per_image import load_eval  # noqa: E402

FULL_HOLDOUT_MAP50 = 0.1420
RECOVER = round(0.90 * FULL_HOLDOUT_MAP50, 3)    # 0.128, as pre-registered
RUNS = [("b1h_f50", 202, "random", 0), ("b1h_f50_seed1", 202, "random", 1), ("b1h_smart50", 202, "smart", 0),
        ("b1h_f75", 302, "random", 0), ("b1h_f75_seed1", 302, "random", 1), ("b1h_smart75", 302, "smart", 0),
        ("b1h_tile1024_holdout40", 403, "full", 0), ("b1h_seed1", 403, "full", 1)]
SPLITS = [("val", "eval"), ("holdout", "eval_holdout40")]


def decision(rows):
    """rows: DataFrame with size, selection, seed and holdout_<metric> columns."""
    out = []
    for size in sorted(rows[rows.selection == "smart"]["size"].unique()):
        g = rows[rows["size"] == size]
        for metric, role in (("holdout_AP50_agnostic", "primary"), ("holdout_mAP50", "secondary")):
            rnd = g[g.selection == "random"].set_index("seed")[metric]
            sm = g[g.selection == "smart"][metric]
            if len(rnd) < 2 or sm.empty:
                out.append(dict(size=size, metric=metric, role=role, note="missing runs"))
                continue
            spread = abs(rnd.get(0) - rnd.get(1))
            margin = float(sm.iloc[0]) - float(rnd.max())
            out.append(dict(size=size, metric=metric, role=role, smart=float(sm.iloc[0]), random_seed0=rnd.get(0),
                            random_seed1=rnd.get(1), seed_spread=spread, margin_over_best_random=margin,
                            exceeds_both_by_more_than_spread=bool(margin > spread)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="figures/subset_compare")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, size, sel, seed in RUNS:
        r = dict(run=run, size=size, selection=sel, seed=seed)
        ok = True
        for split, d in SPLITS:
            ev = Path(a.results_root) / run / d
            if not (ev / "predictions.csv").exists():
                print(f"[subset] skip {run}: no {d}/predictions.csv")
                ok = False
                break
            m = run_metrics(load_eval(ev), a.bootstrap)
            mj = json.loads((ev / "metrics.json").read_text())
            assert abs(m["mAP50"][0] - mj["mAP50"]) < 1e-9, (run, d)
            for k in ("mAP50", "AP50_agnostic"):
                v, bd = m[k]
                lo, hi = np.nanpercentile(bd, [2.5, 97.5])
                r.update({f"{split}_{k}": v, f"{split}_{k}_ci95_lo": lo, f"{split}_{k}_ci95_hi": hi})
        if not ok:
            continue
        r["holdout_mAP50_pct_of_full"] = 100 * r["holdout_mAP50"] / FULL_HOLDOUT_MAP50
        r["recovers_90pct"] = bool(r["holdout_mAP50"] >= RECOVER)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "subset_compare.csv", index=False)
    dr = decision(t) if len(t) else pd.DataFrame()
    dr.to_csv(out / "decision_rule.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("holdout_AP50_agnostic", "holdout_mAP50")):
        for k, (_, r) in enumerate(t.iterrows()):
            col = dict(random="#2a6fbb", smart="#d1495b", full="#666")[r.selection]
            ax.errorbar(k, r[metric], yerr=[[r[metric] - r[f"{metric}_ci95_lo"]], [r[f"{metric}_ci95_hi"] - r[metric]]],
                        fmt="o", color=col, capsize=3)
        ax.set_xticks(range(len(t)), [f"{r.size} {r.selection} s{r.seed}" for _, r in t.iterrows()], rotation=45,
                      ha="right", fontsize=7)
        if metric == "holdout_mAP50":
            ax.axhline(RECOVER, ls="--", color="#888", lw=1, label=f"90% of full ({RECOVER})")
            ax.legend(fontsize=7)
        ax.set_title(metric, fontsize=9)
    fig.suptitle("S54: smart vs random subsets (held-out, 95% bootstrap CIs over images)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "subset_compare.png", dpi=110); plt.close(fig)
    cols = ["run", "size", "selection", "seed", "val_mAP50", "val_AP50_agnostic", "holdout_mAP50",
            "holdout_AP50_agnostic", "holdout_mAP50_pct_of_full", "recovers_90pct"]
    if len(t):
        print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(dr):
        print(dr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[subset] wrote {out}")


if __name__ == "__main__":
    main()
