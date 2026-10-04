"""Figures for the human-readable EXPERIMENTS.md (figures/story/). CPU only; reads copies of committed result files
in analysis/inputs/story/ (no new numbers). Each output is kept under ~300 KB.

  python analysis/story_figures.py --out figures/story

  scoreboard.png        train40 vs holdout40 mAP50 per run (last.pt), with the y = x line: the overfitting story
  holdout_curves.png    holdout40 mAP50 at every 10-epoch checkpoint, one line per run with a curve file
  recall_by_size.png    B1h class-agnostic recall by box size, val vs holdout40 (conf 0.001)
  overfit_tile.png, label_mismatch_tile.png, fp_audit_sheet.png   downscaled copies of existing renders
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "analysis" / "inputs" / "story"
Image.MAX_IMAGE_PIXELS = None
RUNS = {"b1h_tile1024_holdout40": "B1h (baseline)", "e1_b1h_150ep": "E1 150 ep", "e2_b1h_150ep_scale02": "E2 150 ep, scale 0.2",
        "e3_b1h_dota": "E3 DOTA init", "e4_b1h_flipud_mixup": "E4 flipud+mixup", "e7_b1h_dota_frozen": "E7 DOTA frozen",
        "e6_b1h_rfs": "E6 rare-class resampling", "e8_b1h_flipud_mixup_100ep": "E8 E4 x 100 ep",
        "e1_b1h_150ep_holdout": "E1", "e2_b1h_150ep_scale02_holdout": "E2"}


def save(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def shrink(src, dst, max_kb=300):
    im = Image.open(src).convert("RGB")
    w = im.width
    while True:
        im2 = im if im.width <= w else im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        im2.save(dst, "JPEG" if dst.suffix == ".jpg" else "PNG", optimize=True, **({"quality": 82} if dst.suffix == ".jpg" else {}))
        if dst.stat().st_size <= max_kb * 1024 or w < 300:
            return
        w = int(w * 0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "figures" / "story"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for r, lab in RUNS.items():
        f1, f2 = IN / "metrics" / f"{r}__eval_train40.json", IN / "metrics" / f"{r}__eval_holdout40.json"
        if f1.exists() and f2.exists():
            rows.append((lab, json.load(open(f1))["mAP50"], json.load(open(f2))["mAP50"]))
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for lab, tr, ho in rows:
        ax.scatter(tr, ho, s=90, color="black" if lab.startswith("B1h") else "tab:blue", zorder=3)
        ax.annotate(f"{lab}\n{tr:.2f} / {ho:.3f}", (tr, ho), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.plot([0, 1], [0, 1], ls=":", color="grey"); ax.axhline(0.1507 + 0.017, ls="--", color="tab:red", lw=1)
    ax.text(0.02, 0.1507 + 0.02, "B1h + noise (0.168): the bar to beat", color="tab:red", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 0.3), xlabel="train40 mAP50 (images it trained on)",
           ylabel="holdout40 mAP50 (unseen train images)", title="Fitting more ≠ generalising more (last.pt)")
    save(fig, out / "scoreboard.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for f in sorted((IN / "curves").glob("*.csv")):
        t = pd.read_csv(f)
        lab = RUNS.get(f.stem, f.stem)
        ax.plot(t.epoch, t.mAP50, marker="o", lw=2.5 if f.stem.startswith("b1h") else 1.5,
                color="black" if f.stem.startswith("b1h") else None, label=lab)
    ax.set(xlabel="epoch", ylabel="holdout40 mAP50", title="Holdout40 at every checkpoint (descriptive)")
    ax.legend(fontsize=8)
    save(fig, out / "holdout_curves.png")

    r = pd.read_csv(IN / "recall_by_factor.csv").query("factor == 'size'")
    order = ["[0.0, 8.0)", "[8.0, 16.0)", "[16.0, 24.0)", "[24.0, 32.0)", "[32.0, 48.0)", "[48.0, 96.0)"]
    p = r.pivot(index="bin", columns="split", values="recall_0.001").reindex(order)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    p[["holdout", "val"]].plot.bar(ax=ax, color=["tab:green", "tab:orange"])
    ax.set(xlabel="box size, sqrt(area) px", ylabel="trucks found (any class)", ylim=(0, 1),
           title="Same size, fewer found on val: a scene gap, not a size gap")
    ax.set_xticklabels([b.replace(".0", "") for b in order], rotation=0, fontsize=8)
    save(fig, out / "recall_by_size.png")

    for src, dst in (("overfit.png", "overfit_tile.jpg"), ("label_mismatch.png", "label_mismatch_tile.jpg"),
                     ("fp_audit.png", "fp_audit_sheet.jpg")):
        shrink(IN / "img" / src, out / dst)
    for f in sorted(out.iterdir()):
        print(f.name, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
