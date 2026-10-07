"""Plain-language charts for EXPERIMENTS.md / REPORT.md, built only from existing result files.

Usage: python analysis/simple_figures.py [--root .] [--out figures/simple]
Writes six charts (PNG at 200 dpi + SVG) and numbers.json with every plotted value and its source file.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#1f6feb"   # the thing that matters
GREY = "#a0a4a8"     # everything else
DARK = "#3a3f44"     # text, target line
BAND = "#dbe7fb"     # "luck alone" band
NOISE = 0.017        # seed spread on the practice test (DETAILED_EXPERIMENTS.md)
TARGET = 0.75

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": DARK,
    "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
    "svg.fonttype": "none",
})

NUMBERS = {}


def metric(root, rel):
    v = json.load(open(root / rel))["mAP50"]
    NUMBERS[rel] = v
    return v


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def progress(root, out):
    rows = [("First try\n(whole photo shrunk)", "results/b0_full640/eval/metrics.json"),
            ("Tiles", "results/b1_tile1024/eval/metrics.json"),
            ("Tiles + own\npractice set", "results/b1h_tile1024_holdout40/eval/metrics.json"),
            ("Final\n(3 models voting)", "results/clean_repro_final/metrics.json")]
    vals = [metric(root, r) for _, r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [GREY, ACCENT, GREY, GREY]
    ax.bar(range(4), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=13)
    ax.axhline(TARGET, ls="--", color=DARK, lw=1.2)
    ax.text(3.35, TARGET + 0.015, "Target 0.75", ha="right", fontsize=12)
    ax.set_xticks(range(4), [n for n, _ in rows])
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Score on the official test\n(0 to 1, higher is better)")
    ax.set_title("Cutting photos into tiles was the biggest improvement")
    save(fig, out, "1_progress")


def memorise(root, out):
    runs = [("Baseline", "b1h_tile1024_holdout40"),
            ("Longer training", "e1_b1h_150ep"),
            ("Longer training,\ngentler zoom", "e2_b1h_150ep_scale02"),
            ("Aerial pretraining", "e3_b1h_dota"),
            ("Aerial pretraining,\nfrozen", "e7_b1h_dota_frozen"),
            ("More photo variety", "e4_b1h_flipud_mixup")]
    fig, ax = plt.subplots(figsize=(10, 6))
    base = metric(root, "results/b1h_tile1024_holdout40/eval_holdout40/metrics.json")
    ax.axvspan(base - NOISE, base + NOISE, color=BAND, zorder=0)
    ax.text(base + NOISE + 0.01, len(runs) - 1.45, "luck alone\n(around baseline)", fontsize=11, color=ACCENT)
    for i, (name, r) in enumerate(runs):
        y = len(runs) - 1 - i
        seen = metric(root, f"results/{r}/eval_train40/metrics.json")
        unseen = metric(root, f"results/{r}/eval_holdout40/metrics.json")
        ax.plot([unseen, seen], [y, y], color=GREY, lw=2, zorder=1)
        ax.scatter([seen], [y], color=GREY, s=80, zorder=2)
        ax.scatter([unseen], [y], color=ACCENT, s=80, zorder=3)
    ax.set_yticks(range(len(runs)), [n for n, _ in reversed(runs)])
    ax.scatter([], [], color=GREY, s=80, label="Photos it trained on")
    ax.scatter([], [], color=ACCENT, s=80, label="Photos it never saw")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (0 to 1, higher is better)")
    ax.set_title("Training harder made it memorise, not learn")
    save(fig, out, "2_memorise_vs_learn")


def mistakes(root, out):
    rel = "figures/b1h_tile1024_holdout40/errors/tide_dAP.csv"
    d = {r["fix"]: float(r["dAP50"]) for r in csv.DictReader(open(root / rel))}
    items = [("Wrong truck type", d["cls"]), ("False alarm (nothing there)", d["bkg"]),
             ("Missed truck", d["missed"]), ("Box slightly off", d["loc"])]
    NUMBERS[rel] = dict(items)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ys = range(len(items))[::-1]
    ax.barh(list(ys), [v for _, v in items], color=[ACCENT, GREY, GREY, GREY], height=0.6)
    for y, (_, v) in zip(ys, items):
        ax.text(v + 0.003, y, f"+{v:.3f}", va="center", fontsize=13)
    ax.set_yticks(list(ys), [n for n, _ in items])
    ax.set_xlim(0, 0.18)
    ax.set_xlabel("How much the score would rise if this mistake were fixed")
    ax.set_title("Naming the wrong truck type is the biggest mistake")
    save(fig, out, "3_mistakes")


def longer(root, out):
    rel = "results/e2_b1h_150ep_scale02/checkpoint_curve/checkpoint_curve.csv"
    rows = [(int(r["epoch"]), float(r["mAP50"])) for r in csv.DictReader(open(root / rel))]
    NUMBERS[rel] = rows
    ep, v = zip(*rows)
    k = max(range(len(v)), key=lambda i: v[i])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, v, color=ACCENT, lw=2.5, marker="o", ms=5)
    ax.annotate(f"Best at about {ep[k]} passes ({v[k]:.2f})", (ep[k], v[k]), xytext=(ep[k] + 15, v[k] + 0.012),
                fontsize=12, arrowprops=dict(arrowstyle="-", color=DARK))
    ax.annotate(f"After {ep[-1]} passes: {v[-1]:.2f}", (ep[-1], v[-1]), xytext=(ep[-1] - 40, v[-1] - 0.018),
                fontsize=12)
    ax.set_ylim(0, max(v) + 0.04)
    ax.set_xlabel("Training passes over the photos")
    ax.set_ylabel("Score on the official test\n(0 to 1, higher is better)")
    ax.set_title("Training longer made it worse")
    save(fig, out, "4_training_longer")


def learning(root, out):
    rel = "figures/learning_curve/learning_curve.csv"
    pts = {}
    for r in csv.DictReader(open(root / rel)):
        if r["split"] == "holdout" and r["metric"] == "mAP50" and r["seed"] == "0":
            pts[int(r["n_images"])] = float(r["value"])
    xs = sorted(pts)
    NUMBERS[rel] = {x: pts[x] for x in xs}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, [pts[x] for x in xs], color=ACCENT, lw=2.5, marker="o", ms=7)
    for x in xs:
        ax.text(x, pts[x] + 0.025, f"{pts[x]:.2f}", ha="center", fontsize=12)
    ax.axhline(TARGET, ls="--", color=DARK, lw=1.2)
    ax.text(xs[0], TARGET + 0.015, "Target 0.75", fontsize=12)
    ax.set_xticks(xs)
    ax.set_ylim(0, 0.82)
    ax.set_xlabel("Photos it trained on")
    ax.set_ylabel("Score on photos it never saw\n(0 to 1, higher is better)")
    ax.set_title("More photos help, but slowly")
    save(fig, out, "5_more_photos")


def confusion(root, out):
    rel = "figures/final_ensemble/errors_conf010/confusion_matrix_conf0.1.csv"
    names = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]
    rows = list(csv.reader(open(root / rel)))[1:6]
    m = [[int(x) for x in r[1:6]] for r in rows]
    share = [[c / sum(r) if sum(r) else 0 for c in r] for r in m]
    NUMBERS[rel] = m
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(share, cmap="Greys", vmin=0, vmax=1.6)
    for i in range(5):
        for j in range(5):
            hi = (i, j) in [(0, 1), (1, 0)]
            ax.text(j, i, f"{share[i][j]:.0%}", ha="center", va="center", fontsize=13,
                    color=ACCENT if hi else DARK, fontweight="bold" if hi else "normal")
            if hi:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ACCENT, lw=2.5))
    ax.set_xticks(range(5), names)
    ax.set_yticks(range(5), names)
    ax.set_xlabel("What the model called it")
    ax.set_ylabel("What the truck really was")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cargo <-> Box is the most frequent mix-up")
    save(fig, out, "6_confusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figures/simple")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (progress, memorise, mistakes, longer, learning, confusion):
        f(root, out)
    json.dump(NUMBERS, open(out / "numbers.json", "w"), indent=1, default=str)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
