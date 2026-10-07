"""Plain-language example sheet: real final-system predictions on val at conf >= 0.10.

Usage: python analysis/example_sheet.py [--data-root data] [--out figures/simple]
Rows: found and named correctly / missed (no box) / found, but wrong type. Four crops per row, chosen with a fixed
seed among all eligible trucks (no hand-picking); trucks whose 200-px crop would leave the image are skipped.
Writes 7_examples.png/.svg and 7_examples.json (the chosen trucks, for traceability).
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

PREDS = "results/e13_final/val_predictions_final.csv"
NAMES = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]
GREEN, ORANGE, DARK = "#1a9850", "#f46d43", "#3a3f44"
CONF, IOU, HALF, SEED = 0.10, 0.5, 100, 0

plt.rcParams.update({"font.size": 13, "figure.facecolor": "white", "text.color": DARK, "svg.fonttype": "none"})


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="figures/simple")
    a = ap.parse_args()
    root, out = Path(a.data_root) / "val", Path(a.out)

    preds = defaultdict(list)
    for r in csv.DictReader(open(PREDS)):
        if float(r["conf"]) >= CONF:
            preds[r["image"]].append((int(r["cls"]), float(r["conf"]),
                                      [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))

    cats = {"ok": [], "missed": [], "wrong": []}
    for img_path in sorted((root / "images").glob("*.png")):
        w, h = Image.open(img_path).size
        for line in open(root / "labels" / f"{img_path.stem}.txt").read().split("\n"):
            if not line.strip():
                continue
            c, xc, yc, bw, bh = map(float, line.split())
            g = [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
            cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            if cx - HALF < 0 or cy - HALF < 0 or cx + HALF > w or cy + HALF > h:
                continue  # crop would be cut off by the image edge
            hits = [p for p in preds[img_path.name] if iou(g, p[2]) >= IOU]
            item = {"image": img_path.name, "true": int(c), "gt": g, "center": [cx, cy]}
            if not hits:
                cats["missed"].append(item)
                continue
            same = [p for p in hits if p[0] == int(c)]
            best = max(same or hits, key=lambda p: p[1])
            item["pred"] = {"cls": best[0], "conf": best[1], "box": best[2]}
            cats["ok" if same else "wrong"].append(item)

    rng = random.Random(SEED)
    chosen = {k: rng.sample(v, 4) for k, v in cats.items()}
    rows = [("Found and named correctly", "ok"), ("Missed (no box)", "missed"), ("Found, but wrong type", "wrong")]

    fig, axes = plt.subplots(3, 4, figsize=(13, 11.5))
    for r, (title, key) in enumerate(rows):
        for c, it in enumerate(chosen[key]):
            ax = axes[r][c]
            cx, cy = it["center"]
            x0, y0 = int(cx - HALF), int(cy - HALF)
            ax.imshow(Image.open(root / "images" / it["image"]).crop((x0, y0, x0 + 2 * HALF, y0 + 2 * HALF)))
            g = it["gt"]
            ax.add_patch(Rectangle((g[0] - x0, g[1] - y0), g[2] - g[0], g[3] - g[1], fill=False, ec=GREEN, lw=2.5))
            label = f"True: {NAMES[it['true']]} / Model: "
            if "pred" in it:
                p = it["pred"]["box"]
                ax.add_patch(Rectangle((p[0] - x0, p[1] - y0), p[2] - p[0], p[3] - p[1], fill=False, ec=ORANGE,
                                       lw=2, ls="--"))
                label += f"{NAMES[it['pred']['cls']]} ({it['pred']['conf']:.2f})"
            else:
                label += "no box"
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(label, fontsize=12)
            if c == 0:
                ax.set_ylabel(title, fontsize=14, fontweight="bold")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, ec=GREEN, lw=2.5, label="True truck"),
                        Rectangle((0, 0), 1, 1, fill=False, ec=ORANGE, lw=2, ls="--", label="Model's box")],
               loc="upper right", frameon=False, ncol=2, fontsize=13)
    fig.suptitle("What the final system gets right and wrong", fontsize=16, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "7_examples.png", dpi=200)
    fig.savefig(out / "7_examples.svg")
    json.dump({"conf": CONF, "seed": SEED, "eligible": {k: len(v) for k, v in cats.items()}, "chosen": chosen},
              open(out / "7_examples.json", "w"), indent=1)
    print({k: len(v) for k, v in cats.items()})


if __name__ == "__main__":
    main()
