"""mAP50 vs epoch with the REAL metric: run eval.py (sliced for B1, full-image for B0, same scorer) on every kept
checkpoint (train/weights/epochNNN.pt, plus last.pt if it is a later epoch).

Usage:
  python analysis/checkpoint_curve.py --config configs/b1.yaml --runs-root /content/drive/MyDrive/auric/runs \
      --data-root /content/data --out-root /content/drive/MyDrive/auric/runs/figures

Per checkpoint, eval.py writes <run>/checkpoint_curve/epNNN/ (bootstrap and pycocotools skipped for speed;
already-evaluated checkpoints with an unchanged sha256 are reused). Summary: figures/<name>/checkpoint_curve.csv
and .png (overall mAP50 and per-class AP50 vs completed epoch). Each eval also appends to <run>/command.txt.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from eval import sha256  # noqa: E402


def checkpoints(wdir, epochs_done):
    out = {int(re.search(r"(\d+)", p.stem).group(1)): p for p in wdir.glob("epoch*.pt")}
    last = wdir / "last.pt"
    if last.exists() and epochs_done and epochs_done not in out:
        out[epochs_done] = last
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-root", default="figures")
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", choices=["val", "holdout"], default="val",
                    help="holdout: score the config's holdout_list images; outputs go to checkpoint_curve_holdout/")
    a = ap.parse_args()
    sub = "checkpoint_curve" if a.split == "val" else "checkpoint_curve_holdout"
    cfg = yaml.safe_load(open(a.config))
    run = Path(a.runs_root) / cfg["name"]
    res = run / "train" / "results.csv"
    done = len(pd.read_csv(res)) if res.exists() else 0
    cks = checkpoints(run / "train" / "weights", done)
    if not cks:
        sys.exit(f"no checkpoints in {run / 'train' / 'weights'}")

    rows = []
    for ep, w in cks.items():
        out = run / sub / f"ep{ep:03d}"
        m = out / "metrics.json"
        h = sha256(w)
        if not (m.exists() and json.load(open(m)).get("weights_sha256") == h):
            cmd = [sys.executable, str(REPO / "eval.py"), "--config", a.config, "--runs-root", a.runs_root,
                   "--data-root", a.data_root, "--weights", str(w), "--out", str(out), "--bootstrap", "0",
                   "--no-coco-crosscheck", "--split", a.split] + (["--device", a.device] if a.device else [])
            print(f"[ckpt] epoch {ep}: {w.name}", flush=True)
            subprocess.run(cmd, check=True)
        r = json.load(open(m))
        rows.append(dict(epoch=ep, checkpoint=w.name, weights_sha256=r["weights_sha256"], mAP50=r["mAP50"],
                         **{f"AP50 {k}": v for k, v in r["per_class_AP50"].items()}))

    t = pd.DataFrame(rows)
    fig_dir = Path(a.out_root) / cfg["name"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(fig_dir / f"{sub}.csv", index=False)
    t.to_csv(run / sub / f"{sub}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in [c for c in t.columns if c.startswith("AP50 ")]:
        ax.plot(t.epoch, t[c], marker="o", lw=1, alpha=.7, label=c[5:])
    ax.plot(t.epoch, t.mAP50, marker="o", lw=2.5, color="black", label="mAP50")
    ax.set(xlabel="completed epoch", ylabel="AP50 (COCO 101-pt, our scorer)", ylim=(0, 1),
           title=f"{cfg['name']}: {cfg['eval']['mode']} eval per checkpoint ({a.split})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / f"{sub}.png", dpi=120); plt.close(fig)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
