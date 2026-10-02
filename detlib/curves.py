"""Per-run training summary: iteration counts and per-epoch loss / mAP curves from Ultralytics results.csv."""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from detlib.data import IMG_EXTS


def summarize_training(run_dir):
    """Write <run>/training_summary.json and <run>/training_curves.png. Safe to call repeatedly."""
    run_dir = Path(run_dir)
    tdir = run_dir / "train"
    args = yaml.safe_load(open(tdir / "args.yaml"))
    res = pd.read_csv(tdir / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    data = yaml.safe_load(open(args["data"]))
    train_dir = Path(data["path"]) / data["train"]
    n_train = sum(1 for p in train_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    batch = int(args["batch"])
    per_epoch = math.ceil(n_train / batch)  # Ultralytics' train loader does not drop the last partial batch
    summary = dict(
        train_images_or_tiles=n_train, batch_used=batch, imgsz=args["imgsz"], epochs_configured=args["epochs"],
        epochs_completed=int(len(res)), iterations_per_epoch=per_epoch, total_iterations=per_epoch * int(len(res)),
        images_seen=n_train * int(len(res)),
        note="per-epoch val metrics are Ultralytics' own val on un-sliced full val images; for curves only, "
             "not the reported metric and not used to pick a checkpoint (last.pt is evaluated)",
    )
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    groups = [("box_loss", "box loss"), ("cls_loss", "cls loss"), ("dfl_loss", "dfl loss")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (key, title) in zip(axes, groups):
        for split in ("train", "val"):
            col = f"{split}/{key}"
            if col in res:
                ax.plot(res.epoch, res[col], label=split)
        ax.set(title=title, xlabel="epoch"); ax.legend()
    for col, lab in (("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")):
        if col in res:
            axes[3].plot(res.epoch, res[col], label=lab)
    axes[3].set(title="Ultralytics val (un-sliced, curves only)", xlabel="epoch"); axes[3].legend()
    fig.suptitle(f"{run_dir.name}: {summary['total_iterations']} iterations, batch {batch}, {n_train} train images/tiles")
    fig.tight_layout(); fig.savefig(run_dir / "training_curves.png", dpi=110); plt.close(fig)
    res.to_csv(run_dir / "training_curves.csv", index=False)
    return summary
