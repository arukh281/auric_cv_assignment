"""Small facts for the final write-up (CPU): per-image median truck size in TRAIN (sqrt area, px), its 5th-95th
percentile across images; and per-class val AP50 of the final system vs B1h (from saved per_class files).

  python analysis/final_diag_extra.py --final-per-class analysis/inputs/final/final_val_per_class.csv --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final-per-class", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(REPO / "figures" / "eda" / "tables" / "boxes.csv")
    med = b.groupby(["split", "image"]).sqrt_area_px.median()
    tr = med.loc["train"]; va = med.loc["val"]
    res = dict(train_per_image_median_px=dict(p5=float(np.percentile(tr, 5)), p50=float(np.percentile(tr, 50)), p95=float(np.percentile(tr, 95)), n=len(tr)),
               val_per_image_median_px={k: float(v) for k, v in va.items()})
    f = pd.read_csv(a.final_per_class); bb = pd.read_csv(REPO / "analysis" / "inputs" / "no_liquid" / "b1h_tile1024_holdout40__val.csv")
    res["val_per_class"] = {r.class_name: dict(final=float(r.AP50_coco), b1h=float(bb[bb.class_name == r.class_name].AP50_coco.iloc[0])) for r in f.itertuples()}
    (out / "final_diag_extra.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
