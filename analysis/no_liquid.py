"""Post-hoc robustness check: mAP50 without Truck w/Liquid (mean AP50 over the other four classes), from
per_class.csv files. CPU only. Inputs: analysis/inputs/no_liquid/<run>__<split>.csv (copies of
results/<run>/{eval,eval_holdout40}/per_class.csv and results/tta_b1h/<set>/per_class.csv).

  python analysis/no_liquid.py [--out results/no_liquid]

Writes <out>/no_liquid.csv: run, split, mAP50 (5 classes), mAP50 without Liquid, Liquid AP50, Liquid n_gt.
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "analysis" / "inputs" / "no_liquid"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "results" / "no_liquid"))
    a = ap.parse_args()
    rows = []
    for f in sorted(IN.glob("*.csv")):
        run, split = f.stem.rsplit("__", 1)
        t = pd.read_csv(f)
        c = t[t.cls >= 0]
        liq = c[c.class_name == "Truck w/Liquid"].iloc[0]
        rows.append(dict(run=run, split=split, mAP50=float(t[t.cls == -1].AP50_coco.iloc[0]),
                         mAP50_no_liquid=float(c[c.class_name != "Truck w/Liquid"].AP50_coco.mean()),
                         liquid_AP50=float(liq.AP50_coco), liquid_n_gt=int(liq.n_gt)))
    r = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "no_liquid.csv", index=False)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
