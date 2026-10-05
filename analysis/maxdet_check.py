"""Detection-cap check for the final E13 system (rule in DETAILED_EXPERIMENTS.md, recorded before running).

  python analysis/maxdet_check.py --data-root "$DATA_ROOT" --clean-root /kaggle/tmp/xvr_data \
      --weights b1h=<pt> e4=<pt> e7=<pt> --out <dir> [--device 0]

For max_det in (902, 3000): each model is run with multi-label NMS and max_det used everywhere (per-tile predict,
per-model merge, top-k after WBF fusion); holdout40-clean mAP50 per model and fused; images hitting the cap.
If fused(3000) - fused(902) >= 0.005: val is scored once with 3000 (fused), with per-class AP50.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "analysis"))
from detlib.data import list_images  # noqa: E402
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from e13_ensemble import fuse  # noqa: E402
from eval import collect, finalize, image_sizes, run_predictions  # noqa: E402

NAMES = ["Cargo Truck", "Truck w/Box", "Truck w/Flatbed", "Truck Tractor", "Truck w/Liquid"]


def score(p, imgs, L, sizes):
    aps, m, _ = ap_from_records(score_images(collect(p[p.conf >= 0.001], imgs, L, sizes), 5), 5, "coco")
    return float(m), [float(x) for x in aps]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True); ap.add_argument("--clean-root", required=True)
    ap.add_argument("--weights", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W = dict(w.split("=", 1) for w in a.weights)
    ev = yaml.safe_load(open(REPO / "configs" / "b1h.yaml"))["eval"]
    hold = [l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip()]
    himgs = [root / "train" / "images" / n for n in hold]; hs = image_sizes(himgs); LC = Path(a.clean_root) / "train" / "labels"

    def run(imgs, sizes, md):
        per = {}
        for k, w in W.items():
            A = SimpleNamespace(weights=w, device=a.device, batch=16, mode="sliced", imgsz=ev["imgsz"], tile=ev["tile"],
                                overlap=ev["overlap"], conf=ev["conf"], nms_iou=ev["nms_iou"], max_det=md, multi_label=True)
            r = run_predictions(A, imgs); r = r[r.cls < 5]
            per[k] = finalize(r, sizes, ev["merge"], ev["merge_thr"], ev["merge_metric"], md)
        return per, fuse(list(per.values()), sizes, max_det=md)

    res = {}
    for md in (902, 3000):
        per, fused = run(himgs, hs, md)
        d = {k: score(p, himgs, LC, hs)[0] for k, p in per.items()}
        d["fused"], d["fused_per_class"] = score(fused, himgs, LC, hs)
        d["images_at_cap_fused"] = int((fused.groupby("image").size() >= md).sum())
        res[str(md)] = d; print(md, json.dumps(d), flush=True)
    gain = res["3000"]["fused"] - res["902"]["fused"]; res["gain"] = gain; res["adopt_3000"] = bool(gain >= 0.005)
    if res["adopt_3000"]:
        vimgs = list_images(root / "val" / "images"); vs = image_sizes(vimgs)
        _, fv = run(vimgs, vs, 3000); fv.to_csv(out / "val_predictions_maxdet3000.csv", index=False)
        m, pc = score(fv, vimgs, root / "val" / "labels", vs)
        res["val_maxdet3000"] = dict(mAP50=m, per_class=dict(zip(NAMES, pc)), images_at_cap=int((fv.groupby("image").size() >= 3000).sum()))
    (out / "maxdet_check.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
