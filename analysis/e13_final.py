"""E13 final (pre-registered E13 + amendments of 4 Oct): select ONE system on holdout40-clean, then score val once.

  python analysis/e13_final.py --data-root "$DATA_ROOT" --clean-root /kaggle/tmp/xvr_data \
      --run b1h=<run dir> --run e4=<dir> --run e7=<dir> --run e10=<dir> --run e15=<dir> --out <dir> [--device 0]

Per model and mode (single-label = saved eval_holdout40/predictions.csv; multi-label = fresh inference with
eval.run_predictions + NMS multi_label=True, same tiling/merge/max_det), only classes 0-4.
Candidates: every combo containing b1h (b1h alone, and b1h with any subset of the others), in each mode; ensembles by
WBF (equal weights, IoU 0.55, skip 0.001, top 902; analysis/e13_ensemble.fuse).
Selection: highest holdout40-clean mAP50. Pass rule: beats B1h single-label on holdout40-clean by > 0.017 AND beats it
on holdout40-clean mAP50 without Liquid. Supplied-label holdout40 is reported for every candidate.
If it passes: checkpoint rule - for each model in the chosen system, among epoch010..epoch040 and last.pt, the one
with the highest holdout40-clean mAP50 alone (in the chosen mode); the system is re-scored with those. Then val once.
If it does not pass: the final system is B1h, single-label, last.pt (val 0.1065, already reported); val is not
re-scored.
"""
import argparse
import itertools
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


def score(p, imgs, L, sizes):
    aps, m, _ = ap_from_records(score_images(collect(p[p.conf >= 0.001], imgs, L, sizes), 5), 5, "coco")
    return float(m), float(np.mean(aps[:4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True); ap.add_argument("--clean-root", required=True)
    ap.add_argument("--run", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    root, clean, out = Path(a.data_root), Path(a.clean_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = dict(r.split("=", 1) for r in a.run)
    ev = yaml.safe_load(open(REPO / "configs" / "b1h.yaml"))["eval"]
    hold = [l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip()]
    himgs = [root / "train" / "images" / n for n in hold]; hs = image_sizes(himgs)
    LS, LC = root / "train" / "labels", clean / "train" / "labels"

    def infer(weights, imgs, sizes, multi):
        A = SimpleNamespace(weights=str(weights), device=a.device, batch=16, mode="sliced", imgsz=ev["imgsz"], tile=ev["tile"],
                            overlap=ev["overlap"], conf=ev["conf"], nms_iou=ev["nms_iou"], max_det=ev["max_det"], multi_label=multi)
        r = run_predictions(A, imgs); r = r[r.cls < 5]
        return finalize(r, sizes, ev["merge"], ev["merge_thr"], ev["merge_metric"], ev["max_det"])

    P = {}
    for k, d in runs.items():
        s = pd.read_csv(Path(d) / "eval_holdout40" / "predictions.csv"); P[(k, "single")] = s[s.cls < 5]
        P[(k, "multi")] = infer(Path(d) / "train" / "weights" / "last.pt", himgs, hs, True)
    others = [k for k in runs if k != "b1h"]
    combos = [("b1h",) + c for r in range(0, len(others) + 1) for c in itertools.combinations(others, r)]
    rows = []
    for mode in ("single", "multi"):
        for c in combos:
            p = P[(c[0], mode)] if len(c) == 1 else fuse([P[(k, mode)] for k in c], hs)
            mc, nlc = score(p, himgs, LC, hs); ms, nls = score(p, himgs, LS, hs)
            rows.append(dict(mode=mode, combo="+".join(c), clean_mAP50=mc, clean_no_liquid=nlc, supplied_mAP50=ms, supplied_no_liquid=nls))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / "holdout40_candidates.csv", index=False)
    base = R[(R["mode"] == "single") & (R.combo == "b1h")].iloc[0]
    best = R.sort_values("clean_mAP50", ascending=False).iloc[0]
    res = dict(chosen_mode=best["mode"], chosen_combo=best.combo, chosen_clean=float(best.clean_mAP50),
               chosen_clean_no_liquid=float(best.clean_no_liquid), b1h_clean=float(base.clean_mAP50),
               b1h_clean_no_liquid=float(base.clean_no_liquid))
    res["passes"] = bool(res["chosen_clean"] - res["b1h_clean"] > 0.017 and res["chosen_clean_no_liquid"] > res["b1h_clean_no_liquid"])
    if res["passes"]:
        multi = best["mode"] == "multi"; members = best.combo.split("+"); ck = {}
        for k in members:
            wd = Path(runs[k]) / "train" / "weights"; cands = sorted(wd.glob("epoch*.pt")) + [wd / "last.pt"]
            sc = {w.name: score(infer(w, himgs, hs, multi), himgs, LC, hs)[0] for w in cands if w.exists()}
            ck[k] = dict(scores=sc, chosen=max(sc, key=sc.get))
        res["checkpoints"] = ck
        hp = [infer(Path(runs[k]) / "train" / "weights" / ck[k]["chosen"], himgs, hs, multi) for k in members]
        ph = hp[0] if len(hp) == 1 else fuse(hp, hs)
        res["final_holdout40_clean"], res["final_holdout40_clean_no_liquid"] = score(ph, himgs, LC, hs)
        res["final_holdout40_supplied"] = score(ph, himgs, LS, hs)[0]
        vimgs = list_images(root / "val" / "images"); vs = image_sizes(vimgs)
        vp = [infer(Path(runs[k]) / "train" / "weights" / ck[k]["chosen"], vimgs, vs, multi) for k in members]
        pv = vp[0] if len(vp) == 1 else fuse(vp, vs); pv.to_csv(out / "val_predictions_final.csv", index=False)
        res["val_mAP50"], res["val_mAP50_no_liquid"] = score(pv, vimgs, root / "val" / "labels", vs)
    (out / "e13_final.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
