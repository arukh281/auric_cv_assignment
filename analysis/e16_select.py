"""E16 decision (pre-registered + 2 amendments): ONE selection on holdout40-clean among
  final | final+E16 | final+seed1 | final+E16+seed1
(final = multi-label WBF of B1h ep40, E4 ep40, E7 ep20; E16 = last.pt at its own geometry, 512 tiles overlap 128 at
imgsz 1024; seed1 = b1h_seed1 last.pt; all multi-label, max_det 902, WBF equal weights IoU 0.55).
Pass: best candidate beats the final by > 0.017 on holdout40-clean AND on holdout40-clean without Liquid. If it passes,
val is scored once for it. Also reports E16 alone (single and multi) vs B1h alone on holdout40-clean.

  python analysis/e16_select.py --data-root "$DATA_ROOT" --clean-root /kaggle/tmp/xvr_data \
      --final b1h=<pt> e4=<pt> e7=<pt> --e16 <pt> --seed1 <pt> --out <dir> [--device 0]
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


def score(p, imgs, L, sizes):
    aps, m, _ = ap_from_records(score_images(collect(p[p.conf >= 0.001], imgs, L, sizes), 5), 5, "coco")
    return float(m), float(np.mean(aps[:4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True); ap.add_argument("--clean-root", required=True)
    ap.add_argument("--final", nargs="+", required=True); ap.add_argument("--e16", required=True)
    ap.add_argument("--seed1", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev = yaml.safe_load(open(REPO / "configs" / "b1h.yaml"))["eval"]
    ev16 = yaml.safe_load(open(REPO / "configs" / "e16_b1h_tile512_up2x.yaml"))["eval"]
    hold = [l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip()]
    himgs = [root / "train" / "images" / n for n in hold]; hs = image_sizes(himgs); LC = Path(a.clean_root) / "train" / "labels"

    def infer(w, imgs, sizes, multi, e):
        A = SimpleNamespace(weights=w, device=a.device, batch=16, mode="sliced", imgsz=e["imgsz"], tile=e["tile"], overlap=e["overlap"],
                            conf=e["conf"], nms_iou=e["nms_iou"], max_det=902, multi_label=multi)
        r = run_predictions(A, imgs); r = r[r.cls < 5]
        return finalize(r, sizes, e["merge"], e["merge_thr"], e["merge_metric"], 902)

    F = dict(x.split("=", 1) for x in a.final)
    P = {k: infer(w, himgs, hs, True, ev) for k, w in F.items()}
    P["e16"] = infer(a.e16, himgs, hs, True, ev16); P["seed1"] = infer(a.seed1, himgs, hs, True, ev)
    res = dict(e16_alone_multi_clean=score(P["e16"], himgs, LC, hs)[0],
               e16_alone_single_clean=score(infer(a.e16, himgs, hs, False, ev16), himgs, LC, hs)[0],
               b1h_alone_single_clean=0.1764591279449225)
    res["e16_hypothesis_passes"] = res["e16_alone_single_clean"] - res["b1h_alone_single_clean"] > 0.017
    fin = list(F)
    cands = {"final": fin, "final+e16": fin + ["e16"], "final+seed1": fin + ["seed1"], "final+e16+seed1": fin + ["e16", "seed1"]}
    rows = []
    for name, mem in cands.items():
        m, nl = score(fuse([P[k] for k in mem], hs), himgs, LC, hs)
        rows.append(dict(candidate=name, clean_mAP50=m, clean_no_liquid=nl)); print(rows[-1], flush=True)
    C = pd.DataFrame(rows); C.to_csv(out / "e16_candidates.csv", index=False)
    base = C[C.candidate == "final"].iloc[0]; best = C[C.candidate != "final"].sort_values("clean_mAP50", ascending=False).iloc[0]
    res.update(candidates=rows, chosen=best.candidate, passes=bool(best.clean_mAP50 - base.clean_mAP50 > 0.017 and best.clean_no_liquid > base.clean_no_liquid))
    if res["passes"]:
        vimgs = list_images(root / "val" / "images"); vs = image_sizes(vimgs); W = {**F, "e16": a.e16, "seed1": a.seed1}
        vp = [infer(W[k], vimgs, vs, True, ev16 if k == "e16" else ev) for k in cands[best.candidate]]
        pv = fuse(vp, vs); pv.to_csv(out / "val_predictions_e16_choice.csv", index=False)
        res["val_mAP50"], res["val_mAP50_no_liquid"] = score(pv, vimgs, root / "val" / "labels", vs)
    (out / "e16_select.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
