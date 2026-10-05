"""E16 selection on CPU from saved per-model predictions (author's instruction, 5 Oct: GPU budget).

  infer:   python analysis/e16_parts.py infer --weights <pt> --config <yaml with eval geometry> --split holdout|val \
               --data-root "$DATA_ROOT" --out <preds.csv> [--device cpu]
           multi-label, max_det 902, the config's eval geometry; merged per model (finalize), classes 0-4 only.
  combine: python analysis/e16_parts.py combine --data-root "$DATA_ROOT" --clean-root <xvr_data> \
               --preds b1h=<csv> e4=<csv> e7=<csv> e16=<csv> seed1=<csv> --out <dir>
           WBF (equal weights, IoU 0.55, skip 0.001, top 902) for: final, final+e16, final+seed1, final+e16+seed1, scored on
           holdout40-clean (and without Liquid). Reproduction check: 'final' must equal 0.2364 (E13 final / maxdet check)
           within 0.0005 before the others are used. Pass: best non-final candidate > final + 0.017 AND better without
           Liquid. Writes e16_selection.json and candidates.csv.
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

FINAL_REF = 0.23639163751484532  # results/maxdet_check/maxdet_check.json, 902 fused


def imgs_for(root, split):
    if split == "val":
        return list_images(root / "val" / "images")
    hold = [l.strip() for l in (REPO / "splits" / "holdout40_seed0.txt").read_text().splitlines() if l.strip()]
    return [root / "train" / "images" / n for n in hold]


def score(p, imgs, L, sizes):
    aps, m, _ = ap_from_records(score_images(collect(p[p.conf >= 0.001], imgs, L, sizes), 5), 5, "coco")
    return float(m), float(np.mean(aps[:4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["infer", "combine"]); ap.add_argument("--data-root", required=True)
    ap.add_argument("--weights"); ap.add_argument("--config"); ap.add_argument("--split", default="holdout")
    ap.add_argument("--clean-root"); ap.add_argument("--preds", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); root = Path(a.data_root)
    if a.mode == "infer":
        ev = yaml.safe_load(open(a.config))["eval"]; imgs = imgs_for(root, a.split); sizes = image_sizes(imgs)
        A = SimpleNamespace(weights=a.weights, device=a.device, batch=16, mode="sliced", imgsz=ev["imgsz"], tile=ev["tile"],
                            overlap=ev["overlap"], conf=ev["conf"], nms_iou=ev["nms_iou"], max_det=902, multi_label=True)
        r = run_predictions(A, imgs); r = r[r.cls < 5]
        p = finalize(r, sizes, ev["merge"], ev["merge_thr"], ev["merge_metric"], 902)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); p.to_csv(a.out, index=False); print(len(p), "predictions"); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = {k: pd.read_csv(v) for k, v in (x.split("=", 1) for x in a.preds)}
    imgs = imgs_for(root, "holdout"); sizes = image_sizes(imgs); LC = Path(a.clean_root) / "train" / "labels"
    fin = ["b1h", "e4", "e7"]
    cands = {"final": fin, "final+e16": fin + ["e16"], "final+seed1": fin + ["seed1"], "final+e16+seed1": fin + ["e16", "seed1"]}
    rows = []
    for n, mem in cands.items():
        m, nl = score(fuse([P[k] for k in mem], sizes), imgs, LC, sizes); rows.append(dict(candidate=n, clean_mAP50=m, clean_no_liquid=nl))
    single = {k: score(P[k], imgs, LC, sizes)[0] for k in P}
    C = pd.DataFrame(rows); C.to_csv(out / "candidates.csv", index=False)
    f = C[C.candidate == "final"].iloc[0]
    res = dict(final_reproduced=float(f.clean_mAP50), final_reference=FINAL_REF, reproduction_ok=bool(abs(f.clean_mAP50 - FINAL_REF) < 0.0005),
               single_model_multi_clean=single, candidates=rows)
    if res["reproduction_ok"]:
        b = C[C.candidate != "final"].sort_values("clean_mAP50", ascending=False).iloc[0]
        res.update(chosen=b.candidate, margin=float(b.clean_mAP50 - f.clean_mAP50),
                   passes=bool(b.clean_mAP50 - f.clean_mAP50 > 0.017 and b.clean_no_liquid > f.clean_no_liquid))
    (out / "e16_selection.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
