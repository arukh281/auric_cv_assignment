"""Known-answer tests for analysis/per_image.py. Run: python tests/test_per_image.py (also pytest-compatible)."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
from per_image import flags_of, image_row, load_eval, subset_map  # noqa: E402
from test_pipeline import DATA, real_gt  # noqa: E402

VAL_EVAL = REPO / "results" / "b1_tile1024" / "eval"


def B(*r):
    return np.array(r, float).reshape(-1, 4)


def test_flags():
    assert flags_of("2308.png") == "blur;scale" and flags_of("2470.png") == "haze;label_issue"
    assert flags_of("31.png") == "label_issue" and flags_of("2139.png") == "none"


def test_known_image_rows():
    g = B([0, 0, 10, 10], [20, 0, 30, 10])
    perfect = dict(image="x", p_xyxy=g.copy(), p_cls=np.array([0, 1]), p_conf=np.array([.9, .8]), g_xyxy=g,
                   g_cls=np.array([0, 1]))
    r = image_row(perfect)
    assert r["AP50_agnostic"] == 1 and r["mAP50_per_class"] == 1 and r["recall_conf0.25"] == 1
    assert r["n_bkg_fp_conf0.25"] == 0 and r["n_gt"] == 2
    swapped = dict(perfect, p_cls=np.array([1, 0]))           # right boxes, wrong classes
    r = image_row(swapped)
    assert r["AP50_agnostic"] == 1 and r["mAP50_per_class"] == 0 and r["recall_conf0.25"] == 0
    assert r["recall_agnostic_conf0.25"] == 1
    extra = dict(perfect, p_xyxy=np.vstack([g, [500, 500, 510, 510]]), p_cls=np.array([0, 1, 0]),
                 p_conf=np.array([.9, .8, .3]))
    low = dict(perfect, p_conf=np.array([.9, .1]))
    assert image_row(extra)["n_bkg_fp_conf0.25"] == 1
    assert image_row(low)["recall_conf0.25"] == 0.5 and image_row(low)["recall_conf0.001"] == 1


def test_coco_gt_matches_val_labels_and_metrics():
    if not (VAL_EVAL / "coco_gt.json").exists():
        print("  skip: no results/b1_tile1024/eval")
        return
    per = {d["image"]: d for d in load_eval(VAL_EVAL)}
    from detlib.data import list_images
    for p, (w, h, c, b) in zip(list_images(DATA / "val" / "images"), real_gt()):
        d = per[p.name]
        assert np.array_equal(d["g_cls"], c) and np.allclose(d["g_xyxy"], b, atol=1e-4), p.name
    m = json.loads((VAL_EVAL / "metrics.json").read_text())["mAP50"]
    assert abs(subset_map(list(per.values()), [True] * len(per), n_boot=10)["mAP50"] - m) < 1e-12


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
