"""Known-answer tests for analysis/errors.py and analysis/gt_box_oracle.py. Run: python tests/test_analysis.py
(also pytest-compatible). Uses data/val labels for realistic GT where noted; no trained weights needed.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from errors import annotate, apply_fix, f1_thresholds, tide_image, tide_table  # noqa: E402
from gt_box_oracle import assign_tiles, compare_modes, letterbox, oracle_tables, scores_at_boxes  # noqa: E402
from test_pipeline import DATA, NC, real_gt  # noqa: E402

NAMES = {c: f"c{c}" for c in range(NC)}
B = lambda *r: np.array(r, float).reshape(-1, 4)  # noqa: E731


def one_image(pb, pc, ps, gb, gc):
    return dict(image="x", p_xyxy=B(*pb) if len(pb) else B(), p_cls=np.array(pc, int), p_conf=np.array(ps, float),
                g_xyxy=B(*gb), g_cls=np.array(gc, int))


def test_tide_types_known():
    gt = [[0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10], [300, 0, 310, 10]]
    d = one_image([[0, 0, 10, 10],          # TP on GT0
                   [0, 0, 10, 10],          # same box, lower conf -> dupe of GT0
                   [100, 0, 110, 10],       # class 1 on a class-0 GT -> cls
                   [200, 0, 206, 10],       # IoU 0.6 with class-0 GT -> TP? no: class 0, GT2 class 0 -> TP
                   [203, 0, 213, 10],       # IoU 7/13 = 0.54, GT2 already taken -> dupe
                   [306, 0, 316, 10],       # IoU 4/16 = 0.25 same class -> loc
                   [500, 0, 510, 10],       # nowhere -> bkg
                   [106, 0, 116, 10]],      # class 2, IoU 0.25 with class-0 GT1 -> both
                  [0, 0, 1, 0, 0, 0, 0, 2], [.9, .8, .7, .95, .6, .5, .4, .3], gt, [0, 0, 0, 0])
    kind, ref, matched, missed = tide_image(d["p_xyxy"], d["p_cls"], d["p_conf"], d["g_xyxy"], d["g_cls"])
    assert list(kind) == ["TP", "dupe", "cls", "TP", "dupe", "loc", "bkg", "both"], kind
    assert list(ref) == [0, 0, 1, 2, 2, 3, -1, 1], ref
    assert list(matched) == [True, False, True, False] and list(missed) == [False, False, False, False]
    # GT3 not covered by loc any more if that prediction is gone -> missed
    k2 = tide_image(d["p_xyxy"][:5], d["p_cls"][:5], d["p_conf"][:5], d["g_xyxy"], d["g_cls"])
    assert list(k2[3]) == [False, False, False, True]


def _perfect(gt):
    return [dict(image=str(i), p_xyxy=b.copy(), p_cls=c.copy(), p_conf=np.linspace(1, 0.5, len(c)),
                 g_xyxy=b, g_cls=c) for i, (w, h, c, b) in enumerate(gt)]


def _inject(per, kind):
    """Add one error of `kind` above every true detection in the first image that has a GT of class 0."""
    d = next(x for x in per if (x["g_cls"] == 0).any())
    j = int(np.where(d["g_cls"] == 0)[0][0])
    g = d["g_xyxy"][j]
    w = g[2] - g[0]
    if kind == "bkg":
        nb, nc_ = np.array([-1000, -1000, -990, -990.]), 0
    elif kind == "dupe":
        nb, nc_ = g.copy(), 0
    elif kind in ("cls", "loc", "both", "missed"):
        # replace the TP of GT j: cls = same box other class; loc = shifted 0.7 w (IoU ~0.18); both = shifted, other class
        shift = np.array([0.7 * w, 0, 0.7 * w, 0]) if kind in ("loc", "both") else 0
        d["p_xyxy"][j] = g + shift
        d["p_cls"][j] = {"cls": 1, "both": 1}.get(kind, 0)
        if kind == "missed":
            d["p_xyxy"][j] = [-5000, -5000, -4990, -4990]  # becomes a bkg FP at its confidence too
        return
    d["p_xyxy"] = np.vstack([nb, d["p_xyxy"]])
    d["p_cls"] = np.concatenate([[nc_], d["p_cls"]])
    d["p_conf"] = np.concatenate([[2.0], d["p_conf"]])


def test_oracle_fix_restores_perfect_score():
    gt = real_gt()
    for kind in ("bkg", "dupe", "cls", "loc", "both", "missed"):
        per = annotate(_perfect(gt))
        _inject(per, kind)
        per = annotate(per)
        t = tide_table(per, NAMES).set_index("fix")
        assert t.loc["none (base)", "mAP50"] < 1 - 1e-6, kind
        assert t.loc[kind, "n_errors"] >= 1, (kind, t.n_errors)
        assert t.loc[kind, "dAP50"] > 0, (kind, t.dAP50)
        # bkg/dupe/cls/loc fixes alone restore 1. "both" removes the box that replaced the TP, so its GT stays
        # missed; "missed" removes the GT but leaves the moved box as a bkg FP. Each needs its partner fix.
        need = {"both": {"both", "missed"}, "missed": {"missed", "bkg"}}.get(kind, {kind})
        fixed = [apply_fix(d, need) for d in per]
        assert abs(ap_from_records(score_images(fixed, NC), NC)[1] - 1) < 1e-12, (kind, need)
        others = [k for k in ("bkg", "dupe", "cls", "loc", "both") if k != kind and t.loc[k, "n_errors"] == 0]
        assert all(abs(t.loc[k, "dAP50"]) < 1e-12 for k in others), (kind, t.dAP50)


def test_tide_priority_matches_reference():
    """tidecv/quantify.py order: Loc before Cls; IoU exactly fg with a used GT is Loc; max IoU exactly bg is Bkg."""
    gt = [[0, 0, 10, 10], [4, 0, 14, 10]]                                  # class 0 and class 1, overlapping
    # class-0 box on GT1: same-class IoU 6/14 = 0.43 (GT0), other-class IoU 1.0 (GT1) -> loc wins over cls
    k = tide_image(B([4, 0, 14, 10]), np.array([0]), np.array([.9]), B(*gt), np.array([1, 1]))[0]
    assert list(k) == ["cls"]  # sanity: no same-class GT -> cls
    k = tide_image(B([0, 0, 10, 10], [4, 0, 14, 10]), np.array([0, 0]), np.array([.9, .8]), B(*gt),
                   np.array([0, 1]))[0]
    assert list(k) == ["TP", "loc"], k
    # IoU exactly 0.5 with an already matched GT -> loc (TIDE uses <= fg), not dupe
    k = tide_image(B([0, 0, 10, 10], [0, 0, 10, 20]), np.array([0, 0]), np.array([.9, .8]), B([0, 0, 10, 10]),
                   np.array([0]))[0]
    assert list(k) == ["TP", "loc"], k
    # max IoU exactly bg (0.1) with any GT -> bkg; just above with another class -> both
    k = tide_image(B([0, 0, 10, 10]), np.array([1]), np.array([.9]), B([0, 0, 10, 2]), np.array([0]))[0]
    assert list(k) == ["both"], k  # other-class IoU 0.2
    k = tide_image(B([0, 0, 10, 10]), np.array([1]), np.array([.9]), B([0, 0, 10, 1]), np.array([0]))[0]
    assert list(k) == ["bkg"], k  # other-class IoU exactly 0.1


def test_f1_threshold_known():
    # class 0: 4 GT; predictions by confidence: TP .9, TP .8, FP .7, TP .6, FP .1
    # F1 at each cut: .4, .667, .571, .75, .667 -> threshold .6 with P = R = .75
    gb = B([0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10])
    pb = B([0, 0, 10, 10], [20, 0, 30, 10], [500, 0, 510, 10], [40, 0, 50, 10], [600, 0, 610, 10])
    per = [dict(image="x", p_xyxy=pb, p_cls=np.zeros(5, int), p_conf=np.array([.9, .8, .7, .6, .1]),
                g_xyxy=gb, g_cls=np.zeros(4, int))]
    t = f1_thresholds(per, NAMES, 0.25).set_index("cls")
    assert t.loc[0, "threshold"] == 0.6 and abs(t.loc[0, "f1"] - 0.75) < 1e-12 and t.loc[0, "recall"] == 0.75
    assert t.loc[1, "threshold"] == 0.25 and t.loc[1, "source"].startswith("fallback")
    # ties: two predictions at the same confidence are kept or dropped together
    per[0]["p_conf"] = np.array([.9, .8, .6, .6, .1])
    t = f1_thresholds(per, NAMES, 0.25).set_index("cls")
    assert t.loc[0, "threshold"] == 0.6 and abs(t.loc[0, "precision"] - 0.75) < 1e-12


def test_compare_modes_known():
    gt_cls = np.array([0, 0, 1, 1, 2])
    flagged = np.array([False, False, False, True, True])
    pool = np.eye(5)[[0, 0, 1, 0, 2]]   # wrong on GT 3 (flagged)
    best = np.eye(5)[[0, 1, 1, 0, 2]]   # wrong on GT 1 and 3
    cms, per, comp, agree = compare_modes(gt_cls, dict(pool=pool, best=best), flagged, NAMES)
    c = comp.set_index(["mode", "subset"])
    assert c.loc[("pool", "all"), "accuracy"] == 0.8 and c.loc[("best", "all"), "accuracy"] == 0.6
    assert c.loc[("pool", "unflagged"), "accuracy"] == 1.0 and c.loc[("best", "unflagged"), "accuracy"] == 2 / 3
    assert c.loc[("pool", "flagged"), "n_gt"] == 2 and c.loc[("pool", "flagged"), "accuracy"] == 0.5
    assert agree == dict(n=5, n_agree=4, agreement=0.8, n_agree_unflagged=2, n_unflagged=3)
    assert cms["best"][0, 1] == 1 and cms["pool"].trace() == 4


def test_cls_fix_drops_when_gt_already_matched():
    d = annotate([one_image([[0, 0, 10, 10], [0, 0, 10, 10]], [0, 1], [.9, .8], [[0, 0, 10, 10]], [0])])[0]
    assert list(d["kind"]) == ["TP", "cls"]
    f = apply_fix(d, {"cls"})
    assert len(f["p_cls"]) == 1 and f["p_cls"][0] == 0


def test_base_equals_scorer():
    rng = np.random.default_rng(0)
    per = []
    for w, h, c, b in real_gt():
        keep = rng.random(len(c)) < 0.7
        pb = b[keep] + rng.normal(0, 2, (keep.sum(), 4))
        per.append(dict(image="x", p_xyxy=pb, p_cls=c[keep], p_conf=rng.random(keep.sum()), g_xyxy=b, g_cls=c))
    t = tide_table(annotate(per), NAMES)
    assert abs(t.mAP50.iloc[0] - ap_from_records(score_images(per, NC), NC)[1]) < 1e-12


def test_letterbox_inverse():
    img = np.zeros((300, 1000, 3), np.uint8)
    img[100:150, 200:260] = 255
    lb, r, (px, py) = letterbox(img, 640)
    ys, xs = np.where(lb[:, :, 0] > 0.9)
    back = (np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) - [px, py, px, py]) / r
    assert np.allclose(back, [200, 100, 260, 150], atol=2), back
    assert abs(lb[0, 0, 0] - 114 / 255) < 1e-6 and r == 0.64


def test_scores_at_boxes_and_tables():
    anchors = B([0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60], [200, 200, 210, 210])
    sc = np.array([[.1, .7, .0, 0, 0], [.6, .2, 0, 0, 0], [0, 0, .9, 0, 0], [0, 0, 0, .3, .4]])
    gt = B([0, 0, 10, 10], [50, 50, 60, 60], [300, 300, 310, 310])
    s, sb, best, n = scores_at_boxes(anchors, sc, gt, 0.5)
    assert np.allclose(s[0], [.6, .7, 0, 0, 0]) and np.allclose(s[1], sc[2])
    assert np.allclose(sb[0], sc[0]) and np.allclose(sb[2], sc[0]) and np.allclose(s[2], sb[2])
    assert list(n) == [2, 1, 0] and best[2] == 0  # no overlap: falls back to argmax of IoU (anchor 0)
    gt_cls = np.array([1, 2, 0, 0, 3, 3, 4, 4])
    scores = np.eye(5)[[1, 2, 0, 1, 3, 4, 4, 4]]
    cm, per, summ = oracle_tables(gt_cls, scores, NAMES)
    assert summ["accuracy"] == 6 / 8 and cm[0, 1] == 1 and cm[3, 4] == 1
    assert list(per.accuracy) == [0.5, 1.0, 1.0, 0.5, 1.0]
    assert abs(per.precision[4] - 2 / 3) < 1e-12


def test_assign_tiles_prefers_largest_margin():
    wins = [(0, 0, 100, 100), (50, 0, 150, 100)]
    assert list(assign_tiles(B([10, 10, 20, 20], [80, 40, 95, 50], [60, 40, 70, 50], [95, 40, 105, 50]), wins)) == [0, 1, 0, 1]


def test_head_outputs_random_model():
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from gt_box_oracle import image_scores
    torch.manual_seed(0)
    net = DetectionModel("yolo11n.yaml", nc=NC, verbose=False).eval()
    w, h, c, b = next(g for g in real_gt() if len(g[2]))
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), np.uint8)
    sel = np.arange(min(len(c), 20))
    s, sb, best, n = image_scores(net, img, b[sel], "sliced", 640, 1024, 256, 0.5, 4)
    assert s.shape == sb.shape == (len(sel), NC) and np.all(s >= sb - 1e-7) and np.all((s >= 0) & (s <= 1)) and np.all(best > 0)


def test_errors_cli_end_to_end():
    """Synthetic predictions on data/val -> errors.py; base mAP must equal eval.py's scorer on the same CSV."""
    from PIL import Image
    from detlib.data import list_images
    Image.MAX_IMAGE_PIXELS = None
    rng = np.random.default_rng(1)
    rows = []
    for p, (w, h, c, b) in zip(list_images(DATA / "val" / "images"), real_gt()):
        for k in range(len(c)):
            if rng.random() < 0.8:
                cl = c[k] if rng.random() < 0.9 else (c[k] + 1) % NC
                bb = b[k] + rng.normal(0, 3, 4)
                rows.append(dict(image=p.name, cls=int(cl), conf=float(rng.random()), x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]))
        for _ in range(5):
            x, y = rng.random() * (w - 30), rng.random() * (h - 30)
            rows.append(dict(image=p.name, cls=int(rng.integers(NC)), conf=float(rng.random()), x1=x, y1=y, x2=x + 25, y2=y + 25))
    with tempfile.TemporaryDirectory() as tmp:
        ev = Path(tmp) / "eval"
        ev.mkdir()
        pd.DataFrame(rows).to_csv(ev / "predictions.csv", index=False)
        subprocess.run([sys.executable, str(REPO / "eval.py"), "--from-preds", str(ev / "predictions.csv"), "--out",
                        str(ev), "--run-dir", tmp, "--data-root", str(DATA), "--bootstrap", "0",
                        "--no-coco-crosscheck"], check=True, capture_output=True, cwd=REPO)
        subprocess.run([sys.executable, str(REPO / "analysis/errors.py"), "--preds", str(ev / "predictions.csv"),
                        "--name", "t", "--out-root", tmp, "--data-root", str(DATA), "--n-crops", "2"],
                       check=True, capture_output=True, cwd=REPO)
        out = Path(tmp) / "t" / "errors"
        args = json.loads((out / "errors_args.json").read_text())
        assert args["base_check"]["abs_diff"] < 1e-9, args["base_check"]
        for f in ["tide_dAP.csv", "tide_counts.csv", "slices.csv", "size_x_class_fn.csv", "size_x_class_fp.csv",
                  "thresholds.csv", "confusion_matrix_conf0.25.csv", "confusion_matrix_f1opt.csv", "op_gt.csv",
                  "op_predictions.csv"] + [f"crops_{t}.png" for t in
                                                                                ("cls", "loc", "both", "dupe", "bkg", "missed")]:
            assert (out / f).exists(), f
        G, SL = pd.read_csv(out / "op_gt.csv"), pd.read_csv(out / "slices.csv")
        thr = pd.read_csv(out / "thresholds.csv")
        assert set(SL.threshold_set) == set(G.threshold_set) == {"conf0.25", "f1opt"}
        assert args["threshold_sets"]["f1opt"] == dict(zip(thr.class_name, thr.threshold))
        P = pd.read_csv(out / "op_predictions.csv")
        assert (P.conf >= P.threshold).all()
        for tag in ("conf0.25", "f1opt"):
            g, sl = G[G.threshold_set == tag], SL[SL.threshold_set == tag]
            cm = pd.read_csv(out / f"confusion_matrix_{tag}.csv", index_col=0).to_numpy()
            assert cm[:-1].sum() == len(g)  # every GT appears once in the GT rows
            assert cm[:-1, :-1].trace() == g.matched.sum()
            assert (~g.matched).sum() == sl[sl.slice == "cls"].n_fn.sum()
            for v in ("cls", "size_bin", "objects_bin", "brightness_bin"):
                assert sl[sl.slice == v].n_gt.sum() == len(g), v

if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
