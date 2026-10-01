"""Known-answer tests for scoring, tiling and tile merging. Run: python tests/test_pipeline.py
(no pytest needed; also pytest-compatible). Uses data/val labels for realistic GT where noted.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detlib.data import list_images, read_yolo_labels  # noqa: E402
from detlib.merge import merge  # noqa: E402
from detlib.scoring import ap_from_records, average_precision, bootstrap, score_images  # noqa: E402
from detlib.tiling import clip_boxes, tile_starts, tile_windows  # noqa: E402

NC = 5
DATA = Path(__file__).resolve().parent.parent / "data"


def real_gt():
    """GT of the local val images, as (w, h, cls, xyxy) per image."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    out = []
    for p in list_images(DATA / "val" / "images"):
        with Image.open(p) as im:
            w, h = im.size
        c, b = read_yolo_labels(DATA / "val" / "labels" / f"{p.stem}.txt", w, h)
        out.append((w, h, c, b))
    return out


def per_image(gt, pred_fn):
    per = []
    for w, h, c, b in gt:
        pb, pc, ps = pred_fn(c, b)
        per.append(dict(p_xyxy=pb, p_cls=pc, p_conf=ps, g_xyxy=b, g_cls=c))
    return per


def maps(per):
    recs = score_images(per, NC)
    return ap_from_records(recs, NC, "coco"), ap_from_records(recs, NC, "ultralytics"), recs


def test_perfect_predictions_give_1():
    (aps, m, _), (apu, mu, _), recs = maps(per_image(real_gt(), lambda c, b: (b, c, np.ones(len(c)))))
    assert np.allclose(aps[np.isfinite(aps)], 1.0) and abs(m - 1) < 1e-12, aps
    assert abs(mu - 0.995) < 1e-9, apu  # Ultralytics caps a perfect AP at 0.995 (last trapezoid ends at (1, 0))
    boot = bootstrap(recs, NC, n=50, seed=0)
    assert np.allclose(boot[np.isfinite(boot)], 1.0)


def test_half_recall_known_values():
    # Keep every other GT box per image, conf 1: precision 1 up to recall 0.5, nothing beyond.
    # COCO 101-pt: recall levels 0.00..0.50 (51 of 101) get precision 1 -> 51/101.
    # Ultralytics: area 0.5 (flat) + 0.25 (linear ramp from (0.5,1) to (1,0)) = 0.75.
    # Built per class with an even count so recall is exactly 0.5.
    rng = np.random.default_rng(0)
    g_cls = np.repeat(np.arange(NC), 10)
    xy = rng.uniform(0, 900, (len(g_cls), 2))
    g = np.hstack([xy, xy + 20])
    keep = np.tile(np.arange(10) % 2 == 0, NC)
    per = [dict(p_xyxy=g[keep], p_cls=g_cls[keep], p_conf=np.ones(keep.sum()), g_xyxy=g, g_cls=g_cls)]
    (aps, m, _), (apu, mu, _), _ = maps(per)
    assert np.allclose(aps, 51 / 101), aps
    assert np.allclose(apu, 0.75, atol=1e-9), apu


def test_wrong_class_and_shifted_boxes_give_0():
    gt = real_gt()
    (_, m, _), _, _ = maps(per_image(gt, lambda c, b: (b, (c + 1) % NC, np.ones(len(c)))))
    assert m == 0, m
    def shift(c, b):  # move every box off the image -> IoU 0 with all GT
        s = b.copy(); s[:, [0, 2]] += 1e5
        return s, c, np.ones(len(c))
    (_, m2, _), _, _ = maps(per_image(gt, shift))
    assert m2 == 0, m2


def test_ranking_matters():
    # 1 GT; FP with higher conf than the TP -> precision 0.5 at recall 1 -> COCO AP = 0.5
    g = np.array([[0, 0, 10, 10.]])
    p = np.array([[100, 100, 110, 110.], [0, 0, 10, 10.]])
    per = [dict(p_xyxy=p, p_cls=np.array([0, 0]), p_conf=np.array([0.9, 0.8]), g_xyxy=g, g_cls=np.array([0]))]
    aps, _, _ = ap_from_records(score_images(per, 1), 1, "coco")
    assert abs(aps[0] - 0.5) < 1e-12, aps
    assert average_precision(np.array([0.9]), np.array([False]), 0) != average_precision(np.array([]), np.array([]), 3)


def test_duplicate_prediction_is_fp():
    g = np.array([[0, 0, 10, 10.]])
    p = np.array([[0, 0, 10, 10.], [0, 0, 10, 10.]])
    per = [dict(p_xyxy=p, p_cls=np.array([0, 0]), p_conf=np.array([0.9, 0.8]), g_xyxy=g, g_cls=np.array([0]))]
    recs = score_images(per, 1)
    assert recs[0]["tp"].tolist() == [True, False]


def test_tiles_cover_image_with_overlap():
    for w, h in [(3378, 2713), (7204, 5932), (1369, 1334), (1024, 1024), (900, 3000)]:
        wins = tile_windows(w, h, 1024, 256)
        cover = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in wins:
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
            cover[y0:y1, x0:x1] = True
        assert cover.all(), (w, h)
        for L in (w, h):
            s = tile_starts(L, 1024, 256)
            assert all(b - a <= 768 for a, b in zip(s, s[1:]))  # overlap >= 256 everywhere


def test_every_box_whole_in_some_tile_and_labels_roundtrip():
    for w, h, c, b in real_gt():
        whole = np.zeros(len(c), bool)
        for win in tile_windows(w, h, 1024, 256):
            cc, local, st = clip_boxes(c, b, win, 0.5)
            back = local + np.array([win[0], win[1], win[0], win[1]])
            assert ((back[:, [0, 2]] >= win[0] - 1e-6) & (back[:, [0, 2]] <= win[2] + 1e-6)).all()
            e = 1e-6  # labels on the image edge round to ~1e-12 px outside it
            inside = (b[:, 0] >= win[0] - e) & (b[:, 1] >= win[1] - e) & (b[:, 2] <= win[2] + e) & (b[:, 3] <= win[3] + e)
            whole |= inside
            # boxes fully inside come back unchanged
            full = [k for k in np.where(inside)[0]]
            for k in full:
                assert np.isclose(back, b[k], atol=1e-4).all(axis=1).any()  # edge labels clip by ~1e-7 px
        assert whole.all(), "a box is not fully contained in any tile"


def test_clip_threshold():
    c = np.array([0, 1])
    b = np.array([[90, 0, 110, 10.], [95, 0, 115, 10.]])  # 50% and 25% inside x<100
    cc, local, st = clip_boxes(c, b, (0, 0, 100, 100), 0.5)
    assert cc.tolist() == [0] and st == dict(kept_whole=0, kept_clipped=1, dropped_partial=1)
    assert np.allclose(local, [[90, 0, 100, 10]])


def test_merge_removes_cross_tile_duplicates():
    xy = np.array([[100, 100, 130, 120.], [101, 100, 131, 120.], [100, 100, 112, 120.], [300, 300, 330, 320.]])
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    cls = np.array([0, 0, 0, 0])
    b, s, c = merge(xy, conf, cls, 1000, 1000, "nms", 0.6, "ios")  # duplicate + edge fragment removed
    assert len(b) == 2 and s.tolist() == [0.9, 0.6]
    b, s, c = merge(xy, conf, cls, 1000, 1000, "nms", 0.6, "iou")  # IoU keeps the fragment (IoU = 0.4)
    assert len(b) == 3
    b, s, c = merge(xy, conf, np.array([0, 1, 0, 0]), 1000, 1000, "nms", 0.6, "ios")  # class-wise
    assert len(b) == 3
    b, s, c = merge(xy, conf, cls, 1000, 1000, "wbf", 0.55)
    assert len(b) == 3  # WBF fuses the IoU>0.55 pair only


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print(f"{len(fns)} tests passed")
