"""Known-answer tests for analysis/label_mismatch.py. Run: python tests/test_label_mismatch.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from label_mismatch import flagged, pair_boxes, source_overlaps  # noqa: E402

B = lambda *r: np.array(r, float).reshape(-1, 4)  # noqa: E731


def test_flag_and_pair_types():
    eb, ec = B([0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]), np.array([0, 1, 2])
    assert not flagged(ec.copy(), eb.copy(), ec, eb)
    gb, gc = B([0, 0, 10, 10], [22, 0, 32, 10], [60, 0, 70, 10]), np.array([0, 3, 2])
    assert flagged(gc, gb, ec, eb)
    t = sorted(r["type"] for r in pair_boxes(gc, gb, ec, eb))
    assert t == ["extra", "missing", "reclass", "same"], t
    gc2 = np.array([0, 1, 2]); gb2 = B([0, 0, 10, 10], [20.9, 0, 30.9, 10], [40, 0, 50, 10])
    r = {x["type"]: x for x in pair_boxes(gc2, gb2, ec, eb)}
    assert "shifted" in r and abs(r["shifted"]["max_abs_d"] - 0.9) < 1e-9


def test_source_overlaps():
    sb, sc = B([0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10.5], [50, 50, 60, 60]), np.array([0, 1, 0, 2])
    o = source_overlaps(sc, sb)
    assert o["pairs_iou_ge_0_9"] == 3 and o["pairs_iou_ge_0_9_diff_class"] == 2 and o["exact_duplicate_pairs"] == 3


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
