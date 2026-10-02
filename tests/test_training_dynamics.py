"""Known-answer tests for analysis/training_dynamics.py. Run: python tests/test_training_dynamics.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))
from training_dynamics import box_metrics, categorize  # noqa: E402

B = lambda *r: np.array(r, float).reshape(-1, 4)  # noqa: E731


def test_categories():
    T, F = True, False
    assert categorize([F] * 5, [F] * 5) == "never-detected"
    assert categorize([F, T, T, F, T], [F] * 5) == "never-learned"
    assert categorize([T] * 5, [T] * 5) == "learned-early"
    assert categorize([T] * 5, [F, F, T, T, T]) == "learned-late"
    assert categorize([T] * 5, [T, T, F, T, T]) == "forgotten"
    assert categorize([T] * 5, [F, T, T, T, F]) == "forgotten"
    assert categorize([T] * 5, [F, F, F, F, T]) == "learned-late"


def test_box_metrics_known():
    gb, gc = B([0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10], [300, 0, 310, 10]), np.array([0, 1, 2, 3])
    pb = B([0, 0, 10, 10],        # GT0: class 0 at 0.9 -> detected + correct
           [0, 0, 10, 10],        # GT0: class 4 at 0.5 -> wrong class candidate
           [100, 0, 110, 10],     # GT1: class 0 at 0.8 (wrong, top) ...
           [100, 0, 110, 10],     # ... and class 1 at 0.3 (right, lower) -> detected, not correct
           [200, 0, 210, 10],     # GT2: class 2 at 0.1 -> below CONF_OP: not detected, true_score 0.1
           [306, 0, 316, 10])     # GT3: IoU 0.25 at 0.9 -> not detected, best_iou 0.25
    pc = np.array([0, 4, 0, 1, 2, 3])
    ps = np.array([.9, .5, .8, .3, .1, .9])
    m = box_metrics(gb, gc, pb, pc, ps)
    assert m["detected"].tolist() == [True, True, False, False]
    assert m["correct"].tolist() == [True, False, False, False]
    assert np.allclose(m["best_iou"], [1, 1, 0, 0.25])
    assert np.allclose(m["true_score"], [.9, .3, .1, 0])
    assert m["wrong_cls"].tolist() == [4, 0, -1, -1] and np.allclose(m["wrong_score"], [.5, .8, 0, 0])
    e = box_metrics(gb, gc, np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
    assert not e["detected"].any() and (e["wrong_cls"] == -1).all()


def test_halves_disjoint_and_cover_pool():
    ins = set((REPO / "splits/s52_inspect_seed0.txt").read_text().split())
    tst = set((REPO / "splits/s52_test_seed0.txt").read_text().split())
    hold = set((REPO / "splits/holdout40_seed0.txt").read_text().split())
    assert len(ins) == 202 and len(tst) == 201 and not ins & tst and not (ins | tst) & hold


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
