"""E3/E4 configs, fp_audit and crop_classifier helpers. Run: python tests/test_e3e4.py (also pytest-compatible)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "analysis"))
from crop_classifier import crop  # noqa: E402
from fp_audit import crop_window, unmatched  # noqa: E402


def _diff(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_configs_differ_only_as_specified():
    b1h = yaml.safe_load((REPO / "configs/b1h.yaml").read_text())
    e3 = yaml.safe_load((REPO / "configs/e3_b1h_dota.yaml").read_text())
    e4 = yaml.safe_load((REPO / "configs/e4_b1h_flipud_mixup.yaml").read_text())
    assert _diff(b1h, e3) == {"name", "model", "init_from", "train_args"}
    assert e3["model"] == "yolo11s.yaml" and e3["init_from"] == "yolo11s-obb.pt"
    assert _diff(b1h["train_args"], e3["train_args"]) == {"patience"} and e3["train_args"]["patience"] == 0
    assert _diff(b1h, e4) == {"name", "train_args"}
    assert _diff(b1h["train_args"], e4["train_args"]) == {"patience", "flipud", "mixup"}
    assert e4["train_args"]["flipud"] == 0.5 and e4["train_args"]["mixup"] == 0.1 and e4["train_args"]["patience"] == 0
    for c in (e3, e4):
        assert c["epochs"] == 50 and c["checkpoint_every"] == 10 and c["seed"] == 0 and c["eval"]["max_det"] == 902


def test_unmatched_and_window():
    p = pd.DataFrame(dict(image=["a", "a", "b"], cls=[0, 1, 2], conf=[.9, .8, .7],
                          x1=[0, 50, 0], y1=[0, 50, 0], x2=[10, 60, 10], y2=[10, 60, 10]))
    u = unmatched(p, {"a": np.array([[0, 0, 10, 10.]])}, 0.1)
    assert list(u.conf) == [.8, .7]  # exact match on "a" removed; "b" has no GT
    assert crop_window(np.array([100, 100, 110, 120.]), 1000, 1000, 4, 96) == (57, 62, 153, 158)
    assert crop_window(np.array([0, 0, 10, 10.]), 50, 50, 4, 96) == (0, 0, 50, 50)


def test_labels_match_pairs_by_iou():
    from sanity_check import labels_match
    # two boxes whose x1 differ by 0.2 px but whose y order is reversed: coordinate sorting mis-pairs them
    eb = np.array([[10.0, 50, 30, 70], [10.2, 0, 40, 20]]); ec = np.array([0, 1])
    gb = eb[::-1] + 0.1; gc = ec[::-1]
    assert labels_match(gc, gb, ec, eb)
    assert not labels_match(np.array([1, 1]), gb, ec, eb)          # class changed
    assert not labels_match(gc, gb + np.array([0, 0, 2, 0]), ec, eb)  # shifted 2 px
    assert not labels_match(gc[:1], gb[:1], ec, eb)                 # missing box
    assert labels_match(np.zeros(0, int), np.zeros((0, 4)), np.zeros(0, int), np.zeros((0, 4)))


def test_repeat_factors():
    from detlib.rfs import build_list, repeat_factors
    cl = {"a": {0}, "b": {0}, "c": {0, 1}, "d": set()} | {f"e{i}": {0} for i in range(16)}  # class 1 in 1/20 tiles
    f, rc, rt = repeat_factors(cl, 2, t=0.1)
    assert abs(f[1] - 0.05) < 1e-12 and abs(rc[1] - np.sqrt(2)) < 1e-12 and rc[0] == 1.0
    assert rt["c"] == rc[1] and rt["d"] == 1.0
    lst = build_list({k: k for k in cl}, rt, seed=0)
    assert lst.count("a") == 1 and lst.count("c") in (1, 2)


def test_crop_pads_and_resizes():
    img = np.zeros((50, 60, 3), np.uint8); img[:, :30] = 200
    c = crop(img, np.array([0, 0, 20, 10.]), 2.0, 96)
    assert c.shape == (96, 96, 3) and c[:, :10].mean() > 150  # left half edge-padded from the bright column


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
