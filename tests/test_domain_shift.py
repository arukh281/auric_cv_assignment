"""Known-answer tests for analysis/domain_shift.py. Run: python tests/test_domain_shift.py (also pytest-compatible).
No data or weights needed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from domain_shift import domain_auc, hf_energy, image_stats, montage_sample, window  # noqa: E402


def test_stats_constant_image():
    img = np.zeros((40, 60, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 200, 100, 50
    s = image_stats(img)
    assert (s["mean_R"], s["mean_G"], s["mean_B"]) == (200, 100, 50)
    assert s["std_R"] == s["std_G"] == s["std_B"] == 0
    assert abs(s["saturation"] - round(255 * (200 - 50) / 200) / 255) < 1e-9  # OpenCV HSV S = 255 (max-min)/max
    assert s["sharpness_lapvar"] == 0 and s["hf_energy"] == 0
    assert (s["width"], s["height"], s["megapixels"]) == (60, 40, 60 * 40 / 1e6)


def test_stats_known_contrast_and_frequency():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, ::2] = 255                                         # columns alternate 0/255: Nyquist frequency
    s = image_stats(img)
    assert abs(s["mean_R"] - 127.5) < 1e-9 and abs(s["std_R"] - 127.5) < 1e-9
    assert s["hf_energy"] > 0.999 and s["sharpness_lapvar"] > 0
    x = np.arange(256)
    low = np.tile(127 + 100 * np.sin(2 * np.pi * 0.05 * x), (256, 1))   # 0.05 cycles/px = 0.1 Nyquist
    high = np.tile(127 + 100 * np.sin(2 * np.pi * 0.2 * x), (256, 1))   # 0.4 Nyquist
    assert hf_energy(low) < 0.01 and hf_energy(high) > 0.99


def test_same_images_in_both_splits_give_auc_half():
    """Every image appears once as train and once as val (same features, same group): nothing to learn."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 12))
    X2 = np.vstack([X, X])
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    g = np.r_[np.arange(20), np.arange(20)]
    for n_splits in (None, 5):
        r, s = domain_auc(X2, y, g, n_splits, n_boot=200)
        assert abs(r["auc"] - 0.5) < 1e-9, (n_splits, r)
        assert np.allclose(s[:20], s[20:])


def test_separable_domains_give_auc_one():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (20, 5)), rng.normal(6, 1, (22, 5))])
    y = np.r_[np.zeros(20, int), np.ones(22, int)]
    r, _ = domain_auc(X, y, np.arange(42), None, n_boot=200)
    assert r["auc"] == 1.0 and r["ci95_lo"] == 1.0


def test_window_padding_and_montage_cap():
    img = np.full((50, 50, 3), 7, np.uint8)
    w = window(img, 0, 0, 20)                                 # centred on the corner: 3/4 padding
    assert w.shape == (20, 20, 3) and (w[:10, :10] == 114).all() and (w[10:, 10:] == 7).all()
    cands = [("dense.png", j) for j in range(100)] + [(f"img{k}.png", 0) for k in range(5)]
    pick = montage_sample(cands, 12)
    assert len(pick) == 12 and sum(n == "dense.png" for n, _ in pick) <= 7  # 2 + top-up only after 5 others
    assert sum(n == "dense.png" for n, _ in montage_sample(cands, 7)) == 2


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
