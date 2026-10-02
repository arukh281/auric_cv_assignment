"""Known-answer tests for analysis/learning_curve.py fitting. Run: python tests/test_learning_curve.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from learning_curve import fit_power, fit_with_uncertainty  # noqa: E402

NS = [101, 202, 302, 403]


def test_power_law_recovered_exactly():
    y = [0.02 * n ** 0.3 for n in NS]
    a, b = fit_power(NS, y)
    assert abs(a - 0.02) < 1e-9 and abs(b - 0.3) < 1e-9
    draws = [np.full(50, v) for v in y]                 # no sampling noise -> zero-width interval
    r = fit_with_uncertainty(NS, y, draws)
    assert abs(r["extrapolated_at_903"] - 0.02 * 903 ** 0.3) < 1e-9
    assert abs(r["extrap_ci95_hi"] - r["extrap_ci95_lo"]) < 1e-9
    assert abs(r["gain_403_to_903"] - 0.02 * (903 ** 0.3 - 403 ** 0.3)) < 1e-9
    assert r["reliable"] is False and "2.24x" in r["flags"]   # +500 is always > 2x the largest n here
    assert fit_with_uncertainty(NS, y, draws, target=700)["reliable"] is True


def test_unreliable_flags():
    draws = [np.full(20, 0.1)] * 4
    assert "monotone" in fit_with_uncertainty(NS, [0.1, 0.15, 0.12, 0.2], draws, target=700)["flags"]
    r = fit_with_uncertainty(NS, [0.0, 0.1, 0.12, 0.13], draws, target=700)
    assert "<= 0" in r["flags"] and r["reliable"] is False and np.isnan(r["extrapolated_at_903"])
    assert "only 2 points" in fit_with_uncertainty(NS[:2], [0.1, 0.12], draws[:2], target=300)["flags"]
    rng = np.random.default_rng(0)
    noisy = [np.clip(v + rng.normal(0, 0.2, 200), 1e-3, None) for v in (0.05, 0.08, 0.1, 0.12)]
    assert "wider than" in fit_with_uncertainty(NS, [0.05, 0.08, 0.1, 0.12], noisy, target=700)["flags"]


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
