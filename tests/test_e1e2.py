"""E1/E2 configs and the paired-run 2-epoch rule. Run: python tests/test_e1e2.py (also pytest-compatible)."""
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from run_pair import decide, project  # noqa: E402


def _diff(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_configs_differ_only_as_specified():
    b1h = yaml.safe_load((REPO / "configs/b1h.yaml").read_text())
    e1 = yaml.safe_load((REPO / "configs/e1_b1h_150ep.yaml").read_text())
    e2 = yaml.safe_load((REPO / "configs/e2_b1h_150ep_scale02.yaml").read_text())
    assert _diff(b1h, e1) == {"name", "epochs"} and e1["epochs"] == 150
    assert _diff(e1, e2) == {"name", "train_args"}
    assert _diff(e1["train_args"], e2["train_args"]) == {"scale"} and e2["train_args"]["scale"] == 0.2
    assert e1["checkpoint_every"] == 10 and e1["seed"] == 0 and e1["eval"]["max_det"] == 902
    assert "warmup_epochs" not in e1["train_args"] and "close_mosaic" not in e1["train_args"]


def test_projection_rule():
    # epoch 2 done at 0.30 h, epoch 2 took 4.2 min -> 0.30 + 148 * 0.07 = 10.66 h > 10.5
    p = project(0.30, 4.2 / 60, 150)
    assert abs(p - (0.30 + 148 * 0.07)) < 1e-9
    assert decide({"e1": 9.0, "e2": 10.5}, 10.5) == "both_continue"
    assert decide({"e1": 9.0, "e2": 10.51}, 10.5) == "stop_second"


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
