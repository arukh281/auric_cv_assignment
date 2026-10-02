"""Known-answer tests for the learning-curve setup (subsets, configs, include-list tiling) and the
merge_sensitivity.py fixes. Run: python tests/test_learning_curve_setup.py (also pytest-compatible).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from detlib.data import list_images  # noqa: E402
from make_subsets import FRACTIONS, plan  # noqa: E402
from test_holdout import _tile  # noqa: E402
from test_pipeline import DATA  # noqa: E402

HOLD = set((REPO / "splits" / "holdout40_seed0.txt").read_text().split())


def test_subsets_nested_stratified_deterministic():
    p, q = plan(0), plan(0)
    assert p == q and p["pool_size"] == 403
    prev = set()
    for f, n in zip(FRACTIONS, (101, 202, 302)):
        s = p["subsets"][f"f{f}"]
        names = set((REPO / s["list"]).read_text().split())
        assert names == set(s["images"]) and len(names) == n
        assert prev <= names and not names & HOLD
        assert len(s["boxes_per_class"]) == 5 and min(s["images_per_class"].values()) >= 5
        prev = names
    assert plan(1)["subsets"]["f25"]["images"] != p["subsets"]["f25"]["images"]


def test_equal_iterations_and_configs():
    p = plan(0)
    full = p["subsets"]["f100"]
    assert (full["tiles"], full["iterations_per_epoch"], full["epochs"], full["total_iterations"]) == (3439, 215, 50, 10750)
    base = yaml.safe_load((REPO / "configs" / "b1h.yaml").read_text())
    for f in FRACTIONS:
        s = p["subsets"][f"f{f}"]
        assert abs(s["total_iterations"] - 10750) <= s["iterations_per_epoch"] / 2
        cfg = yaml.safe_load((REPO / "configs" / f"b1h_f{f}.yaml").read_text())
        assert cfg["epochs"] == s["epochs"] and cfg["train_list"] == s["list"]
        diff = {k for k in set(cfg) | set(base) if cfg.get(k) != base.get(k)}
        assert diff == {"name", "train_list", "epochs", "checkpoint_every", "train_args"}, diff
        ta = {k for k in set(cfg["train_args"]) | set(base["train_args"]) if cfg["train_args"].get(k) != base["train_args"].get(k)}
        assert ta == {"warmup_epochs", "close_mosaic"}, ta
    s1 = yaml.safe_load((REPO / "configs" / "b1h_seed1.yaml").read_text())
    assert {k for k in set(s1) | set(base) if s1.get(k) != base.get(k)} == {"name", "seed"} and s1["seed"] == 1


def test_include_list_tiles_only_listed_images_unchanged():
    imgs = [p.name for p in list_images(DATA / "train" / "images")][:3]
    from train import tiles_dir_name
    t = dict(tile=1024, overlap=256, empty_keep=0.2, min_vis=0.5, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.txt").write_text(f"{imgs[0]}\n{imgs[2]}\n")
        assert tiles_dir_name(t) == "tiles_1024_ov256_e0.2_v0.5_s0"
        assert "_in" in tiles_dir_name(t, None, tmp / "in.txt")
        full, _ = _tile(tmp / "full", 3)
        cmd = [sys.executable, str(REPO / "tools" / "make_tiles.py"), "--data-root", str(DATA), "--out", str(tmp / "inc"),
               "--max-images", "3", "--ext", ".jpg", "--workers", "2", "--include-list", str(tmp / "in.txt")]
        subprocess.run(cmd, check=True, capture_output=True, cwd=REPO)
        inc = pd.read_csv(tmp / "inc" / "tiles_index.csv")
        params = json.loads((tmp / "inc" / "tiling_params.json").read_text())
        assert set(inc.source) == {imgs[0], imgs[2]} and params["n_included_found"] == 2
        pd.testing.assert_frame_equal(full[full.source != imgs[1]].reset_index(drop=True), inc.reset_index(drop=True))


def test_merge_sensitivity_max_det_and_default_flag():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    rng = np.random.default_rng(0)
    rows = []
    for p in list_images(DATA / "val" / "images")[:3]:
        with Image.open(p) as im:
            w, h = im.size
        for _ in range(40):
            x, y = rng.random() * (w - 40), rng.random() * (h - 40)
            rows.append(dict(image=p.name, tile_x0=0, tile_y0=0, cls=int(rng.integers(5)), conf=float(rng.random()),
                             x1=x, y1=y, x2=x + 30, y2=y + 30))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "eval").mkdir()
        pd.DataFrame(rows).to_csv(tmp / "eval" / "predictions_raw.csv", index=False)
        (tmp / "eval" / "eval_args.json").write_text(json.dumps(dict(max_det=7)))
        subprocess.run([sys.executable, "analysis/merge_sensitivity.py", "--raw", str(tmp / "eval" / "predictions_raw.csv"),
                        "--name", "t", "--out-root", str(tmp), "--data-root", str(DATA)], check=True, capture_output=True,
                       cwd=REPO)
        t = pd.read_csv(tmp / "t" / "merge_sensitivity.csv")
        assert t.default.tolist() == [False, True, False, False, False]
        assert (t.max_det == 7).all() and (t.n_kept <= 7 * 3).all()
        d = t[t.default].iloc[0]
        assert (d["merge"], d["metric"], d["thr"]) == ("nms", "ios", 0.6)


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
