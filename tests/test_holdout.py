"""Known-answer tests for the B1h held-out experiment. Run: python tests/test_holdout.py (also pytest-compatible).
Tiling tests use a few data/ train images and write to a temp folder.
"""
import filecmp
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from detlib.data import EDA_TABLES, list_images  # noqa: E402
from make_holdout import choose_holdout, expected_tiles  # noqa: E402
from test_pipeline import DATA  # noqa: E402

HOLDOUT = REPO / "splits" / "holdout40_seed0.txt"
B1 = dict(tile=1024, overlap=256, empty_keep=0.2, min_vis=0.5, seed=0)


def test_holdout_list_deterministic_and_complete():
    boxes = pd.read_csv(EDA_TABLES / "boxes.csv")
    a, b = choose_holdout(boxes, 40, 0), choose_holdout(boxes, 40, 0)
    assert a == b == HOLDOUT.read_text().split()
    assert len(set(a)) == 40 and "1938.png" not in a
    tr = boxes[(boxes.split == "train") & boxes.image.isin(a)]
    assert set(tr.class_name) == set(boxes.class_name)          # every class, Truck w/Liquid included
    assert choose_holdout(boxes, 40, 1) != a


def test_expected_tiles_reproduce_b1_and_tile_dir_name():
    images, boxes = pd.read_csv(EDA_TABLES / "images.csv"), pd.read_csv(EDA_TABLES / "boxes.csv")
    full = expected_tiles(images, boxes)
    rec = REPO / "results" / "b1_tile1024" / "tiling_params.json"
    if rec.exists():
        c = json.loads(rec.read_text())["counts"]
        assert (full["tiles_written"], full["tiles_with_boxes"], full["empty_tiles_kept"]) == \
            (c["tiles_written"], c["tiles_with_boxes"], c["empty_tiles_kept"])
    ho = expected_tiles(images, boxes, HOLDOUT.read_text().split())
    assert ho["n_source_images"] == 403 and ho["tiles_written"] < full["tiles_written"]
    from train import tiles_dir_name
    assert tiles_dir_name(B1) == "tiles_1024_ov256_e0.2_v0.5_s0"     # B1's cache name, unchanged
    assert tiles_dir_name(B1, HOLDOUT).startswith("tiles_1024_ov256_e0.2_v0.5_s0_ho")


def _tile(out, n, exclude=None):
    cmd = [sys.executable, str(REPO / "tools" / "make_tiles.py"), "--data-root", str(DATA), "--out", str(out),
           "--max-images", str(n), "--ext", ".jpg", "--workers", "2"]
    if exclude:
        cmd += ["--exclude-list", str(exclude)]
    subprocess.run(cmd, check=True, capture_output=True, cwd=REPO)
    return pd.read_csv(out / "tiles_index.csv"), json.loads((out / "tiling_params.json").read_text())


def test_excluded_image_never_tiled_and_others_unchanged():
    imgs = [p.name for p in list_images(DATA / "train" / "images")][:3]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ex.txt").write_text(imgs[1] + "\n")
        full, _ = _tile(tmp / "full", 3)
        ex, params = _tile(tmp / "ex", 3, tmp / "ex.txt")
        assert imgs[1] not in set(ex.source) and imgs[1] in set(full.source)
        assert params["n_excluded_found"] == 1 and params["n_source_images"] == 2
        assert not any(f.name.startswith(Path(imgs[1]).stem + "__") for f in (tmp / "ex" / "train" / "images").iterdir())
        same = full[full.source != imgs[1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(same, ex.reset_index(drop=True))   # identical tiles incl. empty-tile draws
        for d in ("images", "labels"):
            names = sorted(f.name for f in (tmp / "ex" / "train" / d).iterdir())
            assert names and all(filecmp.cmp(tmp / "full" / "train" / d / n, tmp / "ex" / "train" / d / n, shallow=False)
                                 for n in names)


def test_eval_holdout_split_and_train_exclusion():
    from test_diagnostics import synthetic_preds
    imgs = [p.name for p in list_images(DATA / "train" / "images")]
    hold = [n for n in HOLDOUT.read_text().split() if n in imgs][:2] or imgs[:2]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ho.txt").write_text("\n".join(hold) + "\n")
        synthetic_preds("train").to_csv(tmp / "p.csv", index=False)
        (tmp / "run").mkdir()
        base = [sys.executable, "eval.py", "--from-preds", str(tmp / "p.csv"), "--run-dir", str(tmp / "run"),
                "--data-root", str(DATA), "--bootstrap", "0", "--holdout-list", str(tmp / "ho.txt"), "--max-det", "902"]
        r = subprocess.run(base + ["--split", "holdout", "--out", str(tmp / "run" / "h")], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr[-1500:]
        m = json.loads((tmp / "run" / "h" / "metrics.json").read_text())
        assert m["split"] == "holdout" and m["sampled_images"] == hold and m["n_images"] == 2 and m["max_det"] == 902
        r = subprocess.run(base + ["--split", "train", "--max-images", str(len(imgs)), "--out", str(tmp / "run" / "t")],
                           capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr[-1500:]
        m = json.loads((tmp / "run" / "t" / "metrics.json").read_text())
        assert not set(hold) & set(m["sampled_images"]) and m["max_det"] == 902
        r = subprocess.run(base + ["--split", "holdout"], capture_output=True, text=True, cwd=REPO)
        assert r.returncode != 0 and "requires an explicit --out" in r.stderr


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
