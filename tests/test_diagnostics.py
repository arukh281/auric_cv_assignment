"""Known-answer tests for eval.py --split train and analysis/class_agnostic.py. Run: python tests/test_diagnostics.py
(also pytest-compatible). Uses data/ labels; no weights needed (eval.py --from-preds).
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
from class_agnostic import scores  # noqa: E402
from detlib.data import list_images, read_yolo_labels  # noqa: E402
from test_pipeline import DATA, NC, real_gt  # noqa: E402

BASELINE_COMMIT = "5914d7e"  # eval.py before --split / --sample-seed were added
OUTPUTS = ("metrics.json", "per_class.csv", "bootstrap_ci.csv", "scorer_comparison.csv", "predictions.csv")


def synthetic_preds(split, seed=1):
    """Noisy copies of the GT (some dropped, some relabelled) plus random boxes, for every image of a split."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    rng = np.random.default_rng(seed)
    rows = []
    for p in list_images(DATA / split / "images"):
        with Image.open(p) as im:
            w, h = im.size
        c, b = read_yolo_labels(DATA / split / "labels" / f"{p.stem}.txt", w, h)
        for k in range(len(c)):
            if rng.random() < 0.8:
                bb = b[k] + rng.normal(0, 2, 4)
                rows.append(dict(image=p.name, cls=int(c[k] if rng.random() < 0.9 else (c[k] + 1) % NC),
                                 conf=float(rng.random()), x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]))
        for _ in range(5):
            x, y = rng.random() * (w - 30), rng.random() * (h - 30)
            rows.append(dict(image=p.name, cls=int(rng.integers(NC)), conf=float(rng.random()),
                             x1=x, y1=y, x2=x + 25, y2=y + 25))
    return pd.DataFrame(rows)


def run_eval(script, preds_csv, out, *extra):
    return subprocess.run([sys.executable, str(script), "--from-preds", str(preds_csv), "--out", str(out),
                           "--run-dir", str(out.parent), "--data-root", str(DATA), "--bootstrap", "50", *extra],
                          capture_output=True, text=True, cwd=REPO)


def test_split_val_default_is_byte_identical_to_baseline():
    old = REPO / f"_eval_{BASELINE_COMMIT}.py"  # next to detlib/ so its imports resolve
    old.write_text(subprocess.run(["git", "show", f"{BASELINE_COMMIT}:eval.py"], capture_output=True, text=True,
                                  check=True, cwd=REPO).stdout)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_preds("val").to_csv(tmp / "preds.csv", index=False)
            for name, script in (("old", old), ("new", REPO / "eval.py")):
                (tmp / name).mkdir()
                r = run_eval(script, tmp / "preds.csv", tmp / name / "eval")
                assert r.returncode == 0, r.stderr[-2000:]
            for f in OUTPUTS:
                a, b = (tmp / "old" / "eval" / f).read_bytes(), (tmp / "new" / "eval" / f).read_bytes()
                if f == "metrics.json":  # weights path differs only because --run-dir differs; compare the rest
                    ja, jb = json.loads(a), json.loads(b)
                    assert ja.pop("weights").replace("/old/", "/x/") == jb.pop("weights").replace("/new/", "/x/")
                    a, b = json.dumps(ja).encode(), json.dumps(jb).encode()
                assert a == b, f
    finally:
        old.unlink()


def test_split_train_samples_and_guards():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic_preds("train").to_csv(tmp / "preds.csv", index=False)
        (tmp / "run").mkdir()
        r = subprocess.run([sys.executable, "eval.py", "--split", "train", "--from-preds", str(tmp / "preds.csv"),
                            "--run-dir", str(tmp / "run"), "--data-root", str(DATA)], capture_output=True, text=True,
                           cwd=REPO)
        assert r.returncode != 0 and "requires an explicit --out" in r.stderr
        r = run_eval(REPO / "eval.py", tmp / "preds.csv", tmp / "run" / "eval", "--split", "train")
        assert r.returncode != 0 and "must not write into" in r.stderr
        outs = []
        for k, seed in enumerate((0, 0, 1)):
            out = tmp / "run" / f"eval_train_{k}"
            r = run_eval(REPO / "eval.py", tmp / "preds.csv", out, "--split", "train", "--max-images", "5",
                         "--sample-seed", str(seed))
            assert r.returncode == 0, r.stderr[-2000:]
            outs.append(json.loads((out / "metrics.json").read_text()))
        m = outs[0]
        pool = [p.name for p in list_images(DATA / "train" / "images") if p.name != "1938.png"]
        assert m["split"] == "train" and m["n_images"] == 5 and len(set(m["sampled_images"])) == 5
        assert set(m["sampled_images"]) <= set(pool) and "1938.png" not in m["sampled_images"]
        assert outs[0]["sampled_images"] == outs[1]["sampled_images"] != outs[2]["sampled_images"]
        assert outs[0]["mAP50"] == outs[1]["mAP50"]
        # class-agnostic script picks up the sampled train images from metrics.json
        r = subprocess.run([sys.executable, "analysis/class_agnostic.py", "--preds",
                            str(tmp / "run" / "eval_train_0" / "predictions.csv"), "--data-root", str(DATA)],
                           capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr[-2000:]
        ca = json.loads((tmp / "run" / "eval_train_0" / "class_agnostic.json").read_text())
        assert ca["split"] == "train" and ca["n_images"] == 5
        assert abs(ca["mAP50_per_class"] - m["mAP50"]) < 1e-12  # same scorer, same images, same CSV
        assert ca["AP50_class_agnostic"] >= ca["mAP50_per_class"]


def test_class_agnostic_class_swaps_only():
    """Perfect boxes, every class wrong: per-class mAP50 0, class-agnostic AP50 1, recall 1 at both thresholds."""
    per = [dict(image=str(i), p_xyxy=b.copy(), p_cls=(c + 1) % NC, p_conf=np.linspace(1, 0.5, len(c)),
                g_xyxy=b, g_cls=c) for i, (w, h, c, b) in enumerate(real_gt())]
    aps, m, m_ag, rec = scores(per, NC)
    assert m == 0.0 and m_ag == 1.0, (m, m_ag)
    assert m_ag >= m
    n_gt = sum(len(d["g_cls"]) for d in per)
    assert rec[0.25] == (n_gt, n_gt, 1.0) and rec[0.001] == (n_gt, n_gt, 1.0)
    # low-confidence half: recall at 0.25 drops to the share of GT whose prediction has conf >= 0.25
    for d in per:
        d["p_conf"] = np.where(np.arange(len(d["p_conf"])) % 2 == 0, 0.9, 0.1)
    _, _, m_ag2, rec2 = scores(per, NC)
    kept = sum(int((np.arange(len(d["g_cls"])) % 2 == 0).sum()) for d in per)
    assert rec2[0.25] == (kept, n_gt, kept / n_gt) and rec2[0.001][2] == 1.0 and m_ag2 == 1.0


def test_class_agnostic_cli_val_matches_eval():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic_preds("val").to_csv(tmp / "preds.csv", index=False)
        out = tmp / "run" / "eval"
        out.parent.mkdir()
        assert run_eval(REPO / "eval.py", tmp / "preds.csv", out).returncode == 0
        r = subprocess.run([sys.executable, "analysis/class_agnostic.py", "--preds", str(out / "predictions.csv"),
                            "--data-root", str(DATA)], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr[-2000:]
        ca = json.loads((out / "class_agnostic.json").read_text())
        ev = json.loads((out / "metrics.json").read_text())
        assert ca["split"] == "val" and abs(ca["mAP50_per_class"] - ev["mAP50"]) < 1e-12
        assert ca["AP50_class_agnostic"] >= ca["mAP50_per_class"]
        assert 0 <= ca["recall_agnostic_conf0.25"] <= ca["recall_agnostic_conf0.001"] <= 1


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
