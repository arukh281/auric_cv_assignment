"""Known-answer tests for tools/make_smart_subsets.py selection. Run: python tests/test_smart_subsets.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from detlib.data import EDA_TABLES  # noqa: E402
from make_smart_subsets import COVER_ORDER, EMPTY, pool_names, select  # noqa: E402


def _setup():
    names = [n for n in pool_names() if n != EMPTY]
    boxes = pd.read_csv(EDA_TABLES / "boxes.csv")
    tr = boxes[(boxes.split == "train") & boxes.image.isin(set(pool_names()))]
    counts = pd.crosstab(tr.image, tr.class_name).reindex(names, fill_value=0)
    emb = np.random.default_rng(0).normal(size=(len(names), 16))
    return names, emb / np.linalg.norm(emb, axis=1, keepdims=True), counts, tr.class_name.value_counts().to_dict()


def test_coverage_shares_met_and_sizes():
    names, emb, counts, totals = _setup()
    assert len(names) == 402 and EMPTY not in names
    for size in (202, 302):
        lst, n_cover, share = select(names, emb, counts, size, totals)
        assert len(lst) == len(set(lst)) == size and EMPTY not in lst
        got = counts.loc[lst].sum()
        assert all(got[c] >= share[c] for c in COVER_ORDER), (got, share)
        assert 0 < n_cover < size
        assert select(names, emb, counts, size, totals)[0] == lst      # deterministic


def test_kcenter_picks_farthest():
    names = ["a", "b", "c", "d"]
    counts = pd.DataFrame(0, index=names, columns=COVER_ORDER + ["Cargo Truck"])
    counts.loc["a", "Truck w/Liquid"] = 1
    emb = np.array([[0, 0], [1, 0], [10, 0], [5, 0]], float)
    totals = dict.fromkeys(counts.columns, 1)
    lst, n_cover, _ = select(names, emb, counts, 3, totals, n_pool=4)
    assert lst[0] == "a" and n_cover == 1 and lst[1:] == ["c", "d"]   # farthest from a is c, then d


def test_s54_configs_and_lists():
    import yaml
    base = yaml.safe_load((REPO / "configs" / "b1h.yaml").read_text())
    for f, n in ((50, 202), (75, 302)):
        lst = (REPO / "splits" / f"train_smart{f}_seed0.txt").read_text().split()
        assert len(set(lst)) == n and EMPTY not in lst and not set(lst) & set(
            (REPO / "splits" / "holdout40_seed0.txt").read_text().split())
        c = yaml.safe_load((REPO / "configs" / f"b1h_smart{f}.yaml").read_text())
        assert {k for k in set(c) | set(base) if c.get(k) != base.get(k)} == {"name", "train_list", "epochs",
                                                                             "checkpoint_every", "train_args"}
        assert {k for k in set(c["train_args"]) | set(base["train_args"])
                if c["train_args"].get(k) != base["train_args"].get(k)} == {"warmup_epochs", "close_mosaic"}
        r = yaml.safe_load((REPO / "configs" / f"b1h_f{f}.yaml").read_text())
        r1 = yaml.safe_load((REPO / "configs" / f"b1h_f{f}_seed1.yaml").read_text())
        assert {k for k in set(r) | set(r1) if r.get(k) != r1.get(k)} == {"name", "seed"} and r1["seed"] == 1


if __name__ == "__main__":
    tests = sorted(k for k, v in dict(globals()).items() if k.startswith("test_") and callable(v))
    for k in tests:
        globals()[k]()
        print("PASS", k)
    print(f"{len(tests)} tests passed")
