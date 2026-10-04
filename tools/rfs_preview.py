"""Preview repeat-factor sampling (detlib/rfs.py) for several thresholds t, on B1h's training tiles. CPU only;
deterministic (seed 0, as train.py).

  python tools/rfs_preview.py --data-root /kaggle/tmp/data --tiles /kaggle/tmp/work/tiles_b1h --t 0.1 0.3 --out <dir>

Tiles are made with B1h's settings if missing (analysis/sanity_check.make_tiles). Per t: f_c, r_c, list length,
extra tile views vs the plain tile count, and expected copies per class-containing tile.
Writes <out>/rfs_preview.json. t = 0.1 must reproduce E6's rfs.json (3496 entries).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "analysis"))
from detlib.rfs import build_list, repeat_factors, tile_classes  # noqa: E402
from sanity_check import make_tiles  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True); ap.add_argument("--tiles", required=True)
    ap.add_argument("--t", type=float, nargs="+", default=[0.1, 0.3]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tiles = Path(a.tiles); make_tiles(Path(a.data_root), tiles)
    imgs = {p.stem: p for p in sorted((tiles / "train" / "images").iterdir()) if p.suffix == ".png"}
    cl = tile_classes(tiles / "train" / "labels", imgs)
    res = {}
    for t in a.t:
        f, rc, rt = repeat_factors(cl, 5, t)
        lst = build_list({k: k for k in imgs}, rt, 0)
        res[str(t)] = dict(tiles=len(imgs), list_entries=len(lst), extra_views=len(lst) - len(imgs),
                           extra_pct=100 * (len(lst) - len(imgs)) / len(imgs), class_tile_fraction=f.tolist(),
                           class_repeat_factor=rc.tolist(),
                           tiles_with_class=[sum(c in s for s in cl.values()) for c in range(5)],
                           list_entries_with_class=[sum(lst.count(k) for k, s in cl.items() if c in s) for c in range(5)])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "rfs_preview.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
