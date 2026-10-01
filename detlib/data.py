"""Dataset I/O: class map, YOLO labels, data.yaml, EDA-derived limits."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EDA_TABLES = Path(__file__).resolve().parent.parent / "figures" / "eda" / "tables"


def load_classes(root):
    names = {}
    for line in (Path(root) / "classmap.txt").read_text().splitlines():
        if line.strip():
            i, name = line.split(maxsplit=1)
            names[int(i)] = name.strip()
    return dict(sorted(names.items()))


def list_images(img_dir):
    return sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def read_yolo_labels(path, w, h):
    """Return (cls int array, xyxy float array in pixels). Missing file = no boxes."""
    path = Path(path)
    rows = [l.split() for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4), float)
    a = np.array(rows, float)
    xc, yc, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return a[:, 0].astype(int), np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)


def write_yolo_labels(path, cls, xyxy, w, h):
    lines = []
    for c, (x1, y1, x2, y2) in zip(cls, xyxy):
        lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.8f} {(y1 + y2) / 2 / h:.8f} {(x2 - x1) / w:.8f} {(y2 - y1) / h:.8f}")
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def ensure_data_yaml(root):
    """Create <root>/data.yaml from classmap.txt if it does not exist. Returns its path."""
    root = Path(root)
    p = root / "data.yaml"
    if not p.exists():
        names = load_classes(root)
        p.write_text(yaml.safe_dump({"train": "train/images", "val": "val/images", "nc": len(names),
                                     "names": names}, sort_keys=False))
        print(f"[data] wrote {p}")
    return p


def write_resolved_data_yaml(dst, root, train, val, names):
    """data.yaml with an absolute `path`, so Ultralytics (and resume) never depends on the cwd."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump({"path": str(Path(root).resolve()), "train": str(train), "val": str(val),
                                   "nc": len(names), "names": names}, sort_keys=False))
    return dst


def eda_max_box_side(split="train"):
    """Largest box width/height in pixels from the Phase 1 EDA table (full dataset)."""
    b = pd.read_csv(EDA_TABLES / "boxes.csv")
    b = b[b.split == split]
    return float(max(b.w_px.max(), b.h_px.max()))


def eda_max_boxes_per_image(split="val"):
    b = pd.read_csv(EDA_TABLES / "boxes_per_image.csv")
    return int(b[b.split == split].n_boxes.max())
