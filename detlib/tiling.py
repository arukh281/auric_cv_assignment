"""Tile geometry shared by tools/make_tiles.py (training) and eval.py (sliced inference)."""
import numpy as np


def tile_starts(length, tile, overlap):
    """Start offsets along one axis. Stride = tile - overlap; last tile is flush with the far edge."""
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_windows(w, h, tile, overlap):
    """List of (x0, y0, x1, y1) windows covering a w x h image. Images smaller than `tile` give a smaller window."""
    return [(x, y, min(x + tile, w), min(y + tile, h))
            for y in tile_starts(h, tile, overlap) for x in tile_starts(w, tile, overlap)]


def clip_boxes(cls, xyxy, win, min_vis):
    """Clip full-image boxes to a window.

    Returns (cls, xyxy_local, stats). A box is kept if visible_area / original_area >= min_vis.
    stats counts boxes kept whole, kept clipped, and dropped (partially visible below threshold).
    """
    x0, y0, x1, y1 = win
    stats = dict(kept_whole=0, kept_clipped=0, dropped_partial=0)
    if len(xyxy) == 0:
        return cls[:0], xyxy[:0], stats
    c = xyxy.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(x0, x1)
    c[:, [1, 3]] = c[:, [1, 3]].clip(y0, y1)
    vis = np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    frac = np.where(area > 0, vis / np.maximum(area, 1e-9), 0)
    touching = vis > 0
    keep = touching & (frac >= min_vis)
    whole = keep & (frac >= 1 - 1e-9)
    stats["kept_whole"] = int(whole.sum())
    stats["kept_clipped"] = int((keep & ~whole).sum())
    stats["dropped_partial"] = int((touching & ~keep).sum())
    local = c[keep] - np.array([x0, y0, x0, y0], float)
    return cls[keep], local, stats
