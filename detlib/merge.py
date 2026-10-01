"""Merge predictions from overlapping tiles into one set per full image."""
import numpy as np


def _overlap(box, boxes, metric):
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if metric == "iou":
        return inter / np.maximum(a + b - inter, 1e-12)
    if metric == "ios":  # intersection over the smaller box: catches a tile-edge fragment inside a full box
        return inter / np.maximum(np.minimum(a, b), 1e-12)
    raise ValueError(metric)


def nms(xyxy, conf, cls, thr, metric="iou"):
    """Class-wise greedy NMS. Returns kept indices sorted by descending confidence."""
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        idx = idx[np.argsort(-conf[idx], kind="mergesort")]
        while len(idx):
            i = idx[0]
            keep.append(i)
            rest = idx[1:]
            idx = rest[_overlap(xyxy[i], xyxy[rest], metric) < thr] if len(rest) else rest
    keep = np.array(keep, int)
    return keep[np.argsort(-conf[keep], kind="mergesort")] if len(keep) else keep


def wbf(xyxy, conf, cls, w, h, thr):
    """Weighted boxes fusion (ensemble_boxes) over all tile predictions of one image."""
    from ensemble_boxes import weighted_boxes_fusion
    if len(xyxy) == 0:
        return xyxy, conf, cls
    norm = np.array([w, h, w, h], float)
    b, s, l = weighted_boxes_fusion([(xyxy / norm).clip(0, 1).tolist()], [conf.tolist()], [cls.tolist()],
                                    iou_thr=thr, skip_box_thr=0.0)
    return np.asarray(b) * norm, np.asarray(s), np.asarray(l).astype(int)


def merge(xyxy, conf, cls, w, h, method="nms", thr=0.6, metric="ios"):
    if method == "nms":
        k = nms(xyxy, conf, cls, thr, metric)
        return xyxy[k], conf[k], cls[k]
    if method == "wbf":
        return wbf(xyxy, conf, cls, w, h, thr)
    if method == "none":
        return xyxy, conf, cls
    raise ValueError(method)
