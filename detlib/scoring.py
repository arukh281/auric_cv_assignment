"""Detection scoring at IoU 0.5, independent of Ultralytics so that full-image (B0) and sliced (B1)
predictions are scored by exactly the same code.

Matching (COCO-style, per image, per class): predictions are visited in descending confidence; each
one takes the still-unmatched GT of the same class with the highest IoU, if that IoU >= thr.

Two AP interpolations are reported:
  * "coco": 101-point interpolated precision; precision is 0 at recall levels never reached.
  * "ultralytics": Ultralytics `compute_ap` (append (r=1, p=0), monotone envelope, trapezoid over a
    101-point grid). It linearly ramps from the last reached recall to (1, 0), so it is higher than
    "coco" when recall saturates below 1. Kept to compare with Ultralytics' own validator.
"""
import numpy as np

AP_METHODS = ("coco", "ultralytics")


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


def match_image(p_xyxy, p_cls, p_conf, g_xyxy, g_cls, iou_thr=0.5):
    """Returns (tp bool per prediction, matched GT index per prediction or -1, IoU of that match)."""
    n = len(p_xyxy)
    tp = np.zeros(n, bool)
    gt_idx = np.full(n, -1)
    m_iou = np.zeros(n)
    if n == 0 or len(g_xyxy) == 0:
        return tp, gt_idx, m_iou
    iou = box_iou(p_xyxy, g_xyxy)
    iou[p_cls[:, None] != g_cls[None, :]] = -1
    taken = np.zeros(len(g_xyxy), bool)
    for i in np.argsort(-p_conf, kind="mergesort"):
        row = np.where(taken, -1, iou[i])
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            tp[i], gt_idx[i], m_iou[i] = True, j, row[j]
            taken[j] = True
    return tp, gt_idx, m_iou


def pr_curve(conf, tp, n_gt):
    order = np.argsort(-conf, kind="mergesort")
    t = tp[order].astype(float)
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    return ctp / max(n_gt, 1e-12), ctp / np.maximum(ctp + cfp, 1e-12)


def average_precision(conf, tp, n_gt, method="coco"):
    if n_gt == 0:
        return np.nan
    if len(conf) == 0:
        return 0.0
    rec, prec = pr_curve(conf, tp, n_gt)
    x = np.linspace(0, 1, 101)
    if method == "coco":
        env = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, x, side="left")
        return float(np.mean([env[i] if i < len(env) else 0.0 for i in idx]))
    if method == "ultralytics":
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(np.interp(x, mrec, mpre), x))
    raise ValueError(method)


def score_images(per_image, nc, iou_thr=0.5):
    """per_image: list of dicts with p_xyxy, p_cls, p_conf, g_xyxy, g_cls (numpy arrays).

    Returns per-image records (cls, conf, tp of each prediction; GT count per class). That is all AP
    needs, so bootstrap resampling does not repeat the matching.
    """
    recs = []
    for d in per_image:
        tp, _, _ = match_image(d["p_xyxy"], d["p_cls"], d["p_conf"], d["g_xyxy"], d["g_cls"], iou_thr)
        recs.append(dict(cls=d["p_cls"], conf=d["p_conf"], tp=tp,
                         n_gt=np.bincount(d["g_cls"].astype(int), minlength=nc)[:nc]))
    return recs


def ap_from_records(recs, nc, method="coco", idx=None):
    """Per-class AP and mAP (mean over classes that have GT in this set of images)."""
    sel = recs if idx is None else [recs[i] for i in idx]
    cls = np.concatenate([r["cls"] for r in sel])
    conf = np.concatenate([r["conf"] for r in sel])
    tp = np.concatenate([r["tp"] for r in sel])
    n_gt = np.sum([r["n_gt"] for r in sel], axis=0)
    aps = np.array([average_precision(conf[cls == c], tp[cls == c], n_gt[c], method) for c in range(nc)])
    return aps, (float(np.nanmean(aps)) if np.isfinite(aps).any() else np.nan), n_gt


def bootstrap(recs, nc, n=1000, seed=0, method="coco"):
    """Resample images with replacement. Returns array (n, nc + 1): per-class AP then mAP."""
    rng = np.random.default_rng(seed)
    out = np.full((n, nc + 1), np.nan)
    for b in range(n):
        idx = rng.integers(0, len(recs), len(recs))
        aps, m, _ = ap_from_records(recs, nc, method, idx)
        out[b, :nc], out[b, nc] = aps, m
    return out
