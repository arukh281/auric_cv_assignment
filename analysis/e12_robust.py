"""E12 amendment: robust inference (scale + photometric rules), B1h weights, no training. Rules use only the image.

  python analysis/e12_robust.py test --weights <last.pt> --data-root "$DATA_ROOT" --out <dir>
  python analysis/e12_robust.py val  --weights <last.pt> --data-root "$DATA_ROOT" --rule <name> --out <dir>

Degraded holdout40 copies (test only; fixed, seed 0):
  clean      original
  s0.5, s2   bilinear rescale by 0.5 / 2
  noise      + Gaussian noise sigma 10 (8-bit), clipped
  blurnoise  Gaussian blur sigma 1.5, then + noise sigma 8
  c0.5, c1.4 contrast x0.5 / x1.4 about each image's mean, clipped
Rules (applied per image, before eval.py's sliced inference; merge and scorer as eval.py):
  plain    nothing
  auto     E12's scale rule (predict at 1x; median sqrt(area) of conf>=0.25 predictions -> s in {0.5,1,2} closest
           to 22 px in log scale; predictions at that s)
  cnorm    per-image, per-channel linear map to the TRAINING images' channel mean/std (403 train images minus
           holdout40, computed once at start)
  denoise  cv2.bilateralFilter(d=5, sigmaColor=20, sigmaSpace=5)
  combo    denoise, then cnorm, then auto
A rule passes if (i) clean holdout40 mAP50 >= plain clean - 0.017 and (ii) its mean over the 6 degraded copies is
above plain's mean over them. Chosen: the passing rule with the highest degraded mean (written to chosen_rule.json);
val is then scored once with it (mode val).
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from detlib.data import EDA_TABLES, list_images  # noqa: E402
from detlib.scoring import ap_from_records, score_images  # noqa: E402
from eval import collect, finalize, image_sizes, run_predictions  # noqa: E402

VERSIONS = ["clean", "s0.5", "s2", "noise", "blurnoise", "c0.5", "c1.4"]
RULES = ["plain", "auto", "cnorm", "denoise", "combo"]
RESCALED_VAL = {"2308.png", "2384.png", "2391.png", "2460.png"}
ALTERED_VAL = RESCALED_VAL | {"2292.png", "2543.png", "1399.png", "2139.png"}


def degrade(im, v, rng):
    if v == "clean":
        return im
    if v in ("s0.5", "s2"):
        s = 0.5 if v == "s0.5" else 2.0
        return cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_LINEAR)
    f = im.astype(np.float32)
    if v == "noise":
        f = f + rng.normal(0, 10, f.shape)
    elif v == "blurnoise":
        f = cv2.GaussianBlur(f, (0, 0), 1.5) + rng.normal(0, 8, f.shape)
    elif v in ("c0.5", "c1.4"):
        k = 0.5 if v == "c0.5" else 1.4; m = f.mean(axis=(0, 1), keepdims=True); f = (f - m) * k + m
    return np.clip(f, 0, 255).astype(np.uint8)


def make_cnorm(stats):
    mt, st = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
    def f(im):
        x = im.astype(np.float32); m = x.mean(axis=(0, 1)); s = x.std(axis=(0, 1)) + 1e-6
        return np.clip((x - m) / s * st + mt, 0, 255).astype(np.uint8)
    return f


def denoise(im):
    return cv2.bilateralFilter(im, 5, 20, 5)


def write(imgs, fn, d):
    d.mkdir(parents=True, exist_ok=True); out = []
    for p in imgs:
        q = d / p.name
        if not q.exists():
            cv2.imwrite(str(q), fn(cv2.imread(str(p), cv2.IMREAD_COLOR)))
        out.append(q)
    return out


class Runner:
    def __init__(s, A, ev, tmp):
        s.A, s.ev, s.tmp = A, ev, tmp
    def raw(s, imgs, tag, scale=1.0):
        src = imgs if scale == 1.0 else write(imgs, lambda im: cv2.resize(im, (round(im.shape[1] * scale), round(im.shape[0] * scale)), interpolation=cv2.INTER_LINEAR), s.tmp / f"{tag}_x{scale}")
        r = run_predictions(s.A, src)
        for c in ("x1", "y1", "x2", "y2"):
            r[c] = r[c] / scale
        r["tile_x0"] = (r.tile_x0 / scale).round().astype(int); r["tile_y0"] = (r.tile_y0 / scale).round().astype(int)
        return r
    def fin(s, r, sizes):
        e = s.ev
        return finalize(r, sizes, e["merge"], e["merge_thr"], e["merge_metric"], e["max_det"])
    def auto(s, imgs, sizes, tag):
        r1 = s.raw(imgs, tag); p1 = s.fin(r1, sizes); out, chosen = [], {}
        for p in imgs:
            q = p1[(p1.image == p.name) & (p1.conf >= 0.25)]
            sc = 1.0 if not len(q) else min((0.5, 1.0, 2.0), key=lambda x: abs(np.log(float(np.median(np.sqrt((q.x2 - q.x1) * (q.y2 - q.y1)))) * x / 22.0)))
            chosen[p.name] = sc
        for sc in (0.5, 2.0):
            sel = [p for p in imgs if chosen[p.name] == sc]
            if sel:
                out.append(s.raw(sel, f"{tag}_auto", sc))
        keep1 = r1[r1.image.isin([n for n, v in chosen.items() if v == 1.0])]
        return s.fin(pd.concat([keep1] + out), sizes), chosen
    def rule(s, name, imgs, sizes, tag, cnorm):
        if name == "plain":
            return s.fin(s.raw(imgs, tag), sizes), {}
        if name == "auto":
            return s.auto(imgs, sizes, tag)
        if name == "cnorm":
            return s.fin(s.raw(write(imgs, cnorm, s.tmp / f"{tag}_cnorm"), tag + "c"), sizes), {}
        if name == "denoise":
            return s.fin(s.raw(write(imgs, denoise, s.tmp / f"{tag}_dn"), tag + "d"), sizes), {}
        if name == "combo":
            return s.auto(write(imgs, lambda im: cnorm(denoise(im)), s.tmp / f"{tag}_combo"), sizes, tag + "x")
        raise ValueError(name)


def score(preds, imgs, ldir, sizes, subset=None):
    keep = [p for p in imgs if subset is None or p.name in subset]
    per = collect(preds[preds.image.isin([p.name for p in keep])], keep, ldir, {p.name: sizes[p.name] for p in keep})
    aps, m, _ = ap_from_records(score_images(per, 5), 5, "coco")
    return float(m), float(np.mean(aps[:4]))


def train_stats(root, hold):
    im = pd.read_csv(EDA_TABLES / "images.csv").query("split == 'train'")
    s, ss, n = np.zeros(3), np.zeros(3), 0
    for r in im.itertuples():
        if r.image in hold:
            continue
        x = cv2.imread(str(root / "train" / "images" / r.image), cv2.IMREAD_COLOR).astype(np.float64).reshape(-1, 3)
        s += x.sum(0); ss += (x ** 2).sum(0); n += len(x)
    m = s / n
    return dict(mean=m.tolist(), std=np.sqrt(ss / n - m ** 2).tolist(), channel_order="BGR (cv2)", n_images=len(im) - len(hold))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["test", "val"]); ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=RULES); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(REPO / "configs" / "b1h.yaml")); ev = cfg["eval"]; root = Path(a.data_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A = SimpleNamespace(weights=a.weights, device=a.device, batch=16, mode="sliced", imgsz=ev["imgsz"], tile=ev["tile"],
                        overlap=ev["overlap"], conf=ev["conf"], nms_iou=ev["nms_iou"], max_det=ev["max_det"])
    tmp = Path("/kaggle/tmp/e12r") if Path("/kaggle").exists() else out / "tmp"
    R = Runner(A, ev, tmp)
    hold = [l.strip() for l in (REPO / cfg["holdout_list"]).read_text().splitlines() if l.strip()]
    stf = out / "train_stats.json"
    if not stf.exists():
        stf.write_text(json.dumps(train_stats(root, set(hold)), indent=2))
    cnorm = make_cnorm(json.loads(stf.read_text()))
    if a.mode == "test":
        base = [root / "train" / "images" / n for n in hold]; rows = []
        for v in VERSIONS:
            rng = np.random.default_rng(0)
            imgs = write(base, lambda im: degrade(im, v, rng), tmp / f"v_{v}") if v != "clean" else base
            sizes = image_sizes(imgs)
            for rule in RULES:
                p, ch = R.rule(rule, imgs, sizes, f"{v}_{rule}", cnorm)
                m, nl = score(p, imgs, root / "train" / "labels", sizes)
                rows.append(dict(version=v, rule=rule, mAP50=m, mAP50_no_liquid=nl,
                                 chosen_scales=pd.Series(ch).astype(str).value_counts().to_dict() if ch else "")); print(rows[-1], flush=True)
        T = pd.DataFrame(rows); T.to_csv(out / "robust_test.csv", index=False)
        piv = T.pivot(index="rule", columns="version", values="mAP50")
        deg = [v for v in VERSIONS if v != "clean"]; piv["degraded_mean"] = piv[deg].mean(1)
        base_clean, base_deg = piv.loc["plain", "clean"], piv.loc["plain", "degraded_mean"]
        piv["passes"] = (piv["clean"] >= base_clean - 0.017) & (piv["degraded_mean"] > base_deg)
        piv.loc["plain", "passes"] = False
        piv.to_csv(out / "robust_test_pivot.csv")
        ok = piv[piv.passes]
        chosen = ok.degraded_mean.idxmax() if len(ok) else "plain"
        (out / "chosen_rule.json").write_text(json.dumps(dict(chosen=chosen, pivot=piv.round(4).reset_index().to_dict("records")), indent=2, default=str))
        print(piv.round(4).to_string()); print("CHOSEN", chosen)
    else:
        imgs = list_images(root / "val" / "images"); sizes = image_sizes(imgs)
        p, ch = R.rule(a.rule, imgs, sizes, f"val_{a.rule}", cnorm); p.to_csv(out / f"val_preds_{a.rule}.csv", index=False)
        L = root / "val" / "labels"; names = {i.name for i in imgs}
        res = dict(rule=a.rule)
        for k, sub in (("all22", None), ("unaltered14", names - ALTERED_VAL), ("rescaled4", RESCALED_VAL), ("photometric4", ALTERED_VAL - RESCALED_VAL)):
            m, nl = score(p, imgs, L, sizes, sub); res[f"val_mAP50_{k}"] = m; res[f"val_mAP50_no_liquid_{k}"] = nl
        (out / "val_score.json").write_text(json.dumps(dict(res, chosen_scales=ch), indent=2)); print(res)


if __name__ == "__main__":
    main()
