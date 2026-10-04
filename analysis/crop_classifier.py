"""Crop classifier (CPU only): ImageNet-pretrained ResNet18 on GT truck crops, then re-label a detector's holdout40
predictions with it. Val is never read.

Usage (Kaggle CPU kernel, after scripts/kaggle_setup.sh):
  python analysis/crop_classifier.py --data-root /kaggle/tmp/data --holdout splits/holdout40_seed0.txt \
      --preds <B1h>/eval_holdout40/predictions.csv --out <dir>

Crops: square, side --context x the box's longer side (min 8 px), centred on the box, edge-padded where it leaves the
image, resized to --size px. Train: every GT box of every train image NOT in the holdout list (fixed --epochs, no
model selection; flips and 90-degree rotations; class-balanced sampling). Test: every GT box of the holdout images.
Re-labelling the detector's predictions (same crops around each predicted box):
  argmax:   class = classifier argmax, score = detector conf x p(argmax)
  allclass: one prediction per class c, score = detector conf x p(c) (5 x the predictions)
Both are written as predictions.csv files to be scored by eval.py --from-preds --split holdout (same scorer as B1h).
Writes <out>/holdout_gt_accuracy.csv, confusion_holdout.csv, train_log.csv, train_counts.csv, preds_argmax.csv,
preds_allclass.csv, summary.json, model.pt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from detlib.data import EDA_TABLES, load_classes, read_yolo_labels  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop(img, b, context, size):
    """img: HxWx3 uint8 array; b: xyxy. Square crop, edge padding outside the image, resized to size x size."""
    H, W = img.shape[:2]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(8.0, context * max(b[2] - b[0], b[3] - b[1]))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    x1, y1 = x0 + int(round(s)), y0 + int(round(s))
    p = max(0, -x0, -y0, x1 - W, y1 - H)
    a = np.pad(img, ((p, p), (p, p), (0, 0)), mode="edge") if p else img
    c = a[y0 + p:y1 + p, x0 + p:x1 + p]
    return np.asarray(Image.fromarray(c).resize((size, size), Image.BILINEAR))


def crops_for(root, images, sizes, boxes_of, context, size):
    X, meta = [], []
    for name in images:
        bx = boxes_of(name)
        if not len(bx[1]):
            continue
        img = np.asarray(Image.open(root / "train" / "images" / name).convert("RGB"))
        for j, (c, b) in enumerate(zip(*bx)):
            X.append(crop(img, b, context, size))
            meta.append((name, j, int(c)))
    return np.stack(X) if X else np.zeros((0, size, size, 3), np.uint8), meta


def to_tensor(x):
    return torch.from_numpy(((x.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2).copy())


def augment(x, rng):
    if rng.random() < .5:
        x = x[:, :, ::-1]
    if rng.random() < .5:
        x = x[:, ::-1]
    return np.rot90(x, rng.integers(4), axes=(1, 2))


@torch.no_grad()
def predict(net, X, bs=512):
    net.eval()
    out = [torch.softmax(net(to_tensor(X[i:i + bs])), 1).numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out) if out else np.zeros((0, 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--preds", required=True, help="detector predictions.csv on the holdout images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = load_classes(root); K = len(names)
    im = pd.read_csv(EDA_TABLES / "images.csv").query("split == 'train'")
    sizes = {r.image: (int(r.width), int(r.height)) for r in im.itertuples()}
    hold = [l.strip() for l in Path(a.holdout).read_text().splitlines() if l.strip()]
    train_imgs = sorted(set(sizes) - set(hold))
    gt = lambda n: read_yolo_labels(root / "train" / "labels" / f"{Path(n).stem}.txt", *sizes[n])
    t0 = time.time()
    Xtr, mtr = crops_for(root, train_imgs, sizes, gt, a.context, a.size)
    ytr = np.array([m[2] for m in mtr])
    Xte, mte = crops_for(root, hold, sizes, gt, a.context, a.size)
    yte = np.array([m[2] for m in mte])
    print(f"[crop] {len(Xtr)} train crops from {len(train_imgs)} images, {len(Xte)} holdout crops; "
          f"{time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(dict(cls=range(K), name=[names[k] for k in range(K)], train=np.bincount(ytr, minlength=K),
                      holdout=np.bincount(yte, minlength=K))).to_csv(out / "train_counts.csv", index=False)

    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, K)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(Xtr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    w = 1.0 / np.bincount(ytr, minlength=K).clip(1)
    p = w[ytr] / w[ytr].sum()  # class-balanced sampling
    log = []
    for ep in range(a.epochs):
        net.train(); idx = rng.choice(len(Xtr), len(Xtr), p=p); tot = 0.0
        for i in range(0, len(idx), a.bs):
            j = idx[i:i + a.bs]
            loss = nn.functional.cross_entropy(net(to_tensor(augment(Xtr[j], rng))), torch.from_numpy(ytr[j]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(j)
        log.append(dict(epoch=ep + 1, train_loss=tot / len(idx), seconds=time.time() - t0))
        print(f"[crop] epoch {ep + 1}: loss {tot / len(idx):.4f} ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    torch.save(net.state_dict(), out / "model.pt")

    P = predict(net, Xte); yhat = P.argmax(1)
    acc = pd.DataFrame([dict(cls=k, name=names[k], n=int((yte == k).sum()),
                             accuracy=float((yhat[yte == k] == k).mean()) if (yte == k).any() else None)
                        for k in range(K)])
    acc.to_csv(out / "holdout_gt_accuracy.csv", index=False)
    cm = pd.crosstab(pd.Series([names[k] for k in yte], name="gt"), pd.Series([names[k] for k in yhat], name="pred"))
    cm.to_csv(out / "confusion_holdout.csv")

    preds = pd.read_csv(a.preds)
    preds = preds[preds.image.isin(hold)].reset_index(drop=True)
    Pp = np.zeros((len(preds), K), np.float32)
    for img, g in preds.groupby("image"):
        arr = np.asarray(Image.open(root / "train" / "images" / img).convert("RGB"))
        X = np.stack([crop(arr, b, a.context, a.size) for b in g[["x1", "y1", "x2", "y2"]].to_numpy(float)])
        Pp[g.index.to_numpy()] = predict(net, X)
    am = preds.copy(); am["cls"] = Pp.argmax(1); am["conf"] = preds.conf.to_numpy() * Pp.max(1)
    am.to_csv(out / "preds_argmax.csv", index=False)
    allc = pd.concat([preds.assign(cls=k, conf=preds.conf.to_numpy() * Pp[:, k]) for k in range(K)])
    allc.sort_values(["image", "conf"], ascending=[True, False]).to_csv(out / "preds_allclass.csv", index=False)
    s = dict(train_images=len(train_imgs), train_crops=len(Xtr), holdout_crops=len(Xte), epochs=a.epochs,
             context=a.context, size=a.size, holdout_overall_accuracy=float((yhat == yte).mean()),
             holdout_mean_class_accuracy=float(acc.accuracy.dropna().mean()),
             detector_preds=len(preds), detector_class_kept_by_argmax=float((am.cls.to_numpy() == preds.cls.to_numpy()).mean()),
             seconds=time.time() - t0)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print("[crop]", json.dumps(s), flush=True)
    print(acc.to_string(index=False)); print(cm.to_string())


if __name__ == "__main__":
    main()
