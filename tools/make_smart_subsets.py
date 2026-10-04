"""§5.4 "smart" training subsets (S54 pre-registration in DETAILED_EXPERIMENTS.md) and their configs.

Two steps:
  python tools/make_smart_subsets.py embed --data-root /kaggle/tmp/data --out <dir>      # GPU (Kaggle): embeddings
  python tools/make_smart_subsets.py select --embeddings results/s54/embeddings.npz [--write-configs]   # CPU

embed: for every pool image (the 403 non-holdout train images except 1938.png, which is never selected), the image
  is cut into 1024 px tiles (detlib.tiling.tile_windows, overlap 256, as training); each tile is resized to 224 x 224
  and embedded with DINOv2-small (Hugging Face facebook/dinov2-small, CLS token of the last hidden state;
  fallback torch.hub dinov2_vits14, then ImageNet ResNet18 global-average-pooled features). The image embedding is
  the mean over its tiles, L2-normalised. Writes <dir>/embeddings.npz (names, emb, model) + embed_args.json.

select (deterministic, seed 0), for each size (202 = 50%, 302 = 75%):
  share_c = (GT boxes of class c in the 403-image pool) * size / 403
  (1) class coverage: for c in Liquid, Tractor, Flatbed, Box: while the selected images hold fewer than share_c boxes
      of class c, add the unselected image with the most class-c boxes (ties: file name).
  (2) greedy k-center: repeatedly add the unselected image whose embedding is farthest (Euclidean on L2-normalised
      vectors) from its nearest selected image, until `size` images (ties: lowest index). If (1) selected nothing,
      the first center is drawn with numpy default_rng(0).
  Writes splits/train_smart{50,75}_seed0.txt and splits/smart_subsets_seed0.json (per-class box and image counts next
  to random f50/f75, tiles, epochs, iterations). With --write-configs: configs/b1h_smart{50,75}.yaml (b1h.yaml with
  train_list, epochs and the iteration-matched schedule fields, exactly as tools/make_subsets.py) and
  configs/b1h_f{50,75}_seed1.yaml (configs/b1h_f{50,75}.yaml with name and seed 1).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from detlib.data import EDA_TABLES  # noqa: E402

SIZES = {50: 202, 75: 302}
COVER_ORDER = ["Truck w/Liquid", "Truck Tractor", "Truck w/Flatbed", "Truck w/Box"]
EMPTY = "1938.png"
B1H_ITERS, B1H_IT_PER_EPOCH, BATCH = 10750, 215, 16


def pool_names():
    im = pd.read_csv(EDA_TABLES / "images.csv")
    hold = set((REPO / "splits" / "holdout40_seed0.txt").read_text().split())
    return sorted(set(im[im.split == "train"].image) - hold)


# ---------------------------------------------------------------- embed (Kaggle)
def load_embedder(device):
    import torch
    try:
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(device)
        return "facebook/dinov2-small (transformers, CLS of last_hidden_state)", \
            lambda x: m(pixel_values=x).last_hidden_state[:, 0]
    except Exception as e:  # noqa: BLE001
        print(f"[smart] transformers DINOv2 failed: {e!r}")
    try:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
        return "dinov2_vits14 (torch.hub, CLS)", lambda x: m(x)
    except Exception as e:  # noqa: BLE001
        print(f"[smart] torch.hub DINOv2 failed: {e!r}")
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    m.fc = torch.nn.Identity()
    return "ImageNet ResNet18 (fallback, global-average-pooled)", lambda x: m(x)


def embed(a):
    import cv2
    import torch
    from detlib.tiling import tile_windows
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, f = load_embedder(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    names = [n for n in pool_names() if n != EMPTY][: a.max_images]
    root = Path(a.data_root)
    embs = []
    for i, n in enumerate(names):
        img = cv2.cvtColor(cv2.imread(str(root / "train" / "images" / n), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        tiles = [cv2.resize(img[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA)
                 for x0, y0, x1, y1 in tile_windows(w, h, 1024, 256)]
        x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float().to(device) / 255.0
        with torch.no_grad():
            e = torch.cat([f((x[k:k + 32] - mean) / std).float() for k in range(0, len(x), 32)]).mean(0)
        embs.append((e / e.norm()).cpu().numpy())
        if i % 50 == 0:
            print(f"[smart] {i + 1}/{len(names)} {n}: {len(tiles)} tiles", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "embeddings.npz", names=np.array(names), emb=np.stack(embs).astype(np.float32),
             model=np.array(model_name))
    (out / "embed_args.json").write_text(json.dumps(dict(model=model_name, n_images=len(names), tile=1024,
                                                         overlap=256, resize=224, device=device), indent=2))
    print(f"[smart] embedded {len(names)} images with {model_name} -> {out}")


# ---------------------------------------------------------------- select (local)
def select(names, emb, counts, size, pool_totals, n_pool=403, seed=0):
    """counts: DataFrame indexed by image name, one column per class (GT boxes). Returns ordered list."""
    share = {c: pool_totals[c] * size / n_pool for c in COVER_ORDER}
    idx = {n: i for i, n in enumerate(names)}
    chosen, have = [], dict.fromkeys(counts.columns, 0)
    for c in COVER_ORDER:
        while have[c] < share[c]:
            cand = counts.loc[[n for n in names if n not in set(chosen)], c]
            cand = cand[cand > 0]
            if cand.empty:
                break
            n = sorted(cand.index, key=lambda x: (-cand[x], x))[0]
            chosen.append(n)
            for k in have:
                have[k] += int(counts.at[n, k])
    n_cover = len(chosen)
    if not chosen:
        chosen.append(names[int(np.random.default_rng(seed).integers(len(names)))])
    sel = np.zeros(len(names), bool)
    sel[[idx[n] for n in chosen]] = True
    d = np.min(np.linalg.norm(emb[:, None, :] - emb[sel][None, :, :], axis=2), axis=1)
    while sel.sum() < size:
        d[sel] = -1
        j = int(np.argmax(d))
        sel[j] = True
        chosen.append(names[j])
        d = np.minimum(d, np.linalg.norm(emb - emb[j], axis=1))
    return chosen, n_cover, share


def subset_plan(lst):
    from make_holdout import expected_tiles
    images, boxes = pd.read_csv(EDA_TABLES / "images.csv"), pd.read_csv(EDA_TABLES / "boxes.csv")
    train = set(images[images.split == "train"].image)
    t = expected_tiles(images, boxes, sorted(train - set(lst)))
    ipe = math.ceil(t["tiles_written"] / BATCH)
    ep = int(round(B1H_ITERS / ipe))
    tr = boxes[(boxes.split == "train") & boxes.image.isin(lst)]
    return dict(n_images=len(lst), tiles=t["tiles_written"], iterations_per_epoch=ipe, epochs=ep,
                total_iterations=ep * ipe, warmup_epochs=round(3 * B1H_IT_PER_EPOCH / ipe, 4),
                close_mosaic=int(round(10 * B1H_IT_PER_EPOCH / ipe)), checkpoint_every=max(1, ep // 5),
                boxes_per_class=tr.class_name.value_counts().sort_index().to_dict(),
                images_per_class=tr.groupby("class_name").image.nunique().sort_index().to_dict())


def write_configs(plans):
    base = yaml.safe_load((REPO / "configs" / "b1h.yaml").read_text())
    for f, p in plans.items():
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg.update(name=f"b1h_smart{f}", train_list=f"splits/train_smart{f}_seed0.txt", epochs=p["epochs"],
                   checkpoint_every=p["checkpoint_every"])
        cfg["train_args"].update(warmup_epochs=p["warmup_epochs"], close_mosaic=p["close_mosaic"])
        head = (f"# b1h_smart{f}: configs/b1h.yaml trained on the {p['n_images']} 'smart' images of "
                f"splits/train_smart{f}_seed0.txt\n# (tools/make_smart_subsets.py; S54 pre-registration), epochs = "
                f"round(10750 / ceil({p['tiles']} tiles / 16)) = {p['epochs']} -> {p['total_iterations']} iterations;\n"
                f"# warmup_epochs / close_mosaic / checkpoint_every rescaled to cover the same iterations as in B1h.\n")
        (REPO / "configs" / f"b1h_smart{f}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
        r = yaml.safe_load((REPO / "configs" / f"b1h_f{f}.yaml").read_text())
        r.update(name=f"b1h_f{f}_seed1", seed=1)
        (REPO / "configs" / f"b1h_f{f}_seed1.yaml").write_text(
            f"# b1h_f{f}_seed1: configs/b1h_f{f}.yaml (random subset) with training seed 1, to measure noise at "
            f"that size.\n" + yaml.safe_dump(r, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--data-root", default="data")
    e.add_argument("--out", required=True)
    e.add_argument("--max-images", type=int)
    s = sub.add_parser("select")
    s.add_argument("--embeddings", default="results/s54/embeddings.npz")
    s.add_argument("--write-configs", action="store_true")
    a = ap.parse_args()
    if a.cmd == "embed":
        return embed(a)
    z = np.load(a.embeddings, allow_pickle=False)
    names, emb, model = [str(n) for n in z["names"]], z["emb"].astype(np.float64), str(z["model"])
    expected = [n for n in pool_names() if n != EMPTY]
    if names != expected:
        sys.exit(f"embeddings cover {len(names)} images, expected the {len(expected)} pool images")
    boxes = pd.read_csv(EDA_TABLES / "boxes.csv")
    tr = boxes[(boxes.split == "train") & boxes.image.isin(set(pool_names()))]
    counts = pd.crosstab(tr.image, tr.class_name).reindex(names, fill_value=0)
    totals = tr.class_name.value_counts().to_dict()
    lc = json.loads((REPO / "splits" / "learning_curve_seed0.json").read_text())
    plans, report = {}, dict(embedding_model=model, n_candidates=len(names), excluded=[EMPTY], sizes={})
    rows = []
    for f, size in SIZES.items():
        lst, n_cover, share = select(names, emb, counts, size, totals)
        assert len(set(lst)) == size and EMPTY not in lst
        (REPO / "splits" / f"train_smart{f}_seed0.txt").write_text("\n".join(sorted(lst)) + "\n")
        plans[f] = p = subset_plan(lst)
        rnd = lc["subsets"][f"f{f}"]
        overlap = len(set(lst) & set(rnd["images"]))
        report["sizes"][f"smart{f}"] = dict(p, n_coverage_images=n_cover, share_targets=share,
                                            overlap_with_random=overlap, order=lst)
        for kind, q in ((f"smart{f}", p), (f"random f{f}", rnd)):
            r = dict(subset=kind, images=q["n_images"], tiles=q["tiles"], epochs=q["epochs"],
                     total_iters=q["total_iterations"])
            r.update({f"boxes {c}": q["boxes_per_class"].get(c, 0) for c in sorted(totals)})
            r.update({f"imgs {c}": q["images_per_class"].get(c, 0) for c in sorted(totals)})
            rows.append(r)
        rows[-2]["coverage_images"] = n_cover
        rows[-2]["overlap_with_random"] = overlap
    (REPO / "splits" / "smart_subsets_seed0.json").write_text(json.dumps(report, indent=1))
    if a.write_configs:
        write_configs(plans)
    print(f"[smart] embedding model: {model}")
    print(pd.DataFrame(rows).set_index("subset").T.to_string())


if __name__ == "__main__":
    main()
