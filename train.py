"""Train one configured run with Ultralytics YOLO. Resumable: re-running the same command continues from
<run>/train/weights/last.pt if it exists.

Examples:
  python train.py --config configs/b0.yaml --data-root /content/data --runs-root /content/drive/MyDrive/auric_runs
  python train.py --config configs/b1.yaml --data-root /content/data --runs-root ... --work-dir /content/work

Run folder <runs-root>/<name>/: config.yaml (resolved), data.yaml, command.txt (every invocation),
env/train_* (pip freeze, GPU/CUDA/torch, git commit), init_weights.json (sha256 of the starting checkpoint),
train/ (Ultralytics output: args.yaml, results.csv, weights/last.pt, plots), training_summary.json
(batch actually used, iterations per epoch, total iterations), training_curves.png/.csv (per-epoch train/val
losses and Ultralytics val mAP).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from detlib.data import ensure_data_yaml, load_classes, write_resolved_data_yaml  # noqa: E402
from detlib.envlog import log_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="data", help="folder with train/ val/ classmap.txt")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--work-dir", default="work", help="scratch for tiles (local disk, not Drive)")
    ap.add_argument("--name", help="override run name")
    ap.add_argument("--no-resume", action="store_true", help="refuse to continue an existing run")
    ap.add_argument("--oom-fallback-batch", type=int, default=8,
                    help="if the configured batch runs out of GPU memory before epoch 1 is saved, retry with this "
                         "batch (recorded in config.yaml and training_summary.json); 0 disables")
    # overrides for smoke tests only; anything overridden is recorded in config.yaml
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device")
    ap.add_argument("--max-tile-images", type=int, help="testing only: tile only N train images")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    overrides = {k: getattr(a, k) for k in ("epochs", "imgsz", "batch", "workers", "device") if getattr(a, k) is not None}
    cfg.update(overrides)
    if a.name:
        cfg["name"] = a.name
    cfg["overrides"] = overrides
    run_dir = (Path(a.runs_root) / cfg["name"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(a.data_root).resolve()
    ensure_data_yaml(root)
    names = load_classes(root)

    tiling = cfg.get("tiling")
    if tiling:
        t = tiling
        holdout = cfg.get("holdout_list")
        subset = cfg.get("train_list")
        tiles = (Path(a.work_dir) / tiles_dir_name(t, REPO / holdout if holdout else None,
                                                   REPO / subset if subset else None)).resolve()
        cmd = [sys.executable, str(REPO / "tools" / "make_tiles.py"), "--data-root", str(root), "--out", str(tiles),
               "--tile", str(t["tile"]), "--overlap", str(t["overlap"]), "--empty-keep", str(t["empty_keep"]),
               "--min-vis", str(t["min_vis"]), "--seed", str(t["seed"])]
        if holdout:
            cmd += ["--exclude-list", str(REPO / holdout)]
        if subset:
            cmd += ["--include-list", str(REPO / subset)]
        if a.max_tile_images:
            cmd += ["--max-images", str(a.max_tile_images)]
            cfg["overrides"]["max_tile_images"] = a.max_tile_images
        subprocess.run(cmd, check=True)
        (run_dir / "tiling_params.json").write_text((tiles / "tiling_params.json").read_text())
        data_yaml = write_resolved_data_yaml(run_dir / "data.yaml", tiles, "train/images",
                                             str(root / "val" / "images"), names)
    else:
        if cfg.get("holdout_list") or cfg.get("train_list"):
            sys.exit("holdout_list / train_list are only supported for tiled configs (tiling: ...)")
        data_yaml = write_resolved_data_yaml(run_dir / "data.yaml", root, "train/images", "val/images", names)

    last = run_dir / "train" / "weights" / "last.pt"
    resuming = last.exists()
    if resuming and a.no_resume:
        sys.exit(f"{last} exists and --no-resume was given")
    if not resuming:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log_env(run_dir, "train_resume" if resuming else "train")

    from ultralytics import YOLO
    results = run_dir / "train" / "results.csv"
    done = max(sum(1 for _ in open(results)) - 1, 0) if results.exists() else 0
    if resuming and done >= cfg["epochs"]:
        print(f"[train] {run_dir.name} already finished ({done}/{cfg['epochs']} epochs); skipping training")
    elif resuming:
        print(f"[train] resuming from {last} ({done}/{cfg['epochs']} epochs done)")
        m = YOLO(str(last))
        add_checkpoint_callback(m, cfg.get("checkpoint_every"))
        m.train(resume=True)
    else:
        def fit(batch):
            model = YOLO(cfg["model"])
            init = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else Path(cfg["model"])
            if cfg.get("init_from"):  # architecture from cfg["model"] (a .yaml), weights from another checkpoint
                info = transfer_weights(model, cfg["init_from"])
                (run_dir / "init_weights.json").write_text(json.dumps({"model": cfg["model"], **info}, indent=2))
                print("[train] init_from:", json.dumps(info))
            elif init.exists():
                from eval import sha256
                (run_dir / "init_weights.json").write_text(json.dumps(
                    {"model": cfg["model"], "path": str(init.resolve()), "sha256": sha256(init)}, indent=2))
            add_checkpoint_callback(model, cfg.get("checkpoint_every"))
            model.train(data=str(data_yaml), project=str(run_dir), name="train", exist_ok=True,
                        epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=batch, workers=cfg.get("workers", 8),
                        seed=cfg["seed"], deterministic=cfg["deterministic"], device=cfg.get("device"),
                        **(cfg.get("train_args") or {}))

        used = train_with_fallback(fit, cfg["batch"], a.oom_fallback_batch, last)
        if used != cfg["batch"]:
            cfg["batch_requested"], cfg["batch"] = cfg["batch"], used
            cfg["batch_note"] = f"batch {cfg['batch_requested']} ran out of GPU memory; fell back to {used}"
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(f"[train] {cfg['batch_note']}")
    from detlib.curves import summarize_training
    print("[train] summary:", json.dumps(summarize_training(run_dir)))


def transfer_weights(model, src):
    """Copy every tensor of checkpoint `src` whose name and shape match into `model` (an Ultralytics YOLO built from a
    .yaml); everything else keeps its fresh initialisation. Returns counts and the names that did not transfer."""
    try:
        from ultralytics.nn.tasks import load_checkpoint
    except ImportError:  # older Ultralytics name
        from ultralytics.nn.tasks import attempt_load_one_weight as load_checkpoint
    from eval import sha256
    ckpt_model, _ = load_checkpoint(src)
    csd = ckpt_model.float().state_dict()
    msd = model.model.state_dict()
    match = {k: v for k, v in csd.items() if k in msd and v.shape == msd[k].shape}
    model.load(src)  # Ultralytics' own transfer (same name+shape rule); sets model.ckpt so train() starts from it
    after = model.model.state_dict()
    assert all(after[k].float().equal(v.float()) for k, v in match.items()), "transfer check failed"
    n_layers = len({k.rsplit(".", 1)[0] for k in msd})
    n_layers_tr = len({k.rsplit(".", 1)[0] for k in match})
    return {"init_from": str(src), "path": str(Path(src).resolve()) if Path(src).exists() else str(src),
            "sha256": sha256(src) if Path(src).exists() else None,
            "tensors_transferred": len(match), "tensors_total": len(msd), "tensors_in_source": len(csd),
            "modules_transferred": n_layers_tr, "modules_total": n_layers,
            "not_transferred": sorted(k for k in msd if k not in match)}


def tiles_dir_name(t, holdout=None, subset=None):
    """Tile cache folder name. Without a holdout list it is the original B1 name, so B1's cache is reused as before;
    with one, the list's content hash is appended, so a holdout run never reuses (or overwrites) full-train tiles."""
    name = f"tiles_{t['tile']}_ov{t['overlap']}_e{t['empty_keep']}_v{t['min_vis']}_s{t['seed']}"
    if holdout:
        name += "_ho" + hashlib.sha256(Path(holdout).read_bytes()).hexdigest()[:10]
    if subset:
        name += "_in" + hashlib.sha256(Path(subset).read_bytes()).hexdigest()[:10]
    return name


def add_checkpoint_callback(model, every):
    """Keep a copy of last.pt after every `every` COMPLETED epochs (epoch010.pt, epoch020.pt, ...).

    Ultralytics' save_period counts from epoch 0, so it would keep epochs 1, 11, 21, ...; this keeps 10, 20, 30.
    """
    if not every:
        return

    def keep(trainer):
        done = trainer.epoch + 1
        if done % every == 0 and trainer.last.exists():
            shutil.copy2(trainer.last, trainer.wdir / f"epoch{done:03d}.pt")

    model.add_callback("on_model_save", keep)


def is_oom(e):
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def train_with_fallback(fit, batch, fallback, last):
    """Call fit(batch); on a CUDA out-of-memory error before any epoch was saved, retry once with `fallback`.

    Returns the batch size that trained. An OOM after an epoch was saved is re-raised (resume instead).
    """
    try:
        fit(batch)
        return batch
    except Exception as e:  # noqa: BLE001 - only OOM is handled, everything else re-raised
        if not fallback or fallback >= batch or not is_oom(e) or Path(last).exists():
            raise
        print(f"[train] out of memory at batch {batch}: {e}. Retrying with batch {fallback}")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        fit(fallback)
        return fallback


if __name__ == "__main__":
    main()
