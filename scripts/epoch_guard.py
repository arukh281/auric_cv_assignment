"""In-kernel time guard: after epoch 2, project the training finish (scripts/run_pair.project: hours at epoch 2 +
(epochs - 2) x duration of epoch 2, measured from this guard's start) and, if it exceeds --limit-hours, kill the
given process group. Writes <run>/epoch_guard.json either way.

  python scripts/epoch_guard.py --run-dir $RUNS/e9_b1h_yolo11m --epochs 50 --limit-hours 5 --pgid <pgid> &
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pair import project, read_epochs  # noqa: E402

T0 = time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--limit-hours", type=float, required=True); ap.add_argument("--pgid", type=int, required=True)
    a = ap.parse_args()
    res = Path(a.run_dir) / "train" / "results.csv"
    seen = {}
    while True:
        for ep, _ in read_epochs(res):
            seen.setdefault(ep, (time.time() - T0) / 3600)
        if {1, 2} <= set(seen):
            proj = project(seen[2], seen[2] - seen[1], a.epochs)
            info = dict(hours_at_epoch1=seen[1], hours_at_epoch2=seen[2], projected_training_hours=proj,
                        limit_hours=a.limit_hours, stopped=proj > a.limit_hours)
            Path(a.run_dir).mkdir(parents=True, exist_ok=True)
            (Path(a.run_dir) / "epoch_guard.json").write_text(json.dumps(info, indent=2))
            print("[guard]", json.dumps(info), flush=True)
            if proj > a.limit_hours:
                os.killpg(a.pgid, signal.SIGTERM)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
