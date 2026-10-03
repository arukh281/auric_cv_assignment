"""Run two configs at the same time on a 2-GPU Kaggle session (E1 on GPU 0, E2 on GPU 1) and apply the 2-epoch rule.

Usage (inside a Kaggle kernel, after scripts/kaggle_setup.sh):
  python scripts/run_pair.py configs/e1_b1h_150ep.yaml configs/e2_b1h_150ep_scale02.yaml [--limit-hours 10.5]

Each run is `bash scripts/run_lc.sh <config>` (train -> sliced eval on val -> prediction review -> merge sensitivity ->
checkpoint curve -> eval on holdout40 -> class-agnostic on both) followed by `bash scripts/run_analysis.sh <name>`,
started as its own process group with CUDA_VISIBLE_DEVICES=<i>, TRAIN_EXTRA="--workers 2" and its own WORK_DIR (own
tiles and disk cache, so the two processes never write the same files). Logs: $RUNS/<name>/pair_process.log.

Per-epoch wall time: every 30 s the driver reads $RUNS/<name>/train/results.csv; each newly finished epoch is
appended to $RUNS/<name>/epoch_times.csv (epoch, wall-clock UTC, hours since kernel start, Ultralytics `time`).

2-epoch rule (pre-agreed): once both runs have finished epoch 2, each run's training finish is projected as
  hours_since_kernel_start_at_epoch_2 + (epochs - 2) * duration_of_epoch_2
(epoch 2, not 1, because epoch 1 includes building the disk cache). If both projections are <= --limit-hours, both
continue. Otherwise the second run is stopped (its whole process group) and only the first continues; the second
must then be relaunched as its own kernel. The decision and the numbers go to $RUNS/pair_decision.json.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
T0 = time.time()


def hours():
    return (time.time() - T0) / 3600


def project(h_at_epoch2, dur_epoch2_h, epochs):
    """Projected hours since kernel start at which training finishes."""
    return h_at_epoch2 + (epochs - 2) * dur_epoch2_h


def decide(projections, limit):
    """projections: {name: hours}. Both continue only if every projection is within the limit."""
    return "both_continue" if all(v <= limit for v in projections.values()) else "stop_second"


def read_epochs(results_csv):
    if not results_csv.exists():
        return []
    lines = [l for l in results_csv.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    head = [c.strip() for c in lines[0].split(",")]
    out = []
    for l in lines[1:]:
        v = dict(zip(head, [x.strip() for x in l.split(",")]))
        try:
            out.append((int(float(v["epoch"])), float(v.get("time", "nan"))))
        except (KeyError, ValueError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs=2)
    ap.add_argument("--limit-hours", type=float, default=10.5)
    a = ap.parse_args()
    runs_root = Path(os.environ.get("RUNS", "/kaggle/working/runs"))
    work_root = Path(os.environ.get("WORK_DIR", "/kaggle/tmp/work"))
    runs = []
    for gpu, cfg in enumerate(a.configs):
        c = yaml.safe_load(open(REPO / cfg))
        name, epochs = c["name"], int(c["epochs"])
        (runs_root / name).mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TRAIN_EXTRA": "--workers 2",
               "WORK_DIR": str(work_root / name), "SKIP_PULL": "1"}
        log = open(runs_root / name / "pair_process.log", "a")
        cmd = f"bash scripts/run_lc.sh {cfg} && bash scripts/run_analysis.sh {name}"
        p = subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        runs.append(dict(name=name, cfg=cfg, epochs=epochs, gpu=gpu, proc=p, seen=0, ep_h={}))
        print(f"[pair] started {name} on GPU {gpu} (pid {p.pid}): {cmd}", flush=True)
    decision = None
    while any(r["proc"].poll() is None for r in runs):
        for r in runs:
            rows = read_epochs(runs_root / r["name"] / "train" / "results.csv")
            for ep, t in rows[r["seen"]:]:
                r["ep_h"][ep] = hours()
                with open(runs_root / r["name"] / "epoch_times.csv", "a") as f:
                    if ep == rows[0][0] and r["seen"] == 0:
                        f.write("epoch,utc,hours_since_kernel_start,ultralytics_time_s\n")
                    f.write(f"{ep},{datetime.now(timezone.utc).isoformat(timespec='seconds')},{hours():.4f},{t}\n")
                print(f"[pair] {r['name']} epoch {ep} done at {hours():.3f} h", flush=True)
            r["seen"] = len(rows)
        if decision is None and all({1, 2} <= set(r["ep_h"]) for r in runs):
            proj = {r["name"]: project(r["ep_h"][2], r["ep_h"][2] - r["ep_h"][1], r["epochs"]) for r in runs}
            per_epoch = {r["name"]: 60 * (r["ep_h"][2] - r["ep_h"][1]) for r in runs}
            decision = decide(proj, a.limit_hours)
            info = dict(decision=decision, limit_hours=a.limit_hours, projected_training_finish_hours=proj,
                        epoch2_minutes=per_epoch, hours_at_decision=hours(),
                        rule="both continue if every projection <= limit, else stop the second run")
            if decision == "stop_second":
                r2 = runs[1]
                os.killpg(r2["proc"].pid, signal.SIGTERM)
                info["stopped"] = r2["name"]
                print(f"[pair] STOPPED {r2['name']}", flush=True)
            (runs_root / "pair_decision.json").write_text(json.dumps(info, indent=2))
            print(f"[pair] decision: {json.dumps(info)}", flush=True)
        time.sleep(30)
    codes = {r["name"]: r["proc"].returncode for r in runs}
    print(f"[pair] finished at {hours():.2f} h; exit codes {codes}", flush=True)
    stopped = decision == "stop_second"
    bad = [n for i, (n, c) in enumerate(codes.items()) if c != 0 and not (stopped and i == 1)]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
