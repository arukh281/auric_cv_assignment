#!/usr/bin/env bash
# Progress of B1/B0 (or the runs named as arguments): last finished epoch and losses, live batch progress,
# GPU usage, and whether training/eval is still running.   Usage: bash scripts/status.sh [run ...]
cd "$(dirname "$0")/.."
source scripts/_env.sh
RUNLIST=${*:-b1_tile1024 b0_full640}

for NAME in $RUNLIST; do
  echo "===== $NAME"
  R="$RUNS/$NAME/train/results.csv"
  if [ -f "$R" ]; then
    $PY - "$R" "$RUNS/$NAME/config.yaml" <<'EOF'
import sys, pandas as pd, yaml
r = pd.read_csv(sys.argv[1]); r.columns = [c.strip() for c in r.columns]
total = yaml.safe_load(open(sys.argv[2])).get("epochs", "?") if len(sys.argv) > 2 else "?"
last = r.iloc[-1]
pick = [c for c in r.columns if c.startswith(("train/", "val/")) or c in ("metrics/mAP50(B)", "time")]
print(f"epochs done: {int(last['epoch'])}/{total}")
print("  ".join(f"{c}={last[c]:.4g}" for c in pick))
if "time" in r and len(r) > 1:
    per = r["time"].diff().iloc[1:].median()
    left = (int(total) - int(last["epoch"])) if str(total).isdigit() else 0
    print(f"median epoch time {per/60:.1f} min -> ~{per*left/3600:.1f} h left (Ultralytics val mAP is un-sliced; curves only)")
EOF
  else
    echo "no finished epoch yet"
  fi
  ls "$RUNS/$NAME/train/weights" 2>/dev/null | tr '\n' ' '; echo
  L="$RUNS/$NAME/run.log"
  [ -f "$L" ] && { echo "log tail:"; tail -c 600 "$L" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 2; }
done

echo "===== processes"
pgrep -af 'train.py|eval.py|checkpoint_curve.py|merge_sensitivity.py|pred_review.py' | grep -v pgrep || echo "nothing running"
echo "===== GPU"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "no GPU visible"
