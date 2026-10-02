#!/usr/bin/env bash
# Copy one run's small result files (CSV/JSON/PNG, nothing over 20 MB, no weights/) from $RUNS into the repo
# (results/<run>/ and figures/<run>/), commit, and push with the Kaggle Secrets token "GH_TOKEN" (never printed).
# Usage: bash scripts/kaggle_push_results.sh <run name>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh
R=${1:?usage: bash scripts/kaggle_push_results.sh <run name>}
[ -d "$RUNS/$R" ] || { echo "no run folder $RUNS/$R"; exit 1; }
$PY - "$RUNS/$R" "results/$R" "$FIGS/$R" "figures/$R" <<'PYEOF'
import shutil, sys
from pathlib import Path
MAX = 20 * 1024 * 1024
for src, dst in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        continue
    n = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file() or "weights" in f.relative_to(src).parts or f.suffix.lower() not in (".csv", ".json", ".png"):
            continue
        if f.stat().st_size > MAX:
            print(f"SKIPPED (over 20 MB): {f}")
            continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
    print(f"copied {n} files {src} -> {dst}")
PYEOF
git add "results/$R" 2>/dev/null || true
git add "figures/$R" 2>/dev/null || true
if git diff --cached --quiet; then echo "nothing new to commit"; exit 0; fi
git -c user.name="aradhya khandelwal" -c user.email="arukhandelwal281@gmail.com" \
  commit -q -m "Results: $R (CSV/JSON/PNG from Kaggle)" && git log -1 --oneline
set +x
T=$($PY -c 'from kaggle_secrets import UserSecretsClient as U; print(U().get_secret("GH_TOKEN"))' 2>/dev/null || true)
[ -n "$T" ] || { echo "no GH_TOKEN secret: committed locally only"; exit 1; }
git pull -q --rebase "https://$T@github.com/arukh281/auric_cv_assignment.git" "$(git rev-parse --abbrev-ref HEAD)" || true
git push -q "https://$T@github.com/arukh281/auric_cv_assignment.git" "HEAD:$(git rev-parse --abbrev-ref HEAD)" \
  && echo "pushed $(git log -1 --oneline)" || { unset T; echo "push failed"; exit 1; }
unset T
