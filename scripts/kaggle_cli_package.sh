#!/usr/bin/env bash
# Mac side, Kaggle via CLI: package the code at HEAD (git archive; no data, weights or results/, and of figures/ only
# figures/eda) and upload it as the PRIVATE Kaggle dataset <user>/auric-cv-code (create once, then new versions with
# message = commit hash). Refuses a dirty working tree. Build folder: kaggle_build/code (gitignored).
# Auth: the Kaggle CLI's own credentials (~/.kaggle); nothing here reads or prints them.
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=${KAGGLE:-.venv/bin/kaggle}; USER_=${KAGGLE_USER:-aradhya1211}; SLUG=auric-cv-code
git diff --quiet HEAD -- || { echo "uncommitted changes to tracked files: commit first"; exit 1; }
C=$(git rev-parse HEAD)
D=kaggle_build/code; rm -rf "$D"; mkdir -p "$D"
git archive --format=tar HEAD -- . ':(exclude)results' ':(exclude)figures' > "$D/code.tar"
git archive --format=tar HEAD -- figures/eda > "$D/eda.tar"
tar -Af "$D/code.tar" "$D/eda.tar" && rm "$D/eda.tar"
printf '%s\n' "$C" > "$D/CODE_COMMIT"
tar -rf "$D/code.tar" -C "$D" CODE_COMMIT && gzip -9 "$D/code.tar"
printf '{"title": "auric-cv-code", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' "$USER_" "$SLUG" > "$D/dataset-metadata.json"
echo "package: $(du -h "$D/code.tar.gz" | cut -f1), commit $C, $(tar -tzf "$D/code.tar.gz" | wc -l | tr -d ' ') entries"
if $KAGGLE datasets status "$USER_/$SLUG" >/dev/null 2>&1; then
  $KAGGLE datasets version -p "$D" -m "$C" -q
else
  $KAGGLE datasets create -p "$D" -q          # private unless --public is given
fi
echo "uploaded $USER_/$SLUG @ $C"
