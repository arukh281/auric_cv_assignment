#!/usr/bin/env bash
# B0: full image 640, 50 epochs -> eval (+ Ultralytics cross-check) -> checkpoint curve. Re-run to resume.
# Log: /content/drive/MyDrive/auric/runs/b0_full640/run.log
exec "$(dirname "$0")/_run.sh" configs/b0.yaml b0_full640 --ultra-crosscheck "$@"
