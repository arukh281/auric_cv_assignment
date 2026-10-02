#!/usr/bin/env bash
# B1: 1024 tiles, 50 epochs -> eval (sliced) -> merge sensitivity -> checkpoint curve. Re-run to resume.
# Log: /content/drive/MyDrive/auric/runs/b1_tile1024/run.log
exec "$(dirname "$0")/_run.sh" configs/b1.yaml b1_tile1024 "$@"
