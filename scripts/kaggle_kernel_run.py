"""Kaggle kernel entry point (pushed with the Kaggle CLI by scripts/kaggle_cli_kernel.sh; not run locally).

Copies the packaged code (private dataset auric-cv-code: code.tar.gz + CODE_COMMIT, or the folder Kaggle extracted
from it) to /kaggle/working/repo, runs scripts/kaggle_setup.sh with SKIP_PULL=1 (no GitHub token on Kaggle), then:
  MODE = "smoke": a 1-epoch training of CONFIG on the tiles of the first SMOKE_IMAGES train images (after the config's
                  holdout/subset lists), into /kaggle/working/smoke_runs (setup + tests + a short GPU check)
  MODE = "full":  SCRIPT (e.g. scripts/run_b1h.sh or scripts/run_lc.sh <config>), then scripts/run_analysis.sh RUN,
                  into /kaggle/working/runs
  MODE = "script": any shell command (scripts/kaggle_cli_script_kernel.sh), run in the repo after setup
MODE, CONFIG, RUN and SCRIPT are filled in by scripts/kaggle_cli_kernel.sh / kaggle_cli_script_kernel.sh.
The code commit is written to <run folder>/code_commit.txt. Weights stay in the kernel output.
"""
import base64
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MODE = "__MODE__"
CONFIG = "__CONFIG__"
SCRIPT = "__SCRIPT__"
if SCRIPT.startswith("b64:"):  # MODE "script": arbitrary shell command, base64 so quotes and $ survive templating
    SCRIPT = base64.b64decode(SCRIPT[4:]).decode()
# smoke: first N train images before the config's lists. b1h: N = 10 -> 7 images, 47 tiles; with a train_list
# (learning-curve subsets) N = 30 -> 6 images of train_f25, 41 tiles (tools/make_holdout.expected_tiles)
SMOKE_IMAGES = 10
SMOKE_IMAGES_WITH_TRAIN_LIST = 30
RUN = "__RUN__"
REPO = Path("/kaggle/working/repo")


def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(["bash", "-c", cmd], cwd=REPO if REPO.exists() else None, env={**os.environ, "SKIP_PULL": "1"})
    if check and r.returncode:
        sys.exit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


def install_code():
    marks = sorted(Path("/kaggle/input").rglob("CODE_COMMIT"))
    # prefer the packaged dataset (code.tar.gz); a mounted kernel output (kernel_sources) also holds an older repo/ copy
    # Kaggle may auto-extract code.tar.gz, so also prefer any copy under the auric-cv-code dataset over kernel outputs
    src = next((m.parent for m in marks if (m.parent / "code.tar.gz").exists()), None) or \
        next((m.parent for m in marks if "auric-cv-code" in str(m) and (m.parent / "scripts").is_dir()), None) or \
        next((m.parent for m in marks if (m.parent / "scripts").is_dir()), None)
    if src is None:
        sys.exit(f"code dataset not found under /kaggle/input (CODE_COMMIT files: {marks})")
    if REPO.exists():
        shutil.rmtree(REPO)
    if (src / "code.tar.gz").exists():
        REPO.mkdir(parents=True)
        with tarfile.open(src / "code.tar.gz") as t:
            t.extractall(REPO)
    else:
        shutil.copytree(src, REPO)
    commit = (src / "CODE_COMMIT").read_text().strip()
    (REPO / "CODE_COMMIT").write_text(commit + "\n")
    print(f"code from {src}, commit {commit}", flush=True)
    return commit


def main():
    commit = install_code()
    sh("nvidia-smi; df -h /kaggle/working /kaggle/input /tmp / | sort -u; nproc; free -g", check=False)
    sh("bash scripts/kaggle_setup.sh")
    if MODE == "smoke":
        n_smoke = SMOKE_IMAGES_WITH_TRAIN_LIST if "train_list:" in (REPO / CONFIG).read_text() else SMOKE_IMAGES
        out = Path("/kaggle/working/smoke_runs") / RUN
        out.mkdir(parents=True, exist_ok=True)
        (out / "code_commit.txt").write_text(commit + "\n")
        sh(f"source scripts/_env.sh && $PY train.py --config {CONFIG} --data-root \"$DATA_ROOT\" "
           f"--runs-root /kaggle/working/smoke_runs --work-dir /kaggle/tmp/smoke_work --epochs 1 "
           f"--max-tile-images {n_smoke} --oom-fallback-batch 8")
        sh(f"cat {out}/tiling_params.json; cat {out}/training_summary.json; ls -la {out}/train/weights; "
           f"grep -h -i 'name\\|gpu' {out}/env/*hardware.json | head -20", check=False)
        print("SMOKE DONE", flush=True)
    elif MODE == "script":
        sh(SCRIPT)
        print("SCRIPT DONE", flush=True)
    elif MODE == "full":
        runs = Path("/kaggle/working/runs") / RUN
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "code_commit.txt").write_text(commit + "\n")
        sh(f"bash {SCRIPT}")
        sh(f"bash scripts/run_analysis.sh {RUN}")
        print("FULL DONE", flush=True)
    else:
        sys.exit(f"unknown MODE {MODE}")


if __name__ == "__main__":
    main()
