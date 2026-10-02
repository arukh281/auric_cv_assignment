"""Kaggle kernel entry point (pushed with the Kaggle CLI by scripts/kaggle_cli_kernel.sh; not run locally).

Copies the packaged code (private dataset auric-cv-code: code.tar.gz + CODE_COMMIT, or the folder Kaggle extracted
from it) to /kaggle/working/repo, runs scripts/kaggle_setup.sh with SKIP_PULL=1 (no GitHub token on Kaggle), then:
  MODE = "smoke": a 1-epoch B1h training on the tiles of the first SMOKE_IMAGES train images, into
                  /kaggle/working/smoke_runs (setup + tests + a short GPU check)
  MODE = "full":  scripts/run_b1h.sh, then scripts/run_analysis.sh b1h_tile1024_holdout40, into /kaggle/working/runs
The code commit is written to <run folder>/code_commit.txt. Weights stay in the kernel output.
"""
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MODE = "__MODE__"
SMOKE_IMAGES = 10  # first 10 train images, 3 are held out -> 7 images, 47 tiles expected (tools/make_holdout.expected_tiles)
RUN = "b1h_tile1024_holdout40"
REPO = Path("/kaggle/working/repo")


def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(["bash", "-c", cmd], cwd=REPO if REPO.exists() else None, env={**os.environ, "SKIP_PULL": "1"})
    if check and r.returncode:
        sys.exit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


def install_code():
    marks = sorted(Path("/kaggle/input").rglob("CODE_COMMIT"))
    src = next((m.parent for m in marks if (m.parent / "code.tar.gz").exists() or (m.parent / "scripts").is_dir()), None)
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
        out = Path("/kaggle/working/smoke_runs") / RUN
        out.mkdir(parents=True, exist_ok=True)
        (out / "code_commit.txt").write_text(commit + "\n")
        sh(f"source scripts/_env.sh && $PY train.py --config configs/b1h.yaml --data-root \"$DATA_ROOT\" "
           f"--runs-root /kaggle/working/smoke_runs --work-dir /kaggle/tmp/smoke_work --epochs 1 "
           f"--max-tile-images {SMOKE_IMAGES} --oom-fallback-batch 8")
        sh(f"cat {out}/tiling_params.json; cat {out}/training_summary.json; ls -la {out}/train/weights; "
           f"grep -h -i 'name\\|gpu' {out}/env/*hardware.json | head -20", check=False)
        print("SMOKE DONE", flush=True)
    elif MODE == "full":
        runs = Path("/kaggle/working/runs") / RUN
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "code_commit.txt").write_text(commit + "\n")
        sh("bash scripts/run_b1h.sh")
        sh(f"bash scripts/run_analysis.sh {RUN}")
        print("FULL DONE", flush=True)
    else:
        sys.exit(f"unknown MODE {MODE}")


if __name__ == "__main__":
    main()
