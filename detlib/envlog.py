"""Write software/hardware details into a run folder."""
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # tool missing (e.g. nvidia-smi on a Mac)
        return f"unavailable: {e}"


def pip_freeze():
    return "\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                             if d.metadata["Name"]}, key=str.lower)) + "\n"


def hardware():
    info = dict(python=sys.version, platform=platform.platform(), processor=platform.processor())
    try:
        import torch
        info.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                    torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    gpu_mem_gib=[round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
                                 for i in range(torch.cuda.device_count())],
                    mps_available=torch.backends.mps.is_available())
    except ImportError:
        pass
    try:
        info["ultralytics"] = metadata.version("ultralytics")
    except metadata.PackageNotFoundError:
        pass
    info["nvidia_smi"] = _run(["nvidia-smi"])
    info["git_commit"] = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    info["git_dirty"] = bool(_run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]))
    return info


def log_env(run_dir, tag):
    """Write <run_dir>/env/{tag}_pip_freeze.txt and {tag}_hardware.json, and append the command line."""
    run_dir = Path(run_dir)
    (run_dir / "env").mkdir(parents=True, exist_ok=True)
    (run_dir / "env" / f"{tag}_pip_freeze.txt").write_text(pip_freeze())
    (run_dir / "env" / f"{tag}_hardware.json").write_text(json.dumps(hardware(), indent=2))
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(run_dir / "command.txt", "a") as f:
        f.write(f"# {stamp} ({tag})\ncd {shlex.quote(str(Path.cwd()))} && {shlex.join([sys.executable] + sys.argv)}\n")
