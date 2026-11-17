"""Load CANN for NPU workflows before loading native dependencies."""
import os
from pathlib import Path
import subprocess
import sys

def ensure_npu_environment(config):
    acc = (config.get("qlib_init") or {}).get("accelerator") or {}
    if acc.get("device_type") != "npu" or os.environ.get("QLIB_CANN_INITIALIZED") == "1":
        return
    override = os.environ.get("QLIB_CANN_SET_ENV")
    candidates = [Path(override)] if override else [
        Path("/usr/local/Ascend/cann/set_env.sh"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"),
        *sorted(Path("/usr/local/Ascend").glob("cann-*/set_env.sh"), reverse=True),
        Path("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
    ]
    script = next((p for p in candidates if p.is_file() and p.stat().st_size), None)
    if script is None:
        raise RuntimeError("Set QLIB_CANN_SET_ENV to the installed CANN set_env.sh.")
    result = subprocess.run(
        ["bash", "-c", 'source "$1" >&2 && env -0', "qlib-cann", str(script)],
        check=True, stdout=subprocess.PIPE,
    )
    env = dict(os.environ)
    for entry in result.stdout.split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            env[os.fsdecode(key)] = os.fsdecode(value)
    env["QLIB_CANN_INITIALIZED"] = "1"
    # Restart: the dynamic loader reads LD_LIBRARY_PATH at process startup.
    os.execve(sys.executable, [sys.executable, "-m", "qlib.cli.run", *sys.argv[1:]], env)
