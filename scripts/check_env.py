#!/usr/bin/env python3
"""Environment validation for the RFML project."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_command(cmd: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 127, ""

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    text = stdout if stdout else stderr
    return completed.returncode, text


def print_section(title: str) -> None:
    print(f"\n[{title}]")


def main() -> int:
    print("RFML environment check")
    print(f"repo_root: {REPO_ROOT}")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"cwd: {Path.cwd()}")
    print(f"src_exists: {SRC_DIR.exists()}")
    print(f"pythonpath_contains_src: {str(SRC_DIR) in sys.path}")
    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")

    print_section("PATH")
    print(f"PATH_python3: {shutil.which('python3')}")
    print(f"PATH_pip: {shutil.which('pip')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    print_section("NVIDIA")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        code, out = run_command(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
        print(f"nvidia_smi: {nvidia_smi}")
        print(f"nvidia_smi_exit_code: {code}")
        print(f"nvidia_smi_output: {out or '<empty>'}")
    else:
        print("nvidia_smi: not found")

    print_section("Package Imports")
    try:
        import rfml  # noqa: F401

        print("rfml_import: ok")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"rfml_import: failed ({exc!r})")

    try:
        import torch

        print(f"torch_import: ok ({torch.__version__})")
        print(f"torch_cuda_available: {torch.cuda.is_available()}")
        print(f"torch_cuda_device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"torch_cuda_device_0: {torch.cuda.get_device_name(0)}")
            capability = torch.cuda.get_device_capability(0)
            print(f"torch_cuda_capability_0: {capability}")
    except Exception as exc:
        print(f"torch_import: failed ({exc!r})")
        print("hint: install PyTorch into the active environment before running training.")
        return 1

    print_section("Result")
    print("environment_check: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
