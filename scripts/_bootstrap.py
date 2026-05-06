#!/usr/bin/env python3
"""Runtime bootstrap helpers for project scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ENV_CANDIDATES = ("rfml", "zdyf")


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _existing_conda_envs(conda_executable: str) -> set[str]:
    completed = subprocess.run(
        [conda_executable, "env", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return set()

    env_names: set[str] = set()
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        if first not in {"*", "+"} and not first.startswith("/"):
            env_names.add(first)
    return env_names


def delegate_to_conda_if_needed(script_path: str) -> None:
    """Re-run the current script in a known-good Conda env if torch is missing.

    This keeps Phase 0 validation runnable on machines where the default
    interpreter is not the project interpreter yet.
    """

    if has_torch():
        return

    if os.environ.get("RFML_DELEGATED") == "1":
        return

    conda = shutil.which("conda")
    if not conda:
        return
    existing_envs = _existing_conda_envs(conda)
    if not existing_envs:
        return

    extra_candidates = []
    requested_env = os.environ.get("RFML_FALLBACK_CONDA_ENV")
    if requested_env:
        extra_candidates.append(requested_env)

    for env_name in [*extra_candidates, *PROJECT_ENV_CANDIDATES]:
        if env_name not in existing_envs:
            continue
        cmd = [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            str(Path(script_path).resolve()),
            *sys.argv[1:],
        ]
        env = dict(os.environ)
        env["RFML_DELEGATED"] = "1"
        env["RFML_DELEGATED_ENV"] = env_name

        completed = subprocess.run(cmd, check=False, env=env)
        if completed.returncode == 0:
            raise SystemExit(0)

    # If all delegation attempts fail, let the original interpreter continue and
    # report the local import failure normally.


def delegated_env_name() -> str | None:
    return os.environ.get("RFML_DELEGATED_ENV")
