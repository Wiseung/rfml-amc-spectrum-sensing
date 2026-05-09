#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from hook_lib import print_json, read_stdin_json, repo_root_from_cwd


MAX_CONTEXT_CHARS = 2000


def read_lines(path: Path, limit: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    snippet = "\n".join(lines[:limit]).strip()
    return snippet


def summarize_pyproject(path: Path) -> str:
    try:
        import tomllib
    except ModuleNotFoundError:
        return ""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    project = data.get("project", {})
    build_system = data.get("build-system", {})
    bits = []
    if project.get("name"):
        bits.append(f"name={project['name']}")
    if project.get("requires-python"):
        bits.append(f"python={project['requires-python']}")
    dependencies = project.get("dependencies") or []
    if dependencies:
        bits.append(f"deps={len(dependencies)}")
    requires = build_system.get("requires") or []
    if requires:
        bits.append("build=" + ",".join(requires[:3]))
    return "pyproject.toml: " + "; ".join(bits) if bits else ""


def summarize_package_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return ""
    bits = []
    if isinstance(data.get("name"), str):
        bits.append(f"name={data['name']}")
    if isinstance(data.get("packageManager"), str):
        bits.append(f"packageManager={data['packageManager']}")
    scripts = data.get("scripts")
    if isinstance(scripts, dict) and scripts:
        bits.append("scripts=" + ",".join(list(scripts.keys())[:5]))
    return "package.json: " + "; ".join(bits) if bits else ""


def summarize_cargo_toml(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("name =", "version =", "edition =")):
            lines.append(stripped)
        if len(lines) >= 3:
            break
    return "Cargo.toml: " + "; ".join(lines) if lines else ""


def summarize_go_mod(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    summary = []
    for line in lines[:20]:
        stripped = line.strip()
        if stripped.startswith(("module ", "go ")):
            summary.append(stripped)
    return "go.mod: " + "; ".join(summary) if summary else ""


def main() -> int:
    payload = read_stdin_json()
    repo_root = Path(repo_root_from_cwd(payload.get("cwd")) or Path.cwd())

    sections: list[str] = []
    agents_path = repo_root / "AGENTS.md"
    if agents_path.exists():
        snippet = read_lines(agents_path, 60)
        if snippet:
            sections.append("AGENTS.md:\n" + snippet)

    readme_path = repo_root / "README.md"
    if readme_path.exists():
        snippet = read_lines(readme_path, 80)
        if snippet:
            sections.append("README.md (first 80 lines):\n" + snippet)

    notes_path = repo_root / ".codex" / "hook-notes.md"
    if notes_path.exists():
        snippet = read_lines(notes_path, 60)
        if snippet:
            sections.append(".codex/hook-notes.md:\n" + snippet)

    for candidate, summarizer in (
        (repo_root / "pyproject.toml", summarize_pyproject),
        (repo_root / "package.json", summarize_package_json),
        (repo_root / "Cargo.toml", summarize_cargo_toml),
        (repo_root / "go.mod", summarize_go_mod),
    ):
        if candidate.exists():
            summary = summarizer(candidate)
            if summary:
                sections.append(summary)

    additional_context = "\n\n".join(sections).strip()
    if len(additional_context) > MAX_CONTEXT_CHARS:
        additional_context = additional_context[: MAX_CONTEXT_CHARS - 3].rstrip() + "..."

    print_json(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": additional_context,
            }
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
