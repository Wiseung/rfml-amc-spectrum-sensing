#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any


SECRET_PATTERNS = (
    (
        "OpenAI API key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        "matches the OpenAI key format",
    ),
    (
        "GitHub token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "matches the GitHub token format",
    ),
    (
        "GitLab token",
        re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
        "matches the GitLab token format",
    ),
    (
        "AWS access key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "matches the AWS access key format",
    ),
    (
        "PEM private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
        "contains a PEM private key block",
    ),
    (
        "Slack token",
        re.compile(r"\bxox(?:b|p|a)-[A-Za-z0-9-]{10,}\b"),
        "matches the Slack token format",
    ),
)

SENSITIVE_FILE_PATTERNS = (
    re.compile(r"^\.env(?:\..+)?$", re.IGNORECASE),
    re.compile(r"^id_(?:rsa|dsa|ecdsa|ed25519)$", re.IGNORECASE),
    re.compile(r"^known_hosts$", re.IGNORECASE),
    re.compile(r"^credentials(?:\..+)?$", re.IGNORECASE),
    re.compile(r"^token(?:\..+)?$", re.IGNORECASE),
    re.compile(r"^secret(?:s)?(?:\..+)?$", re.IGNORECASE),
    re.compile(r"^.*\.(?:pem|key|p12|pfx)$", re.IGNORECASE),
)

FORBIDDEN_EDIT_SEGMENTS = {".git", "node_modules", "venv", ".venv", "dist", "build"}
HIGH_RISK_ABS_PREFIXES = ("/etc", "/usr", "/var")


def read_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_stdin": raw}
    return payload if isinstance(payload, dict) else {"payload": payload}


def print_json(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")


def block_pretool(reason: str) -> None:
    print_json(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


def deny_permission(reason: str) -> None:
    print_json(
        {
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": {
                    "behavior": "deny",
                    "message": f"Blocked by repository hook policy: {reason}",
                },
            }
        }
    )


def posttool_feedback(reason: str, additional_context: str | None = None) -> None:
    payload: dict[str, Any] = {
        "decision": "block",
        "reason": reason,
        "hookSpecificOutput": {"hookEventName": "PostToolUse"},
    }
    if additional_context:
        payload["hookSpecificOutput"]["additionalContext"] = additional_context
    print_json(payload)


def stop_continue(reason: str | None = None) -> None:
    payload: dict[str, Any] = {"continue": True}
    if reason:
        payload["reason"] = reason
    print_json(payload)


def success_no_output() -> None:
    return None


def detect_secrets(text: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not text:
        return findings
    for secret_type, pattern, reason in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append({"type": secret_type, "reason": reason})
    return findings


def extract_command(payload: dict[str, Any]) -> str:
    command = _find_first_string_by_keys(
        payload,
        ("command", "cmd", "bash_command", "script", "shell_command"),
    )
    if command:
        return command
    tool_input = _find_first_dict_by_keys(payload, ("tool_input", "toolInput", "input", "arguments", "params"))
    if tool_input:
        return _find_first_string_by_keys(
            tool_input,
            ("command", "cmd", "bash_command", "script", "shell_command"),
        )
    return ""


def repo_root_from_cwd(cwd: str | None) -> str:
    if not cwd:
        return ""
    current = Path(cwd).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return ""


def get_tool_name(payload: dict[str, Any]) -> str:
    name = _find_first_string_by_keys(payload, ("tool_name", "toolName", "tool", "name"))
    return name or ""


def get_prompt_text(payload: dict[str, Any]) -> str:
    prompt = _find_first_string_by_keys(payload, ("prompt", "user_prompt", "message", "text"))
    if prompt:
        return prompt
    return flatten_strings(payload)


def get_tool_output_text(payload: dict[str, Any]) -> str:
    output = _find_first_string_by_keys(
        payload,
        (
            "output",
            "stdout",
            "stderr",
            "combined_output",
            "message",
            "result",
            "response",
        ),
    )
    if output:
        return output
    return flatten_strings(payload)


def flatten_strings(value: Any) -> str:
    parts: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, str):
            parts.append(node)
            return
        if isinstance(node, dict):
            for inner in node.values():
                visit(inner)
            return
        if isinstance(node, list):
            for inner in node:
                visit(inner)

    visit(value)
    return "\n".join(parts)


def collect_paths(value: Any) -> list[str]:
    paths: list[str] = []

    def visit(node: Any, key_hint: str | None = None) -> None:
        if isinstance(node, dict):
            for key, inner in node.items():
                visit(inner, key)
            return
        if isinstance(node, list):
            for inner in node:
                visit(inner, key_hint)
            return
        if not isinstance(node, str):
            return
        if key_hint and key_hint.lower() in {"path", "paths", "destination", "source", "uri", "filename", "file"}:
            paths.append(node)
            return
        if node.startswith("/") or node.startswith("./") or node.startswith("../") or "/" in node:
            paths.append(node)

    visit(value)
    return paths


def find_policy_violation(payload: dict[str, Any]) -> str | None:
    tool_name = get_tool_name(payload)
    tool_name_lower = tool_name.lower()
    repo_root = repo_root_from_cwd(payload.get("cwd") or os.getcwd())

    if tool_name == "Bash":
        return find_high_risk_bash_reason(extract_command(payload), repo_root)

    if tool_name == "apply_patch":
        patch_text = flatten_strings(payload.get("tool_input", payload))
        return find_patch_violation_reason(patch_text, repo_root)

    if "edit" in tool_name_lower or "write" in tool_name_lower:
        return find_edit_violation_reason(payload, repo_root)

    if tool_name_lower.startswith("mcp__"):
        return find_mcp_violation_reason(payload, repo_root)

    return None


def find_high_risk_bash_reason(command: str, repo_root: str) -> str | None:
    if not command.strip():
        return None

    if detect_secrets(command):
        return "The command text appears to include a secret. Remove it before continuing."

    rm_reason = _find_destructive_rm_reason(command)
    if rm_reason:
        return rm_reason

    patterns = (
        (
            re.compile(r"\bsudo\s+(?:rm|chmod|chown)\b"),
            "Refusing a privileged destructive command.",
        ),
        (
            re.compile(r"\bchmod\s+-R\s+777\b"),
            "Refusing a recursive chmod 777 operation.",
        ),
        (
            re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sh|bash)\b"),
            "Refusing to pipe a remote script directly into a shell.",
        ),
        (
            re.compile(r"\bdd\b[^\n]*\bif=[^\s]+\b[^\n]*\bof=/dev/"),
            "Refusing a raw disk write command.",
        ),
        (
            re.compile(r"\b(?:mkfs(?:\.[A-Za-z0-9_+-]+)?|fdisk|parted)\b"),
            "Refusing a disk partitioning or formatting command.",
        ),
        (
            re.compile(r"\b(?:mount|umount)\b"),
            "Refusing a system-level mount or unmount command.",
        ),
    )
    for pattern, reason in patterns:
        if pattern.search(command):
            return reason

    if _looks_like_sensitive_file_read(command):
        return "Refusing to read or print a sensitive file."

    if _is_git_force_command(command) and not _has_safety_annotation(command):
        return "Refusing a force push or git clean -fdx without an explicit safety annotation."

    return _find_absolute_write_violation(command, repo_root)


def find_patch_violation_reason(patch_text: str, repo_root: str) -> str | None:
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("*** Add File: ") or line.startswith("*** Update File: ") or line.startswith("*** Delete File: "):
            paths.append(line.split(": ", 1)[1].strip())
        elif line.startswith("*** Move to: "):
            paths.append(line.split(": ", 1)[1].strip())
    for path in paths:
        reason = classify_edit_path(path, repo_root)
        if reason:
            return reason
    return None


def find_edit_violation_reason(payload: dict[str, Any], repo_root: str) -> str | None:
    for path in collect_paths(payload):
        reason = classify_edit_path(path, repo_root)
        if reason:
            return reason
    return None


def find_mcp_violation_reason(payload: dict[str, Any], repo_root: str) -> str | None:
    serialized = flatten_strings(payload)
    if detect_secrets(serialized):
        return "Refusing an MCP call that appears to include a secret."

    tool_name = get_tool_name(payload).lower()
    paths = collect_paths(payload)
    for path in paths:
        sensitive_reason = classify_sensitive_read_path(path)
        if sensitive_reason:
            return sensitive_reason
        if any(action in tool_name for action in ("write", "edit", "update", "create", "move", "rename", "delete")):
            reason = classify_edit_path(path, repo_root)
            if reason:
                return reason
    return None


def classify_edit_path(path_str: str, repo_root: str) -> str | None:
    normalized = _normalize_candidate_path(path_str, repo_root)
    path_obj = Path(normalized)
    basename = path_obj.name
    if _matches_sensitive_file_name(basename):
        return "Refusing to modify a sensitive file."

    lower_parts = {part.lower() for part in path_obj.parts}
    forbidden = FORBIDDEN_EDIT_SEGMENTS.intersection(lower_parts)
    if forbidden:
        segment = sorted(forbidden)[0]
        return f"Refusing to modify files under {segment}."

    if repo_root:
        repo_root_path = Path(repo_root).resolve()
        try:
            resolved = path_obj if path_obj.is_absolute() else (repo_root_path / path_obj).resolve(strict=False)
            if os.path.commonpath((str(repo_root_path), str(resolved))) != str(repo_root_path):
                return "Refusing to modify a path outside the repository root."
        except ValueError:
            return "Refusing to modify a path outside the repository root."

    if path_obj.is_absolute():
        for prefix in HIGH_RISK_ABS_PREFIXES:
            if str(path_obj).startswith(prefix):
                return f"Refusing to modify a high-risk system path under {prefix}."
        if str(path_obj).startswith(str(Path.home() / ".ssh")):
            return "Refusing to modify files under ~/.ssh."

    return None


def classify_sensitive_read_path(path_str: str) -> str | None:
    normalized = _normalize_candidate_path(path_str, "")
    path_obj = Path(normalized)
    if _matches_sensitive_file_name(path_obj.name):
        return "Refusing to access a sensitive file path."
    if "/.ssh/" in normalized or normalized.endswith("/.ssh"):
        return "Refusing to access a sensitive SSH path."
    return None


def _find_first_string_by_keys(node: Any, keys: tuple[str, ...]) -> str:
    if isinstance(node, dict):
        for key in keys:
            value = node.get(key)
            if isinstance(value, str):
                return value
        for value in node.values():
            found = _find_first_string_by_keys(value, keys)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_first_string_by_keys(value, keys)
            if found:
                return found
    return ""


def _find_first_dict_by_keys(node: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(node, dict):
        for key in keys:
            value = node.get(key)
            if isinstance(value, dict):
                return value
        for value in node.values():
            found = _find_first_dict_by_keys(value, keys)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_first_dict_by_keys(value, keys)
            if found:
                return found
    return {}


def _normalize_candidate_path(path_str: str, repo_root: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    if not expanded:
        return expanded
    if repo_root and not os.path.isabs(expanded):
        return str((Path(repo_root) / expanded).resolve(strict=False))
    return str(Path(expanded).resolve(strict=False) if os.path.isabs(expanded) else Path(expanded))


def _matches_sensitive_file_name(name: str) -> bool:
    for pattern in SENSITIVE_FILE_PATTERNS:
        if pattern.match(name):
            return True
    return False


def _looks_like_sensitive_file_read(command: str) -> bool:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return False
    if tokens[0] not in {"cat", "sed", "awk", "grep", "rg", "head", "tail", "less", "more", "nl", "tee"}:
        return False
    for token in tokens[1:]:
        cleaned = token.strip("\"'")
        if classify_sensitive_read_path(cleaned):
            return True
    return False


def _find_destructive_rm_reason(command: str) -> str | None:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    if not tokens or Path(tokens[0]).name != "rm":
        return None

    has_recursive = any(token.startswith("-") and "r" in token for token in tokens[1:])
    has_force = any(token.startswith("-") and "f" in token for token in tokens[1:])
    if not (has_recursive and has_force):
        return None

    critical_targets = {"/", "~", "$HOME", ".git"}
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        candidate = token.strip("\"'")
        if candidate in critical_targets or candidate.startswith(".git/"):
            return "Refusing a destructive rm -rf command against a critical path."
    return None


def _is_git_force_command(command: str) -> bool:
    return bool(
        re.search(r"\bgit\s+push\b[^\n]*\s--force(?:-with-lease)?\b", command)
        or re.search(r"\bgit\s+clean\b[^\n]*\s-(?:[^\n]*f[^\n]*d[^\n]*x|[^\n]*x[^\n]*f[^\n]*d)\b", command)
    )


def _has_safety_annotation(command: str) -> bool:
    markers = ("SAFE_FORCE_OK", "SAFE_CLEAN_OK", "--dry-run", "DRY_RUN")
    return any(marker in command for marker in markers)


def _find_absolute_write_violation(command: str, repo_root: str) -> str | None:
    if not repo_root:
        return None
    repo_root_path = Path(repo_root).resolve()

    for target in _extract_redirection_targets(command):
        violation = _absolute_write_target_violation(target, repo_root_path)
        if violation:
            return violation

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return None

    program = Path(tokens[0]).name
    non_option_args = [token for token in tokens[1:] if not token.startswith("-")]
    destination_candidates: list[str] = []

    if program in {"cp", "mv", "install", "ln"} and len(non_option_args) >= 2:
        destination_candidates.append(non_option_args[-1])
    elif program in {"touch", "mkdir", "truncate"}:
        destination_candidates.extend(non_option_args)
    elif program == "tee":
        destination_candidates.extend(non_option_args)

    for target in destination_candidates:
        violation = _absolute_write_target_violation(target, repo_root_path)
        if violation:
            return violation
    return None


def _extract_redirection_targets(command: str) -> list[str]:
    pattern = re.compile(r"(?:>|>>)\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s;&|]+))")
    targets: list[str] = []
    for match in pattern.finditer(command):
        for group in match.groups():
            if group:
                targets.append(group)
                break
    tee_pattern = re.compile(r"\btee\b(?:\s+-a)?\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s;&|]+))")
    for match in tee_pattern.finditer(command):
        for group in match.groups():
            if group:
                targets.append(group)
                break
    return targets


def _absolute_write_target_violation(target: str, repo_root_path: Path) -> str | None:
    expanded = os.path.expandvars(os.path.expanduser(target))
    if not os.path.isabs(expanded):
        return None
    resolved = Path(expanded).resolve(strict=False)
    try:
        if os.path.commonpath((str(repo_root_path), str(resolved))) == str(repo_root_path):
            return None
    except ValueError:
        pass
    for prefix in HIGH_RISK_ABS_PREFIXES:
        if str(resolved).startswith(prefix):
            return f"Refusing to write outside the repository root to {prefix}."
    if str(resolved).startswith(str(Path.home() / ".ssh")):
        return "Refusing to write outside the repository root under ~/.ssh."
    return "Refusing to write to an absolute path outside the repository root."
