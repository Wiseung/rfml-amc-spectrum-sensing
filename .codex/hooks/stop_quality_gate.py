#!/usr/bin/env python3
from __future__ import annotations

import re

from hook_lib import print_json, read_stdin_json, stop_continue


COMPLETE_PATTERNS = (
    re.compile(r"\b(?:done|completed|finished|implemented|installed)\b", re.IGNORECASE),
    re.compile(r"(?:已完成|完成了|已经完成|已安装|已实现)"),
)

VALIDATION_PATTERNS = (
    re.compile(r"`[^`]*(?:python|pytest|npm|cargo|go test|make test|uv run|poetry run)[^`]*`", re.IGNORECASE),
    re.compile(r"(?:测试命令|验证命令|运行测试|已验证|验证结果)"),
)

NOT_RUN_PATTERNS = (
    re.compile(r"(?:未运行|未验证|没有运行|无法运行)"),
    re.compile(r"(?:did not run|not run|unable to run)", re.IGNORECASE),
)

FILE_SUMMARY_PATTERNS = (
    re.compile(r"(?:修改文件|创建文件|文件列表|files changed|created files|modified files)", re.IGNORECASE),
    re.compile(r"\.codex/(?:config\.toml|hooks\.json|hooks/)"),
)


def claims_completion(message: str) -> bool:
    return any(pattern.search(message) for pattern in COMPLETE_PATTERNS)


def has_validation_summary(message: str) -> bool:
    has_file_summary = any(pattern.search(message) for pattern in FILE_SUMMARY_PATTERNS)
    has_validation = any(pattern.search(message) for pattern in VALIDATION_PATTERNS)
    has_not_run_reason = any(pattern.search(message) for pattern in NOT_RUN_PATTERNS)
    return has_file_summary and (has_validation or has_not_run_reason)


def main() -> int:
    payload = read_stdin_json()
    if payload.get("stop_hook_active") is True:
        stop_continue()
        return 0

    message = str(payload.get("last_assistant_message") or "")
    if claims_completion(message) and not has_validation_summary(message):
        print_json(
            {
                "decision": "block",
                "reason": "Before finalizing, provide a concise completion report: files changed, validation/tests run, results, and any remaining risks. If tests were not run, say why.",
            }
        )
        return 0

    stop_continue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
