#!/usr/bin/env python3
from __future__ import annotations

import re

from hook_lib import detect_secrets, get_tool_output_text, posttool_feedback, read_stdin_json


FAILURE_PATTERNS = (
    (re.compile(r"command not found", re.IGNORECASE), "Command execution failed because a command was missing."),
    (re.compile(r"ModuleNotFoundError"), "Python reported a missing module."),
    (re.compile(r"ImportError"), "Python reported an import failure."),
    (re.compile(r"Traceback \(most recent call last\):"), "The tool output includes a Python traceback."),
    (re.compile(r"\bTypeError\b"), "The tool output includes a TypeError."),
    (re.compile(r"\bSyntaxError\b"), "The tool output includes a SyntaxError."),
    (re.compile(r"\bnpm ERR!\b"), "npm reported an error."),
    (re.compile(r"\bcargo test failed\b", re.IGNORECASE), "cargo test reported a failure."),
    (re.compile(r"\bpytest\b.*\bfailed\b", re.IGNORECASE), "pytest reported failures."),
    (re.compile(r"\btests? failed\b", re.IGNORECASE), "The tool output reports test failures."),
    (re.compile(r"=+\s+FAILURES\s+=+"), "The tool output includes pytest failure sections."),
)


def main() -> int:
    payload = read_stdin_json()
    output = get_tool_output_text(payload)

    if detect_secrets(output):
        posttool_feedback(
            "The last tool output may contain a secret. Stop propagating it and advise credential rotation if exposure is confirmed.",
            "The output appears to include a possible secret. Do not repeat it. Contain the exposure and recommend rotating the credential.",
        )
        return 0

    for pattern, summary in FAILURE_PATTERNS:
        if pattern.search(output):
            posttool_feedback(
                "The last tool output indicates a failure that should be addressed before continuing.",
                f"{summary} Summarize the failure, fix the root cause, and rerun the relevant check.",
            )
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
