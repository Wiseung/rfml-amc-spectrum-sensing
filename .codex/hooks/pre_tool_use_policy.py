#!/usr/bin/env python3
from __future__ import annotations

from hook_lib import block_pretool, find_policy_violation, read_stdin_json


def main() -> int:
    payload = read_stdin_json()
    reason = find_policy_violation(payload)
    if reason:
        block_pretool(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
