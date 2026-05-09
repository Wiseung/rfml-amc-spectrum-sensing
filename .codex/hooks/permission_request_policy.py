#!/usr/bin/env python3
from __future__ import annotations

from hook_lib import deny_permission, find_policy_violation, read_stdin_json


def main() -> int:
    payload = read_stdin_json()
    reason = find_policy_violation(payload)
    if reason:
        deny_permission(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
