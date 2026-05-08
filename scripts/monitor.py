#!/usr/bin/env python3
"""Serve a lightweight local web UI for training progress."""

from __future__ import annotations

import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfml.monitor import collect_gpu_stats, list_run_snapshots, now_local_iso, render_dashboard_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/runs", help="Run directory root to scan")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    return parser.parse_args()


class MonitorHandler(BaseHTTPRequestHandler):
    root: Path
    refresh_seconds: float

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/", "/index.html"}:
            self.send_error(404, "Not Found")
            return
        root = type(self).root
        refresh_seconds = type(self).refresh_seconds
        runs = sorted(
            list_run_snapshots(root),
            key=lambda run: (run.live_status or {}).get("updated_at", ""),
            reverse=True,
        )
        html = render_dashboard_html(
            runs,
            root=root,
            gpu_stats=collect_gpu_stats(),
            refreshed_at=now_local_iso(),
            refresh_seconds=refresh_seconds,
        )
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    MonitorHandler.root = root
    MonitorHandler.refresh_seconds = float(args.refresh_seconds)
    server = ReusableThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
    print(f"monitor_root: {root}")
    print(f"monitor_url: http://{args.host}:{args.port}")
    print(f"auto_refresh_seconds: {args.refresh_seconds}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
