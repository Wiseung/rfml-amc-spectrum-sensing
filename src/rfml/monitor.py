"""Lightweight training monitor utilities."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSnapshot:
    run_dir: Path
    train_log: list[dict[str, Any]]
    history: list[dict[str, Any]]
    live_status: dict[str, Any] | None
    checkpoint_info: dict[str, Any]
    summary: dict[str, Any] | None
    accuracy_vs_snr: list[dict[str, Any]] | None


def discover_run_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return []
    run_dirs = [path.parent for path in root_path.rglob("train_log.csv")]
    run_dirs.extend(path.parent for path in root_path.rglob("live_status.json"))
    return sorted(set(run_dirs))


def load_run_snapshot(run_dir: str | Path) -> RunSnapshot:
    run_path = Path(run_dir).expanduser().resolve()
    train_log = _read_csv(run_path / "train_log.csv")
    history = _read_json_list(run_path / "history.json")
    live_status = _read_json_dict(run_path / "live_status.json")
    checkpoint_info = _checkpoint_info(run_path)
    summary = _read_json_dict(run_path / "summary.json")
    accuracy_vs_snr = _read_csv_optional(run_path / "accuracy_vs_snr.csv")
    if accuracy_vs_snr is None:
        accuracy_vs_snr = _read_csv_optional(run_path / "modulation_accuracy_vs_snr.csv")
    return RunSnapshot(
        run_dir=run_path,
        train_log=train_log,
        history=history,
        live_status=live_status,
        checkpoint_info=checkpoint_info,
        summary=summary,
        accuracy_vs_snr=accuracy_vs_snr,
    )


def list_run_snapshots(root: str | Path) -> list[RunSnapshot]:
    return [load_run_snapshot(run_dir) for run_dir in discover_run_dirs(root)]


def latest_rows(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    return rows[-limit:]


def render_dashboard_html(
    runs: list[RunSnapshot],
    *,
    root: str | Path,
    gpu_stats: list[dict[str, str]],
    refreshed_at: str,
    refresh_seconds: float,
) -> str:
    run_cards = "\n".join(_render_run_card(run) for run in runs) or "<p>No training runs found.</p>"
    gpu_table = _render_gpu_table(gpu_stats)
    root_text = _escape(str(Path(root).expanduser().resolve()))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{max(1, int(refresh_seconds))}">
  <title>RFML Training Monitor</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #22c55e;
      --warn: #f59e0b;
      --error: #ef4444;
      --line-a: #38bdf8;
      --line-b: #f97316;
      --line-c: #a78bfa;
      --grid: rgba(148, 163, 184, 0.18);
      --border: rgba(148, 163, 184, 0.15);
      --mono: "JetBrains Mono", "Fira Code", monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top, #172554 0%, var(--bg) 45%, #020617 100%);
      color: var(--text);
      font-family: var(--sans);
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 20px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .muted {{ color: var(--muted); }}
    .mono {{ font-family: var(--mono); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }}
    .panel {{
      background: linear-gradient(180deg, rgba(31,41,55,0.95), rgba(15,23,42,0.95));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 18px 50px rgba(2, 6, 23, 0.35);
    }}
    .panel h2, .panel h3 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .metric {{
      background: rgba(15,23,42,0.7);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric .value {{
      font-size: 18px;
      font-weight: 600;
    }}
    .status {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border: 1px solid var(--border);
    }}
    .status-running {{ color: #4ade80; background: rgba(34, 197, 94, 0.12); }}
    .status-completed {{ color: #38bdf8; background: rgba(56, 189, 248, 0.12); }}
    .status-failed, .status-interrupted {{ color: #f87171; background: rgba(239, 68, 68, 0.12); }}
    svg {{
      width: 100%;
      height: auto;
      background: rgba(2, 6, 23, 0.5);
      border: 1px solid var(--border);
      border-radius: 12px;
      display: block;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .section {{
      margin-top: 12px;
    }}
    .small {{ font-size: 12px; }}
    .nowrap {{ white-space: nowrap; }}
    @media (max-width: 800px) {{
      .hero {{ flex-direction: column; }}
      .wrap {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero panel">
      <div>
        <h1>RFML Training Monitor</h1>
        <div class="muted">Run root: <span class="mono">{root_text}</span></div>
        <div class="muted">Refreshed: <span class="mono">{_escape(refreshed_at)}</span> | Auto refresh: {refresh_seconds:.1f}s</div>
      </div>
      <div style="min-width: 360px;">
        <h2>GPU</h2>
        {gpu_table}
      </div>
    </section>
    <section class="grid">
      {run_cards}
    </section>
  </div>
</body>
</html>"""


def collect_gpu_stats() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []
    if completed.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "temperature_c": parts[2],
                "utilization_pct": parts[3],
                "memory_used_mb": parts[4],
                "memory_total_mb": parts[5],
                "power_w": parts[6],
            }
        )
    return rows


def now_local_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    return [_normalize_row(row) for row in rows]


def _read_csv_optional(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return _read_csv(path)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            try:
                if value.isdigit():
                    normalized[key] = int(value)
                else:
                    normalized[key] = float(value)
            except ValueError:
                normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def _checkpoint_info(run_dir: Path) -> dict[str, Any]:
    items = {}
    for name in ("best.pt", "last.pt", "live_status.json"):
        path = run_dir / name
        if path.exists():
            stat = path.stat()
            items[name] = {
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "size_kb": round(stat.st_size / 1024.0, 1),
            }
    return items


def _render_run_card(run: RunSnapshot) -> str:
    latest = run.train_log[-1] if run.train_log else {}
    best = _best_metrics(run.train_log)
    live = run.live_status or {}
    status = str(live.get("status", "completed" if run.train_log else "idle")).lower()
    phase = _escape(str(live.get("phase", "-")))
    live_epoch = live.get("epoch")
    latest_epoch = _fmt(live_epoch if live_epoch is not None else latest.get("epoch"))
    best_val_acc = _fmt(best.get("best_val_acc"), precision=4)
    best_val_loss = _fmt(best.get("best_val_loss"), precision=4)
    latest_train_loss = _fmt(
        latest.get("train_loss") if run.train_log else live.get("running_loss"),
        precision=4,
    )
    latest_val_acc = _fmt(
        latest.get("val_acc") if run.train_log else live.get("running_acc"),
        precision=4,
    )
    progress_text = _escape(_progress_text(live))
    chart_svg = _render_training_svg(run.train_log)
    snr_svg = _render_snr_svg(run.accuracy_vs_snr)
    checkpoints = _render_checkpoint_table(run.checkpoint_info)
    summary_table = _render_summary_table(run.summary)
    recent_rows = _render_recent_rows(run.train_log)
    run_name = _escape(run.run_dir.name)
    run_path = _escape(str(run.run_dir))
    return f"""
    <article class="panel">
      <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
        <div>
          <h2>{run_name}</h2>
          <div class="muted mono small">{run_path}</div>
        </div>
        <div class="status status-{status}">{_escape(status)}</div>
      </div>
      <div class="meta">
        <div class="metric"><div class="label">Phase</div><div class="value">{phase}</div></div>
        <div class="metric"><div class="label">Epoch</div><div class="value">{latest_epoch}</div></div>
        <div class="metric"><div class="label">Best val acc</div><div class="value">{best_val_acc}</div></div>
        <div class="metric"><div class="label">Best val loss</div><div class="value">{best_val_loss}</div></div>
        <div class="metric"><div class="label">Train loss</div><div class="value">{latest_train_loss}</div></div>
        <div class="metric"><div class="label">Val acc</div><div class="value">{latest_val_acc}</div></div>
      </div>
      <div class="muted small" style="margin-bottom:12px;">{progress_text}</div>
      <div class="section">
        <h3>Loss / Accuracy</h3>
        {chart_svg}
      </div>
      <div class="section">
        <h3>Accuracy vs SNR</h3>
        {snr_svg}
      </div>
      <div class="section">
        <h3>Checkpoints</h3>
        {checkpoints}
      </div>
      <div class="section">
        <h3>Evaluation Summary</h3>
        {summary_table}
      </div>
      <div class="section">
        <h3>Recent Epochs</h3>
        {recent_rows}
      </div>
    </article>"""


def _best_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {"best_val_acc": None, "best_val_loss": None}
    val_accs = [float(row["val_acc"]) for row in rows if "val_acc" in row]
    val_losses = [float(row["val_loss"]) for row in rows if "val_loss" in row]
    return {
        "best_val_acc": max(val_accs) if val_accs else None,
        "best_val_loss": min(val_losses) if val_losses else None,
    }


def _render_checkpoint_table(info: dict[str, Any]) -> str:
    if not info:
        return "<p class='muted'>No checkpoint files found.</p>"
    rows = []
    for name, item in info.items():
        rows.append(
            f"<tr><td class='mono nowrap'>{_escape(name)}</td><td>{_escape(str(item.get('mtime', '-')))}</td><td>{_escape(str(item.get('size_kb', '-')))} KB</td></tr>"
        )
    body = "".join(rows)
    return f"<table><thead><tr><th>File</th><th>Modified</th><th>Size</th></tr></thead><tbody>{body}</tbody></table>"


def _render_summary_table(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "<p class='muted'>No evaluation summary yet.</p>"
    rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            continue
        rows.append(f"<tr><td class='mono'>{_escape(str(key))}</td><td>{_escape(str(value))}</td></tr>")
    body = "".join(rows[:12]) or "<tr><td colspan='2' class='muted'>No scalar summary fields.</td></tr>"
    return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{body}</tbody></table>"


def _render_recent_rows(rows: list[dict[str, Any]]) -> str:
    recent = latest_rows(rows, limit=8)
    if not recent:
        return "<p class='muted'>No epoch rows yet.</p>"
    body = "".join(
        "<tr>"
        f"<td>{_fmt(row.get('epoch'), precision=0)}</td>"
        f"<td>{_fmt(row.get('train_loss'), precision=4)}</td>"
        f"<td>{_fmt(row.get('train_acc'), precision=4)}</td>"
        f"<td>{_fmt(row.get('val_loss'), precision=4)}</td>"
        f"<td>{_fmt(row.get('val_acc'), precision=4)}</td>"
        f"<td>{_fmt(row.get('lr'), precision=6)}</td>"
        "</tr>"
        for row in recent
    )
    return (
        "<table><thead><tr><th>Epoch</th><th>Train loss</th><th>Train acc</th><th>Val loss</th><th>Val acc</th><th>LR</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _render_gpu_table(gpu_stats: list[dict[str, str]]) -> str:
    if not gpu_stats:
        return "<p class='muted'>nvidia-smi unavailable.</p>"
    body = "".join(
        "<tr>"
        f"<td>{_escape(gpu['index'])}</td>"
        f"<td>{_escape(gpu['name'])}</td>"
        f"<td>{_escape(gpu['temperature_c'])} C</td>"
        f"<td>{_escape(gpu['utilization_pct'])}%</td>"
        f"<td>{_escape(gpu['memory_used_mb'])} / {_escape(gpu['memory_total_mb'])} MB</td>"
        f"<td>{_escape(gpu['power_w'])} W</td>"
        "</tr>"
        for gpu in gpu_stats
    )
    return (
        "<table><thead><tr><th>ID</th><th>GPU</th><th>Temp</th><th>Util</th><th>Memory</th><th>Power</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _render_training_svg(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No training log yet.</p>"
    width, height = 900, 260
    margin_left, margin_right, margin_top, margin_bottom = 52, 18, 18, 34
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    epochs = [float(row.get("epoch", idx + 1)) for idx, row in enumerate(rows)]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_acc"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows]

    loss_min = min(train_loss + val_loss)
    loss_max = max(train_loss + val_loss)
    acc_min = min(train_acc + val_acc)
    acc_max = max(train_acc + val_acc)
    loss_min, loss_max = _expand_bounds(loss_min, loss_max)
    acc_min, acc_max = _expand_bounds(acc_min, acc_max)

    def x_map(value: float) -> float:
        if len(epochs) == 1:
            return margin_left + chart_width / 2.0
        return margin_left + (value - epochs[0]) / max(1e-8, epochs[-1] - epochs[0]) * chart_width

    def y_loss(value: float) -> float:
        return margin_top + chart_height * (1.0 - (value - loss_min) / max(1e-8, loss_max - loss_min))

    def y_acc(value: float) -> float:
        return margin_top + chart_height * (1.0 - (value - acc_min) / max(1e-8, acc_max - acc_min))

    grid_lines = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + chart_height * frac
        grid_lines.append(f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' stroke='var(--grid)' stroke-width='1' />")
    paths = [
        _svg_path(epochs, train_loss, x_map, y_loss, "var(--line-a)", "train_loss"),
        _svg_path(epochs, val_loss, x_map, y_loss, "var(--line-b)", "val_loss"),
        _svg_path(epochs, train_acc, x_map, y_acc, "var(--line-c)", "train_acc"),
        _svg_path(epochs, val_acc, x_map, y_acc, "#22c55e", "val_acc"),
    ]
    legend = (
        "<g font-size='12' font-family='var(--mono)'>"
        "<text x='66' y='22' fill='var(--line-a)'>train_loss</text>"
        "<text x='170' y='22' fill='var(--line-b)'>val_loss</text>"
        "<text x='252' y='22' fill='var(--line-c)'>train_acc</text>"
        "<text x='350' y='22' fill='#22c55e'>val_acc</text>"
        "</g>"
    )
    axes = (
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='var(--muted)' stroke-width='1' />"
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='var(--muted)' stroke-width='1' />"
    )
    return f"<svg viewBox='0 0 {width} {height}'>{''.join(grid_lines)}{axes}{''.join(paths)}{legend}</svg>"


def _render_snr_svg(rows: list[dict[str, Any]] | None) -> str:
    if not rows:
        return "<p class='muted'>No accuracy-vs-SNR file yet.</p>"
    width, height = 900, 220
    margin_left, margin_right, margin_top, margin_bottom = 50, 18, 18, 34
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    parsed = []
    for row in rows:
        if "snr" not in row or "accuracy" not in row:
            continue
        parsed.append((float(row["snr"]), float(row["accuracy"])))
    if not parsed:
        return "<p class='muted'>No valid SNR curve rows yet.</p>"
    parsed.sort(key=lambda item: item[0])
    xs = [item[0] for item in parsed]
    ys = [item[1] for item in parsed]
    y_min, y_max = _expand_bounds(min(ys), max(ys))
    x_min, x_max = min(xs), max(xs)

    def x_map(value: float) -> float:
        if x_max == x_min:
            return margin_left + chart_width / 2.0
        return margin_left + (value - x_min) / (x_max - x_min) * chart_width

    def y_map(value: float) -> float:
        return margin_top + chart_height * (1.0 - (value - y_min) / max(1e-8, y_max - y_min))

    grid_lines = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + chart_height * frac
        grid_lines.append(f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' stroke='var(--grid)' stroke-width='1' />")
    path = _svg_path(xs, ys, x_map, y_map, "#f59e0b", "accuracy_vs_snr", with_points=True)
    axes = (
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='var(--muted)' stroke-width='1' />"
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='var(--muted)' stroke-width='1' />"
    )
    return f"<svg viewBox='0 0 {width} {height}'>{''.join(grid_lines)}{axes}{path}</svg>"


def _svg_path(
    xs: list[float],
    ys: list[float],
    x_map,
    y_map,
    color: str,
    name: str,
    *,
    with_points: bool = False,
) -> str:
    commands = []
    points = []
    for idx, (x_value, y_value) in enumerate(zip(xs, ys, strict=True)):
        x = x_map(x_value)
        y = y_map(y_value)
        commands.append(f"{'M' if idx == 0 else 'L'} {x:.2f} {y:.2f}")
        if with_points:
            points.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='2.8' fill='{color}' />")
    label = f"<title>{_escape(name)}</title>"
    path = f"<path d='{' '.join(commands)}' fill='none' stroke='{color}' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>{label}</path>"
    return path + "".join(points)


def _expand_bounds(vmin: float, vmax: float) -> tuple[float, float]:
    if abs(vmax - vmin) < 1e-8:
        pad = 1.0 if abs(vmax) < 1e-8 else abs(vmax) * 0.1
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.08
    return vmin - pad, vmax + pad


def _fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if precision == 0:
            return str(int(round(float(value))))
        return f"{float(value):.{precision}f}"
    return _escape(str(value))


def _progress_text(live: dict[str, Any]) -> str:
    if not live:
        return "No live status available."
    epoch = live.get("epoch")
    num_epochs = live.get("num_epochs")
    batch = live.get("batch")
    num_batches = live.get("num_batches")
    elapsed_seconds = live.get("elapsed_seconds")
    parts = []
    if epoch is not None and num_epochs is not None:
        parts.append(f"epoch {epoch}/{num_epochs}")
    if batch is not None and num_batches is not None:
        parts.append(f"batch {batch}/{num_batches}")
    if elapsed_seconds is not None:
        try:
            parts.append(f"elapsed {_format_duration(float(elapsed_seconds))}")
        except (TypeError, ValueError):
            pass
    updated_at = live.get("updated_at")
    if updated_at:
        parts.append(f"updated {updated_at}")
    return " | ".join(parts) if parts else "No live progress yet."


def _format_duration(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
