"""Lightweight training monitor utilities."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


@dataclass(frozen=True)
class RunSnapshot:
    run_dir: Path
    eval_dir: Path | None
    train_log: list[dict[str, Any]]
    history: list[dict[str, Any]]
    live_status: dict[str, Any] | None
    checkpoint_info: dict[str, Any]
    summary: dict[str, Any] | None
    accuracy_vs_snr: list[dict[str, Any]] | None


@dataclass(frozen=True)
class SweepRow:
    family: str
    task: str
    count: int
    best_run_name: str
    best_metric_name: str
    best_metric_value: float
    latest_status: str
    latest_updated_at: str


@dataclass(frozen=True)
class DashboardFilters:
    task: str = "all"
    status: str = "all"
    family: str = "all"


def discover_run_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return []
    run_dirs = [path.parent for path in root_path.rglob("train_log.csv")]
    run_dirs.extend(path.parent for path in root_path.rglob("live_status.json"))
    return sorted(set(run_dirs))


def load_run_snapshot(run_dir: str | Path) -> RunSnapshot:
    run_path = Path(run_dir).expanduser().resolve()
    eval_dir = _find_eval_dir(run_path)
    train_log = _read_csv(run_path / "train_log.csv")
    history = _read_json_list(run_path / "history.json")
    live_status = _read_json_dict(run_path / "live_status.json")
    checkpoint_info = _checkpoint_info(run_path)
    summary = _read_json_dict(run_path / "summary.json")
    accuracy_vs_snr = _read_csv_optional(run_path / "accuracy_vs_snr.csv")
    if accuracy_vs_snr is None:
        accuracy_vs_snr = _read_csv_optional(run_path / "modulation_accuracy_vs_snr.csv")
    if eval_dir is not None:
        if summary is None:
            summary = _read_json_dict(eval_dir / "summary.json")
        if accuracy_vs_snr is None:
            accuracy_vs_snr = _read_csv_optional(eval_dir / "accuracy_vs_snr.csv")
        if accuracy_vs_snr is None:
            accuracy_vs_snr = _read_csv_optional(eval_dir / "modulation_accuracy_vs_snr.csv")
    return RunSnapshot(
        run_dir=run_path,
        eval_dir=eval_dir,
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
    filters: DashboardFilters | None = None,
) -> str:
    active_filters = filters or DashboardFilters()
    filtered_runs = filter_runs(runs, active_filters)
    filter_options = build_filter_options(runs)
    overview = summarize_runs(filtered_runs)
    leaderboard = build_leaderboard_rows(filtered_runs)
    sweep_rows = build_sweep_rows(filtered_runs)
    family_trends = build_family_trends(filtered_runs)
    recent_table = build_recent_run_rows(filtered_runs)
    run_cards = "\n".join(_render_run_card(run) for run in filtered_runs) or "<p>No training runs found for the selected filters.</p>"
    gpu_table = _render_gpu_table(gpu_stats)
    overview_cards = _render_overview_cards(overview)
    leaderboard_tables = _render_leaderboard_tables(leaderboard)
    sweep_table = _render_sweep_table(sweep_rows)
    filter_panel = _render_filter_panel(active_filters, filter_options, len(filtered_runs), len(runs))
    family_trend_grid = _render_family_trend_grid(family_trends)
    recent_runs_table = _render_recent_runs_table(recent_table)
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
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }}
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 16px 0;
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
    .scroll-x {{ overflow-x: auto; }}
    .task-chip {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid var(--border);
      color: #7dd3fc;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      align-items: end;
    }}
    .toolbar label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .toolbar select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.9);
      color: var(--text);
      font-family: var(--sans);
    }}
    .toolbar-actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 0 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
      color: var(--text);
      text-decoration: none;
      background: rgba(56, 189, 248, 0.12);
      font-weight: 600;
    }}
    .button-secondary {{
      background: rgba(15, 23, 42, 0.8);
    }}
    .family-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
    .family-mini {{
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.6);
    }}
    .family-mini h3 {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .family-mini svg {{
      min-height: 126px;
    }}
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
    <section class="panel">
      <h2>Filters</h2>
      {filter_panel}
    </section>
    <section class="panel">
      <h2>Experiment Overview</h2>
      <div class="grid-4">
        {overview_cards}
      </div>
    </section>
    <section class="grid-2">
      <article class="panel">
        <h2>Task Leaderboard</h2>
        {leaderboard_tables}
      </article>
      <article class="panel">
        <h2>Sweep Families</h2>
        {sweep_table}
      </article>
    </section>
    <section class="panel">
      <h2>Family Trends</h2>
      {family_trend_grid}
    </section>
    <section class="panel">
      <h2>Recent Runs</h2>
      {recent_runs_table}
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


def filter_runs(runs: list[RunSnapshot], filters: DashboardFilters) -> list[RunSnapshot]:
    filtered: list[RunSnapshot] = []
    for run in runs:
        task = _infer_task(run)
        status = _run_status(run)
        family = _family_name(run.run_dir.name)
        if filters.task != "all" and task != filters.task:
            continue
        if filters.status != "all" and status != filters.status:
            continue
        if filters.family != "all" and family != filters.family:
            continue
        filtered.append(run)
    return filtered


def build_filter_options(runs: list[RunSnapshot]) -> dict[str, list[str]]:
    tasks = sorted({_infer_task(run) for run in runs})
    statuses = sorted({_run_status(run) for run in runs})
    families = sorted({_family_name(run.run_dir.name) for run in runs})
    return {
        "task": ["all", *tasks],
        "status": ["all", *statuses],
        "family": ["all", *families],
    }


def summarize_runs(runs: list[RunSnapshot]) -> dict[str, Any]:
    running = 0
    completed = 0
    with_summary = 0
    best_amc = None
    best_sensing_auc = None
    best_multitask = None
    for run in runs:
        status = _run_status(run)
        if status == "running":
            running += 1
        if status == "completed":
            completed += 1
        if run.summary:
            with_summary += 1
            metric_name, metric_value = _primary_metric(run)
            if metric_name == "overall_accuracy":
                if best_amc is None or metric_value > best_amc[1]:
                    best_amc = (run.run_dir.name, metric_value)
            elif metric_name == "roc_auc":
                if best_sensing_auc is None or metric_value > best_sensing_auc[1]:
                    best_sensing_auc = (run.run_dir.name, metric_value)
            elif metric_name == "modulation_accuracy":
                if best_multitask is None or metric_value > best_multitask[1]:
                    best_multitask = (run.run_dir.name, metric_value)
    return {
        "run_count": len(runs),
        "running_count": running,
        "completed_count": completed,
        "evaluated_count": with_summary,
        "best_amc": best_amc,
        "best_sensing_auc": best_sensing_auc,
        "best_multitask": best_multitask,
    }


def build_leaderboard_rows(runs: list[RunSnapshot]) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "amc": [],
        "spectrum_sensing": [],
        "multitask": [],
    }
    for run in runs:
        if not run.summary:
            continue
        task = str(run.summary.get("task", _infer_task(run))).lower()
        row = {
            "run_name": run.run_dir.name,
            "task": task,
            "status": _run_status(run),
            "updated_at": _run_updated_at(run),
        }
        if task == "amc":
            row["metric_name"] = "overall_accuracy"
            row["metric_value"] = float(run.summary.get("overall_accuracy", float("nan")))
            buckets["amc"].append(row)
        elif task == "spectrum_sensing":
            row["metric_name"] = "roc_auc"
            row["metric_value"] = float(run.summary.get("roc_auc", run.summary.get("overall_accuracy", float("nan"))))
            row["secondary"] = float(run.summary.get("overall_accuracy", float("nan")))
            buckets["spectrum_sensing"].append(row)
        elif task == "multitask":
            row["metric_name"] = "modulation_accuracy"
            row["metric_value"] = float(run.summary.get("modulation_accuracy", float("nan")))
            row["secondary"] = float(run.summary.get("roc_auc", float("nan")))
            buckets["multitask"].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda item: item["metric_value"], reverse=True)
    return buckets


def build_sweep_rows(runs: list[RunSnapshot]) -> list[SweepRow]:
    families: dict[tuple[str, str], list[RunSnapshot]] = {}
    for run in runs:
        family = _family_name(run.run_dir.name)
        task = _infer_task(run)
        families.setdefault((family, task), []).append(run)

    rows: list[SweepRow] = []
    for (family, task), family_runs in families.items():
        best_run = None
        best_metric_name = "-"
        best_metric_value = float("-inf")
        latest_run = max(family_runs, key=_run_updated_epoch_key)
        for run in family_runs:
            metric_name, metric_value = _primary_metric(run)
            if metric_value == metric_value and metric_value > best_metric_value:
                best_metric_value = metric_value
                best_metric_name = metric_name
                best_run = run
        if best_run is None:
            best_run = latest_run
            best_metric_name = "-"
            best_metric_value = float("nan")
        rows.append(
            SweepRow(
                family=family,
                task=task,
                count=len(family_runs),
                best_run_name=best_run.run_dir.name,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                latest_status=_run_status(latest_run),
                latest_updated_at=_run_updated_at(latest_run),
            )
        )
    rows.sort(
        key=lambda row: (
            row.task,
            -(row.best_metric_value if row.best_metric_value == row.best_metric_value else float("-inf")),
            row.family,
        )
    )
    return rows


def build_family_trends(runs: list[RunSnapshot]) -> list[dict[str, Any]]:
    families: dict[tuple[str, str], list[RunSnapshot]] = {}
    for run in runs:
        families.setdefault((_family_name(run.run_dir.name), _infer_task(run)), []).append(run)

    trends: list[dict[str, Any]] = []
    for (family, task), family_runs in sorted(families.items()):
        points = []
        for run in sorted(family_runs, key=_family_run_sort_key):
            metric_name, metric_value = _primary_metric(run)
            if metric_value != metric_value:
                continue
            points.append(
                {
                    "run_name": run.run_dir.name,
                    "label": _short_round_label(run.run_dir.name),
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "status": _run_status(run),
                    "updated_at": _run_updated_at(run),
                }
            )
        if not points:
            continue
        trends.append(
            {
                "family": family,
                "task": task,
                "metric_name": points[-1]["metric_name"],
                "points": points,
                "best_value": max(point["metric_value"] for point in points),
                "latest_value": points[-1]["metric_value"],
            }
        )
    trends.sort(key=lambda item: (item["task"], item["family"]))
    return trends


def build_recent_run_rows(runs: list[RunSnapshot]) -> list[dict[str, Any]]:
    rows = []
    for run in sorted(runs, key=_run_updated_epoch_key, reverse=True):
        metric_name, metric_value = _primary_metric(run)
        rows.append(
            {
                "run_name": run.run_dir.name,
                "task": _infer_task(run),
                "status": _run_status(run),
                "updated_at": _run_updated_at(run),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "epoch": _run_latest_epoch(run),
            }
        )
    return rows[:12]


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


def _find_eval_dir(run_dir: Path) -> Path | None:
    parent = run_dir.parent
    candidates = []
    for path in parent.glob(f"{run_dir.name}_eval*"):
        if not path.is_dir():
            continue
        if (path / "summary.json").exists() or (path / "accuracy_vs_snr.csv").exists() or (path / "modulation_accuracy_vs_snr.csv").exists():
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _render_overview_cards(overview: dict[str, Any]) -> str:
    best_amc = overview.get("best_amc")
    best_sensing_auc = overview.get("best_sensing_auc")
    best_multitask = overview.get("best_multitask")
    cards = [
        ("runs", _fmt(overview.get("run_count"), precision=0), "discovered run directories"),
        ("running", _fmt(overview.get("running_count"), precision=0), "currently active runs"),
        ("evaluated", _fmt(overview.get("evaluated_count"), precision=0), "runs with summary.json"),
        (
            "best AMC",
            _fmt(best_amc[1], precision=4) if best_amc else "-",
            best_amc[0] if best_amc else "no evaluated AMC run",
        ),
        (
            "best sensing AUC",
            _fmt(best_sensing_auc[1], precision=4) if best_sensing_auc else "-",
            best_sensing_auc[0] if best_sensing_auc else "no sensing run",
        ),
        (
            "best multitask AMC",
            _fmt(best_multitask[1], precision=4) if best_multitask else "-",
            best_multitask[0] if best_multitask else "no multitask run",
        ),
    ]
    return "".join(
        f"<div class='metric'><div class='label'>{_escape(label)}</div><div class='value'>{_escape(value)}</div><div class='muted small'>{_escape(desc)}</div></div>"
        for label, value, desc in cards
    )


def _render_filter_panel(
    filters: DashboardFilters,
    options: dict[str, list[str]],
    filtered_count: int,
    total_count: int,
) -> str:
    task_options = _render_select_options(options.get("task", ["all"]), filters.task)
    status_options = _render_select_options(options.get("status", ["all"]), filters.status)
    family_options = _render_select_options(options.get("family", ["all"]), filters.family)
    return f"""
    <form method="get" action="/" class="toolbar">
      <div>
        <label for="task">Task</label>
        <select id="task" name="task">{task_options}</select>
      </div>
      <div>
        <label for="status">Status</label>
        <select id="status" name="status">{status_options}</select>
      </div>
      <div>
        <label for="family">Family</label>
        <select id="family" name="family">{family_options}</select>
      </div>
      <div>
        <label>Result Set</label>
        <div class="muted small">{filtered_count} / {total_count} runs shown</div>
      </div>
      <div class="toolbar-actions">
        <button class="button" type="submit">Apply Filters</button>
        <a class="button button-secondary" href="/">Clear</a>
      </div>
    </form>
    """


def _render_leaderboard_tables(leaderboard: dict[str, list[dict[str, Any]]]) -> str:
    sections = []
    titles = {
        "amc": "AMC",
        "spectrum_sensing": "Spectrum Sensing",
        "multitask": "Multi-task",
    }
    for task_key in ("amc", "spectrum_sensing", "multitask"):
        rows = leaderboard.get(task_key, [])
        if not rows:
            sections.append(f"<div class='section'><h3>{titles[task_key]}</h3><p class='muted'>No evaluated runs.</p></div>")
            continue
        table_rows = []
        for idx, row in enumerate(rows[:5], start=1):
            secondary = ""
            if "secondary" in row and row["secondary"] == row["secondary"]:
                if task_key == "spectrum_sensing":
                    secondary = f" | acc {_fmt(row['secondary'], precision=4)}"
                elif task_key == "multitask":
                    secondary = f" | AUC {_fmt(row['secondary'], precision=4)}"
            table_rows.append(
                "<tr>"
                f"<td>{idx}</td>"
                f"<td class='mono'>{_escape(row['run_name'])}</td>"
                f"<td>{_fmt(row['metric_value'], precision=4)}{_escape(secondary)}</td>"
                f"<td>{_escape(row['status'])}</td>"
                "</tr>"
            )
        sections.append(
            f"<div class='section'><h3>{titles[task_key]}</h3>"
            "<div class='scroll-x'><table><thead><tr><th>#</th><th>Run</th><th>Primary Metric</th><th>Status</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody></table></div></div>"
        )
    return "".join(sections)


def _render_sweep_table(rows: list[SweepRow]) -> str:
    if not rows:
        return "<p class='muted'>No sweep families found.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td class='mono'>{_escape(row.family)}</td>"
            f"<td><span class='task-chip'>{_escape(row.task)}</span></td>"
            f"<td>{row.count}</td>"
            f"<td class='mono'>{_escape(row.best_run_name)}</td>"
            f"<td>{_escape(row.best_metric_name)} = {_fmt(row.best_metric_value, precision=4)}</td>"
            f"<td>{_escape(row.latest_status)}</td>"
            f"<td class='nowrap'>{_escape(row.latest_updated_at)}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll-x'><table><thead><tr><th>Family</th><th>Task</th><th>Runs</th><th>Best Run</th>"
        "<th>Best Metric</th><th>Latest Status</th><th>Latest Update</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _render_family_trend_grid(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No family trends found for the selected filters.</p>"
    cards = []
    for row in rows:
        cards.append(
            "<article class='family-mini'>"
            f"<h3><span class='mono'>{_escape(row['family'])}</span><span class='task-chip'>{_escape(row['task'])}</span></h3>"
            f"<div class='muted small' style='margin-bottom:8px;'>metric: {_escape(row['metric_name'])} | latest {_fmt(row['latest_value'], precision=4)} | best {_fmt(row['best_value'], precision=4)}</div>"
            f"{_render_family_trend_svg(row)}"
            "</article>"
        )
    return f"<div class='family-grid'>{''.join(cards)}</div>"


def _render_recent_runs_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No runs found.</p>"
    body = []
    for row in rows:
        metric_text = "-"
        if row["metric_value"] == row["metric_value"]:
            metric_text = f"{row['metric_name']}={_fmt(row['metric_value'], precision=4)}"
        body.append(
            "<tr>"
            f"<td class='mono'>{_escape(row['run_name'])}</td>"
            f"<td><span class='task-chip'>{_escape(row['task'])}</span></td>"
            f"<td>{_escape(row['status'])}</td>"
            f"<td>{_fmt(row['epoch'], precision=0)}</td>"
            f"<td>{_escape(metric_text)}</td>"
            f"<td class='nowrap'>{_escape(row['updated_at'])}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll-x'><table><thead><tr><th>Run</th><th>Task</th><th>Status</th><th>Epoch</th><th>Primary Metric</th><th>Updated</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _render_family_trend_svg(row: dict[str, Any]) -> str:
    points = row["points"]
    if not points:
        return "<p class='muted'>No trend points.</p>"
    width, height = 360, 126
    margin_left, margin_right, margin_top, margin_bottom = 18, 14, 14, 28
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    xs = list(range(len(points)))
    ys = [float(point["metric_value"]) for point in points]
    y_min, y_max = _expand_bounds(min(ys), max(ys))

    def x_map(value: float) -> float:
        if len(xs) == 1:
            return margin_left + chart_width / 2.0
        return margin_left + value / max(1, len(xs) - 1) * chart_width

    def y_map(value: float) -> float:
        return margin_top + chart_height * (1.0 - (value - y_min) / max(1e-8, y_max - y_min))

    grid_lines = []
    for frac in (0.0, 0.5, 1.0):
        y = margin_top + chart_height * frac
        grid_lines.append(f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' stroke='var(--grid)' stroke-width='1' />")
    path = _svg_path(xs, ys, x_map, y_map, "#22c55e", f"{row['family']}_trend", with_points=True)
    labels = []
    for idx, point in enumerate(points):
        x = x_map(idx)
        labels.append(
            f"<text x='{x:.2f}' y='{height - 8}' text-anchor='middle' fill='var(--muted)' font-size='10' font-family='var(--mono)'>{_escape(str(point['label']))}</text>"
        )
    value_labels = (
        f"<text x='{margin_left}' y='11' fill='var(--muted)' font-size='10' font-family='var(--mono)'>{_fmt(max(ys), precision=4)}</text>"
        f"<text x='{margin_left}' y='{height - margin_bottom + 12}' fill='var(--muted)' font-size='10' font-family='var(--mono)'>{_fmt(min(ys), precision=4)}</text>"
    )
    axes = (
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='var(--muted)' stroke-width='1' />"
        f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='var(--muted)' stroke-width='1' />"
    )
    return f"<svg viewBox='0 0 {width} {height}'>{''.join(grid_lines)}{axes}{path}{''.join(labels)}{value_labels}</svg>"


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
    eval_path = _escape(str(run.eval_dir)) if run.eval_dir is not None else "-"
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
      <div class="muted small" style="margin-bottom:12px;">eval_dir: <span class="mono">{eval_path}</span></div>
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


def _render_select_options(values: list[str], selected: str) -> str:
    options = []
    for value in values:
        label = value
        if value == "all":
            label = "all"
        is_selected = " selected" if value == selected else ""
        options.append(f"<option value='{_escape(value)}'{is_selected}>{_escape(label)}</option>")
    return "".join(options)


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


def _run_status(run: RunSnapshot) -> str:
    live = run.live_status or {}
    return str(live.get("status", "completed" if run.train_log else "idle")).lower()


def _run_updated_at(run: RunSnapshot) -> str:
    live = run.live_status or {}
    updated_at = live.get("updated_at")
    if updated_at:
        return str(updated_at)
    if run.checkpoint_info:
        return max(str(item.get("mtime", "")) for item in run.checkpoint_info.values())
    return "-"


def _run_latest_epoch(run: RunSnapshot) -> float | None:
    live = run.live_status or {}
    if live.get("epoch") is not None:
        return float(live["epoch"])
    if run.train_log:
        return float(run.train_log[-1].get("epoch", 0))
    return None


def _run_updated_epoch_key(run: RunSnapshot) -> tuple[str, float]:
    return (_run_updated_at(run), _run_latest_epoch(run) or -1.0)


def _family_run_sort_key(run: RunSnapshot) -> tuple[int, str, float]:
    return (_extract_round_number(run.run_dir.name), _run_updated_at(run), _run_latest_epoch(run) or -1.0)


def _infer_task(run: RunSnapshot) -> str:
    if run.summary and run.summary.get("task"):
        return str(run.summary["task"])
    if run.live_status and run.live_status.get("task"):
        return str(run.live_status["task"])
    name = run.run_dir.name.lower()
    if "multitask" in name:
        return "multitask"
    if "sensing" in name:
        return "spectrum_sensing"
    return "amc"


def _primary_metric(run: RunSnapshot) -> tuple[str, float]:
    if not run.summary:
        return "-", float("nan")
    task = str(run.summary.get("task", _infer_task(run))).lower()
    if task == "amc":
        return "overall_accuracy", float(run.summary.get("overall_accuracy", float("nan")))
    if task == "spectrum_sensing":
        return "roc_auc", float(run.summary.get("roc_auc", run.summary.get("overall_accuracy", float("nan"))))
    if task == "multitask":
        return "modulation_accuracy", float(run.summary.get("modulation_accuracy", float("nan")))
    return "-", float("nan")


def _family_name(run_name: str) -> str:
    name = run_name
    for suffix in ("_eval_rerun", "_eval_current", "_eval", "_seed42"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    parts = name.split("_")
    trimmed = parts[:]
    for marker in ("round1", "round2", "round3", "round4", "round5", "round6"):
        if marker in trimmed:
            trimmed = trimmed[: trimmed.index(marker)]
            break
    trimmed = [part for part in trimmed if part]
    return "_".join(trimmed) if trimmed else name


def _extract_round_number(run_name: str) -> int:
    match = re.search(r"round(\d+)", run_name)
    if match is None:
        return 10**9
    return int(match.group(1))


def _short_round_label(run_name: str) -> str:
    match = re.search(r"(round\d+)", run_name)
    if match is not None:
        return match.group(1)
    return run_name[:12]
