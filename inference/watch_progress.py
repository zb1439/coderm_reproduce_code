#!/usr/bin/env python3
"""Simple terminal dashboard for inference_api live progress files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


FINAL_STATUSES = {"completed", "failed", "skipped"}


def load_progress_rows(progress_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(progress_dir.glob("progress_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            rows.append(
                {
                    "model_tag": path.stem.replace("progress_", ""),
                    "status": "invalid",
                    "reason": "parse_error",
                }
            )
    return rows


def format_row(row: Dict[str, Any]) -> str:
    tag = str(row.get("model_tag", "unknown"))
    status = str(row.get("status", "unknown"))
    completed = int(row.get("tasks_completed", 0))
    total = int(row.get("tasks_total", 0))
    task_index = int(row.get("current_task_index", 0))
    resp = int(row.get("current_task_responses", 0))
    resp_target = int(row.get("current_task_target", 0))
    reason = str(row.get("reason", ""))

    base = f"{tag:<24} {status:<10} tasks {completed:>3}/{total:<3}"
    if status == "running":
        return f"{base} | current {task_index:>3}/{total:<3} responses {resp:>3}/{resp_target:<3}"
    if reason:
        return f"{base} | {reason}"
    return base


def all_done(rows: List[Dict[str, Any]]) -> bool:
    return bool(rows) and all(str(row.get("status", "")) in FINAL_STATUSES for row in rows)


def render(progress_dir: Path, rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"[{now}] progress_dir={progress_dir}")
    if not rows:
        lines.append("No progress files yet.")
    else:
        for row in rows:
            lines.append(format_row(row))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch live progress_*.json files.")
    parser.add_argument(
        "--progress_dir",
        type=Path,
        required=True,
        help="Directory that contains progress_*.json files.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print once and exit.")
    parser.add_argument(
        "--exit_when_done",
        action="store_true",
        help="Exit automatically when all tracked models are in a final status.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress_dir: Path = args.progress_dir.resolve()
    progress_dir.mkdir(parents=True, exist_ok=True)

    last = ""
    while True:
        rows = load_progress_rows(progress_dir)
        text = render(progress_dir, rows)
        if text != last:
            if sys.stdout.isatty():
                sys.stdout.write("\x1b[2J\x1b[H")
            print(text, flush=True)
            last = text
        if args.once:
            return
        if args.exit_when_done and all_done(rows):
            return
        time.sleep(max(0.2, args.interval))


if __name__ == "__main__":
    main()
