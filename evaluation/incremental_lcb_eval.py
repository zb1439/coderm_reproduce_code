#!/usr/bin/env python3
"""Incrementally label new LCB solutions (only new ones appended, not interleaved).

Steps:
  1. Read existing graded_*.json -> per task, how many solutions were labeled
  2. Read current func_converted.jsonl
  3. For each task, take solutions[old_count:] as "new"
  4. Write a subset func.jsonl with only new solutions
  5. Run run_lcb_eval.py on subset
  6. Merge new graded lists into existing graded.

Must be run with PYTHONPATH=/tmp/lcb_repo
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_path", required=True, help="Current (post-backfill) converted func.jsonl")
    parser.add_argument("--existing_graded", required=True, help="Existing graded_*.json to extend")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--release_version", default="release_v4")
    args = parser.parse_args()

    func_rows = [json.loads(l) for l in open(args.func_path)]
    func_map = {r["task_id"]: r["solutions"] for r in func_rows}

    with open(args.existing_graded) as f:
        existing = json.load(f)

    # Per task, figure out new solutions
    new_per_task = {}
    for tid, sols in func_map.items():
        old = existing.get(tid, [])
        if len(sols) > len(old):
            new_per_task[tid] = sols[len(old):]

    if not new_per_task:
        print("No new solutions to label.")
        return

    print(f"New solutions needed for {len(new_per_task)} tasks, total "
          f"{sum(len(v) for v in new_per_task.values())} solutions")

    # Write subset func file in same format as input
    subset_path = args.func_path.replace(".jsonl", "_increment.jsonl")
    with open(subset_path, "w") as f:
        for tid, sols in new_per_task.items():
            f.write(json.dumps({"task_id": tid, "solutions": sols}, ensure_ascii=False) + "\n")
    print(f"Wrote subset to {subset_path}")

    subset_graded = subset_path.replace(".jsonl", "_graded.json")
    if os.path.exists(subset_graded):
        os.remove(subset_graded)

    repo_root = Path(__file__).parent.parent
    cmd = [
        ".venv-local/bin/python", "evaluation/run_lcb_eval.py",
        "--func_path", subset_path,
        "--output_path", subset_graded,
        "--num_workers", str(args.num_workers),
        "--timeout", str(args.timeout),
        "--release_version", args.release_version,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":/tmp/lcb_repo" if env.get("PYTHONPATH") else "/tmp/lcb_repo"
    print(f"Running: PYTHONPATH=/tmp/lcb_repo {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)

    # Merge
    with open(subset_graded) as f:
        subset = json.load(f)

    merged_count = 0
    for tid, entries in subset.items():
        old = existing.get(tid, [])
        old.extend(entries)
        existing[tid] = old
        merged_count += len(entries)

    with open(args.existing_graded, "w") as f:
        json.dump(existing, f)
    print(f"Merged {merged_count} new entries into {args.existing_graded}")


if __name__ == "__main__":
    main()
