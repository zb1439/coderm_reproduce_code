#!/usr/bin/env python3
"""Incrementally label new solutions without re-running evalplus on everything.

Assumes:
  - Existing eval_results.json labels a prefix of each task's solutions
    (i.e., new solutions are appended at the end of func file, not interleaved).
  - Current func file has all solutions (old + new appended).

Steps:
  1. Read existing eval_results.json -> per task, how many solutions were labeled
  2. Read current func file
  3. For each task, take solutions[old_count:] as "new"
  4. Write new solutions to a subset jsonl
  5. Run evalplus on subset
  6. Merge new labels into existing eval_results.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--func_path", required=True,
                        help="Current (post-backfill) func.jsonl")
    parser.add_argument("--existing_results", required=True,
                        help="Existing evalplus_*_eval_results.json to extend")
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    func_rows = [json.loads(l) for l in open(args.func_path)]
    func_map = {r["task_id"]: r["solutions"] for r in func_rows}

    with open(args.existing_results) as f:
        existing = json.load(f)
    existing_eval = existing.get("eval", {})

    # Identify new solutions per task
    new_pairs = []  # list of (task_id, sol_idx_in_func, solution_text)
    for tid, sols in func_map.items():
        old_labels = existing_eval.get(tid, [])
        if len(sols) > len(old_labels):
            for i in range(len(old_labels), len(sols)):
                new_pairs.append((tid, i, sols[i]))
    if not new_pairs:
        print("No new solutions to label. Nothing to do.")
        return

    print(f"Found {len(new_pairs)} new solutions across {len(set(p[0] for p in new_pairs))} tasks")

    # Write subset evalplus jsonl
    subset_path = args.func_path.replace(".jsonl", "_increment.jsonl")
    with open(subset_path, "w") as f:
        for tid, idx, sol in new_pairs:
            f.write(json.dumps({"task_id": tid, "solution": sol}) + "\n")
    print(f"Wrote subset to {subset_path}")

    # Remove any stale increment results file
    subset_results = subset_path.replace(".jsonl", "_eval_results.json")
    if os.path.exists(subset_results):
        os.remove(subset_results)

    # Run evalplus on subset
    cmd = [
        ".venv-local/bin/evalplus.evaluate",
        "--dataset", args.dataset,
        "--samples", subset_path,
        "--parallel", str(args.parallel),
        "--i-just-wanna-run",
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=True)

    # Load subset results and merge
    with open(subset_results) as f:
        subset = json.load(f)
    subset_eval = subset.get("eval", {})

    # subset_eval[tid] is a list of N solution results for that task_id (the new ones)
    # We just extend existing_eval[tid] with them
    merged_count = 0
    for tid, entries in subset_eval.items():
        old_list = existing_eval.get(tid, [])
        old_list.extend(entries)
        existing_eval[tid] = old_list
        merged_count += len(entries)

    existing["eval"] = existing_eval
    # Overwrite the existing results file
    with open(args.existing_results, "w") as f:
        json.dump(existing, f)
    print(f"Merged {merged_count} new entries into {args.existing_results}")


if __name__ == "__main__":
    main()
