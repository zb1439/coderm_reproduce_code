#!/usr/bin/env python3
"""Evaluate LiveCodeBench solutions using LCB's official evaluator.

Usage:
    PYTHONPATH=/tmp/lcb_repo python evaluation/run_lcb_eval.py \
        --func_path data/result/livecodebench/sol_gemma-3-4b_100_func_converted.jsonl \
        --output_path output/livecodebench/graded_gemma-3-4b.json \
        --num_workers 4
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--release_version", default="release_v4")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    print("Loading benchmark...", flush=True)
    benchmark = load_code_generation_dataset(release_version=args.release_version)

    func_rows = [json.loads(l) for l in open(args.func_path)]
    func_map = {str(r["task_id"]): r["solutions"] for r in func_rows}
    print(f"Loaded {len(func_rows)} tasks from {args.func_path}", flush=True)

    # Only evaluate problems where we have solutions. Keep equal-length lists
    # (codegen_metrics requires all generation lists to match generations_list[0] length).
    samples = []
    custom_outputs = []
    task_ids = []
    target_n = None
    for p in benchmark:
        qid = str(p.question_id)
        sols = func_map.get(qid)
        if not sols:
            continue
        if target_n is None:
            target_n = len(sols)
        # Pad/truncate so every list has the same length
        if len(sols) < target_n:
            sols = sols + [""] * (target_n - len(sols))
        elif len(sols) > target_n:
            sols = sols[:target_n]
        samples.append(p.get_evaluation_sample())
        custom_outputs.append(sols)
        task_ids.append(qid)

    print(f"Evaluating {len(samples)} problems with {target_n} solutions each", flush=True)

    metrics, results, metadata = codegen_metrics(
        samples,
        custom_outputs,
        num_process_evaluate=args.num_workers,
        timeout=args.timeout,
    )

    print(f"\nOverall metrics: {json.dumps(metrics, indent=2)}", flush=True)

    # Extract graded_list per task (results keyed by index into our filtered list)
    graded_map = {}
    for i, qid in enumerate(task_ids):
        if i in results:
            graded_map[qid] = results[i]

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(graded_map, f)
    print(f"Saved graded for {len(graded_map)} tasks to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
