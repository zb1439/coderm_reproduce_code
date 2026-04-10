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

    # Convert problem objects to evaluation dicts and pad to match benchmark size
    samples = []
    custom_outputs = []
    our_indices = []  # track which indices are ours
    for i, p in enumerate(benchmark):
        qid = str(p.question_id)
        samples.append(p.get_evaluation_sample())
        sols = func_map.get(qid, [])
        custom_outputs.append(sols)
        if sols:
            our_indices.append(i)

    print(f"Evaluating {len(our_indices)}/{len(benchmark)} problems with solutions", flush=True)

    metrics, results, metadata = codegen_metrics(
        samples,
        custom_outputs,
        num_process_evaluate=args.num_workers,
        timeout=args.timeout,
    )

    print(f"\nOverall metrics: {json.dumps(metrics, indent=2)}", flush=True)

    # Extract graded_list for our tasks only
    graded_map = {}
    for i, p in enumerate(benchmark):
        qid = str(p.question_id)
        if qid in func_map and i in results:
            graded_map[qid] = results[i]

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(graded_map, f)
    print(f"Saved graded for {len(graded_map)} tasks to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
