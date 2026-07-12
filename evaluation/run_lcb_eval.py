#!/usr/bin/env python3
"""Evaluate LiveCodeBench solutions using LCB's official evaluator.

Usage:
    PYTHONPATH=/tmp/lcb_repo python evaluation/run_lcb_eval.py \
        --func_path data/result/livecodebench/sol_gemma-3-4b_100_func_converted.jsonl \
        --output_path output/livecodebench/graded_gemma-3-4b.json \
        --num_workers 4

Processes tasks in chunks to bound memory usage and supports resume.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--release_version", default="release_v4")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=20,
                        help="Tasks per chunk. Smaller keeps memory bounded.")
    args = parser.parse_args()

    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    print("Loading benchmark...", flush=True)
    benchmark = load_code_generation_dataset(release_version=args.release_version)

    func_rows = [json.loads(l) for l in open(args.func_path)]
    func_map = {str(r["task_id"]): r["solutions"] for r in func_rows}
    print(f"Loaded {len(func_rows)} tasks from {args.func_path}", flush=True)

    # Build evaluation list
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
        if len(sols) < target_n:
            sols = sols + [""] * (target_n - len(sols))
        elif len(sols) > target_n:
            sols = sols[:target_n]
        samples.append(p.get_evaluation_sample())
        custom_outputs.append(sols)
        task_ids.append(qid)

    total = len(samples)
    print(f"Evaluating {total} problems with {target_n} solutions each, chunk_size={args.chunk_size}", flush=True)

    # Load existing graded (resume)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graded_map = {}
    if output_path.exists():
        try:
            graded_map = json.load(open(output_path))
            print(f"Resume: loaded {len(graded_map)} already-graded tasks", flush=True)
        except Exception:
            graded_map = {}

    # Filter out already-done tasks from this run
    pending_indices = [i for i, qid in enumerate(task_ids) if qid not in graded_map]
    print(f"Pending: {len(pending_indices)} tasks to grade", flush=True)

    # Process in chunks
    for start in range(0, len(pending_indices), args.chunk_size):
        chunk_idx = pending_indices[start : start + args.chunk_size]
        chunk_samples = [samples[i] for i in chunk_idx]
        chunk_outputs = [custom_outputs[i] for i in chunk_idx]
        chunk_tids = [task_ids[i] for i in chunk_idx]
        print(f"\n[chunk {start // args.chunk_size + 1}/"
              f"{(len(pending_indices) + args.chunk_size - 1) // args.chunk_size}] "
              f"{len(chunk_idx)} tasks, first tid={chunk_tids[0]}", flush=True)

        try:
            metrics, results, metadata = codegen_metrics(
                chunk_samples,
                chunk_outputs,
                num_process_evaluate=args.num_workers,
                timeout=args.timeout,
            )
        except Exception as e:
            print(f"  chunk error: {e} - skipping chunk", flush=True)
            continue

        # Merge and save
        for i, qid in enumerate(chunk_tids):
            if i in results:
                real_n = len(func_map.get(qid, []))
                graded_map[qid] = results[i][:real_n]

        # Flush to disk every chunk
        with open(output_path, "w") as f:
            json.dump(graded_map, f)
        print(f"  saved {len(graded_map)}/{total} so far", flush=True)

    print(f"\nDone. Saved graded for {len(graded_map)} tasks to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
