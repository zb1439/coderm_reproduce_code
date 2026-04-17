#!/usr/bin/env python3
"""Build prompt messages file for LCB v6 new problems (not in v4).

Usage:
    PYTHONPATH=/tmp/lcb_repo python preprocess/build_lcb_v6_new_prompts.py \
        --output data/benchmark/input_livecodebench_v6new_sol.jsonl
"""
import argparse
import json
from pathlib import Path


SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem specification) "
    "and will generate a correct Python program that matches the specification and passes all tests. "
    "You will NOT return anything except for the program."
)


def build_prompt(problem) -> str:
    """Build a user prompt for a CodeGenerationProblem.

    For class Solution (LeetCode) problems, include starter_code.
    For stdin/stdout problems (AtCoder/Codeforces), ask for a standalone script.
    """
    content = problem.question_content
    starter = (problem.starter_code or "").strip()

    if starter:
        instruction = (
            f"### Question\n{content}\n\n"
            f"### Format: You will use the following starter code to write the solution "
            f"to the problem and enclose your code within delimiters.\n"
            f"```python\n{starter}\n```\n\n"
            f"### Answer: (use the provided format with backticks)\n"
        )
    else:
        instruction = (
            f"### Question\n{content}\n\n"
            f"### Format: Read the inputs from stdin, solve the problem and write the "
            f"answer to stdout (do not directly test on the sample inputs). Enclose your "
            f"code within delimiters as follows. Ensure that when the Python program runs, "
            f"it reads the inputs, runs the algorithm and writes output to stdout.\n"
            f"```python\n# YOUR CODE HERE\n```\n\n"
            f"### Answer: (use the provided format with backticks)\n"
        )
    return instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--release_version", default="release_v6")
    parser.add_argument("--exclude_version", default="release_v4",
                        help="Exclude problems present in this version (to get new-only)")
    args = parser.parse_args()

    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    print(f"Loading {args.release_version}...", flush=True)
    v_full = load_code_generation_dataset(release_version=args.release_version)
    print(f"Loading {args.exclude_version}...", flush=True)
    v_old = load_code_generation_dataset(release_version=args.exclude_version)

    old_ids = set(str(p.question_id) for p in v_old)
    new_problems = [p for p in v_full if str(p.question_id) not in old_ids]

    print(f"{args.release_version}: {len(v_full)} problems")
    print(f"{args.exclude_version}: {len(v_old)} problems")
    print(f"New problems (in {args.release_version} but not {args.exclude_version}): {len(new_problems)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_starter = 0
    no_starter = 0
    with output_path.open("w") as f:
        for p in new_problems:
            user_prompt = build_prompt(p)
            if (p.starter_code or "").strip():
                has_starter += 1
            else:
                no_starter += 1
            row = {
                "task_id": str(p.question_id),
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt},
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(new_problems)} problems to {output_path}")
    print(f"  With starter_code (class Solution): {has_starter}")
    print(f"  Without starter_code (stdin/stdout): {no_starter}")


if __name__ == "__main__":
    main()
