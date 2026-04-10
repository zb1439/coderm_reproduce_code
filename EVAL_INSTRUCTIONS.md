# Evaluation Instructions

This document describes how to run the remaining evaluation tasks for the CodeRM reproduction experiment.

## What needs to be run

There are **3 evaluation tasks** remaining. All solution generation is complete.

| # | Task | Input File | Est. Time (16 workers) |
|---|------|-----------|----------------------|
| 1 | LiveCodeBench Gemma-3-4B | `data/result/livecodebench/sol_gemma-3-4b_100_func_converted.jsonl` | ~30 min |
| 2 | LiveCodeBench Ministral-3B | `data/result/livecodebench/sol_ministral-3b_100_func_converted.jsonl` | ~30 min |
| 3 | MBPP+ Ministral-3B | `data/result/mbpp_plus/sol_ministral-3b_100_func.jsonl` | ~60 min |

## Quick Start

```bash
# Clone this repo and checkout the correct branch
git clone https://github.com/zb1439/coderm_reproduce_code.git
cd coderm_reproduce_code
git checkout codex/solution-api

# Install dependencies
pip install evalplus datasets tqdm

# Run everything
bash scripts/run_eval_remote.sh
```

That's it. The script handles everything automatically: cloning LiveCodeBench, patching compatibility issues, running evaluations, and generating output files.

## Environment Variables (optional)

```bash
PY=python3.10          # Python executable (default: python3)
NUM_WORKERS=16         # Parallel workers for evaluation (default: 16)
LCB_REPO=/tmp/lcb_repo # Where to clone LiveCodeBench (default: /tmp/lcb_repo)
```

Example with custom settings:
```bash
NUM_WORKERS=32 bash scripts/run_eval_remote.sh
```

## What the script does

1. **LiveCodeBench evaluation** (tasks 1 & 2):
   - Clones [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) repo
   - Patches two compatibility issues automatically (relative paths, resource limits)
   - Runs `evaluation/run_lcb_eval.py` which loads the `release_v4` benchmark (713 problems), matches our 168 tasks, and evaluates each solution against official test cases
   - Outputs: `output/livecodebench/graded_gemma-3-4b.json` and `graded_ministral-3b.json`
   - Generates anno files: `data/result/livecodebench/sol_*_100_anno.jsonl`

2. **MBPP+ Ministral-3B evaluation** (task 3):
   - Converts func.jsonl to evalplus flat format (one solution per line)
   - Runs `evalplus.evaluate --dataset mbpp` to grade each solution
   - Outputs: `output/mbpp_plus/evalplus_ministral-3b_eval_results.json`

## Output files to collect

After the script finishes, these files contain the results:

```
output/livecodebench/graded_gemma-3-4b.json          # LCB Gemma graded results
output/livecodebench/graded_ministral-3b.json         # LCB Ministral graded results
data/result/livecodebench/sol_gemma-3-4b_100_anno.jsonl    # LCB Gemma anno
data/result/livecodebench/sol_ministral-3b_100_anno.jsonl  # LCB Ministral anno
output/mbpp_plus/evalplus_ministral-3b_eval_results.json   # MBPP+ Ministral results
```

## Already completed evaluations

These evaluations were already run locally and results are in the repo:

| Benchmark | Model | Result File | Solution Pass Rate (plus) | Pass@100 (plus) |
|-----------|-------|------------|--------------------------|----------------|
| HumanEval+ | Gemma-3-4B | `output/humaneval_plus/evalplus_gemma-3-4b_eval_results.json` | 65.5% | 129/164 (78.7%) |
| HumanEval+ | Ministral-3B | `output/humaneval_plus/evalplus_ministral-3b_eval_results.json` | 74.4% | 155/164 (94.5%) |
| MBPP+ | Gemma-3-4B | `output/mbpp_plus/evalplus_gemma-3-4b_eval_results.json` | 68.8% | 295/378 (78.0%) |

## Troubleshooting

- **`BrokenProcessPool` or `setrlimit` errors**: The script patches this automatically. If it still happens, try `NUM_WORKERS=1` to debug.
- **`Dataset scripts are no longer supported`**: Run `pip install "datasets<4"` to downgrade.
- **Out of memory**: Reduce `NUM_WORKERS`.
