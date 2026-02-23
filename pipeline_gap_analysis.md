# CodeRM Pipeline Gap Analysis

## Context
Analysis performed 2026-02-16. Goal: identify what's missing to reproduce a LiveCodeBench table score end-to-end.

## Complete Product Chain
```
prompt → inference_mp.py → merge_output.py → extract_solution.py / extract_unit_test.py
  → sol_*_func.jsonl + ut_*_100.jsonl
  → evaluate.py (sol×ut execution matrix) → 100_sol_100_ut_result.jsonl
  → [ANNOTATION GAP] → sol_*_anno.jsonl (ground truth pass/fail per solution)
  → calculate_result.py (best-of-n majority voting) → final accuracy number
```

## Gap 1 (P0): Anno Generation — `func.jsonl` → `anno.jsonl`

### What's missing
No script in the original repo (or coderm_reproduce_code) to convert `sol_*_100_func.jsonl` into `sol_*_100_anno.jsonl`. The anno files were committed as pre-computed artifacts.

### What anno.jsonl contains
Each solution annotated with official LiveCodeBench pass/fail:
```json
{"task_id": "3228", "solutions": [{"sol_id": 0, "code": "class Solution:...", "result": "fail"}, ...]}
```

### The two transformations needed
1. **Code cleaning**: raw LLM output (with `from xxx import *` preamble) → clean `class Solution:...` only
2. **Ground truth evaluation**: run each solution against LiveCodeBench's official test cases (from HuggingFace dataset `livecodebench/code_generation_lite`)

### How to implement
- `generate_livecodebench_anno.py` exists as an untracked file in `/Users/xinyuan/code/coderm/evaluation/` (created ~2026-02-06, NOT part of any git commit)
- It calls `lcb_runner.runner.custom_evaluator` to get graded results, then converts to anno format
- Two modes: `--run_evaluator` (automatic) or `--graded_path` (from pre-existing eval_all.json)
- Uses AST parsing to normalize code (extract Solution class or wrap bare functions)
- Key changes made to improve robustness:
  - `Optional[str]` instead of `str | None` for Python 3.9 compat
  - `load_json_or_jsonl` handles single dict JSON
  - `load_graded_map` uses `iter_dict_nodes` recursive traversal for nested evaluator output

### Verification
Compare generated anno against existing anno files — pass/fail should match.

## Gap 2 (P1): Documentation Inconsistencies

- README says `extract_unit_test.py --data_path` but script uses `--input_path`
- README Step 4 says "WIP"
- `docker_source/` referenced in README but directory doesn't exist

## Gap 3 (P2): Execution Sandbox Missing

- `evaluate.py` uses bare `exec()` in subprocess — no Docker/container isolation
- `docker_source/` directory referenced in README doesn't exist in repo
- Not a blocker for reproducing results, but a security concern

## Non-Gaps (things that look missing but aren't)

- **Execution matrix file** (`100_sol_100_ut_result.jsonl`): Not missing implementation — `evaluate.py` generates this. README offers pre-computed download as convenience.
- **Closed-source model inference**: GPT results are pre-computed in `data/result/`. `inference_mp.py` only supports vLLM, by design.
- **Dynamic scaling**: Research contribution, not a reproduction prerequisite. README "WIP" refers to documentation.

## What coderm_reproduce_code-master Covers vs Doesn't

### Covers ✅
- Preprocessing (merge, extract_solution, extract_unit_test)
- Execution matrix (evaluate.py with sol×ut)
- Best-of-n statistics (calculate_result.py)
- Pre-annotated data files

### Doesn't Cover ❌
- Inference (no inference/ directory — by design)
- Anno generation script (generate_livecodebench_anno.py — the P0 gap)

## Data Format Reference

### func.jsonl (raw solutions)
```json
{"task_id": "3228", "solutions": ["from string import *\nfrom re import *\n...class Solution:...", ...]}
```
- solutions are raw strings with import preamble
- 100 solutions per task, 168 tasks

### anno.jsonl (annotated solutions)
```json
{"task_id": "3228", "solutions": [{"sol_id": 0, "code": "class Solution:...", "result": "pass"}, ...]}
```
- code field: cleaned, class-only
- result field: "pass" or "fail" from official LiveCodeBench evaluation

### Execution matrix result
```json
{"task_id": "3228", "sol_id": 0, "ut_id": 0, "result": "pass", "details": {...}}
```

## Statistics (Llama3-8B on LiveCodeBench)
- 168 tasks, 100 solutions each = 16,800 total solutions
- Overall pass rate: 2,012/16,800 = 11.98%
- Tasks with ≥1 passing solution: 59/168
