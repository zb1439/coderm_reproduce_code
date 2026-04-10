#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# CodeRM Reproduce: Evaluation Script (for running on a remote server)
#
# This script runs 3 evaluation tasks that are too slow on a Mac:
#   1. LiveCodeBench annotation (Gemma-3-4B)    - ~16700 solutions
#   2. LiveCodeBench annotation (Ministral-3B)   - ~16600 solutions
#   3. MBPP+ annotation (Ministral-3B)           - ~37800 solutions
#
# Prerequisites:
#   - Python 3.10+
#   - pip install evalplus datasets tqdm
#   - Clone LiveCodeBench: git clone https://github.com/LiveCodeBench/LiveCodeBench.git /tmp/lcb_repo
#   - Patch lcb_runner (see below)
#
# Usage:
#   cd /path/to/coderm_reproduce_code
#   bash scripts/run_eval_remote.sh
##############################################################################

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python3}"
LCB_REPO="${LCB_REPO:-/tmp/lcb_repo}"
NUM_WORKERS="${NUM_WORKERS:-16}"

echo "================================================"
echo " CodeRM Reproduce - Remote Evaluation"
echo " Repo: $REPO_ROOT"
echo " Python: $PY"
echo " LCB Repo: $LCB_REPO"
echo " Workers: $NUM_WORKERS"
echo "================================================"

##############################################################################
# Step 0: Setup
##############################################################################

# Clone LCB if not present
if [ ! -d "$LCB_REPO/lcb_runner" ]; then
  echo "[setup] Cloning LiveCodeBench..."
  git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_REPO"
fi

# Patch lcb_runner's relative path issue (if not already patched)
if grep -q 'open("lcb_runner/prompts' "$LCB_REPO/lcb_runner/prompts/code_generation.py" 2>/dev/null; then
  echo "[setup] Patching lcb_runner code_generation.py..."
  python3 -c "
import re
path = '$LCB_REPO/lcb_runner/prompts/code_generation.py'
with open(path) as f: src = f.read()
if 'import os' not in src.split('import json')[0]:
    src = src.replace('import json', 'import json\nimport os', 1)
src = src.replace(
    'open(\"lcb_runner/prompts/few_shot_examples/generation/func.json\")',
    'open(os.path.join(os.path.dirname(__file__), \"few_shot_examples\", \"generation\", \"func.json\"))'
)
src = src.replace(
    'open(\"lcb_runner/prompts/few_shot_examples/generation/stdin.json\")',
    'open(os.path.join(os.path.dirname(__file__), \"few_shot_examples\", \"generation\", \"stdin.json\"))'
)
with open(path, 'w') as f: f.write(src)
print('  patched')
"
fi

# Patch lcb_runner's setrlimit issue (macOS/some Linux)
if grep -q 'resource.setrlimit' "$LCB_REPO/lcb_runner/evaluation/testing_util.py" 2>/dev/null; then
  if ! grep -q 'try:' "$LCB_REPO/lcb_runner/evaluation/testing_util.py" 2>/dev/null | grep -q 'setrlimit'; then
    echo "[setup] Patching lcb_runner testing_util.py setrlimit..."
    python3 -c "
path = '$LCB_REPO/lcb_runner/evaluation/testing_util.py'
with open(path) as f: src = f.read()
src = src.replace(
    '''        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == \"Darwin\":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )''',
    '''        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
            )
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
            )
        except (ValueError, OSError):
            pass
        if not platform.uname().system == \"Darwin\":
            try:
                resource.setrlimit(
                    resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
                )
            except (ValueError, OSError):
                pass'''
)
with open(path, 'w') as f: f.write(src)
print('  patched')
"
  fi
fi

# Install deps
$PY -m pip install evalplus datasets tqdm 2>/dev/null || true

##############################################################################
# Step 1: LiveCodeBench evaluation (Gemma + Ministral)
##############################################################################

echo ""
echo "========================================"
echo " [1/3] LiveCodeBench - Gemma-3-4B"
echo "========================================"
PYTHONPATH="$LCB_REPO" $PY evaluation/run_lcb_eval.py \
  --func_path data/result/livecodebench/sol_gemma-3-4b_100_func_converted.jsonl \
  --output_path output/livecodebench/graded_gemma-3-4b.json \
  --num_workers "$NUM_WORKERS" --timeout 30
echo "[1/3] DONE"

echo ""
echo "========================================"
echo " [2/3] LiveCodeBench - Ministral-3B"
echo "========================================"
PYTHONPATH="$LCB_REPO" $PY evaluation/run_lcb_eval.py \
  --func_path data/result/livecodebench/sol_ministral-3b_100_func_converted.jsonl \
  --output_path output/livecodebench/graded_ministral-3b.json \
  --num_workers "$NUM_WORKERS" --timeout 30
echo "[2/3] DONE"

##############################################################################
# Step 2: MBPP+ Ministral evalplus
##############################################################################

echo ""
echo "========================================"
echo " [3/3] MBPP+ Ministral-3B (evalplus)"
echo "========================================"

# Convert to evalplus format if not exists
if [ ! -f output/mbpp_plus/evalplus_ministral-3b.jsonl ]; then
  $PY -c "
import json
src = 'data/result/mbpp_plus/sol_ministral-3b_100_func.jsonl'
dst = 'output/mbpp_plus/evalplus_ministral-3b.jsonl'
rows = [json.loads(l) for l in open(src)]
with open(dst, 'w') as f:
    for row in rows:
        for sol in row['solutions']:
            f.write(json.dumps({'task_id': row['task_id'], 'solution': sol}) + '\n')
print(f'Wrote {dst}')
"
fi

$PY -m evalplus.evaluate --dataset mbpp \
  --samples output/mbpp_plus/evalplus_ministral-3b.jsonl \
  --parallel "$NUM_WORKERS" --i-just-wanna-run

echo "[3/3] DONE"

##############################################################################
# Step 3: Generate anno files from graded results
##############################################################################

echo ""
echo "========================================"
echo " Generating LCB anno files..."
echo "========================================"

for MODEL in gemma-3-4b ministral-3b; do
  GRADED="output/livecodebench/graded_${MODEL}.json"
  FUNC="data/result/livecodebench/sol_${MODEL}_100_func_converted.jsonl"
  ANNO="data/result/livecodebench/sol_${MODEL}_100_anno.jsonl"
  if [ -f "$GRADED" ]; then
    PYTHONPATH="$LCB_REPO" $PY evaluation/generate_livecodebench_anno.py \
      --func_path "$FUNC" \
      --output_path "$ANNO" \
      --graded_path "$GRADED" \
      --overwrite --allow_missing
    echo "  $ANNO generated"
  else
    echo "  WARNING: $GRADED not found, skipping $MODEL"
  fi
done

echo ""
echo "================================================"
echo " ALL DONE"
echo "================================================"
echo ""
echo "Results:"
echo "  output/livecodebench/graded_gemma-3-4b.json"
echo "  output/livecodebench/graded_ministral-3b.json"
echo "  output/mbpp_plus/evalplus_ministral-3b_eval_results.json"
echo "  data/result/livecodebench/sol_gemma-3-4b_100_anno.jsonl"
echo "  data/result/livecodebench/sol_ministral-3b_100_anno.jsonl"
