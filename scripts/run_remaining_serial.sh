#!/usr/bin/env bash
# Run all remaining tasks serially to avoid CPU/heat competition.
#
# Order:
#   1. Resume v6 new inference for 3 models (API, runs concurrently as these don't eat CPU)
#   2. LCB 168 labeling: Gemma → Ministral (Qwen already done)
#   3. Wait for v6 new inferences to finish
#   4. Extract + convert v6 new func files
#   5. LCB v6 new labeling: Gemma → Ministral → Qwen
#
# Usage:
#   bash scripts/run_remaining_serial.sh

set -u
cd "$(dirname "$0")/.."

LCB_REPO="${LCB_REPO:-/tmp/lcb_repo}"
PY=".venv-local/bin/python"
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY env var not set"
  exit 1
fi

# Ensure LCB repo is cloned and patched
if [ ! -d "$LCB_REPO/lcb_runner" ]; then
  git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_REPO"
  python3 -c "
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

p2 = '$LCB_REPO/lcb_runner/evaluation/testing_util.py'
with open(p2) as f: src = f.read()
old = '''        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == \"Darwin\":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )'''
new = '''        try:
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
src = src.replace(old, new)
with open(p2, 'w') as f: f.write(src)
print('patched')
"
fi

ts() { date '+%H:%M:%S'; }

# --- Step 1: start v6 new inference resume (background, API-only so they can overlap) ---
echo "[$(ts)] Step 1/5: starting v6 new inference (background)"
for M in qwen3-8b gemma-3-4b ministral-3b; do
  EXTRA=""
  if [ "$M" = "qwen3-8b" ]; then EXTRA="--disable_thinking"; fi
  (OPENROUTER_API_KEY="$OPENROUTER_API_KEY" "$PY" inference/inference_api.py \
    --messages_file data/benchmark/input_livecodebench_v6new_sol.jsonl \
    --models "$M" \
    --raw_output_dir output/livecodebench_v6new \
    --func_output_dir data/result/livecodebench_v6new \
    --run_report_path "output/livecodebench_v6new/run_report_${M}_serial.json" \
    --num_candidates 100 --batch_size 1 --max_tokens 4096 \
    --progress_dir output/livecodebench_v6new/progress \
    $EXTRA --checkpoint_interval 1 > "output/livecodebench_v6new/serial_${M}.log" 2>&1) &
  echo "  → started v6 new ${M} (PID $!)"
done

# --- Step 2: LCB 168 labeling serially ---
echo "[$(ts)] Step 2/5: LCB 168 Gemma labeling"
if [ ! -f output/livecodebench/graded_gemma-3-4b.json ]; then
  PYTHONPATH="$LCB_REPO" "$PY" evaluation/run_lcb_eval.py \
    --func_path data/result/livecodebench/sol_gemma-3-4b_100_func_converted.jsonl \
    --output_path output/livecodebench/graded_gemma-3-4b.json \
    --num_workers 4 --timeout 20
  echo "[$(ts)]   → Gemma done"
fi

echo "[$(ts)] Step 2/5: LCB 168 Ministral labeling"
if [ ! -f output/livecodebench/graded_ministral-3b.json ]; then
  PYTHONPATH="$LCB_REPO" "$PY" evaluation/run_lcb_eval.py \
    --func_path data/result/livecodebench/sol_ministral-3b_100_func_converted.jsonl \
    --output_path output/livecodebench/graded_ministral-3b.json \
    --num_workers 4 --timeout 20
  echo "[$(ts)]   → Ministral done"
fi

# --- Step 3: wait for v6 new inference background jobs ---
echo "[$(ts)] Step 3/5: waiting for v6 new inference to finish"
wait
echo "[$(ts)]   → all v6 new inference jobs finished"

# --- Step 4: extract + convert v6 new func files ---
echo "[$(ts)] Step 4/5: extract + convert v6 new func files"
for M in gemma-3-4b ministral-3b qwen3-8b; do
  echo "  → extracting $M"
  "$PY" preprocess/extract_solution.py \
    --data_path "output/livecodebench_v6new/sol_${M}_provider_raw.jsonl" \
    --id_path data/benchmark/input_livecodebench_v6new_sol.jsonl \
    --output_path "data/result/livecodebench_v6new/sol_${M}_100_func.jsonl" > /dev/null 2>&1 || true
  "$PY" preprocess/convert_class_to_func.py \
    --input "data/result/livecodebench_v6new/sol_${M}_100_func.jsonl" \
    --output "data/result/livecodebench_v6new/sol_${M}_100_func_converted.jsonl" > /dev/null 2>&1 || true
done

# --- Step 5: v6 new labeling serially ---
echo "[$(ts)] Step 5/5: LCB v6 new labeling (serial)"
for M in qwen3-8b gemma-3-4b ministral-3b; do
  if [ ! -f "output/livecodebench_v6new/graded_${M}.json" ]; then
    echo "[$(ts)]   → labeling v6 new ${M}"
    PYTHONPATH="$LCB_REPO" "$PY" evaluation/run_lcb_eval.py \
      --func_path "data/result/livecodebench_v6new/sol_${M}_100_func_converted.jsonl" \
      --output_path "output/livecodebench_v6new/graded_${M}.json" \
      --num_workers 4 --timeout 20
    echo "[$(ts)]   → ${M} done"
  fi
done

echo "[$(ts)] ALL DONE"
