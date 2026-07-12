#!/usr/bin/env bash
set -u

if [[ -z "${DEEPINFRA_TOKEN:-}" ]]; then
  echo "DEEPINFRA_TOKEN is not set"
  exit 1
fi

PY=".venv-local/bin/python"
CMD="inference/inference_api.py"
MODELS="qwen3.5-4b,qwen3.5-0.8b"

run_one() {
  local name="$1"
  local msg_file="$2"
  local raw_dir="$3"
  local func_dir="$4"
  local report="$5"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name}"
  "$PY" "$CMD" \
    --messages_file "$msg_file" \
    --models "$MODELS" \
    --num_candidates 100 \
    --raw_output_dir "$raw_dir" \
    --func_output_dir "$func_dir" \
    --run_report_path "$report"
  local ec=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END ${name} exit=${ec}"
  return $ec
}

run_one "livecodebench" \
  "/Users/xinyuan/code/coderm/data/benchmark/input_livecodebench_sol.jsonl" \
  "output/livecodebench" \
  "data/result/livecodebench" \
  "output/livecodebench/run_report_deepinfra_qwen35.json"

run_one "mbpp_plus" \
  "/Users/xinyuan/code/coderm/data/benchmark/input_mbpp+_sol.jsonl" \
  "output/mbpp_plus" \
  "data/result/mbpp_plus" \
  "output/mbpp_plus/run_report_deepinfra_qwen35.json"

run_one "humaneval_plus" \
  "/Users/xinyuan/code/coderm/data/benchmark/input_humaneval+_sol.jsonl" \
  "output/humaneval_plus" \
  "data/result/humaneval_plus" \
  "output/humaneval_plus/run_report_deepinfra_qwen35.json"
