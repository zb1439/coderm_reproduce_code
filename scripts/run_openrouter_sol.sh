#!/usr/bin/env bash
set -u

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set"
  exit 1
fi

PY=".venv-local/bin/python"
CMD="inference/inference_api.py"
MSG="/Users/xinyuan/code/coderm/data/benchmark/input_livecodebench_sol.jsonl"

run_model() {
  local tag="$1"
  echo "[$(date '+%H:%M:%S')] START $tag"
  "$PY" "$CMD" \
    --messages_file "$MSG" \
    --models "$tag" \
    --raw_output_dir output/livecodebench \
    --func_output_dir data/result/livecodebench \
    --run_report_path "output/livecodebench/run_report_${tag}.json" \
    --num_candidates 100 \
    --batch_size 1 \
    --max_tokens 4096 \
    --progress_dir output/livecodebench/progress \
    --checkpoint_interval 1
  echo "[$(date '+%H:%M:%S')] DONE $tag (exit=$?)"
}

run_model "gemma-3-4b"
run_model "ministral-3b"
