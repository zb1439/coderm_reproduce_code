#!/usr/bin/env bash
set -u

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set"
  exit 1
fi

PY=".venv-local/bin/python"
CMD="inference/inference_api.py"
CODERM="/Users/xinyuan/code/coderm"

run_one() {
  local model_tag="$1"
  local benchmark="$2"
  local msg_file="$3"

  echo "[$(date '+%H:%M:%S')] START ${model_tag} / ${benchmark}"
  mkdir -p "output/${benchmark}" "data/result/${benchmark}" "output/${benchmark}/progress"

  "$PY" "$CMD" \
    --messages_file "$msg_file" \
    --models "$model_tag" \
    --raw_output_dir "output/${benchmark}" \
    --func_output_dir "data/result/${benchmark}" \
    --run_report_path "output/${benchmark}/run_report_${model_tag}.json" \
    --num_candidates 100 \
    --batch_size 1 \
    --max_tokens 4096 \
    --progress_dir "output/${benchmark}/progress" \
    --checkpoint_interval 1

  echo "[$(date '+%H:%M:%S')] DONE ${model_tag} / ${benchmark} (exit=$?)"
}

# ---- HumanEval+ (164 tasks) ----
run_one "gemma-3-4b"   "humaneval_plus" "${CODERM}/data/benchmark/input_humaneval+_sol.jsonl"
run_one "ministral-3b" "humaneval_plus" "${CODERM}/data/benchmark/input_humaneval+_sol.jsonl"

# ---- MBPP+ (378 tasks) ----
run_one "gemma-3-4b"   "mbpp_plus" "${CODERM}/data/benchmark/input_mbpp+_sol.jsonl"
run_one "ministral-3b" "mbpp_plus" "${CODERM}/data/benchmark/input_mbpp+_sol.jsonl"
