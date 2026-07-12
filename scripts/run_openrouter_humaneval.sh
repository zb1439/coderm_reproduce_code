#!/usr/bin/env bash
set -u

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set"
  exit 1
fi

PY=".venv-local/bin/python"
CMD="inference/inference_api.py"
MSG="/Users/xinyuan/code/coderm/data/benchmark/input_humaneval+_sol.jsonl"
BENCH="humaneval_plus"

mkdir -p "output/${BENCH}" "data/result/${BENCH}" "output/${BENCH}/progress"

for TAG in gemma-3-4b ministral-3b; do
  echo "[$(date '+%H:%M:%S')] START ${TAG} / ${BENCH}"
  "$PY" "$CMD" \
    --messages_file "$MSG" \
    --models "$TAG" \
    --raw_output_dir "output/${BENCH}" \
    --func_output_dir "data/result/${BENCH}" \
    --run_report_path "output/${BENCH}/run_report_${TAG}.json" \
    --num_candidates 100 \
    --batch_size 1 \
    --max_tokens 4096 \
    --progress_dir "output/${BENCH}/progress" \
    --checkpoint_interval 1
  echo "[$(date '+%H:%M:%S')] DONE ${TAG} / ${BENCH} (exit=$?)"
done
