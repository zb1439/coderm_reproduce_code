# Provider Inference Runner

This directory now contains an OpenAI-compatible provider runner for LiveCodeBench solution generation.

## Script

- `/Users/xinyuan/code/coderm_reproduce_code/inference/inference_api.py`

## What it does

- Reads solution prompts from `/Users/xinyuan/code/coderm/data/benchmark/input_livecodebench_sol.jsonl`.
- Falls back to `/Users/xinyuan/code/coderm_reproduce_code/data/benchmark/input_livecodebench_sol.jsonl` if the primary file is missing.
- Tries these model-provider pairs:
  - `qwen3.5-4b` on DeepInfra (auto) or DashScope (fallback)
  - `qwen3-4b-instruct-2507` on Nscale
  - `qwen3.5-0.8b` on DeepInfra (auto) or local Transformers backend (fallback)
- Writes provider raw output to `output/livecodebench/sol_*_provider_raw.jsonl`.
- Runs `preprocess/extract_solution.py` to produce `data/result/livecodebench/sol_*_100_func.jsonl`.
- Writes a run summary report to `output/livecodebench/run_report.json`.

## Environment variables

- `DEEPINFRA_TOKEN` (or `DEEPINFRA_API_KEY`)
- `NSCALE_SERVICE_TOKEN`
- `DASHSCOPE_API_KEY` (only used if DeepInfra key is not set)

Local backend (0.8B) uses your local Python environment (`torch` + `transformers`) and is only used when no DeepInfra key is present.

## Commands

Smoke test (2 tasks x 3 candidates):

```bash
python3 inference/inference_api.py --smoke
```

Full run (168 tasks x 100 candidates):

```bash
python3 inference/inference_api.py
```

Run only selected models:

```bash
python3 inference/inference_api.py --models qwen3.5-4b,qwen3-4b-instruct-2507
```

Run local 0.8B only:

```bash
python3 inference/inference_api.py --models qwen3.5-0.8b
```

Force local backend on CPU (if MPS has issues):

```bash
unset DEEPINFRA_TOKEN DEEPINFRA_API_KEY
python3 inference/inference_api.py --models qwen3.5-0.8b --local_device cpu
```

Skip extraction step:

```bash
python3 inference/inference_api.py --skip_extract
```

Enable live progress files:

```bash
python3 inference/inference_api.py --progress_dir output/livecodebench/progress
```

Watch progress in terminal:

```bash
python3 inference/watch_progress.py --progress_dir output/livecodebench/progress --interval 2
```

## Notes

- The script supports resume/checkpoint behavior by default.
- If a provider key is missing, that model is marked `skipped` in `run_report.json`.
- If `DEEPINFRA_TOKEN` or `DEEPINFRA_API_KEY` is set, `qwen3.5-4b` and `qwen3.5-0.8b` are routed to DeepInfra automatically.
- If provider/model endpoint returns `404`, that model is marked `skipped` with reason `provider_model_unavailable`.
- If local 0.8B dependencies or model weights are unavailable, that model is marked `skipped` with an explicit reason.
- The script runs a subprocess probe before local runtime init; probe failure also marks local model as `skipped`.
