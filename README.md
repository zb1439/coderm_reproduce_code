# Unified Evaluation Script - Instruction Guide

## Overview

The `unified_eval.py` script provides a one-command solution for running the complete CodeRM evaluation pipeline. It integrates all steps from inference to final evaluation, with optimizations for speed and parallelization.

## Quick Start

### Basic Usage

```bash
python unified_eval.py \
    --model_path /path/to/coderm-8b \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_llama-8b-instruct_200.jsonl \
    --benchmark humaneval \
    --num_unit_tests 100 \
    --num_solutions 100
```

### Using Previously Generated Unit Tests

If you've already generated unit tests and want to simulate top-p or top-k sampling:

```bash
python unified_eval.py \
    --model_path /path/to/coderm-8b \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_llama-8b-instruct_200.jsonl \
    --use_previous_ut \
    --previous_ut_path output/humaneval/inference/raw_inference_results.jsonl \
    --sample_top_k 50 \
    --num_unit_tests 50
```

Or with top-p sampling:

```bash
python unified_eval.py \
    --model_path /path/to/coderm-8b \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_llama-8b-instruct_200.jsonl \
    --use_previous_ut \
    --previous_ut_path output/humaneval/inference/raw_inference_results.jsonl \
    --sample_top_p 0.8 \
    --num_unit_tests 50
```

## Critical Paths and Functions

### 1. Inference Pipeline (`run_inference`)

**Location**: Lines 200-310

**Purpose**: Generates unit tests using vLLM with probability recording.

**Key Functions**:
- `run_inference()`: Main inference function that:
  - Loads prompts from JSONL file
  - Initializes vLLM with specified configuration
  - Generates unit tests with logprobs enabled
  - Calculates generation probabilities
  - Saves raw inference results with probabilities

**Critical Parameters**:
- `logprobs=5`: Enables probability calculation (top 5 logprobs)
- `n`: Number of samples per prompt (affects total generation count)
- `num_unit_tests`: Target number of unit tests per task

**Output**: 
- Raw inference results with probabilities saved to `output/{benchmark}/inference/raw_inference_results.jsonl`

### 2. Unit Test Extraction (`extract_unit_tests`)

**Location**: Lines 312-360

**Purpose**: Extracts valid unit tests from inference responses.

**Key Functions**:
- `extract_unit_test()`: Extracts unit test code from markdown-formatted responses
- `extract_class_names()`: Finds test class names using AST parsing
- `remove_func()`: Removes function definitions, keeps only test classes

**Critical Logic**:
- Parses markdown code blocks (```python ... ```)
- Validates unit test structure (must contain exactly one test class)
- Formats unit tests for execution with unittest framework

**Output**:
- Extracted unit tests saved to `output/{benchmark}/unit_tests/unit_tests_{num}.jsonl`

### 3. Unit Test Execution (`execute_unit_tests`)

**Location**: Lines 500-600

**Purpose**: Executes unit tests on solutions in parallel.

**Key Functions**:
- `execute_unit_tests()`: Main execution coordinator
- `run_unit_tests()`: Parallel execution using ProcessPoolExecutor
- `handle_execute()`: Wraps execution in isolated process with timeout
- `execute_unittest()`: Actual test execution with result capture

**Critical Features**:
- **Parallelization**: Uses `mp_num` processes for concurrent execution
- **Timeout Protection**: Each test has `time_limit_seconds` timeout
- **Isolation**: Each test runs in separate process to prevent interference
- **Chunking**: Processes data in chunks to manage memory

**Output**:
- Execution results saved to `output/{benchmark}/execution/{num_sol}_sol_{num_ut}_ut_result.jsonl`

### 4. Solution Selection (`select_solutions`)

**Location**: Lines 602-680

**Purpose**: Selects best solutions using majority voting.

**Key Algorithm**:
1. For each task, count passed unit tests per solution
2. Find solutions with maximum passed tests
3. Among top solutions, select by consistency (majority voting)
4. If multiple solutions have same consistency, select all

**Critical Logic**:
- Uses set operations to track which unit tests pass for each solution
- Implements two-stage selection: max pass count → max consistency
- Handles edge cases (no solutions, all fail, etc.)

**Output**:
- Selected solutions saved to `output/{benchmark}/selection/select_in_{num_sol}_sol_by_{num_ut}_ut_max+vote.jsonl`

### 5. EvalPlus Evaluation (`run_evalplus`)

**Location**: Lines 682-720

**Purpose**: Runs official EvalPlus evaluation to compute pass@1.

**Key Features**:
- Calls `evalplus.evaluate` command
- Sets `EVALPLUS_MAX_MEMORY_BYTES=-1` to avoid memory errors
- Supports parallel execution

**Output**:
- EvalPlus results printed to console and log file

### 6. Top-p/Top-k Sampling (`load_previous_and_sample`)

**Location**: Lines 312-360

**Purpose**: Samples from previously generated unit tests.

**Key Functions**:
- `load_previous_and_sample()`: Loads previous results and applies sampling
- Supports both top-k (select k highest probability) and top-p (cumulative probability threshold)

**Critical Logic**:
- Sorts unit tests by probability (descending)
- For top-k: selects first k tests
- For top-p: selects tests until cumulative probability exceeds threshold

## Configuration Parameters

### Required Arguments

- `--model_path`: Path to the model for unit test generation
- `--prompt_path`: Path to prompt JSONL file (e.g., `data/benchmark/input_humaneval+_ut.jsonl`)
- `--solution_path`: Path to solution JSONL file

### Inference Parameters

- `--num_unit_tests`: Number of unit tests to generate per task (default: 100)
- `--num_solutions`: Number of solutions to evaluate (default: 100)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_p`: Top-p sampling parameter (default: 0.95)
- `--top_k`: Top-k sampling parameter (default: -1, disabled)
- `--max_tokens`: Maximum tokens to generate (default: 2048)
- `--n`: Number of samples per prompt (default: 1)

### Hardware Parameters

- `--num_gpus`: Number of GPUs to use (default: 1)
- `--tensor_parallel_size`: Tensor parallel size (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization (default: 0.8)
- `--max_num_seqs`: Maximum number of sequences (default: 512)

### Execution Parameters

- `--mp_num`: Number of processes for unit test execution (default: 8)
- `--chunk_size`: Chunk size for processing (default: 1000)
- `--time_limit_seconds`: Time limit per unit test (default: 1.0)
- `--save_details`: Save detailed execution results

### Sampling from Previous Results

- `--use_previous_ut`: Enable using previously generated unit tests
- `--previous_ut_path`: Path to previous results with probabilities
- `--sample_top_p`: Top-p value for sampling (mutually exclusive with top_k)
- `--sample_top_k`: Top-k value for sampling (mutually exclusive with top_p)

### Output Parameters

- `--output_dir`: Output directory (default: "output")
- `--log_file`: Log file path (default: `{output_dir}/{benchmark}/eval.log`)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Output Structure

```
output/
└── {benchmark}/
    ├── eval.log                          # Main log file
    ├── inference/
    │   └── raw_inference_results.jsonl   # Raw inference with probabilities
    ├── unit_tests/
    │   └── unit_tests_{num}.jsonl        # Extracted unit tests
    ├── execution/
    │   └── {num_sol}_sol_{num_ut}_ut_result.jsonl  # Execution results
    └── selection/
        └── select_in_{num_sol}_sol_by_{num_ut}_ut_max+vote.jsonl  # Selected solutions
```

## Performance Optimization

### Parallelization Strategy

1. **Inference**: Uses vLLM's built-in batching and tensor parallelism
2. **Unit Test Execution**: Uses `ProcessPoolExecutor` with `mp_num` workers
3. **Chunking**: Processes data in chunks to manage memory efficiently

### Speed Improvements Over Original Pipeline

1. **Eliminated Intermediate Files**: No need to merge outputs from multiple processes
2. **In-Memory Processing**: Reduces I/O overhead
3. **Parallel Execution**: All steps are optimized for parallel processing
4. **Single Script**: No need to run multiple separate scripts

## Error Handling

The script includes comprehensive error handling:

- **GPU Detection**: Falls back to default GPU if nvidia-smi fails
- **File Operations**: Checks for file existence and handles overwrites
- **Process Timeouts**: Each unit test has timeout protection
- **Logging**: All errors are logged with full stack traces

## Logging

Logging is configured to write to both console and file:

- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Timestamp**: Included in every log message
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **File**: Defaults to `{output_dir}/{benchmark}/eval.log`

## GPU Memory Management for unified_eval.py

This section describes how to manage GPU memory when running the CodeRM evaluation pipeline with `unified_eval.py`. It covers quantization, free GPU detection, and vLLM parameters tuned for common GPU sizes.

---

### 1. Quantization

Quantization reduces model precision to lower VRAM usage. Use `--quantization` to enable it.

#### Options

| Option | Use Case | VRAM Savings | Notes |
|-------|----------|--------------|-------|
| `bitsandbytes` | Any model | ~50% (FP16→4-bit) | Dynamic 4-bit quantization. Works with any HuggingFace model. Requires `bitsandbytes` package. |
| `awq` | Pre-quantized models | ~75% | Use models already quantized with AWQ (e.g. from TheBloke, RedHat AI on HuggingFace). |
| `gptq` | Pre-quantized models | ~75% | Use models already quantized with GPTQ. |

#### Examples

```bash
# 4-bit quantization for any model (recommended for 8–16GB GPUs)
python unified_eval.py --model_path KAKA22/CodeRM-8B --quantization bitsandbytes ...

# Pre-quantized AWQ model (if available for your model)
python unified_eval.py --model_path path/to/model-AWQ --quantization awq ...

# Pre-quantized GPTQ model
python unified_eval.py --model_path path/to/model-GPTQ --quantization gptq ...
```

#### Dependencies

- **bitsandbytes**: `pip install bitsandbytes` (already in requirements.txt)
- **awq/gptq**: Use models from HuggingFace that are published in AWQ/GPTQ format

---

### 2. Free GPU Detection Threshold

The script selects GPUs with at least a given amount of free memory (in MiB). The threshold is used in `get_free_gpus(threshold=...)` and is **hardcoded** at 8192 MiB (8 GB) in `run_inference()`.

#### Current Behavior

- Default threshold: **8192 MiB (8 GB)**
- GPUs with less free memory are skipped
- Location: `unified_eval.py` line ~303: `free_gpus = get_free_gpus(threshold=8192)`

#### Adjusting the Threshold

**Option A: Edit the code**

In `unified_eval.py`, change the threshold in the `run_inference` function:

```python
# More lenient (e.g. 4 GB) – useful when other processes use GPU
free_gpus = get_free_gpus(threshold=4096)

# Stricter (e.g. 16 GB) – for large models or multi-GPU
free_gpus = get_free_gpus(threshold=16384)
```

**Option B: Add a CLI argument**

Add to the argument parser:

```python
parser.add_argument("--gpu_memory_threshold", type=int, default=8192,
                    help="Minimum free GPU memory (MiB) to consider a GPU available")
```

Then in `run_inference`:

```python
free_gpus = get_free_gpus(threshold=config.gpu_memory_threshold)
```

#### Suggested Thresholds by Scenario

| Scenario | Threshold (MiB) | Notes |
|---------|-----------------|-------|
| Shared GPU, light load | 4096 (4 GB) | Other processes may use the rest |
| Dedicated GPU, 8B model | 8192 (8 GB) | Default |
| Dedicated GPU, 13B+ model | 12288 (12 GB) | Avoid GPUs with little headroom |
| Multi-GPU, large model | 16384 (16 GB) | Ensure enough free memory per GPU |

---

### 3. vLLM Parameters by GPU Size

Tune these parameters based on your GPU VRAM. All can be set via command-line arguments.

#### Parameter Overview

| Parameter | Default | Effect | Trade-off |
|-----------|---------|--------|-----------|
| `--gpu_memory_utilization` | 0.8 | Fraction of GPU memory vLLM may use | Higher = more KV cache, more OOM risk |
| `--max_model_len` | 2048 | Max context length | Lower = less KV cache, less memory |
| `--max_num_seqs` | 64 | Max batch size | Lower = less memory, slower inference |
| `--enforce_eager` | True | Disable CUDA graphs | True = less memory, slightly slower |
| `--quantization` | None | 4-bit or pre-quantized | Reduces model weight memory |
| `--tensor_parallel_size` | 1 | Split model across GPUs | >1 for models that don’t fit on one GPU |

### 8 GB GPU (e.g. RTX 3070, RTX 4060)

```bash
python unified_eval.py \
  --model_path KAKA22/CodeRM-8B \
  --quantization bitsandbytes \
  --gpu_memory_utilization 0.75 \
  --max_model_len 2048 \
  --max_num_seqs 16 \
  --enforce_eager \
  ...
```

- Use **bitsandbytes** quantization
- Lower `gpu_memory_utilization` (0.7–0.75)
- Lower `max_num_seqs` (8–16)
- Keep `enforce_eager` (default)

#### 12 GB GPU (e.g. RTX 3060 12GB, RTX 4070)

```bash
python unified_eval.py \
  --model_path KAKA22/CodeRM-8B \
  --quantization bitsandbytes \
  --gpu_memory_utilization 0.8 \
  --max_model_len 2048 \
  --max_num_seqs 32 \
  --enforce_eager \
  ...
```

- Quantization still recommended for 8B models
- Can increase `max_num_seqs` to 24–32

#### 16–24 GB GPU (e.g. RTX 4080, RTX 4090, A10)

```bash
# 8B model – no quantization needed
python unified_eval.py \
  --model_path KAKA22/CodeRM-8B \
  --gpu_memory_utilization 0.85 \
  --max_model_len 2048 \
  --max_num_seqs 64 \
  --enforce_eager \
  ...

# 8B model – faster with CUDA graphs (if OOM, add --enforce_eager)
python unified_eval.py \
  --model_path KAKA22/CodeRM-8B \
  --gpu_memory_utilization 0.9 \
  --max_model_len 4096 \
  --max_num_seqs 64 \
  --no-enforce_eager \
  ...
```

- 8B fits without quantization
- Can try `--no-enforce_eager` for speed
- Can raise `max_model_len` to 4096 if needed

#### 40–48 GB GPU (e.g. A100 40GB, A6000)

```bash
python unified_eval.py \
  --model_path KAKA22/CodeRM-8B \
  --gpu_memory_utilization 0.95 \
  --max_model_len 8192 \
  --max_num_seqs 128 \
  --no-enforce_eager \
  ...
```

- High utilization (0.9–0.95)
- Larger `max_model_len` and `max_num_seqs`
- CUDA graphs usually safe

#### 80 GB GPU (e.g. A100 80GB, H100)

```bash
python unified_eval.py \
  --model_path KAKA22/CodeRM-8B \
  --gpu_memory_utilization 0.98 \
  --max_model_len 16384 \
  --max_num_seqs 256 \
  --no-enforce_eager \
  ...
```

- Near-maximum utilization
- Large context and batch sizes

#### Multi-GPU (Tensor Parallelism)

For models that don’t fit on one GPU (e.g. 70B):

```bash
python unified_eval.py \
  --model_path meta-llama/Llama-3-70B-Instruct \
  --tensor_parallel_size 2 \
  --num_gpus 2 \
  --gpu_memory_utilization 0.9 \
  ...
```

- Set `tensor_parallel_size` and `num_gpus` to the number of GPUs
- Ensure `get_free_gpus` threshold is high enough for all GPUs

---

### 4. Quick Reference

| GPU VRAM | Quantization | gpu_memory_util | max_num_seqs | enforce_eager |
|----------|--------------|-----------------|--------------|---------------|
| 8 GB     | bitsandbytes | 0.70–0.75       | 8–16         | Yes           |
| 12 GB    | bitsandbytes | 0.75–0.80       | 24–32        | Yes           |
| 16 GB    | Optional     | 0.80–0.85       | 48–64        | Yes           |
| 24 GB    | No           | 0.85–0.90       | 64           | Optional      |
| 40+ GB   | No           | 0.90–0.95       | 128+         | No            |
| 80 GB    | No           | 0.95–0.98       | 256+         | No            |

---

### 5. Troubleshooting

#### OOM (Out of Memory)

1. Enable quantization: `--quantization bitsandbytes`
2. Lower `--gpu_memory_utilization` (e.g. 0.6–0.7)
3. Lower `--max_num_seqs` (e.g. 8 or 16)
4. Lower `--max_model_len` (e.g. 1024)
5. Ensure `--enforce_eager` is set (default)

#### "No GPUs available with sufficient free memory"

- Lower the free-GPU threshold in `get_free_gpus(threshold=...)` (e.g. 4096)
- Free GPU memory: close other processes, `nvidia-smi` to inspect
- Set `CUDA_VISIBLE_DEVICES` to a specific GPU: `export CUDA_VISIBLE_DEVICES=0`

#### Slow Inference

- Try `--no-enforce_eager` if you have enough VRAM
- Increase `--max_num_seqs` if memory allows
- Increase `--gpu_memory_utilization` (e.g. 0.9) if stable


## Common Use Cases

### 1. Full Evaluation Pipeline

```bash
python unified_eval.py \
    --model_path /path/to/model \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_model_200.jsonl \
    --benchmark humaneval \
    --num_unit_tests 100 \
    --num_solutions 100 \
    --mp_num 16
```

### 2. Quick Test with Fewer Samples

```bash
python unified_eval.py \
    --model_path /path/to/model \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_model_200.jsonl \
    --num_unit_tests 10 \
    --num_solutions 10 \
    --skip_evalplus
```

### 3. Scaling Law Evaluation

Generate 100 unit tests first:

```bash
python unified_eval.py \
    --model_path /path/to/model \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_model_200.jsonl \
    --num_unit_tests 100 \
    --skip_evalplus
```

Then evaluate with different numbers using top-k sampling:

```bash
# Evaluate with 50 unit tests (top-50)
python unified_eval.py \
    --model_path /path/to/model \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_model_200.jsonl \
    --use_previous_ut \
    --previous_ut_path output/humaneval/inference/raw_inference_results.jsonl \
    --sample_top_k 50 \
    --num_unit_tests 50

# Evaluate with 25 unit tests (top-25)
python unified_eval.py \
    --model_path /path/to/model \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_model_200.jsonl \
    --use_previous_ut \
    --previous_ut_path output/humaneval/inference/raw_inference_results.jsonl \
    --sample_top_k 25 \
    --num_unit_tests 25
```

## Best Practices

1. **Start Small**: Test with `--num_unit_tests 10` and `--num_solutions 10` first
2. **Monitor Logs**: Check log file for warnings and errors
3. **Save Intermediate Results**: The script saves all intermediate results automatically
4. **Use Previous Results**: For scaling law evaluation, generate once and sample multiple times
5. **Parallelization**: Adjust `--mp_num` based on your CPU cores

## Advanced Usage

### Custom Logging

```bash
python unified_eval.py \
    --model_path /path/to/model \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_model_200.jsonl \
    --log_file custom/path/eval.log \
    --log_level DEBUG
```

### Multi-GPU Inference

```bash
python unified_eval.py \
    --model_path /path/to/model \
    --prompt_path data/benchmark/input_humaneval+_ut.jsonl \
    --solution_path data/result/humaneval+/sol_model_200.jsonl \
    --num_gpus 4 \
    --tensor_parallel_size 2
```

## Notes

- The script automatically detects free GPUs and uses them
- All intermediate files are saved for debugging and resumption
- The script is designed to be idempotent (can be re-run safely)
- Probability recording enables scaling law evaluation without re-generation
