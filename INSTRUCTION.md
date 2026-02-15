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

## Troubleshooting

### GPU Memory Issues

If you encounter GPU memory errors:

1. Reduce `--gpu_memory_utilization` (e.g., 0.6)
2. Reduce `--max_num_seqs` (e.g., 256)
3. Reduce `--num_gpus` or use smaller model

### Slow Execution

To speed up execution:

1. Increase `--mp_num` (number of parallel processes)
2. Increase `--chunk_size` for larger batches
3. Use more GPUs with `--num_gpus`

### EvalPlus Errors

If EvalPlus fails:

1. Set `EVALPLUS_MAX_MEMORY_BYTES=-1` (already done automatically)
2. Reduce `--evalplus_parallel`
3. Use `--skip_evalplus` to skip this step

### Probability Calculation

The script calculates probabilities from logprobs. If probabilities seem incorrect:

1. Check that `logprobs=5` is set in sampling params (it is by default)
2. Verify model supports logprobs
3. Check log file for warnings

## Integration with Existing Pipeline

The script is designed to be compatible with existing data formats:

- **Input**: Uses same JSONL format as original pipeline
- **Output**: Generates same format as original pipeline
- **Solutions**: Compatible with existing solution files

You can use the output files with existing scripts if needed.

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
