#!/usr/bin/env python3
"""
Unified Evaluation Script for CodeRM

This script provides a one-command solution for running the complete evaluation pipeline:
1. Inference with vLLM (with probability recording)
2. Unit test extraction
3. Unit test execution
4. Solution selection
5. EvalPlus evaluation

It supports:
- Generating unit tests with probability recording
- Top-p/top-k sampling from previously generated tests
- Comprehensive logging
- Parallel execution optimization
"""

import os
import sys
import json
import re
import ast
import time
import signal
import ctypes
import logging
import argparse
import subprocess
import contextlib
import unittest
import multiprocessing
from datetime import datetime
from io import StringIO
from pathlib import Path
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Value, Array
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ==================== Configuration ====================

@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    model_path: str
    prompt_path: str
    solution_path: str
    benchmark: str = "humaneval"
    num_unit_tests: int = 100
    num_solutions: int = 100
    output_dir: str = "output"
    log_file: Optional[str] = None
    log_level: str = "INFO"
    
    # Inference parameters
    dtype: str = "auto"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.8
    max_num_seqs: int = 64
    tensor_parallel_size: int = 1
    num_gpus: int = 1
    enforce_eager: bool = True  # Disable CUDA graphs to save GPU memory (helps on 16-24GB GPUs)
    quantization: Optional[str] = None  # "bitsandbytes" for 4-bit (saves ~50% VRAM), "awq"/"gptq" for pre-quantized models
    
    # Sampling parameters
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 2048
    n: int = 1  # Number of samples per prompt
    
    # Execution parameters
    mp_num: int = 8
    chunk_size: int = 1000
    time_limit_seconds: float = 1.0
    save_details: bool = False
    
    # Top-p/top-k sampling from previous results
    use_previous_ut: bool = False
    previous_ut_path: Optional[str] = None
    sample_top_p: Optional[float] = None
    sample_top_k: Optional[int] = None
    
    # Output file names
    raw_inference_filename: str = "raw_inference_results.jsonl"
    sampled_inference_filename: str = "sampled_inference_results.jsonl"
    
    # Unit test extraction parameters
    coderm: bool = True  # Whether to use CodeRM extraction mode
    
    # EvalPlus parameters
    evalplus_parallel: int = 8
    skip_evalplus: bool = False

    # LiveCodeBench evaluation
    anno_path: Optional[str] = None  # Path to anno file (sol_*_anno.jsonl) for LiveCodeBench evaluation
    anno_scenario: str = "codegeneration"  # Evaluator scenario for anno generation
    anno_release_version: Optional[str] = None  # LiveCodeBench release version for anno generation


# ==================== Logging Setup ====================

def setup_logging(config: EvalConfig) -> logging.Logger:
    """Setup logging to both console and file"""
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if config.log_file:
        os.makedirs(os.path.dirname(config.log_file) if os.path.dirname(config.log_file) else '.', exist_ok=True)
        file_handler = logging.FileHandler(config.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ==================== Utility Functions ====================

def compute_total_logprob(logprobs: Optional[List[Optional[Dict[int, Any]]]]) -> float:
    """
    Compute total logprob from vLLM logprobs structure.
    Each element is a dict mapping token_id -> Logprob (with .logprob attribute).
    Extracts .logprob to avoid TypeError when comparing Logprob instances.
    """
    if not logprobs:
        return 0.0
    return sum(
        max((lp.logprob for lp in token_logprobs.values()), default=0.0)
        if token_logprobs else 0.0
        for token_logprobs in logprobs
    )


def load_jsonl(filename: str) -> List[Dict]:
    """Load JSONL file"""
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(filename: str, dataset: List[Dict], overwrite: bool = False):
    """Save data to JSONL file"""
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f"The file '{filename}' already exists.")
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, "w", encoding="UTF-8") as fp:
        for data in tqdm(dataset, desc=f"Saving {os.path.basename(filename)}"):
            fp.write(json.dumps(data, ensure_ascii=False) + "\n")


def get_free_gpus(threshold: int = 8192) -> List[int]:
    """Get list of free GPUs with at least threshold MiB free memory.
    Default 4096 MiB (4GB) is sufficient for most inference workloads.
    """
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
            shell=True
        )
        gpu_free_memory = [int(x) for x in output.decode("utf-8").strip().split("\n")]
        free_gpus = [i for i, mem in enumerate(gpu_free_memory) if mem > threshold]
        return free_gpus
    except Exception as e:
        logging.warning(f"Could not query GPU memory: {e}. Using default GPU 0.")
        return [0]


# ==================== Unit Test Extraction ====================

UNITTEST_FORMAT = """{code}

suite = unittest.TestLoader().loadTestsFromTestCase({class_name})
runner = unittest.TextTestRunner(stream=output, verbosity=2)
result = runner.run(suite)
locals_dict['result'] = result
"""


def extract_code(markdown_text: str) -> List[str]:
    """Extract code from markdown"""
    pattern = r'```python\n(.*?)\n```'
    matches = re.findall(pattern, markdown_text, re.DOTALL)
    
    if len(matches) == 0:
        pattern = r'```\n(.*?)\n```'
        matches = re.findall(pattern, markdown_text, re.DOTALL)
    
    return matches


def extract_class_names(code: str) -> List[str]:
    """Extract class names from code"""
    try:
        tree = ast.parse(code)
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        return class_names
    except Exception:
        return []


def remove_import_lines(code: str) -> str:
    """Remove import lines for your_module"""
    pattern = r'^from\s+your_module\s+import\s+\w+(\s+#.*)?$'
    cleaned_code = re.sub(pattern, '', code, flags=re.MULTILINE)
    return cleaned_code


def extract_imports(code: str) -> List[str]:
    """Extract import statements"""
    imports = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
    except:
        return ['import unittest']
    return imports


def remove_func(code: str) -> str:
    """Remove function definitions, keep only class"""
    imports = '\n'.join(extract_imports(code))
    code = code[code.find('\nclass'):]   
    if '\ndef' in code:
        code = code[:code.find('\ndef')]
        lines = code.splitlines()
        while '#' in lines[-1]:
            lines = lines[:-1]
        code = '\n'.join(lines)
    return imports + '\n' + code


def extract_unit_test(response: str, coderm: bool = True) -> str:
    """Extract unit test from response"""
    code = extract_code(response)
    unit_test = ""
    if coderm:
        code = [response]
    if len(code) == 1:
        code = code[0]
        class_names = extract_class_names(code)
        if len(class_names) == 1:
            if "__name__ == '__main__'" in code:
                code = code.replace("if __name__ == '__main__':\n    unittest.main()", "")
            if '__name__ == "__main__"' in code:
                code = code.replace('if __name__ == "__main__":\n    unittest.main()', "")
            code = remove_import_lines(code)
            code = remove_func(code)
            unit_test = UNITTEST_FORMAT.format_map({'code': code.rstrip('\n'), 'class_name': class_names[0]})
    return unit_test


# ==================== Inference with Probability Recording ====================

def run_inference(
    config: EvalConfig,
    logger: logging.Logger
) -> Tuple[List[Dict], str]:
    """Run inference with vLLM and record probabilities"""
    logger.info("=" * 60)
    logger.info("STEP 1: Running Inference")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Load prompts
    logger.info(f"Loading prompts from {config.prompt_path}")
    prompts_data = load_jsonl(config.prompt_path)
    logger.info(f"Loaded {len(prompts_data)} prompts")
    
    # Check if we should use previous unit tests
    if config.use_previous_ut and config.previous_ut_path:
        logger.info(f"Using previous unit tests from {config.previous_ut_path}")
        return load_previous_and_sample(config, logger)
    
    # Setup vLLM
    logger.info(f"Initializing vLLM with model: {config.model_path}")
    free_gpus = get_free_gpus(threshold=8192)  # 4GB min free memory for inference
    if len(free_gpus) == 0:
        raise RuntimeError(
            "No GPUs available with sufficient free memory (need >4GB). "
            "Check: (1) nvidia-smi shows your GPU, (2) GPU has enough free memory, "
            "(3) CUDA drivers are properly installed for WSL."
        )
    if len(free_gpus) < config.num_gpus:
        logger.warning(f"Only {len(free_gpus)} GPUs available, requested {config.num_gpus}")
        config.num_gpus = len(free_gpus)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in free_gpus[:config.num_gpus])
    
    llm_kwargs: Dict[str, Any] = {
        "model": config.model_path,
        "trust_remote_code": True,
        "dtype": config.dtype,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": config.tensor_parallel_size,
        "max_num_seqs": config.max_num_seqs,
        "enforce_eager": config.enforce_eager,
    }
    if config.quantization:
        llm_kwargs["quantization"] = config.quantization
        logger.info(f"Using quantization: {config.quantization} (reduces GPU memory usage)")
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    
    # Prepare prompts
    logger.info("Preparing prompts")
    prompts = []
    for data in prompts_data:
        messages = data["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompts.append(prompt)
    
    # Setup sampling params with logprobs
    sampling_params = SamplingParams(
        n=config.n,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        top_k=config.top_k if config.top_k > 0 else -1,
        temperature=config.temperature,
        logprobs=5,  # Get top 5 logprobs for probability calculation
    )
    
    # Generate
    logger.info(f"Generating {config.num_unit_tests} unit tests per prompt")
    all_outputs = []
    
    # Generate multiple samples per prompt
    num_samples_per_prompt = config.num_unit_tests // config.n
    if config.num_unit_tests % config.n != 0:
        num_samples_per_prompt += 1
    
    for sample_idx in range(num_samples_per_prompt):
        logger.info(f"Generating batch {sample_idx + 1}/{num_samples_per_prompt}")
        outputs_list = llm.generate(prompts, sampling_params)
        
        for prompt_idx, outputs in enumerate(outputs_list):
            for output in outputs.outputs:
                # Calculate probability from logprobs
                logprobs = output.logprobs
                total_logprob = compute_total_logprob(logprobs)
                probability = float(torch.exp(torch.tensor(total_logprob)))
                
                all_outputs.append({
                    'task_id': prompts_data[prompt_idx]['task_id'],
                    'messages': prompts_data[prompt_idx]['messages'],
                    'prompt': prompts[prompt_idx],
                    'response': output.text,
                    'probability': probability,
                    'logprob': total_logprob,
                })
    
    # Limit to requested number
    all_outputs = all_outputs[:config.num_unit_tests * len(prompts_data)]
    
    elapsed = time.time() - start_time
    logger.info(f"Inference completed in {elapsed:.2f} seconds")
    logger.info(f"Generated {len(all_outputs)} unit test candidates")
    
    # Save raw inference results
    output_dir = os.path.join(config.output_dir, config.benchmark, "inference")
    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, config.raw_inference_filename)
    save_jsonl(raw_output_path, all_outputs, overwrite=True)
    logger.info(f"Saved raw inference results to {raw_output_path}")
    
    return all_outputs, raw_output_path


def load_previous_and_sample(
    config: EvalConfig,
    logger: logging.Logger
) -> Tuple[List[Dict], str]:
    """Load previous unit tests and apply top-p/top-k sampling"""
    logger.info("Loading previous unit test results")
    
    # Check if file exists
    if not os.path.exists(config.previous_ut_path):
        error_msg = (
            f"Previous unit test file does not exist: {config.previous_ut_path}\n"
            f"Please ensure the file exists or generate unit tests first."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    previous_data = load_jsonl(config.previous_ut_path)
    
    if not previous_data:
        error_msg = (
            f"Previous unit test file is empty: {config.previous_ut_path}\n"
            f"Please ensure the file contains valid unit test data."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Group by task_id
    task_to_tests = {}
    for item in previous_data:
        task_id = item['task_id']
        if task_id not in task_to_tests:
            task_to_tests[task_id] = []
        task_to_tests[task_id].append(item)
    
    # Check if we have enough unit tests per task
    insufficient_tasks = []
    for task_id, tests in task_to_tests.items():
        available_count = len(tests)
        required_count = config.num_unit_tests
        
        # If using top-k sampling, check against top-k limit
        if config.sample_top_k:
            available_count = min(available_count, config.sample_top_k)
        
        if available_count < required_count:
            insufficient_tasks.append({
                'task_id': task_id,
                'available': available_count,
                'required': required_count
            })
    
    if insufficient_tasks:
        error_msg = (
            f"Insufficient unit tests available for {len(insufficient_tasks)} task(s).\n"
            f"Required: {config.num_unit_tests} unit tests per task.\n"
            f"Tasks with insufficient unit tests:\n"
        )
        for task_info in insufficient_tasks:
            error_msg += (
                f"  - {task_info['task_id']}: "
                f"available={task_info['available']}, "
                f"required={task_info['required']}\n"
            )
        if config.sample_top_k:
            error_msg += (
                f"\nNote: Top-k sampling is set to {config.sample_top_k}, "
                f"which may limit available tests."
            )
        elif config.sample_top_p:
            error_msg += (
                f"\nNote: Top-p sampling is set to {config.sample_top_p}, "
                f"which may limit available tests."
            )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Apply sampling
    sampled_outputs = []
    for task_id, tests in task_to_tests.items():
        # Sort by probability (descending)
        tests_sorted = sorted(tests, key=lambda x: x.get('probability', 0.0), reverse=True)
        
        if config.sample_top_k:
            # Top-k sampling
            sampled = tests_sorted[:config.sample_top_k]
        elif config.sample_top_p:
            # Top-p sampling
            cumulative_prob = 0.0
            sampled = []
            for test in tests_sorted:
                prob = test.get('probability', 0.0)
                if cumulative_prob + prob <= config.sample_top_p:
                    sampled.append(test)
                    cumulative_prob += prob
                else:
                    break
        else:
            # No sampling, use all
            sampled = tests_sorted
        
        sampled_outputs.extend(sampled[:config.num_unit_tests])
    
    logger.info(f"Sampled {len(sampled_outputs)} unit tests from previous results")
    
    output_dir = os.path.join(config.output_dir, config.benchmark, "inference")
    os.makedirs(output_dir, exist_ok=True)
    raw_output_path = os.path.join(output_dir, config.sampled_inference_filename)
    save_jsonl(raw_output_path, sampled_outputs, overwrite=True)
    
    return sampled_outputs, raw_output_path


# ==================== Unit Test Extraction ====================

def extract_unit_tests(
    inference_results: List[Dict],
    config: EvalConfig,
    logger: logging.Logger
) -> str:
    """Extract unit tests from inference results"""
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting Unit Tests")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Group by task_id
    task_to_tests = {}
    for item in inference_results:
        task_id = item['task_id']
        if task_id not in task_to_tests:
            task_to_tests[task_id] = []
        task_to_tests[task_id].append(item)
    
    # Extract unit tests
    output = []
    for task_id in tqdm(sorted(task_to_tests.keys()), desc="Extracting unit tests"):
        tests = task_to_tests[task_id]
        ut_set = set()
        for item in tests:
            unit_test = extract_unit_test(item['response'], coderm=config.coderm)
            if unit_test != "" and unit_test not in ut_set:
                ut_set.add(unit_test)
        
        output.append({
            'task_id': task_id,
            'unit_tests': list(ut_set)
        })
    
    # Save extracted unit tests
    output_dir = os.path.join(config.output_dir, config.benchmark, "unit_tests")
    os.makedirs(output_dir, exist_ok=True)
    ut_path = os.path.join(output_dir, f"unit_tests_{config.num_unit_tests}.jsonl")
    save_jsonl(ut_path, output, overwrite=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Extracted {len(output)} unit test sets in {elapsed:.2f} seconds")
    logger.info(f"Saved to {ut_path}")
    
    return ut_path


# ==================== Unit Test Execution ====================

class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout"""
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout = old_stdout


@contextlib.contextmanager
def time_limit(seconds: float):
    """Time limit context manager"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def execute_unittest(
    code: str,
    time_limits: float,
    is_pass: Value,
    total_num: Value,
    pass_num: Value,
    fail_num: Value,
    error_num: Value,
    save_detail: bool,
    shared_array: Array,
):
    """Execute unit test in isolated process"""
    output = StringIO()
    locals_dict = {}
    try:
        with time_limit(time_limits), suppress_stdout():
            exec(
                code,
                {"unittest": unittest, "output": output, "locals_dict": locals_dict},
            )
        
        if save_detail:
            encoded_output = output.getvalue().encode("utf-8")
            shared_array.value = encoded_output[: len(shared_array)]
        result = locals_dict.get("result")
        if result is None:
            raise RuntimeError("No unittest result captured from executed code.")
        is_pass.value = result.wasSuccessful()
        total_num.value = result.testsRun
        pass_num.value = result.testsRun - len(result.failures) - len(result.errors)
        fail_num.value = len(result.failures)
        error_num.value = len(result.errors)
    except TimeoutException:
        is_pass.value = False
        if save_detail:
            shared_array.value = "Timed out!".encode("utf-8")
    except Exception as error:
        is_pass.value = False
        if save_detail:
            encoded_output = f"{output.getvalue()}\n{error}"
            shared_array.value = encoded_output.encode("utf-8")[: len(shared_array)]


def handle_execute(
    task_id: str,
    solution_id: int,
    test_case_id: int,
    code: str,
    time_limits: float,
    save_detail: bool,
):
    """Handle execution of a single unit test"""
    is_pass = Value("b", False)
    total_num = Value("i", -1)
    pass_num = Value("i", 0)
    fail_num = Value("i", 0)
    error_num = Value("i", 0)
    shared_array = Array(ctypes.c_char, 2000)
    
    process = multiprocessing.Process(
        target=execute_unittest,
        args=(
            code,
            time_limits,
            is_pass,
            total_num,
            pass_num,
            fail_num,
            error_num,
            save_detail,
            shared_array,
        ),
    )
    process.start()
    process.join(time_limits + 1)
    
    if process.is_alive():
        process.terminate()
        process.join(0.1)
    if process.is_alive():
        process.kill()
        process.join(0.1)
    
    details_text = ""
    if save_detail:
        try:
            details_text = shared_array.value.decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            details_text = ""
    
    details = {
        "total_num": total_num.value,
        "pass_num": pass_num.value,
        "fail_num": fail_num.value,
        "error_num": error_num.value,
        "text": details_text,
    }
    
    return task_id, solution_id, test_case_id, bool(is_pass.value), details


def read_jsonline_in_chunks(file_path: str, chunk_size: int):
    """Read JSONL file in chunks"""
    with open(file_path, "r", encoding="utf-8") as fp:
        chunk = []
        for index, line in enumerate(fp):
            chunk.append(json.loads(line))
            if (index + 1) % chunk_size == 0:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def run_unit_tests(
    input_path: str,
    output_path: str,
    mp_num: int,
    chunk_size: int = 1000,
    recover: int = 0,
    details: bool = False,
    time_limit_seconds: float = 1,
    logger: Optional[logging.Logger] = None,
):
    """Run unit tests in parallel"""
    if logger:
        logger.info(f"Running unit tests from {input_path}")
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    processed_records = 0
    for chunk in read_jsonline_in_chunks(input_path, chunk_size=chunk_size):
        processed_records += len(chunk)
        if logger:
            logger.info(f"Processing chunk: {processed_records} records")
        if recover >= processed_records:
            continue
        
        max_workers = mp_num or max(1, multiprocessing.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for data in tqdm(chunk, desc="Scheduling unit tests", leave=False):
                futures.append(
                    executor.submit(
                        handle_execute,
                        data["task_id"],
                        data["sol_id"],
                        data["ut_id"],
                        data["code"],
                        time_limit_seconds,
                        details,
                    )
                )
            
            raw_results = []
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Collecting results", leave=False
            ):
                task_id, solution_id, test_case_id, is_pass, details_payload = future.result()
                raw_results.append(
                    {
                        "task_id": task_id,
                        "sol_id": solution_id,
                        "ut_id": test_case_id,
                        "result": "pass" if is_pass else "fail",
                        "details": details_payload,
                    }
                )
        
        sorted_results = sorted(raw_results, key=itemgetter("task_id", "sol_id", "ut_id"))
        
        with open(output_path, "a", encoding="UTF-8") as fp:
            for result in sorted_results:
                fp.write(json.dumps(result, ensure_ascii=False) + "\n")


def execute_unit_tests(
    solution_path: str,
    unit_test_path: str,
    config: EvalConfig,
    logger: logging.Logger
) -> str:
    """Execute unit tests on solutions"""
    logger.info("=" * 60)
    logger.info("STEP 3: Executing Unit Tests")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Load solutions and unit tests
    logger.info(f"Loading solutions from {solution_path}")
    sol_dataset = load_jsonl(solution_path)
    logger.info(f"Loading unit tests from {unit_test_path}")
    ut_dataset = load_jsonl(unit_test_path)
    
    # Create task_id mapping
    sol_dict = {item['task_id']: item for item in sol_dataset}
    ut_dict = {item['task_id']: item for item in ut_dataset}
    
    # Combine solutions and unit tests
    logger.info("Combining solutions and unit tests")
    output = []
    for task_id in tqdm(sorted(sol_dict.keys()), desc="Combining"):
        if task_id not in ut_dict:
            logger.warning(f"Task {task_id} not found in unit tests")
            continue
        
        solutions = sol_dict[task_id]['solutions'][:config.num_solutions]
        unit_tests = ut_dict[task_id]['unit_tests'][:config.num_unit_tests]
        
        for sol_id in range(len(solutions)):
            for ut_id in range(len(unit_tests)):
                code = solutions[sol_id] + "\n\n" + unit_tests[ut_id]
                output.append({
                    "task_id": task_id,
                    "sol_id": sol_id,
                    "ut_id": ut_id,
                    "code": code,
                })
    
    # Save combined file
    output_dir = os.path.join(config.output_dir, config.benchmark, "execution")
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, "sol_ut_combined.jsonl")
    save_jsonl(combined_path, output, overwrite=True)
    logger.info(f"Created {len(output)} solution-unit test pairs")
    
    # Execute unit tests
    result_path = os.path.join(output_dir, f"{config.num_solutions}_sol_{config.num_unit_tests}_ut_result.jsonl")
    logger.info(f"Executing unit tests with {config.mp_num} processes")
    run_unit_tests(
        input_path=combined_path,
        output_path=result_path,
        mp_num=config.mp_num,
        chunk_size=config.chunk_size,
        recover=0,
        details=config.save_details,
        time_limit_seconds=config.time_limit_seconds,
        logger=logger,
    )
    
    # Clean up combined file
    if os.path.exists(combined_path):
        os.remove(combined_path)
    
    elapsed = time.time() - start_time
    logger.info(f"Execution completed in {elapsed:.2f} seconds")
    logger.info(f"Results saved to {result_path}")
    
    return result_path


# ==================== Solution Selection ====================

def select_solutions(
    result_path: str,
    solution_path: str,
    config: EvalConfig,
    logger: logging.Logger
) -> str:
    """Select best solutions using majority voting"""
    logger.info("=" * 60)
    logger.info("STEP 4: Selecting Solutions")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Load results
    logger.info(f"Loading execution results from {result_path}")
    dataset = load_jsonl(result_path)
    dataset = sorted(
        dataset,
        key=lambda x: (int(x["task_id"].split("/")[1]) if "/" in x["task_id"] else 0, x["sol_id"], x["ut_id"]),
    )
    
    # Load solutions
    logger.info(f"Loading solutions from {solution_path}")
    sol_dataset = load_jsonl(solution_path)
    task_id_to_sol_entry = {entry["task_id"]: entry for entry in sol_dataset}
    
    # Select solutions
    current_task = None
    if config.benchmark == "humaneval":
        current_task = "HumanEval/0"
    elif config.benchmark == "mbpp":
        current_task = "Mbpp/2"
    else:
        current_task = dataset[0]["task_id"] if dataset else None
    
    solution_dict = {i: set() for i in range(config.num_solutions)}
    chosen_solution = []
    
    for dataset_idx in range(len(dataset) + 1):
        data = dataset[dataset_idx] if dataset_idx < len(dataset) else None
        if data and data["task_id"] == current_task:
            if data["result"] == "pass":
                solution_dict[data["sol_id"]].add(data["ut_id"])
        else:
            # Calculate best solution
            if current_task:
                sorted_solution_dict = sorted(
                    solution_dict.items(), key=lambda item: len(item[1]), reverse=True
                )
                highest_pass = len(sorted_solution_dict[0][1]) if sorted_solution_dict else 0
                potential_solutions = []
                for sol_id_and_ut_ids in sorted_solution_dict:
                    if len(sol_id_and_ut_ids[1]) < highest_pass:
                        break
                    potential_solutions.append(sol_id_and_ut_ids)
                
                max_consistency = 0
                per_task_chosen_solutions = []
                for sol_id, ut_ids in potential_solutions:
                    consistency = 0
                    for _, other_ut_ids in potential_solutions:
                        if ut_ids == other_ut_ids:
                            consistency += 1
                    if consistency > max_consistency:
                        max_consistency = consistency
                        per_task_chosen_solutions = [sol_id]
                    elif consistency == max_consistency and max_consistency > 0:
                        per_task_chosen_solutions.append(sol_id)
                
                for sol_id in per_task_chosen_solutions:
                    chosen_solution.append({
                        "task_id": current_task,
                        "chosen_solution": sol_id
                    })
            
            # Initialize for next task
            if dataset_idx < len(dataset):
                current_task = dataset[dataset_idx]["task_id"]
                solution_dict = {i: set() for i in range(config.num_solutions)}
    
    # Create output
    output = []
    for chosen_sol_entry in chosen_solution:
        task_id = chosen_sol_entry["task_id"]
        if task_id in task_id_to_sol_entry:
            solutions = task_id_to_sol_entry[task_id]["solutions"]
            sol_id = chosen_sol_entry["chosen_solution"]
            if sol_id < len(solutions):
                output.append({
                    "task_id": task_id,
                    "solution": solutions[sol_id],
                    "sol_id": sol_id,
                })
    
    # Save selected solutions
    output_dir = os.path.join(config.output_dir, config.benchmark, "selection")
    os.makedirs(output_dir, exist_ok=True)
    save_name = f"select_in_{config.num_solutions}_sol_by_{config.num_unit_tests}_ut_max+vote.jsonl"
    selected_path = os.path.join(output_dir, save_name)
    save_jsonl(selected_path, output, overwrite=True)
    
    elapsed = time.time() - start_time
    logger.info(f"Selected {len(output)} solutions in {elapsed:.2f} seconds")
    logger.info(f"Saved to {selected_path}")
    
    return selected_path


# ==================== EvalPlus Evaluation ====================

def _parse_evalplus_stdout(stdout: str) -> Dict[str, Dict[str, float]]:
    """Parse pass@k results from EvalPlus stdout. Returns dict like {'base': {'pass@1': 0.714}, 'plus': {'pass@1': 0.634}}."""
    results = {}
    # Match lines like "pass@1:\t0.714" or "pass@10: 0.xxx"
    # EvalPlus prints (base tests) first, then (base + extra tests)
    lines = stdout.strip().split('\n')
    current_suite = None
    for line in lines:
        line = line.strip()
        if 'base tests' in line.lower() and 'extra' not in line.lower():
            current_suite = 'base'
            results[current_suite] = {}
        elif 'base + extra' in line.lower() or 'humaneval+' in line.lower():
            current_suite = 'plus'
            results[current_suite] = {}
        match = re.match(r'pass@(\d+)\s*:\s*([\d.]+)', line, re.IGNORECASE)
        if match and current_suite:
            k, val = match.group(1), float(match.group(2))
            results[current_suite][f'pass@{k}'] = val
    return results


def _load_evalplus_results(selected_path: str) -> Optional[Dict[str, Any]]:
    """Load pass_at_k from EvalPlus eval_results.json if it exists.
    EvalPlus writes to either {base}_eval_results.json or {base}.eval_results.json."""
    base = selected_path.replace(".jsonl", "")
    for suffix in ("_eval_results.json", ".eval_results.json"):
        path = base + suffix
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return data.get('pass_at_k')
            except (json.JSONDecodeError, IOError):
                pass
    return None


def _log_final_eval_results(
    parsed_stdout: Dict[str, Dict[str, float]],
    selected_path: str,
    logger: logging.Logger,
    benchmark: str = "humaneval",
):
    """Log final evaluation results prominently."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 60)

    # Prefer eval_results.json (authoritative) over parsed stdout
    pass_at_k = _load_evalplus_results(selected_path)
    bench = benchmark.upper().replace("+", "") if benchmark else ""

    if pass_at_k:
        for suite_name, metrics in pass_at_k.items():
            display_name = f"{bench} (base tests)" if suite_name == "base" else f"{bench}+ (base + extra tests)"
            logger.info(f"  {display_name}:")
            for k, v in sorted(metrics.items(), key=lambda x: int(x[0].split('@')[1])):
                logger.info(f"    {k}: {v:.3f}")
    elif parsed_stdout:
        for suite_name, metrics in parsed_stdout.items():
            display_name = f"{bench} (base tests)" if suite_name == "base" else f"{bench}+ (base + extra tests)"
            logger.info(f"  {display_name}:")
            for k, v in sorted(metrics.items(), key=lambda x: int(x[0].split('@')[1])):
                logger.info(f"    {k}: {v:.3f}")
    else:
        logger.info("  (Results could not be parsed)")

    logger.info("=" * 60)
    logger.info("")


# ==================== LiveCodeBench Evaluation ====================

def generate_anno_from_solution(
    solution_path: str,
    config: EvalConfig,
    logger: logging.Logger,
) -> str:
    """Auto-generate anno.jsonl from solution func.jsonl via LiveCodeBench evaluator.

    Returns the path to the generated anno file.
    """
    from evaluation.generate_livecodebench_anno import (
        load_json_or_jsonl,
        build_custom_output,
        run_custom_evaluator,
        load_graded_map,
        convert_to_anno,
        save_jsonl,
        save_json,
    )

    # Determine output path: put anno in the output directory
    anno_dir = os.path.join(config.output_dir, config.benchmark, "anno")
    os.makedirs(anno_dir, exist_ok=True)
    sol_basename = os.path.basename(solution_path)
    anno_basename = sol_basename.replace("_func.jsonl", "_anno.jsonl")
    if anno_basename == sol_basename:
        # Fallback if filename doesn't contain _func
        anno_basename = sol_basename.replace(".jsonl", "_anno.jsonl")
    anno_path = os.path.join(anno_dir, anno_basename)

    logger.info(f"Loading solutions from {solution_path}")
    func_rows = load_json_or_jsonl(Path(solution_path))
    if not func_rows:
        raise ValueError(f"No data found in {solution_path}")
    logger.info(f"Loaded {len(func_rows)} tasks")

    # Build custom output for evaluator
    custom_output_path = os.path.join(anno_dir, "custom_output.json")
    custom_rows = build_custom_output(func_rows)
    save_json(Path(custom_output_path), custom_rows, overwrite=True)
    logger.info(f"Built custom evaluator input: {custom_output_path}")

    # Run LiveCodeBench evaluator
    logger.info("Running LiveCodeBench custom evaluator...")
    graded_path, evaluator_logs = run_custom_evaluator(
        custom_output_file=Path(custom_output_path),
        scenario=config.anno_scenario,
        release_version=config.anno_release_version,
        evaluator_module="lcb_runner.runner.custom_evaluator",
        python_executable=sys.executable,
        preferred_eval_file=None,
    )
    logger.info(f"Evaluator finished. Graded output: {graded_path}")
    if evaluator_logs.strip():
        # Log last 500 chars of evaluator output
        logger.debug(f"Evaluator logs (tail): {evaluator_logs[-500:]}")

    # Convert graded results to anno format
    graded_map = load_graded_map(graded_path)
    anno_rows, warnings = convert_to_anno(func_rows, graded_map, strict=False)
    save_jsonl(Path(anno_path), anno_rows, overwrite=True)

    logger.info(f"Generated anno: {len(anno_rows)} tasks -> {anno_path}")
    if warnings:
        for w in warnings[:10]:
            logger.warning(f"Anno generation: {w}")

    return anno_path


def run_livecodebench_eval(
    selected_path: str,
    config: EvalConfig,
    logger: logging.Logger,
):
    """Evaluate selected solutions against LiveCodeBench anno (ground truth).

    For each task, the selected solution's sol_id is looked up in the anno file
    to determine pass/fail.  When multiple solutions are selected for a task
    (majority-voting tie), accuracy is the fraction that pass — consistent with
    calculate_result.py.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Running LiveCodeBench Evaluation (anno lookup)")
    logger.info("=" * 60)

    if not config.anno_path:
        logger.info("No --anno_path provided. Auto-generating anno from solution_path...")
        config.anno_path = generate_anno_from_solution(
            config.solution_path, config, logger
        )

    start_time = time.time()

    # Load anno → build lookup {task_id: {sol_id: "pass"/"fail"}}
    anno_data = load_jsonl(config.anno_path)
    anno_lookup: Dict[str, Dict[int, str]] = {}
    for entry in anno_data:
        task_id = str(entry["task_id"])
        anno_lookup[task_id] = {}
        for sol in entry["solutions"]:
            anno_lookup[task_id][sol["sol_id"]] = sol["result"]
    logger.info(f"Loaded anno for {len(anno_lookup)} tasks from {config.anno_path}")

    # Load selected solutions
    selected = load_jsonl(selected_path)
    logger.info(f"Loaded {len(selected)} selected solutions from {selected_path}")

    # Group by task_id (a task may have multiple selected solutions on tie)
    from collections import defaultdict
    task_selections: Dict[str, List[int]] = defaultdict(list)
    for entry in selected:
        task_id = str(entry["task_id"])
        sol_id = entry.get("sol_id")
        if sol_id is None:
            logger.warning(f"task_id={task_id}: missing sol_id in selection output, skipping")
            continue
        task_selections[task_id].append(sol_id)

    # Calculate accuracy
    total_tasks = len(task_selections)
    total_accuracy = 0.0
    pass_count = 0
    fail_count = 0
    missing_count = 0

    for task_id, sol_ids in task_selections.items():
        if task_id not in anno_lookup:
            logger.warning(f"task_id={task_id}: not found in anno, treating as fail")
            missing_count += 1
            continue

        # For each selected solution, check ground truth
        task_pass = 0
        for sol_id in sol_ids:
            result = anno_lookup[task_id].get(sol_id, "fail")
            if result == "pass":
                task_pass += 1

        # Accuracy contribution: fraction of selected solutions that pass
        task_accuracy = task_pass / len(sol_ids)
        total_accuracy += task_accuracy
        if task_accuracy > 0:
            pass_count += 1
        else:
            fail_count += 1

    accuracy = total_accuracy / total_tasks if total_tasks > 0 else 0.0

    elapsed = time.time() - start_time

    # Log results
    logger.info("")
    logger.info("=" * 60)
    logger.info("LIVECODEBENCH EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total tasks evaluated: {total_tasks}")
    logger.info(f"  Tasks with passing solution: {pass_count}")
    logger.info(f"  Tasks with failing solution: {fail_count}")
    if missing_count:
        logger.info(f"  Tasks missing from anno: {missing_count}")
    logger.info(f"  pass@1: {accuracy:.4f}")
    logger.info("=" * 60)
    logger.info(f"LiveCodeBench evaluation completed in {elapsed:.2f} seconds")

    return {"pass@1": accuracy}


def run_evalplus(
    selected_path: str,
    config: EvalConfig,
    logger: logging.Logger
):
    """Run EvalPlus evaluation"""
    logger.info("=" * 60)
    logger.info("STEP 5: Running EvalPlus Evaluation")
    logger.info("=" * 60)

    if config.skip_evalplus:
        logger.info("Skipping EvalPlus evaluation (--skip_evalplus flag set)")
        return

    # LiveCodeBench uses anno-based evaluation instead of EvalPlus
    if config.benchmark == "livecodebench":
        return run_livecodebench_eval(selected_path, config, logger)

    start_time = time.time()
    
    # Run evalplus (use python -m for reliable invocation across environments)
    benchmark_name = config.benchmark.replace("+", "")
    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        benchmark_name,
        "--samples", selected_path,
        "--parallel", str(config.evalplus_parallel),
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["EVALPLUS_MAX_MEMORY_BYTES"] = "-1"
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("EvalPlus evaluation completed successfully")

        # Parse pass@k from stdout for display
        parsed = _parse_evalplus_stdout(result.stdout)
        if parsed:
            logger.info("Pass@k from EvalPlus stdout:")
            for suite, metrics in parsed.items():
                for k, v in metrics.items():
                    logger.info(f"  {suite}: {k} = {v:.3f}")

        # Suppress verbose stderr (tqdm progress bars) - only log a brief summary
        if result.stderr:
            stderr_lines = result.stderr.strip().split('\n')
            n_lines = len(stderr_lines)
            if n_lines > 20:
                logger.info(
                    f"EvalPlus progress output: {n_lines} lines (tqdm progress bars suppressed)"
                )
            else:
                logger.info(f"EvalPlus stderr: {result.stderr[:500]}")
    except subprocess.CalledProcessError as e:
        logger.error(f"EvalPlus evaluation failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("evalplus.evaluate command not found. Please install evalplus: pip install evalplus")
        raise
    
    elapsed = time.time() - start_time
    logger.info(f"EvalPlus evaluation completed in {elapsed:.2f} seconds")

    # Log final results prominently
    _log_final_eval_results(parsed, selected_path, logger, config.benchmark)


# ==================== Main Pipeline ====================

def main():
    """Main evaluation pipeline"""
    parser = argparse.ArgumentParser(
        description="Unified Evaluation Script for CodeRM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model for unit test generation")
    parser.add_argument("--prompt_path", type=str, required=True,
                       help="Path to the prompt JSONL file (e.g., data/benchmark/input_humaneval+_ut.jsonl)")
    parser.add_argument("--solution_path", type=str, required=True,
                       help="Path to the solution JSONL file")
    
    # Optional arguments
    parser.add_argument("--benchmark", type=str, default="humaneval",
                       choices=["humaneval", "mbpp", "livecodebench"],
                       help="Benchmark name")
    parser.add_argument("--num_unit_tests", type=int, default=100,
                       help="Number of unit tests to generate per task")
    parser.add_argument("--num_solutions", type=int, default=100,
                       help="Number of solutions to evaluate")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path (default: output_dir/benchmark/eval.log)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Inference parameters
    parser.add_argument("--dtype", type=str, default="auto",
                       help="Model dtype")
    parser.add_argument("--max_model_len", type=int, default=2048,
                       help="Maximum model length (default 2048 for memory-constrained GPUs)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                       help="GPU memory utilization")
    parser.add_argument("--max_num_seqs", type=int, default=64,
                       help="Maximum number of sequences (default 64 for memory-constrained GPUs)")
    parser.add_argument("--no-enforce_eager", dest="enforce_eager", action="store_false", default=True,
                       help="Disable enforce_eager (faster but uses more GPU memory)")
    parser.add_argument("--quantization", type=str, default=None,
                       choices=["bitsandbytes", "awq", "gptq"],
                       help="Quantization: bitsandbytes (4-bit, works with any model, ~50%% VRAM savings), "
                            "awq/gptq (for pre-quantized models on HuggingFace)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use")
    
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1,
                       help="Top-k sampling parameter (-1 to disable)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum tokens to generate")
    parser.add_argument("--n", type=int, default=1,
                       help="Number of samples per prompt")
    
    # Execution parameters
    parser.add_argument("--mp_num", type=int, default=8,
                       help="Number of processes for unit test execution")
    parser.add_argument("--chunk_size", type=int, default=1000,
                       help="Chunk size for processing")
    parser.add_argument("--time_limit_seconds", type=float, default=1.0,
                       help="Time limit per unit test execution")
    parser.add_argument("--save_details", action="store_true",
                       help="Save detailed execution results")
    
    # Top-p/top-k sampling from previous results
    parser.add_argument("--use_previous_ut", action="store_true",
                       help="Use previously generated unit tests")
    parser.add_argument("--previous_ut_path", type=str, default=None,
                       help="Path to previous unit test results (with probabilities)")
    parser.add_argument("--sample_top_p", type=float, default=None,
                       help="Top-p value for sampling from previous results")
    parser.add_argument("--sample_top_k", type=int, default=None,
                       help="Top-k value for sampling from previous results")
    
    # Output file names
    parser.add_argument("--raw_inference_filename", type=str, default="raw_inference_results.jsonl",
                       help="Filename for raw inference results (default: raw_inference_results.jsonl)")
    parser.add_argument("--sampled_inference_filename", type=str, default="sampled_inference_results.jsonl",
                       help="Filename for sampled inference results (default: sampled_inference_results.jsonl)")
    
    # Unit test extraction parameters
    parser.add_argument("--no-coderm", dest="coderm", action="store_false", default=True,
                       help="Disable CodeRM extraction mode (default: coderm=True)")
    
    # EvalPlus parameters
    parser.add_argument("--evalplus_parallel", type=int, default=8,
                       help="Number of parallel processes for EvalPlus")
    parser.add_argument("--skip_evalplus", action="store_true",
                       help="Skip EvalPlus evaluation")

    # LiveCodeBench parameters
    parser.add_argument("--anno_path", type=str, default=None,
                       help="Path to anno JSONL file for LiveCodeBench evaluation. "
                            "If omitted, anno is auto-generated from --solution_path via LiveCodeBench evaluator.")
    parser.add_argument("--anno_scenario", type=str, default="codegeneration",
                       help="LiveCodeBench evaluator scenario (for auto anno generation)")
    parser.add_argument("--anno_release_version", type=str, default=None,
                       help="LiveCodeBench release version (for auto anno generation, e.g. release_v1)")

    # Resume support
    parser.add_argument("--resume_from", type=str, default=None,
                       choices=["inference", "unit_tests", "execution", "selection", "evalplus"],
                       help="Resume from a specific step (requires --resume_dir)")
    parser.add_argument("--resume_dir", type=str, default=None,
                       help="Path to existing run directory (e.g. output/2026-02-15_15-23-28). Defaults to --output_dir when --resume_from is set.")
    
    args = parser.parse_args()
    
    # Resume mode: load config from existing run
    if args.resume_from:
        resume_dir = os.path.abspath(args.resume_dir or args.output_dir)
        config_path = os.path.join(resume_dir, "config.json")
        if not os.path.exists(config_path):
            parser.error(f"Config not found at {config_path}. Cannot resume.")
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)
        config = EvalConfig(**saved_config)
        config.output_dir = resume_dir
        if config.log_file is None:
            config.log_file = os.path.join(config.output_dir, config.benchmark, "eval.log")
        logger = setup_logging(config)
        logger.info("=" * 60)
        logger.info("RESUME MODE")
        logger.info("=" * 60)
        logger.info(f"Resuming from step: {args.resume_from}")
        logger.info(f"Resume directory: {resume_dir}")
    else:
        # Create config
        config = EvalConfig(**{k: v for k, v in vars(args).items() if k in EvalConfig.__dataclass_fields__})
        
        # Create timestamped output directory under output_dir
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.output_dir = os.path.join(config.output_dir, timestamp)
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save config to file
        config_path = os.path.join(config.output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        
        # Set default log file
        if config.log_file is None:
            config.log_file = os.path.join(config.output_dir, config.benchmark, "eval.log")
        
        logger = setup_logging(config)
        
        # Log configuration
        logger.info("=" * 60)
        logger.info("EVALUATION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Config saved to: {config_path}")
        for key, value in asdict(config).items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 60)
    
    try:
        total_start_time = time.time()
        selected_path = None
        
        if args.resume_from == "evalplus":
            # Resume: run only EvalPlus
            selected_path = os.path.join(
                config.output_dir, config.benchmark, "selection",
                f"select_in_{config.num_solutions}_sol_by_{config.num_unit_tests}_ut_max+vote.jsonl"
            )
            if not os.path.exists(selected_path):
                raise FileNotFoundError(
                    f"Selection file not found: {selected_path}. "
                    "Ensure previous steps completed successfully."
                )
            run_evalplus(selected_path, config, logger)
        elif args.resume_from == "selection":
            # Resume: run selection and EvalPlus
            result_path = os.path.join(
                config.output_dir, config.benchmark, "execution",
                f"{config.num_solutions}_sol_{config.num_unit_tests}_ut_result.jsonl"
            )
            if not os.path.exists(result_path):
                raise FileNotFoundError(f"Execution results not found: {result_path}")
            selected_path = select_solutions(result_path, config.solution_path, config, logger)
            run_evalplus(selected_path, config, logger)
        elif args.resume_from == "execution":
            # Resume: run execution, selection, EvalPlus
            unit_test_path = os.path.join(
                config.output_dir, config.benchmark, "unit_tests",
                f"unit_tests_{config.num_unit_tests}.jsonl"
            )
            if not os.path.exists(unit_test_path):
                raise FileNotFoundError(f"Unit tests not found: {unit_test_path}")
            result_path = execute_unit_tests(config.solution_path, unit_test_path, config, logger)
            selected_path = select_solutions(result_path, config.solution_path, config, logger)
            run_evalplus(selected_path, config, logger)
        elif args.resume_from == "unit_tests":
            # Resume: run extraction, execution, selection, EvalPlus
            raw_path = os.path.join(
                config.output_dir, config.benchmark, "inference",
                config.raw_inference_filename if not config.use_previous_ut else config.sampled_inference_filename
            )
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Inference results not found: {raw_path}")
            inference_results = load_jsonl(raw_path)
            unit_test_path = extract_unit_tests(inference_results, config, logger)
            result_path = execute_unit_tests(config.solution_path, unit_test_path, config, logger)
            selected_path = select_solutions(result_path, config.solution_path, config, logger)
            run_evalplus(selected_path, config, logger)
        elif args.resume_from == "inference":
            # Resume: run full pipeline from inference
            inference_results, _ = run_inference(config, logger)
            unit_test_path = extract_unit_tests(inference_results, config, logger)
            result_path = execute_unit_tests(config.solution_path, unit_test_path, config, logger)
            selected_path = select_solutions(result_path, config.solution_path, config, logger)
            run_evalplus(selected_path, config, logger)
        else:
            # Full pipeline
            inference_results, raw_output_path = run_inference(config, logger)
            unit_test_path = extract_unit_tests(inference_results, config, logger)
            result_path = execute_unit_tests(config.solution_path, unit_test_path, config, logger)
            selected_path = select_solutions(result_path, config.solution_path, config, logger)
            run_evalplus(selected_path, config, logger)
        
        total_elapsed = time.time() - total_start_time
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_elapsed:.2f} seconds")
        if selected_path:
            logger.info(f"Selected solutions: {selected_path}")
        # Re-print final results at the very end for visibility
        if selected_path and not config.skip_evalplus:
            if config.benchmark == "livecodebench":
                # LiveCodeBench results already logged by run_livecodebench_eval()
                pass
            else:
                pass_at_k = _load_evalplus_results(selected_path)
                if pass_at_k:
                    bench = config.benchmark.upper().replace("+", "")
                    logger.info("")
                    logger.info(">>> FINAL EVALUATION RESULTS <<<")
                    for suite_name, metrics in pass_at_k.items():
                        name = f"{bench} (base)" if suite_name == "base" else f"{bench}+ (base+extra)"
                        logger.info(f"  {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items(), key=lambda x: int(x[0].split('@')[1]))))
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
