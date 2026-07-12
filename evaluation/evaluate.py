import os
import sys
import json
import time
import signal
import ctypes
import contextlib
import unittest
import multiprocessing
from io import StringIO
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Value, Array
from tqdm import tqdm


class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout = old_stdout


def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(filename, dataset):
    if os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists.")
    with open(filename, "w", encoding="UTF-8") as fp:
        for data in tqdm(dataset):
            fp.write(json.dumps(data, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def time_limit(seconds: float):
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


def read_jsonline_in_chunks(file_path, chunk_size):
    with open(file_path, "r", encoding="utf-8") as fp:
        chunk = []
        for index, line in enumerate(fp):
            chunk.append(json.loads(line))
            if (index + 1) % chunk_size == 0:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


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
    output = StringIO()
    locals_dict = {}
    try:
        with time_limit(time_limits), suppress_stdout():
            exec(  # noqa: S102  # pylint: disable=exec-used
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
    except Exception as error:  # noqa: BLE001  # pylint: disable=broad-except
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


def run_unit_tests(
    input_path: str,
    output_path: str,
    mp_num: int,
    chunk_size: int = 1000,
    recover: int = 0,
    details: bool = False,
    time_limit_seconds: float = 1,
):
    if os.path.exists(output_path):
        os.remove(output_path)

    processed_records = 0
    for chunk in read_jsonline_in_chunks(input_path, chunk_size=chunk_size):
        processed_records += len(chunk)
        print(f"Processed {processed_records} records")
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


def save_sol_and_ut_comb(benchmark, sol_model, ut_model, sol_num, ut_num, sol_path=None, ut_path=None):
    print("========== START PREPROCESSING ==========")
    output = []

    if benchmark != "livecodebench":
        sol_dataset = sol_path or f"data/result/{benchmark}/sol_{sol_model}_200_anno.jsonl"
        sol_dataset = load_jsonl(sol_dataset)
    else:
        sol_dataset = sol_path or f"data/result/{benchmark}/sol_{sol_model}_100_func.jsonl"
        sol_dataset = load_jsonl(sol_dataset)

    ut_dataset = ut_path or f"data/result/{benchmark}/ut_{ut_model}_100.jsonl"
    ut_dataset = load_jsonl(ut_dataset)

    for i in tqdm(range(len(sol_dataset))):
        actual_sol_num = min(sol_num, len(sol_dataset[i]["solutions"]))
        for sol_id in range(actual_sol_num):
            for ut_id in range(len(ut_dataset[i]["unit_tests"])):
                if ut_id == ut_num:
                    break
                code = (
                    sol_dataset[i]["solutions"][sol_id]
                    + "\n\n"
                    + ut_dataset[i]["unit_tests"][ut_id]
                )
                output.append(
                    {
                        "task_id": sol_dataset[i]["task_id"],
                        "sol_id": sol_id,
                        "ut_id": ut_id,
                        "code": code,
                    }
                )

    save_jsonl(
        f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/sol_ut.jsonl", output
    )


def exec_ut(
    benchmark,
    sol_model,
    ut_model,
    sol_num,
    ut_num,
    mp_num,
    chunk_size,
    recover,
    details,
    time_limit_seconds,
):
    print("========== START EXECUTE UNIT TEST ==========")
    input_path = f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/sol_ut.jsonl"
    output_path = (
        f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/"
        f"{sol_num}_sol_{ut_num}_ut_result.jsonl"
    )
    run_unit_tests(
        input_path=input_path,
        output_path=output_path,
        mp_num=mp_num,
        chunk_size=chunk_size,
        recover=recover,
        details=details,
        time_limit_seconds=time_limit_seconds,
    )
    if os.path.exists(input_path):
        os.remove(input_path)


def select_sol(benchmark, sol_model, ut_model, sol_num, ut_num, sol_path=None):
    print("========== START SELECT SOLUTION ==========")
    dataset = load_jsonl(
        f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/{sol_num}_sol_{ut_num}_ut_result.jsonl"
    )
    dataset = sorted(
        dataset,
        key=lambda x: (int(x["task_id"].split("/")[1]), x["sol_id"], x["ut_id"]),
    )

    if benchmark == "humaneval":
        current_task = "HumanEval/0"
    elif benchmark == "mbpp":
        current_task = "Mbpp/2"
    solution_dict = {i: 0 for i in range(sol_num)}
    chosen_solution = []
    for data in tqdm(dataset):
        if data["task_id"] == current_task:
            if data["result"] == "pass":
                solution_dict[data["sol_id"]] += 1
        else:
            # calculate
            sorted_solution_dict = sorted(
                solution_dict.items(), key=lambda item: item[1], reverse=True
            )
            chosen_solution.append(
                {"task_id": current_task, "chosen_solution": sorted_solution_dict[0][0]}
            )

            # initialize
            current_task = data["task_id"]
            solution_dict = {i: 0 for i in range(sol_num)}

    # the last task
    sorted_solution_dict = sorted(
        solution_dict.items(), key=lambda item: item[1], reverse=True
    )
    chosen_solution.append(
        {"task_id": current_task, "chosen_solution": sorted_solution_dict[0][0]}
    )

    sol_dataset_path = sol_path or f"data/{benchmark}/sol_{sol_model}_200.jsonl"
    sol_dataset = load_jsonl(sol_dataset_path)
    output = []
    for i in range(len(chosen_solution)):
        output.append(
            {
                "task_id": chosen_solution[i]["task_id"],
                "solution": sol_dataset[i]["solutions"][
                    chosen_solution[i]["chosen_solution"]
                ],
            }
        )

    save_name = f"select_in_{sol_num}_sol_by_{ut_num}_ut.jsonl"
    save_jsonl(
        f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/{save_name}", output
    )
    return save_name

def select_sol_multi(benchmark, sol_model, ut_model, sol_num, ut_num, sol_path=None):
    print("========== START SELECT SOLUTION ==========")
    dataset = load_jsonl(
        f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/{sol_num}_sol_{ut_num}_ut_result.jsonl"
    )
    dataset = sorted(
        dataset,
        key=lambda x: (int(x["task_id"].split("/")[1]), x["sol_id"], x["ut_id"]),
    )

    current_task = None
    if benchmark == "humaneval":
        current_task = "HumanEval/0"
    elif benchmark == "mbpp":
        current_task = "Mbpp/2"
    solution_dict = {i: set() for i in range(sol_num)}
    chosen_solution = []
    for dataset_idx in range(len(dataset) + 1):
        data = dataset[dataset_idx] if dataset_idx < len(dataset) else None
        if data and data["task_id"] == current_task:
            if data["result"] == "pass":
                solution_dict[data["sol_id"]].add(data["ut_id"])
        else:
            # calculate
            sorted_solution_dict = sorted(
                solution_dict.items(), key=lambda item: len(item[1]), reverse=True
            )
            highest_pass = len(sorted_solution_dict[0][1])
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
                    "task_id": current_task, "chosen_solution": sol_id
                })

            # initialize
            if dataset_idx < len(dataset):
                current_task = dataset[dataset_idx]["task_id"]
                solution_dict = {i: set() for i in range(sol_num)}

    sol_dataset_path = sol_path or f"data/{benchmark}/sol_{sol_model}_200.jsonl"
    sol_dataset = load_jsonl(sol_dataset_path)
    task_id_to_sol_entry = {entry["task_id"]: entry for entry in sol_dataset}
    output = []
    for chosen_sol_entry in chosen_solution:
        task_id = chosen_sol_entry["task_id"]
        solutions = task_id_to_sol_entry[task_id]["solutions"]
        output.append({
            "task_id": task_id,
            "solution": solutions[chosen_sol_entry["chosen_solution"]],
        })

    save_name = f"select_in_{sol_num}_sol_by_{ut_num}_ut_max+vote.jsonl"
    save_jsonl(
        f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/{save_name}", output
    )
    return save_name


def main(options):
    benchmark = options.benchmark
    sol_model = options.sol_model
    ut_model = options.ut_model
    sol_num = options.sol_num
    ut_num = options.ut_num
    mp_num = options.mp_num
    sol_path = options.sol_path
    ut_path = options.ut_path
    chunk_size = options.chunk_size
    recover = options.recover
    details = options.details
    time_limit_seconds = options.time_limit

    output_dir = f"output/{benchmark}/{sol_model}_sol_{ut_model}_ut"
    details_dir = os.path.join(output_dir, "details")
    os.makedirs(details_dir, exist_ok=True)

    st = time.time()
    # get solution_unit_test data
    if not recover:
        save_sol_and_ut_comb(benchmark, sol_model, ut_model, sol_num, ut_num, sol_path, ut_path)

    # execute unit test
    exec_ut(
        benchmark,
        sol_model,
        ut_model,
        sol_num,
        ut_num,
        mp_num,
        chunk_size,
        recover,
        details,
        time_limit_seconds,
    )

    print(time.time() - st)


if __name__ == "__main__":
    import argparse

    # parse parameter
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--benchmark", type=str, help="evaluate benchmark")
    parser.add_argument(
        "--sol_model", type=str, help="the model that generate solutions"
    )
    parser.add_argument(
        "--ut_model", type=str, help="the model that generate unit test"
    )
    parser.add_argument("--sol_num", type=int, help="the number of generated solutions")
    parser.add_argument("--ut_num", type=int, help="the number of generated unit test")
    parser.add_argument(
        "--mp_num", type=int, help="the number of process used for code execution"
    )
    parser.add_argument("--sol_path", type=str, default=None, help="the path of solutions")
    parser.add_argument("--ut_path", type=str, default=None, help="the path of unit tests")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="number of solution-test pairs processed per batch",
    )
    parser.add_argument(
        "--recover",
        type=int,
        default=0,
        help="number of historical records to skip when resuming execution",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="persist detailed unittest output for each execution",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=1.0,
        help="per-test execution time limit in seconds",
    )
    main(parser.parse_args())
