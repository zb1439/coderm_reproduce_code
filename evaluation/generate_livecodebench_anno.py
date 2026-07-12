import argparse
import ast
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            return [loaded]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def save_json(path: Path, payload, overwrite: bool = False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows, overwrite: bool = False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_code_fence(text: str) -> str:
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return text.strip()


def normalize_code_for_anno(raw_solution: str) -> str:
    code = strip_code_fence(raw_solution)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            class_src = ast.get_source_segment(code, node)
            return (class_src or code).strip()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_node = copy.deepcopy(node)
            if not method_node.args.args or method_node.args.args[0].arg != "self":
                method_node.args.args.insert(0, ast.arg(arg="self"))

            class_node = ast.ClassDef(
                name="Solution",
                bases=[],
                keywords=[],
                body=[method_node],
                decorator_list=[],
            )
            module = ast.Module(body=[class_node], type_ignores=[])
            ast.fix_missing_locations(module)
            return ast.unparse(module).strip()

    return code


def build_custom_output(func_rows, benchmark_problems=None):
    """Build custom_output for LCB evaluator.

    If benchmark_problems is provided, we pad the output so that every problem
    in the benchmark is present (with an empty code_list for missing ones).
    This is needed because the evaluator asserts len(custom) == len(benchmark).
    """
    func_map = {str(row["task_id"]): row for row in func_rows}
    custom_rows = []

    if benchmark_problems is not None:
        for problem in benchmark_problems:
            qid = str(problem.question_id)
            if qid in func_map:
                row = func_map[qid]
                custom_rows.append({
                    "question_id": qid,
                    "code_list": [strip_code_fence(x) for x in row["solutions"]],
                })
            else:
                custom_rows.append({
                    "question_id": qid,
                    "code_list": [],
                })
    else:
        for row in func_rows:
            custom_rows.append({
                "question_id": str(row["task_id"]),
                "code_list": [strip_code_fence(x) for x in row["solutions"]],
            })
    return custom_rows


def find_eval_all_path(log_text: str, started_at: float, preferred: Optional[str]):
    if preferred:
        preferred_path = Path(preferred)
        if preferred_path.exists():
            return preferred_path

    path_pattern = re.compile(r"([^\s\"']*eval_all[^\s\"']*\.json)")
    candidates = []
    for match in path_pattern.findall(log_text):
        candidate = Path(match)
        if candidate.exists():
            candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    disk_candidates = [
        path
        for path in Path(".").rglob("*eval_all*.json")
        if path.is_file() and path.stat().st_mtime >= started_at - 1
    ]
    if disk_candidates:
        return max(disk_candidates, key=lambda p: p.stat().st_mtime)

    return None


def run_custom_evaluator(
    custom_output_file: Path,
    scenario: str,
    release_version: Optional[str],
    evaluator_module: str,
    python_executable: str,
    preferred_eval_file: Optional[str],
):
    cmd = [
        python_executable,
        "-m",
        evaluator_module,
        "--custom_output_file",
        str(custom_output_file),
        "--scenario",
        scenario,
    ]
    if release_version:
        cmd.extend(["--release_version", release_version])

    # lcb_runner uses relative paths to load few-shot examples, so we must
    # run the evaluator from the LiveCodeBench repo root.
    lcb_repo_root = os.environ.get("LCB_REPO_ROOT", "/tmp/lcb_repo")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = lcb_repo_root + (":" + existing if existing else "")

    started_at = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=lcb_repo_root, env=env)
    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(
            "custom_evaluator failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {proc.returncode}\n"
            f"Logs:\n{logs}"
        )

    eval_all_path = find_eval_all_path(logs, started_at, preferred_eval_file)
    if not eval_all_path:
        raise RuntimeError(
            "custom_evaluator finished but no eval_all json was found. "
            "Pass --graded_path explicitly or check evaluator logs."
        )
    return eval_all_path, logs


def to_bool_result(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "pass", "passed", "success", "accepted"}
    if isinstance(value, dict):
        for key in ("is_pass", "passed", "result", "status"):
            if key in value:
                return to_bool_result(value[key])
    return bool(value)


def extract_graded_list(entry: dict):
    for key in (
        "graded_list",
        "pass_list",
        "passed_list",
        "is_pass_list",
        "result_list",
        "results",
    ):
        value = entry.get(key)
        if isinstance(value, list):
            return [to_bool_result(x) for x in value]
    return None


def iter_dict_nodes(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from iter_dict_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_dict_nodes(item)


def load_graded_map(graded_path: Path):
    roots = load_json_or_jsonl(graded_path)
    rows = []
    for root in roots:
        rows.extend(iter_dict_nodes(root))
    graded_map = {}
    for row in rows:
        question_id = row.get("question_id", row.get("task_id"))
        if question_id is None:
            continue
        graded_list = extract_graded_list(row)
        if graded_list is None:
            continue
        graded_map[str(question_id)] = graded_list
    if not graded_map:
        raise ValueError(
            f"No per-solution graded list found in {graded_path}. "
            "Expected keys like graded_list or pass_list."
        )
    return graded_map


def convert_to_anno(func_rows, graded_map, strict: bool = True):
    anno_rows = []
    warnings = []

    for row in func_rows:
        task_id = str(row["task_id"])
        solutions = row["solutions"]
        graded = graded_map.get(task_id)
        if graded is None:
            if strict:
                raise KeyError(f"Missing graded results for task_id={task_id}")
            warnings.append(f"Missing graded results for task_id={task_id}; marking all fail.")
            graded = [False] * len(solutions)

        if len(graded) < len(solutions):
            if strict:
                raise ValueError(
                    f"task_id={task_id}: graded_list length {len(graded)} "
                    f"< solution length {len(solutions)}"
                )
            warnings.append(
                f"task_id={task_id}: graded_list shorter than solutions; "
                "missing items are marked fail."
            )
            graded = graded + [False] * (len(solutions) - len(graded))
        elif len(graded) > len(solutions):
            warnings.append(
                f"task_id={task_id}: graded_list longer than solutions; extra items are ignored."
            )
            graded = graded[: len(solutions)]

        anno_solutions = []
        for sol_id, solution in enumerate(solutions):
            anno_solutions.append(
                {
                    "sol_id": sol_id,
                    "code": normalize_code_for_anno(solution),
                    "result": "pass" if graded[sol_id] else "fail",
                }
            )
        anno_rows.append({"task_id": task_id, "solutions": anno_solutions})

    return anno_rows, warnings


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate LiveCodeBench anno jsonl from func jsonl using per-solution grader output "
            "(typically produced by lcb_runner.runner.custom_evaluator)."
        )
    )
    parser.add_argument("--func_path", required=True, help="Path to sol_*_func.jsonl.")
    parser.add_argument("--output_path", required=True, help="Path to output sol_*_anno.jsonl.")
    parser.add_argument(
        "--graded_path",
        default=None,
        help=(
            "Path to evaluator output containing per-solution lists (e.g. eval_all.json). "
            "If omitted, use --run_evaluator."
        ),
    )
    parser.add_argument(
        "--run_evaluator",
        action="store_true",
        help="Run lcb_runner.runner.custom_evaluator to produce graded output automatically.",
    )
    parser.add_argument(
        "--custom_output_path",
        default=None,
        help="Path to write custom_output json used by evaluator (optional).",
    )
    parser.add_argument("--scenario", default="codegeneration", help="Evaluator scenario.")
    parser.add_argument(
        "--release_version",
        default=None,
        help="Release version passed to evaluator, e.g. release_v1/release_v6.",
    )
    parser.add_argument(
        "--evaluator_module",
        default="lcb_runner.runner.custom_evaluator",
        help="Python module path for custom evaluator.",
    )
    parser.add_argument(
        "--python_executable",
        default=sys.executable,
        help="Python executable used to run evaluator module.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Do not fail on missing/incomplete graded_list; fill missing entries with fail.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output/custom_output files.",
    )
    parser.add_argument(
        "--keep_custom_output",
        action="store_true",
        help="Keep temporary custom_output json when --run_evaluator is used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    func_path = Path(args.func_path)
    output_path = Path(args.output_path)

    func_rows = load_json_or_jsonl(func_path)
    if not func_rows:
        raise ValueError(f"No data found in {func_path}")

    graded_path = Path(args.graded_path) if args.graded_path else None
    temp_custom_output = None
    evaluator_logs = ""

    if args.run_evaluator:
        # Load benchmark problems so we can pad custom_output to match evaluator expectations
        benchmark_problems = None
        try:
            from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
            benchmark_problems = load_code_generation_dataset(
                release_version=args.release_version
            )
        except Exception as e:
            print(f"Warning: could not load benchmark dataset: {e}")
        custom_rows = build_custom_output(func_rows, benchmark_problems)
        if args.custom_output_path:
            custom_output_path = Path(args.custom_output_path)
            save_json(custom_output_path, custom_rows, overwrite=args.overwrite)
        else:
            fd, tmp_name = tempfile.mkstemp(prefix="lcb_custom_output_", suffix=".json")
            os.close(fd)
            Path(tmp_name).unlink(missing_ok=True)
            custom_output_path = Path(tmp_name)
            temp_custom_output = custom_output_path
            save_json(custom_output_path, custom_rows, overwrite=True)

        graded_path, evaluator_logs = run_custom_evaluator(
            custom_output_file=custom_output_path,
            scenario=args.scenario,
            release_version=args.release_version,
            evaluator_module=args.evaluator_module,
            python_executable=args.python_executable,
            preferred_eval_file=str(graded_path) if graded_path else None,
        )

    if graded_path is None:
        raise ValueError("graded output is required. Set --graded_path or use --run_evaluator.")

    graded_map = load_graded_map(graded_path)
    anno_rows, warnings = convert_to_anno(
        func_rows,
        graded_map,
        strict=not args.allow_missing,
    )
    save_jsonl(output_path, anno_rows, overwrite=args.overwrite)

    if temp_custom_output and not args.keep_custom_output:
        temp_custom_output.unlink(missing_ok=True)

    print(f"Loaded func rows: {len(func_rows)}")
    print(f"Loaded graded rows: {len(graded_map)} from {graded_path}")
    print(f"Wrote anno rows: {len(anno_rows)} to {output_path}")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings[:20]:
            print(f"- {warning}")
        if len(warnings) > 20:
            print(f"- ... {len(warnings) - 20} more warnings")
    if evaluator_logs.strip():
        print("Evaluator logs captured (truncated):")
        print(evaluator_logs[-2000:])


if __name__ == "__main__":
    main()
