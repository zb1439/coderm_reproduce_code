#!/usr/bin/env python3
"""Generate LiveCodeBench solution candidates via OpenAI-compatible providers.

This script is provider-agnostic and can run multiple model/provider pairs in one command.
It writes intermediate raw responses and optionally extracts `sol_*_func.jsonl` files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import http.client
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PRIMARY_MESSAGES = Path("/Users/xinyuan/code/coderm/data/benchmark/input_livecodebench_sol.jsonl")
DEFAULT_FALLBACK_MESSAGES = Path("data/benchmark/input_livecodebench_sol.jsonl")


class ApiRequestError(RuntimeError):
    """HTTP/API level failure for provider request."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class ModelUnavailableError(RuntimeError):
    """Model is not available on this provider endpoint."""


class FatalModelError(RuntimeError):
    """Non-retryable or exhausted retry failure for one model run."""


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    model: str
    provider: str
    base_url: str
    api_key_env: str
    runtime: str = "api"  # "api" or "local_transformers"
    enabled: bool = True
    skip_reason: Optional[str] = None


@dataclass
class ModelRunResult:
    tag: str
    model: str
    provider: str
    status: str
    reason: str
    candidates_per_task: int
    tasks_requested: int
    tasks_completed: int
    raw_output_path: Optional[str] = None
    func_output_path: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class LocalModelState:
    tokenizer: Any
    model: Any
    device: str


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_progress(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    data = dict(payload)
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(path, data)


def resolve_messages_file(primary: Path, fallback: Path, repo_root: Path) -> Path:
    primary_abs = primary if primary.is_absolute() else (repo_root / primary).resolve()
    if primary_abs.exists():
        return primary_abs

    fallback_abs = fallback if fallback.is_absolute() else (repo_root / fallback).resolve()
    if fallback_abs.exists():
        return fallback_abs

    raise FileNotFoundError(
        f"messages file not found at either {primary_abs} or {fallback_abs}"
    )


def render_prompt(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            content_str = "\n".join(text_parts)
        else:
            content_str = str(content)
        parts.append(f"[{role}] {content_str}")
    return "\n\n".join(parts)


def _has_code_fence(text: str) -> bool:
    return bool(re.search(r"```(?:python)?\s*\n.*?```", text, flags=re.DOTALL | re.IGNORECASE))


def _extract_first_code_block(text: str) -> Optional[str]:
    s = text.strip()
    m = re.search(r"```python\s*\n(.*?)\n```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)\n```", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def normalize_response_for_extraction(text: str) -> str:
    extracted = _extract_first_code_block(text)
    if extracted is not None:
        content = extracted
    else:
        content = text.strip()
    # Avoid nested markdown fences breaking downstream regex-based extraction.
    content = content.replace("```", "'''").strip()
    return f"```python\n{content}\n```"


def normalize_choice_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif "text" in item:
                    chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def is_qwen_model(spec: ModelSpec) -> bool:
    model = spec.model.lower()
    return "qwen" in model


def with_no_think_instruction(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return messages
    copied: List[Dict[str, Any]] = [dict(m) for m in messages]

    for i, msg in enumerate(copied):
        role = str(msg.get("role", ""))
        if role != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if "/no_think" not in content:
                copied[i]["content"] = f"/no_think\n{content}"
            copied[i]["content"] += "\n\nDo NOT include any comments or explanations in your code. Output only clean, executable code."
            return copied
        if isinstance(content, list):
            updated_list: List[Any] = []
            injected = False
            for item in content:
                if (
                    not injected
                    and isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    text = item["text"]
                    if "/no_think" not in text:
                        item = dict(item)
                        item["text"] = f"/no_think\n{text}"
                    injected = True
                updated_list.append(item)
            copied[i]["content"] = updated_list
            return copied
    return copied


def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def resolve_api_key_from_env(env_names: str) -> tuple[str, List[str]]:
    names = [name.strip() for name in env_names.split(",") if name.strip()]
    if not names:
        return "", []
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value, names
    return "", names


def post_chat_completions(
    endpoint: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ApiRequestError(e.code, body) from e


def request_with_retry(
    spec: ModelSpec,
    api_key: str,
    messages: List[Dict[str, Any]],
    disable_thinking: bool,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> List[str]:
    endpoint = spec.base_url.rstrip("/") + "/chat/completions"
    headers = build_headers(api_key)

    request_messages = messages
    if disable_thinking and is_qwen_model(spec):
        request_messages = with_no_think_instruction(messages)

    # OpenRouter does not support n>1; force n=1 and let the outer loop retry.
    effective_n = 1 if spec.provider == "openrouter" else n

    payload: Dict[str, Any] = {
        "model": spec.model,
        "messages": request_messages,
        "n": effective_n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    if disable_thinking and is_qwen_model(spec):
        payload["enable_thinking"] = False
        payload["reasoning"] = {"enabled": False}

    if disable_thinking and spec.provider == "dashscope":
        payload["extra_body"] = {"enable_thinking": False}
    elif disable_thinking and spec.provider == "deepinfra" and spec.model.startswith("Qwen/"):
        # Qwen3.5 defaults to thinking mode on some providers; disable it explicitly
        # to reduce long rationale output and improve code-only response rate.
        payload["extra_body"] = {"enable_thinking": False}

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = post_chat_completions(endpoint, headers, payload, timeout_seconds)
            choices = result.get("choices", [])
            texts: List[str] = []
            for choice in choices:
                msg = choice.get("message", {}) if isinstance(choice, dict) else {}
                content = normalize_choice_content(msg.get("content", ""))
                if content.strip():
                    texts.append(content)
            if not texts:
                raise FatalModelError("empty_choices")
            return texts
        except ApiRequestError as e:
            last_error = e
            if e.status_code in (401, 403):
                raise FatalModelError("auth_failed") from e
            if e.status_code == 404:
                raise ModelUnavailableError("provider_model_unavailable") from e
            if e.status_code in (400, 422):
                raise FatalModelError("bad_request") from e
            if attempt >= max_retries:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.IncompleteRead, ConnectionError, OSError) as e:
            last_error = e
            if attempt >= max_retries:
                break

        time.sleep(retry_backoff_seconds * (2**attempt))

    detail = str(last_error)[:200]
    if isinstance(last_error, ApiRequestError):
        detail = f"HTTP {last_error.status_code}: {last_error.body[:200]}"
    print(f"[request_with_retry] giving up after {max_retries + 1} attempts: {detail}", flush=True)
    raise FatalModelError(f"request_failed: {type(last_error).__name__}") from last_error


def choose_local_device(local_device_arg: str) -> str:
    if local_device_arg != "auto":
        return local_device_arg

    try:
        import torch  # type: ignore

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def probe_local_runtime(timeout_seconds: int = 20) -> Optional[str]:
    try:
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(create=True, size=16)
        shm.close()
        shm.unlink()
    except Exception as e:
        return f"local_runtime_probe_failed:shared_memory_unavailable:{type(e).__name__}"

    cmd = [
        sys.executable,
        "-c",
        "import torch, transformers; print('local_runtime_ok')",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "local_runtime_probe_failed:timeout"
    if proc.returncode == 0:
        return None
    reason = proc.stderr.strip() or proc.stdout.strip() or f"exit_code_{proc.returncode}"
    reason = reason.replace("\n", " | ")
    return f"local_runtime_probe_failed:{reason[:160]}"


def load_local_model_state(
    spec: ModelSpec,
    local_device_arg: str,
    local_cache_dir: Optional[Path],
) -> LocalModelState:
    # Keep local runtime resilient on constrained environments (e.g. sandboxed shared memory).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise FatalModelError("missing_local_dependency:torch_or_transformers") from e

    device = choose_local_device(local_device_arg)

    tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if local_cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = str(local_cache_dir)
        model_kwargs["cache_dir"] = str(local_cache_dir)

    if device == "mps":
        model_kwargs["torch_dtype"] = torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.model, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(spec.model, **model_kwargs)
        model.to(device)
        model.eval()
    except Exception as e:
        detail = str(e).replace("\n", " ")
        raise FatalModelError(f"local_model_load_failed:{type(e).__name__}:{detail[:180]}") from e

    return LocalModelState(tokenizer=tokenizer, model=model, device=device)


def render_local_prompt(tokenizer: Any, messages: List[Dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def generate_local_with_transformers(
    state: LocalModelState,
    messages: List[Dict[str, Any]],
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise FatalModelError("missing_local_dependency:torch") from e

    prompt = render_local_prompt(state.tokenizer, messages)
    encoded = state.tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(state.device) for k, v in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[-1])

    do_sample = temperature > 0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "num_return_sequences": n,
        "do_sample": do_sample,
        "pad_token_id": state.tokenizer.eos_token_id,
        "eos_token_id": state.tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    try:
        with torch.inference_mode():
            output_ids = state.model.generate(**encoded, **gen_kwargs)
        generated_ids = output_ids[:, prompt_len:]
        texts = state.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        detail = str(e).replace("\n", " ")
        raise FatalModelError(f"local_generation_failed:{type(e).__name__}:{detail[:180]}") from e

    cleaned: List[str] = []
    for text in texts:
        if text.strip():
            cleaned.append(normalize_response_for_extraction(text))
    if not cleaned:
        raise FatalModelError("local_empty_choices")
    return cleaned


def write_checkpoint(
    output_path: Path,
    tasks: List[Dict[str, Any]],
    row_map: Dict[str, Dict[str, Any]],
) -> None:
    ordered_rows: List[Dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in row_map:
            ordered_rows.append(row_map[task_id])
    save_jsonl(output_path, ordered_rows)


def run_extract_solution(
    repo_root: Path,
    data_path: Path,
    id_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "preprocess/extract_solution.py",
        "--data_path",
        str(data_path),
        "--id_path",
        str(id_path),
        "--output_path",
        str(output_path),
    ]
    subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)


def validate_raw_rows(rows: List[Dict[str, Any]], expected_tasks: int, expected_candidates: int) -> None:
    if len(rows) != expected_tasks:
        raise ValueError(f"raw rows mismatch: expected {expected_tasks}, got {len(rows)}")

    for row in rows:
        if "task_id" not in row or "messages" not in row or "responses" not in row:
            raise ValueError("raw row missing required keys")
        if not isinstance(row["responses"], list):
            raise ValueError("raw row responses must be a list")
        if len(row["responses"]) != expected_candidates:
            print(
                f"WARNING: task {row.get('task_id')} has {len(row['responses'])} responses, expected {expected_candidates}",
                flush=True,
            )


def validate_func_rows(path: Path, expected_tasks: int, expected_candidates: int) -> None:
    rows = load_jsonl(path)
    if len(rows) != expected_tasks:
        raise ValueError(f"func rows mismatch: expected {expected_tasks}, got {len(rows)}")

    short_tasks = []
    for row in rows:
        sols = row.get("solutions")
        if not isinstance(sols, list):
            raise ValueError("func row missing solutions list")
        if len(sols) < expected_candidates:
            short_tasks.append((row.get("task_id"), len(sols)))

    if short_tasks:
        print(f"WARNING: {len(short_tasks)} tasks have fewer solutions than expected ({expected_candidates}):")
        for tid, n in short_tasks[:10]:
            print(f"  task {tid}: {n} solutions")
        if len(short_tasks) > 10:
            print(f"  ... and {len(short_tasks) - 10} more")


def sample_prompt_consistency(
    tasks: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    sample_size: int,
) -> None:
    sample_size = max(1, min(sample_size, len(tasks)))
    indexes = [int(i * len(tasks) / sample_size) for i in range(sample_size)]
    for idx in indexes:
        source_messages = tasks[idx]["messages"]
        row_messages = rows[idx]["messages"]
        if source_messages != row_messages:
            raise ValueError(f"message mismatch at task index {idx}")


def build_model_specs() -> List[ModelSpec]:
    has_deepinfra = bool(
        os.getenv("DEEPINFRA_TOKEN", "").strip()
        or os.getenv("DEEPINFRA_API_KEY", "").strip()
    )

    if has_deepinfra:
        qwen35_4b_spec = ModelSpec(
            tag="qwen3.5-4b",
            model="Qwen/Qwen3.5-4B",
            provider="deepinfra",
            base_url="https://api.deepinfra.com/v1/openai",
            api_key_env="DEEPINFRA_TOKEN,DEEPINFRA_API_KEY",
        )
        qwen35_08_spec = ModelSpec(
            tag="qwen3.5-0.8b",
            model="Qwen/Qwen3.5-0.8B",
            provider="deepinfra",
            base_url="https://api.deepinfra.com/v1/openai",
            api_key_env="DEEPINFRA_TOKEN,DEEPINFRA_API_KEY",
        )
    else:
        qwen35_4b_spec = ModelSpec(
            tag="qwen3.5-4b",
            model="Qwen3.5-4B",
            provider="dashscope",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="DASHSCOPE_API_KEY",
        )
        qwen35_08_spec = ModelSpec(
            tag="qwen3.5-0.8b",
            model="Qwen/Qwen3.5-0.8B",
            provider="local_transformers",
            base_url="",
            api_key_env="",
            runtime="local_transformers",
            enabled=True,
        )

    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY", "").strip())

    return [
        qwen35_4b_spec,
        ModelSpec(
            tag="qwen3-4b-instruct-2507",
            model="Qwen/Qwen3-4B-Instruct-2507",
            provider="nscale",
            base_url="https://inference.api.nscale.com/v1",
            api_key_env="NSCALE_SERVICE_TOKEN",
        ),
        qwen35_08_spec,
        # --- OpenRouter models ---
        ModelSpec(
            tag="gemma-3-4b",
            model="google/gemma-3-4b-it",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            enabled=has_openrouter,
        ),
        ModelSpec(
            tag="ministral-3b",
            model="mistralai/ministral-3b-2512",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            enabled=has_openrouter,
        ),
    ]


def run_model(
    spec: ModelSpec,
    tasks: List[Dict[str, Any]],
    target_candidates: int,
    args: argparse.Namespace,
    repo_root: Path,
    messages_file: Path,
) -> ModelRunResult:
    start_time = time.time()
    local_state: Optional[LocalModelState] = None

    suffix = "_smoke" if args.smoke else ""
    raw_output_path = (repo_root / args.raw_output_dir / f"sol_{spec.tag}_provider_raw{suffix}.jsonl").resolve()
    func_output_path = (
        repo_root / args.func_output_dir / f"sol_{spec.tag}_{target_candidates}_func{suffix}.jsonl"
    ).resolve()
    progress_path: Optional[Path] = None
    if args.progress_dir is not None:
        progress_root = args.progress_dir
        if not progress_root.is_absolute():
            progress_root = (repo_root / progress_root).resolve()
        progress_root.mkdir(parents=True, exist_ok=True)
        progress_path = (progress_root / f"progress_{spec.tag}{suffix}.json").resolve()

    def emit_progress(
        *,
        status: str,
        reason: str,
        tasks_completed_now: int,
        current_task_index: int = 0,
        current_task_id: str = "",
        current_task_responses: int = 0,
    ) -> None:
        save_progress(
            progress_path,
            {
                "model_tag": spec.tag,
                "model": spec.model,
                "provider": spec.provider,
                "status": status,
                "reason": reason,
                "tasks_total": len(tasks),
                "tasks_completed": tasks_completed_now,
                "current_task_index": current_task_index,
                "current_task_id": current_task_id,
                "current_task_responses": current_task_responses,
                "current_task_target": target_candidates,
                "raw_output_path": str(raw_output_path),
                "func_output_path": str(func_output_path) if not args.skip_extract else None,
            },
        )

    if not spec.enabled:
        emit_progress(
            status="skipped",
            reason=spec.skip_reason or "disabled",
            tasks_completed_now=0,
        )
        return ModelRunResult(
            tag=spec.tag,
            model=spec.model,
            provider=spec.provider,
            status="skipped",
            reason=spec.skip_reason or "disabled",
            candidates_per_task=target_candidates,
            tasks_requested=len(tasks),
            tasks_completed=0,
            duration_seconds=time.time() - start_time,
        )

    api_key = ""
    if spec.runtime == "api":
        api_key, api_env_names = resolve_api_key_from_env(spec.api_key_env)
        if not api_key:
            missing_env_hint = "|".join(api_env_names) if api_env_names else spec.api_key_env
            emit_progress(
                status="skipped",
                reason=f"missing_api_key:{missing_env_hint}",
                tasks_completed_now=0,
            )
            return ModelRunResult(
                tag=spec.tag,
                model=spec.model,
                provider=spec.provider,
                status="skipped",
                reason=f"missing_api_key:{missing_env_hint}",
                candidates_per_task=target_candidates,
                tasks_requested=len(tasks),
                tasks_completed=0,
                duration_seconds=time.time() - start_time,
            )
    elif spec.runtime == "local_transformers":
        probe_reason = probe_local_runtime()
        if probe_reason is not None:
            emit_progress(
                status="skipped",
                reason=probe_reason,
                tasks_completed_now=0,
            )
            return ModelRunResult(
                tag=spec.tag,
                model=spec.model,
                provider=spec.provider,
                status="skipped",
                reason=probe_reason,
                candidates_per_task=target_candidates,
                tasks_requested=len(tasks),
                tasks_completed=0,
                duration_seconds=time.time() - start_time,
            )
        try:
            local_state = load_local_model_state(
                spec=spec,
                local_device_arg=args.local_device,
                local_cache_dir=args.local_cache_dir,
            )
            print(
                f"[{spec.tag}] local backend ready: device={local_state.device}",
                flush=True,
            )
        except FatalModelError as e:
            emit_progress(
                status="skipped",
                reason=str(e),
                tasks_completed_now=0,
            )
            return ModelRunResult(
                tag=spec.tag,
                model=spec.model,
                provider=spec.provider,
                status="skipped",
                reason=str(e),
                candidates_per_task=target_candidates,
                tasks_requested=len(tasks),
                tasks_completed=0,
                duration_seconds=time.time() - start_time,
            )
    else:
        emit_progress(
            status="skipped",
            reason=f"unsupported_runtime:{spec.runtime}",
            tasks_completed_now=0,
        )
        return ModelRunResult(
            tag=spec.tag,
            model=spec.model,
            provider=spec.provider,
            status="skipped",
            reason=f"unsupported_runtime:{spec.runtime}",
            candidates_per_task=target_candidates,
            tasks_requested=len(tasks),
            tasks_completed=0,
            duration_seconds=time.time() - start_time,
        )

    existing_rows: Dict[str, Dict[str, Any]] = {}
    if args.resume and raw_output_path.exists():
        for row in load_jsonl(raw_output_path):
            task_id = str(row.get("task_id"))
            if task_id:
                existing_rows[task_id] = row

    completed_tasks = 0
    emit_progress(
        status="running",
        reason="in_progress",
        tasks_completed_now=completed_tasks,
    )
    try:
        for idx, task in enumerate(tasks):
            task_id = str(task["task_id"])
            messages = task["messages"]

            current = existing_rows.get(task_id, {})
            responses = current.get("responses", []) if isinstance(current, dict) else []
            if not isinstance(responses, list):
                responses = []

            if current.get("messages") != messages:
                responses = []

            responses = [str(r) for r in responses if str(r).strip()]
            responses = responses[:target_candidates]

            task_failed = False
            while len(responses) < target_candidates:
                need = target_candidates - len(responses)
                batch_n = min(args.batch_size, need)
                try:
                    if spec.runtime == "api":
                        generated = request_with_retry(
                            spec=spec,
                            api_key=api_key,
                            messages=messages,
                            disable_thinking=args.disable_thinking,
                            n=batch_n,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            timeout_seconds=args.timeout_seconds,
                            max_retries=args.max_retries,
                            retry_backoff_seconds=args.retry_backoff_seconds,
                        )
                    else:
                        if local_state is None:
                            raise FatalModelError("local_state_uninitialized")
                        generated = generate_local_with_transformers(
                            state=local_state,
                            messages=messages,
                            n=batch_n,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                except FatalModelError as e:
                    print(
                        f"[{spec.tag}] task {idx + 1}/{len(tasks)} ({task_id}) failed "
                        f"with {len(responses)}/{target_candidates} responses: {e}",
                        flush=True,
                    )
                    task_failed = True
                    break
                responses.extend(generated)
                responses = responses[:target_candidates]
                emit_progress(
                    status="running",
                    reason="in_progress",
                    tasks_completed_now=completed_tasks,
                    current_task_index=idx + 1,
                    current_task_id=task_id,
                    current_task_responses=len(responses),
                )

                if (
                    args.progress_every > 0
                    and (
                        len(responses) == target_candidates
                        or len(responses) % args.progress_every == 0
                    )
                ):
                    print(
                        f"[{spec.tag}] task {idx + 1}/{len(tasks)} "
                        f"responses {len(responses)}/{target_candidates}",
                        flush=True,
                    )

                if args.sleep_between_requests > 0:
                    time.sleep(args.sleep_between_requests)

            if task_failed and not responses:
                print(f"[{spec.tag}] skipping task {task_id} (no responses)", flush=True)

            existing_rows[task_id] = {
                "task_id": task_id,
                "messages": messages,
                "prompt": render_prompt(messages),
                "responses": responses,
            }
            completed_tasks = idx + 1

            if args.checkpoint_interval > 0 and completed_tasks % args.checkpoint_interval == 0:
                write_checkpoint(raw_output_path, tasks, existing_rows)
                emit_progress(
                    status="running",
                    reason="in_progress",
                    tasks_completed_now=completed_tasks,
                    current_task_index=idx + 1,
                    current_task_id=task_id,
                    current_task_responses=target_candidates,
                )
                print(
                    f"[{spec.tag}] checkpoint: {completed_tasks}/{len(tasks)} tasks",
                    flush=True,
                )

    except ModelUnavailableError:
        emit_progress(
            status="skipped",
            reason="provider_model_unavailable",
            tasks_completed_now=completed_tasks,
        )
        return ModelRunResult(
            tag=spec.tag,
            model=spec.model,
            provider=spec.provider,
            status="skipped",
            reason="provider_model_unavailable",
            candidates_per_task=target_candidates,
            tasks_requested=len(tasks),
            tasks_completed=completed_tasks,
            raw_output_path=str(raw_output_path),
            duration_seconds=time.time() - start_time,
        )
    except FatalModelError as e:
        emit_progress(
            status="failed",
            reason=str(e),
            tasks_completed_now=completed_tasks,
        )
        return ModelRunResult(
            tag=spec.tag,
            model=spec.model,
            provider=spec.provider,
            status="failed",
            reason=str(e),
            candidates_per_task=target_candidates,
            tasks_requested=len(tasks),
            tasks_completed=completed_tasks,
            raw_output_path=str(raw_output_path),
            duration_seconds=time.time() - start_time,
        )

    final_rows = [existing_rows[str(task["task_id"])] for task in tasks]
    save_jsonl(raw_output_path, final_rows)

    validate_raw_rows(final_rows, expected_tasks=len(tasks), expected_candidates=target_candidates)
    sample_prompt_consistency(tasks, final_rows, sample_size=args.consistency_sample_size)

    if not args.skip_extract:
        run_extract_solution(
            repo_root=repo_root,
            data_path=raw_output_path,
            id_path=messages_file,
            output_path=func_output_path,
        )
        validate_func_rows(func_output_path, expected_tasks=len(tasks), expected_candidates=target_candidates)

    emit_progress(
        status="completed",
        reason="ok",
        tasks_completed_now=len(tasks),
        current_task_index=len(tasks),
        current_task_id=str(tasks[-1]["task_id"]) if tasks else "",
        current_task_responses=target_candidates,
    )
    return ModelRunResult(
        tag=spec.tag,
        model=spec.model,
        provider=spec.provider,
        status="completed",
        reason="ok",
        candidates_per_task=target_candidates,
        tasks_requested=len(tasks),
        tasks_completed=len(tasks),
        raw_output_path=str(raw_output_path),
        func_output_path=str(func_output_path) if not args.skip_extract else None,
        duration_seconds=time.time() - start_time,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LiveCodeBench solution candidates from OpenAI-compatible providers."
    )
    parser.add_argument("--messages_file", type=Path, default=DEFAULT_PRIMARY_MESSAGES)
    parser.add_argument("--fallback_messages_file", type=Path, default=DEFAULT_FALLBACK_MESSAGES)

    parser.add_argument("--raw_output_dir", type=Path, default=Path("output/livecodebench"))
    parser.add_argument("--func_output_dir", type=Path, default=Path("data/result/livecodebench"))
    parser.add_argument("--run_report_path", type=Path, default=Path("output/livecodebench/run_report.json"))

    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument(
        "--disable_thinking",
        dest="disable_thinking",
        action="store_true",
        default=True,
        help="Try to disable Qwen thinking/reasoning mode (default: enabled).",
    )
    parser.add_argument(
        "--allow_thinking",
        dest="disable_thinking",
        action="store_false",
        help="Allow model thinking/reasoning mode.",
    )

    parser.add_argument("--timeout_seconds", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_backoff_seconds", type=float, default=1.0)
    parser.add_argument("--sleep_between_requests", type=float, default=0.0)
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print in-task progress every N collected responses (0 disables).",
    )
    parser.add_argument(
        "--local_device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Device for local_transformers runtime.",
    )
    parser.add_argument(
        "--local_cache_dir",
        type=Path,
        default=None,
        help="Optional cache directory for local model weights.",
    )

    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--checkpoint_interval", type=int, default=1)

    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--consistency_sample_size", type=int, default=3)
    parser.add_argument(
        "--progress_dir",
        type=Path,
        default=None,
        help="Optional directory for live progress files (progress_<model>.json).",
    )

    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_tasks", type=int, default=2)
    parser.add_argument("--smoke_candidates", type=int, default=3)

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated tags to run, e.g. qwen3.5-4b,qwen3-4b-instruct-2507",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    messages_file = resolve_messages_file(args.messages_file, args.fallback_messages_file, repo_root)
    tasks = load_jsonl(messages_file)

    if args.smoke:
        tasks = tasks[: args.smoke_tasks]
        target_candidates = args.smoke_candidates
    else:
        target_candidates = args.num_candidates

    all_specs = build_model_specs()
    if args.models.strip().lower() != "all":
        selected = {item.strip() for item in args.models.split(",") if item.strip()}
        specs = [spec for spec in all_specs if spec.tag in selected]
    else:
        specs = all_specs

    if not specs:
        raise ValueError("no model specs selected")

    results: List[ModelRunResult] = []
    for spec in specs:
        print(f"Running model={spec.tag} provider={spec.provider}", flush=True)
        result = run_model(
            spec=spec,
            tasks=tasks,
            target_candidates=target_candidates,
            args=args,
            repo_root=repo_root,
            messages_file=messages_file,
        )
        results.append(result)
        print(
            f"Done model={spec.tag} status={result.status} reason={result.reason} "
            f"tasks={result.tasks_completed}/{result.tasks_requested}",
            flush=True,
        )

    report_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "messages_file": str(messages_file),
        "smoke": args.smoke,
        "models": [asdict(item) for item in results],
        "summary": {
            "completed": sum(1 for x in results if x.status == "completed"),
            "skipped": sum(1 for x in results if x.status == "skipped"),
            "failed": sum(1 for x in results if x.status == "failed"),
        },
    }

    report_path = args.run_report_path
    if args.smoke and report_path.name == "run_report.json":
        report_path = report_path.with_name("run_report_smoke.json")
    report_path = report_path if report_path.is_absolute() else (repo_root / report_path)
    save_json(report_path, report_payload)

    print(f"run_report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
