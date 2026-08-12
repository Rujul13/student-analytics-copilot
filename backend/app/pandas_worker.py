from __future__ import annotations

import multiprocessing
import os
import sys
from dataclasses import dataclass
from queue import Empty
from typing import Any, Literal

import numpy as np
import pandas as pd

MAX_RESULT_ROWS = 100
MAX_SERIALIZED_BYTES = 200_000
DEFAULT_TIMEOUT_SECONDS = 5.0

_SAFE_BUILTIN_NAMES = (
    "len", "range", "min", "max", "sum", "sorted", "round", "abs",
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "enumerate", "zip", "map", "filter", "isinstance",
)


@dataclass
class WorkerExecutionResult:
    status: Literal["ok", "error", "timeout"]
    result_type: str | None = None
    rows: list[dict[str, Any]] | None = None
    truncated: bool = False
    error: str | None = None


def _safe_builtins() -> dict[str, Any]:
    import builtins as builtins_module

    return {name: getattr(builtins_module, name) for name in _SAFE_BUILTIN_NAMES}


def _apply_resource_limits() -> None:
    if sys.platform == "win32":
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    except (ImportError, ValueError, OSError):
        pass


def _normalize_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return None if np.isnan(result) else result
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_normalize_value(item) for item in value.tolist()]
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _normalize_result(result: Any) -> tuple[str, list[dict[str, Any]], bool]:
    if isinstance(result, pd.DataFrame):
        frame = result
        truncated = len(frame) > MAX_RESULT_ROWS
        if truncated:
            frame = frame.head(MAX_RESULT_ROWS)
        rows = [{str(k): _normalize_value(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]
        return "table", rows, truncated
    if isinstance(result, pd.Series):
        series = result
        truncated = len(series) > MAX_RESULT_ROWS
        if truncated:
            series = series.head(MAX_RESULT_ROWS)
        rows = [{"key": str(index), "value": _normalize_value(value)} for index, value in series.items()]
        return "table", rows, truncated
    if isinstance(result, dict):
        return "table", [{"key": str(k), "value": _normalize_value(v)} for k, v in result.items()], False
    if isinstance(result, (list, tuple)):
        truncated = len(result) > MAX_RESULT_ROWS
        items = list(result)[:MAX_RESULT_ROWS]
        if items and isinstance(items[0], dict):
            rows = [{str(k): _normalize_value(v) for k, v in item.items()} for item in items]
        else:
            rows = [{"value": _normalize_value(item)} for item in items]
        return "table", rows, truncated
    return "scalar", [{"value": _normalize_value(result)}], False


def _shrink_to_budget(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    import json

    try:
        size = len(json.dumps(rows, default=str))
    except (TypeError, ValueError):
        return rows[:1], True
    if size <= MAX_SERIALIZED_BYTES or len(rows) <= 1:
        return rows, False
    return rows[: max(1, len(rows) // 2)], True


def _worker_entry(code: str, frames: dict[str, pd.DataFrame], result_queue: Any) -> None:
    os.environ.clear()
    _apply_resource_limits()
    try:
        namespace: dict[str, Any] = {"__builtins__": _safe_builtins(), "pd": pd, "np": np}
        namespace.update(frames)
        compiled = compile(code, "<generated_pandas_program>", "exec")
        exec(compiled, namespace)  # noqa: S102 - namespace is restricted; code is AST-validated upstream
        if "result" not in namespace:
            result_queue.put(("error", "Generated code did not assign a value to `result`"))
            return
        result_type, rows, truncated = _normalize_result(namespace["result"])
        rows, shrunk = _shrink_to_budget(rows)
        result_queue.put(("ok", {"result_type": result_type, "rows": rows, "truncated": truncated or shrunk}))
    except Exception as error:  # noqa: BLE001 - process boundary: must not leak internals or crash silently
        result_queue.put(("error", f"{type(error).__name__}: execution failed"))


def run_pandas_code(
    code: str, frames: dict[str, pd.DataFrame], timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> WorkerExecutionResult:
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    frame_copies = {name: frame.copy(deep=True) for name, frame in frames.items()}
    process = ctx.Process(target=_worker_entry, args=(code, frame_copies, result_queue), daemon=True)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join(1)
        return WorkerExecutionResult(status="timeout", error="Execution exceeded the time limit")
    try:
        status, payload = result_queue.get(timeout=1)
    except Empty:
        return WorkerExecutionResult(status="error", error="Worker process exited without a result")
    if status == "error":
        return WorkerExecutionResult(status="error", error=str(payload))
    return WorkerExecutionResult(
        status="ok",
        result_type=payload["result_type"],
        rows=payload["rows"],
        truncated=payload["truncated"],
    )
