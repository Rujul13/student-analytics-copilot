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
    total_count: int | None = None
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
    if value is pd.NaT:
        return None
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
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Last-resort fallback for any other Pandas/NumPy extension type (Period, Interval,
    # Categorical, complex, etc.) not already covered above - stringify rather than let a
    # raw non-JSON-serializable object reach the multiprocessing Queue or the caller.
    return str(value)


def _normalize_result(result: Any) -> tuple[str, list[dict[str, Any]], bool, int | None]:
    if isinstance(result, pd.DataFrame):
        frame = result
        total_count = len(frame)
        truncated = len(frame) > MAX_RESULT_ROWS
        if truncated:
            frame = frame.head(MAX_RESULT_ROWS)
        rows = [{str(k): _normalize_value(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]
        return "table", rows, truncated, total_count
    if isinstance(result, pd.Series):
        series = result
        total_count = len(series)
        truncated = len(series) > MAX_RESULT_ROWS
        if truncated:
            series = series.head(MAX_RESULT_ROWS)
        rows = [{"key": str(index), "value": _normalize_value(value)} for index, value in series.items()]
        return "table", rows, truncated, total_count
    if isinstance(result, dict):
        return "table", [{"key": str(k), "value": _normalize_value(v)} for k, v in result.items()], False, len(result)
    if isinstance(result, (list, tuple)):
        total_count = len(result)
        truncated = len(result) > MAX_RESULT_ROWS
        items = list(result)[:MAX_RESULT_ROWS]
        if items and isinstance(items[0], dict):
            rows = [{str(k): _normalize_value(v) for k, v in item.items()} for item in items]
        else:
            rows = [{"value": _normalize_value(item)} for item in items]
        return "table", rows, truncated, total_count
    return "scalar", [{"value": _normalize_value(result)}], False, None


def _shrink_to_budget(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    import json

    if not rows:
        return rows, False

    def _serialized_size(candidate: list[dict[str, Any]]) -> int:
        try:
            return len(json.dumps(candidate, default=str))
        except (TypeError, ValueError):
            return MAX_SERIALIZED_BYTES + 1

    truncated = False
    while len(rows) > 1 and _serialized_size(rows) > MAX_SERIALIZED_BYTES:
        rows = rows[: max(1, len(rows) // 2)]
        truncated = True

    if _serialized_size(rows) > MAX_SERIALIZED_BYTES:
        # Even a single row is over budget (e.g. one very large string value) - the row
        # count is already at the floor, so shrink the oversized values within it instead
        # of returning an unbounded payload.
        shrunk_row: dict[str, Any] = {}
        for key, value in rows[0].items():
            if isinstance(value, str) and len(value) > 500:
                shrunk_row[key] = value[:500] + "...(truncated)"
            else:
                shrunk_row[key] = value
        rows = [shrunk_row]
        truncated = True

    return rows, truncated


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _reject_unsafe_constructs(code: str) -> str | None:
    """A minimal, self-contained static pre-check the worker runs on its own, independent
    of backend/app/pandas_code_validation.py's full AST validator (this module must not
    import that one - see Task 3's brief). This exists as defense-in-depth: even if this
    worker is ever invoked without the upstream validator running first, it still refuses
    to run code that reaches for imports or dunder attributes - closing the classic
    `().__class__.__bases__[0].__subclasses__()` sandbox-escape family - inside its own
    process. It is intentionally narrower than the full validator (imports + dunders only);
    it is not a replacement for it. Returns a sanitized error message, or None if the code
    passes this check.
    """
    import ast

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return "Generated code is not valid Python"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Import statements are not allowed"
        if isinstance(node, ast.Name) and _is_dunder(node.id):
            return f"Disallowed dunder identifier: {node.id}"
        if isinstance(node, ast.Attribute) and _is_dunder(node.attr):
            return f"Disallowed dunder attribute: {node.attr}"
    return None


def _worker_entry(code: str, frames: dict[str, pd.DataFrame], result_queue: Any) -> None:
    os.environ.clear()
    _apply_resource_limits()
    rejection = _reject_unsafe_constructs(code)
    if rejection is not None:
        result_queue.put(("error", rejection))
        return
    try:
        namespace: dict[str, Any] = {"__builtins__": _safe_builtins(), "pd": pd, "np": np}
        namespace.update(frames)
        compiled = compile(code, "<generated_pandas_program>", "exec")
        exec(compiled, namespace)  # noqa: S102 - namespace is restricted; code is AST-validated upstream and by _reject_unsafe_constructs above
        if "result" not in namespace:
            result_queue.put(("error", "Generated code did not assign a value to `result`"))
            return
        result_type, rows, truncated, total_count = _normalize_result(namespace["result"])
        rows, shrunk = _shrink_to_budget(rows)
        result_queue.put(("ok", {
            "result_type": result_type,
            "rows": rows,
            "total_count": total_count,
            "truncated": truncated or shrunk,
        }))
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
    # Read from the queue BEFORE joining the process - do not call `process.join(timeout)`
    # first. `multiprocessing.Queue.put()` writes through a background feeder thread onto an
    # OS pipe with a bounded buffer; if the parent joins first and the payload is larger than
    # that buffer (this bit for real during Task 3's own fix-round testing: a 100-row,
    # ~500KB result reproducibly hung for minutes on this machine), the child blocks writing
    # to a pipe nobody is draining yet while the parent blocks waiting for the child to exit -
    # a classic multiprocessing deadlock, only bounded by the outer timeout escalating to
    # terminate()/kill() and then reporting a false "timeout" for work that had actually
    # already completed. Reading first drains the pipe as data arrives and also serves as the
    # wait-for-completion step, so the deadlock cannot occur.
    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        # Distinguish "still running past the deadline" (a real timeout - kill it) from
        # "already exited without ever queueing a result" (e.g. an interpreter crash or an
        # uncaught BaseException that bypassed _worker_entry's own exception handling) - both
        # raise Empty here, but only the first is actually a timeout.
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
            return WorkerExecutionResult(status="timeout", error="Execution exceeded the time limit")
        process.join(1)
        return WorkerExecutionResult(status="error", error="Worker process exited without a result")
    process.join(1)
    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join(1)
    if status == "error":
        return WorkerExecutionResult(status="error", error=str(payload))
    return WorkerExecutionResult(
        status="ok",
        result_type=payload["result_type"],
        rows=payload["rows"],
        total_count=payload["total_count"],
        truncated=payload["truncated"],
    )
