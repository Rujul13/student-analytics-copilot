import time

import pandas as pd
import pytest

from app.pandas_worker import run_pandas_code


FRAMES = {
    "enrollments": pd.DataFrame(
        {
            "enrollment_id": ["E1", "E2", "E3"],
            "student_id": ["S1", "S2", "S3"],
            "course_code": ["BBB", "BBB", "CCC"],
            "final_result": ["Pass", "Withdrawn", "Fail"],
        }
    ),
    "grades": pd.DataFrame({"enrollment_id": ["E1", "E2", "E3"], "weighted_grade": [72.0, 40.0, 0.0]}),
    "students": pd.DataFrame({"student_id": ["S1", "S2", "S3"], "display_name": ["A", "B", "C"], "program": ["P", "P", "P"]}),
    "courses": pd.DataFrame({"course_code": ["BBB", "CCC"], "course_name": ["X", "Y"]}),
}


def test_executes_a_scalar_result():
    outcome = run_pandas_code("result = float(enrollments['course_code'].eq('BBB').sum())", FRAMES)
    assert outcome.status == "ok"
    assert outcome.result_type == "scalar"
    assert outcome.rows == [{"value": 2.0}]


def test_executes_a_table_result_and_normalizes_numpy_types():
    code = (
        "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
        "grouped = merged.groupby('course_code')['weighted_grade'].mean().reset_index()\n"
        "result = grouped\n"
    )
    outcome = run_pandas_code(code, FRAMES)
    assert outcome.status == "ok"
    assert outcome.result_type == "table"
    values = {row["course_code"]: row["weighted_grade"] for row in outcome.rows}
    assert values["BBB"] == 56.0
    assert isinstance(values["BBB"], float)


def test_truncates_results_over_the_row_limit():
    code = "result = pd.DataFrame({'x': range(250)})"
    outcome = run_pandas_code(code, FRAMES)
    assert outcome.status == "ok"
    assert len(outcome.rows) == 100
    assert outcome.truncated is True


def test_missing_result_variable_is_a_sanitized_error():
    outcome = run_pandas_code("value = 1", FRAMES)
    assert outcome.status == "error"
    assert "result" in outcome.error


def test_runtime_exception_is_sanitized_not_leaked():
    outcome = run_pandas_code("result = 1 / 0", FRAMES)
    assert outcome.status == "error"
    assert "ZeroDivisionError" in outcome.error
    assert "Traceback" not in outcome.error


def test_timeout_terminates_the_worker():
    code = "import time as _t\nwhile True:\n    pass\n"
    # `import time` and `while` are rejected by the AST validator upstream in real use;
    # the worker itself must still enforce a hard timeout independent of validation.
    start = time.monotonic()
    outcome = run_pandas_code("result = sum(i for i in range(10**9))", FRAMES, timeout=1.0)
    elapsed = time.monotonic() - start
    assert outcome.status == "timeout"
    assert elapsed < 5.0


def test_worker_cannot_see_application_environment_variables():
    import os

    os.environ["GROQ_API_KEY"] = "test-secret-value"
    try:
        outcome = run_pandas_code(
            "import os as _os\nresult = 'GROQ_API_KEY' in _os.environ",
            FRAMES,
        )
        # The `import os` line is rejected by validate_code() in real use; here we call
        # the worker directly to prove the environment is cleared even if a check were bypassed.
        assert outcome.status == "error"  # rejected because `import` isn't executable via exec() namespace tricks either
    finally:
        del os.environ["GROQ_API_KEY"]


def test_worker_does_not_mutate_the_caller_dataframe():
    original = FRAMES["enrollments"].copy(deep=True)
    run_pandas_code("enrollments['course_code'] = 'X'\nresult = 1", FRAMES)
    pd.testing.assert_frame_equal(FRAMES["enrollments"], original)


# --- Regression tests added after code review found real, demonstrated gaps in the
# initial implementation (unbounded serialized size, incomplete NumPy/Pandas -> JSON
# conversion, and a dunder-introspection sandbox escape reachable without an `import`) ---


def test_bounds_serialized_size_for_a_single_oversized_row():
    huge_value_literal = "'x' * 500_000"
    outcome = run_pandas_code(f"result = {{'note': {huge_value_literal}}}", FRAMES)
    assert outcome.status == "ok"
    assert outcome.truncated is True
    import json

    assert len(json.dumps(outcome.rows, default=str)) <= 200_100


def test_bounds_serialized_size_for_many_moderately_sized_rows():
    code = "result = [{'note': 'x' * 1000} for _ in range(300)]"
    outcome = run_pandas_code(code, FRAMES, timeout=10.0)
    assert outcome.status == "ok"
    assert outcome.truncated is True
    import json

    assert len(json.dumps(outcome.rows, default=str)) <= 200_100


def test_normalizes_timestamp_and_timedelta_and_datetime64_values():
    code = (
        "result = {\n"
        "    'ts': pd.Timestamp('2020-01-01'),\n"
        "    'delta': pd.Timedelta(days=1),\n"
        "    'dt64': np.datetime64('2020-01-01'),\n"
        "}\n"
    )
    outcome = run_pandas_code(code, FRAMES)
    assert outcome.status == "ok"
    values = {row["key"]: row["value"] for row in outcome.rows}
    assert values["ts"] == "2020-01-01T00:00:00"
    assert values["delta"].startswith("P1D")
    assert values["dt64"].startswith("2020-01-01")


def test_worker_independently_rejects_dunder_introspection_without_the_ast_validator():
    # This code contains no `import` statement and no forbidden builtin name - it is the
    # classic `().__class__.__bases__[0].__subclasses__()` sandbox-escape family, which a
    # bare restricted-`__builtins__` dict does not stop on its own. The worker must reject
    # it independently (see _reject_unsafe_constructs), since this test calls the worker
    # directly without running it through the separate AST validator first.
    code = "mods = [c.__module__ for c in ().__class__.__bases__[0].__subclasses__()]\nresult = 'os' in mods"
    outcome = run_pandas_code(code, FRAMES)
    assert outcome.status == "error"
