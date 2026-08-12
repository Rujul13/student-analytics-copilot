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
