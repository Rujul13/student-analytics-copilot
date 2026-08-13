import json
from types import SimpleNamespace

import pandas as pd
import pytest

from app.data_agent import (
    GeneratedPandasProgram,
    build_schema_context,
    deterministic_answer_from_rows,
    generate_pandas_program,
    generate_validate_execute,
)
from app.repository import DatasetContext
from app.scope_validation import extract_scope


@pytest.fixture
def context():
    frames = {
        "students": pd.DataFrame({"student_id": ["OULAD-242636", "S2"], "display_name": ["Learner 242636", "B"], "program": ["P", "P"]}),
        "courses": pd.DataFrame({"course_code": ["BBB", "CCC"], "course_name": ["X", "Y"]}),
        "enrollments": pd.DataFrame(
            {
                "enrollment_id": ["E1", "E2"],
                "student_id": ["OULAD-242636", "S2"],
                "course_code": ["BBB", "CCC"],
                "presentation": ["2014J", "2013J"],
                "final_result": ["Pass", "Withdrawn"],
            }
        ),
        "grades": pd.DataFrame({"enrollment_id": ["E1", "E2"], "weighted_grade": [70.0, 30.0]}),
    }
    return DatasetContext("Test", "v1", "test", frames)


def _fake_client(program: GeneratedPandasProgram):
    async def create(**kwargs):
        content = program.model_dump_json()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_build_schema_context_includes_required_fields(context):
    schema = build_schema_context(context)
    assert schema["dataset_version"] == "v1"
    assert set(schema["tables"]) == {"students", "courses", "enrollments", "grades"}
    assert "students.student_id = enrollments.student_id" in schema["relationships"]
    assert any("withdrawal rate" in item for item in schema["metric_definitions"])
    assert schema["tables"]["enrollments"]["row_count"] == 2


def test_build_schema_context_never_includes_row_data(context):
    schema = build_schema_context(context)
    serialized = json.dumps(schema)
    assert "OULAD-242636" not in serialized  # no full student rows, only bounded categorical examples


@pytest.mark.asyncio
async def test_generate_pandas_program_parses_the_structured_response(context):
    program = GeneratedPandasProgram(
        interpretation="Average grade for BBB",
        code="result = float(enrollments.merge(grades, on='enrollment_id')[enrollments['course_code']=='BBB']['weighted_grade'].mean())",
        result_type="scalar",
        referenced_tables=["enrollments", "grades"],
        referenced_columns=["course_code", "weighted_grade"],
    )
    client = _fake_client(program)
    result = await generate_pandas_program(client, "test-model", "average grade in BBB", build_schema_context(context), [])
    assert result.interpretation == "Average grade for BBB"
    assert result.result_type == "scalar"


@pytest.mark.asyncio
async def test_generate_validate_execute_returns_execution_on_valid_program(context):
    program = GeneratedPandasProgram(
        interpretation="Average grade for BBB",
        code=(
            "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
            "result = float(merged[merged['course_code'] == 'BBB']['weighted_grade'].mean())"
        ),
        result_type="scalar",
        referenced_tables=["enrollments", "grades"],
        referenced_columns=["course_code", "weighted_grade"],
    )
    client = _fake_client(program)
    scope = extract_scope("What is the average grade in module BBB?", context)
    result_program, execution, error, _ = await generate_validate_execute(
        client, "test-model", "What is the average grade in module BBB?", build_schema_context(context), [], context, scope, None, None,
    )
    assert error is None
    assert execution.status == "ok"
    assert execution.rows == [{"value": 70.0}]


@pytest.mark.asyncio
async def test_generate_validate_execute_reports_unsafe_code_as_an_error(context):
    program = GeneratedPandasProgram(
        interpretation="malicious",
        code="import os\nresult = 1",
        result_type="scalar",
        referenced_tables=["enrollments"],
        referenced_columns=[],
    )
    client = _fake_client(program)
    scope = extract_scope("What is the average grade in module BBB?", context)
    result_program, execution, error, code_for_repair = await generate_validate_execute(
        client, "test-model", "What is the average grade in module BBB?", build_schema_context(context), [], context, scope, None, None,
    )
    assert result_program is None
    assert execution is None
    assert error is not None
    assert code_for_repair == "import os\nresult = 1"


@pytest.mark.asyncio
async def test_generate_validate_execute_rejects_a_program_that_drops_the_course_scope(context):
    program = GeneratedPandasProgram(
        interpretation="cohort average, ignoring the module filter",
        code="merged = enrollments.merge(grades, on='enrollment_id', how='left')\nresult = float(merged['weighted_grade'].mean())",
        result_type="scalar",
        referenced_tables=["enrollments", "grades"],
        referenced_columns=["weighted_grade"],
    )
    client = _fake_client(program)
    scope = extract_scope("What is the average grade in module BBB?", context)
    result_program, execution, error, _ = await generate_validate_execute(
        client, "test-model", "What is the average grade in module BBB?", build_schema_context(context), [], context, scope, None, None,
    )
    assert result_program is None
    assert "BBB" in error


def test_deterministic_answer_from_rows_handles_scalar():
    answer = deterministic_answer_from_rows([{"value": 42.0}], "scalar")
    assert "42" in answer


def test_deterministic_answer_from_rows_handles_empty():
    answer = deterministic_answer_from_rows([], "table")
    assert "empty" in answer.lower()


def test_deterministic_answer_distinguishes_total_matches_from_evidence_preview():
    answer = deterministic_answer_from_rows(
        [{"student_id": "S1", "failed_courses": 2}], "table", total_count=190, rows_truncated=True
    )
    assert "190 total matches" in answer
    assert "1 evidence rows shown" in answer
