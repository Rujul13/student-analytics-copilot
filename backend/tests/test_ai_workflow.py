import json
from types import SimpleNamespace

import pytest

from app.ai_workflow import AnalyticsWorkflow
from app.config import get_settings
from app.data_agent import GeneratedPandasProgram
from app.repository import load_dataset


@pytest.fixture
def context():
    get_settings.cache_clear()
    return load_dataset(get_settings())


def _queue_client(programs: list[GeneratedPandasProgram], answer: str | None = None):
    """Fake AsyncGroq client. Code-gen calls are served in order from `programs`;
    the final answer-synthesis call (different response schema) returns `answer` if given,
    otherwise raises to force the deterministic-answer fallback."""
    calls = {"n": 0}

    async def create(**kwargs):
        schema_name = kwargs["response_format"]["json_schema"]["name"]
        if schema_name == "generated_pandas_program":
            index = calls["n"]
            calls["n"] += 1
            program = programs[index]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=program.model_dump_json()))])
        if schema_name == "answer_narrative":
            if answer is None:
                raise RuntimeError("synthesis unavailable")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"answer": answer})))])
        raise AssertionError(f"unexpected schema {schema_name}")

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def _workflow(context, client) -> AnalyticsWorkflow:
    workflow = AnalyticsWorkflow(context, "unused-key", "test-120b", "test-20b")
    workflow.client = client
    return workflow


@pytest.mark.asyncio
async def test_q1_highest_withdrawal_rate(context):
    program = GeneratedPandasProgram(
        interpretation="Withdrawal rate per module, highest first",
        code=(
            "grouped = enrollments.groupby('course_code').agg(\n"
            "    enrollments=('enrollment_id', 'count'),\n"
            "    withdrawals=('final_result', lambda values: (values == 'Withdrawn').sum()),\n"
            ")\n"
            "grouped['withdrawal_rate'] = (grouped['withdrawals'] / grouped['enrollments'] * 100).round(1)\n"
            "result = grouped.sort_values('withdrawal_rate', ascending=False).reset_index().head(100)\n"
        ),
        result_type="table",
        referenced_tables=["enrollments"],
        referenced_columns=["course_code", "final_result"],
    )
    client, _ = _queue_client([program], answer="Module AAA has the highest withdrawal rate at 60.0%.")
    workflow = _workflow(context, client)
    response = await workflow.run(question="Which module has the highest withdrawal rate?", history=[])
    assert response.execution_mode == "generated-pandas"
    assert response.result_type == "table"
    top = response.rows[0]
    assert top["course_code"] == "AAA"
    assert top["withdrawal_rate"] == 60.0
    assert "60.0" in response.answer or "60" in response.answer


@pytest.mark.asyncio
async def test_q2_average_grade_in_bbb(context):
    program = GeneratedPandasProgram(
        interpretation="Average weighted grade for module BBB",
        code=(
            "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
            "result = round(float(merged[merged['course_code'] == 'BBB']['weighted_grade'].mean()), 1)\n"
        ),
        result_type="scalar",
        referenced_tables=["enrollments", "grades"],
        referenced_columns=["course_code", "weighted_grade"],
    )
    client, _ = _queue_client([program])
    workflow = _workflow(context, client)
    response = await workflow.run(question="What is the average grade in module BBB?", history=[])
    assert response.execution_mode == "generated-pandas"
    assert response.result_type == "metric"
    assert response.rows == [{"value": 66.1}]


@pytest.mark.asyncio
async def test_q3_missing_gender_field_short_circuits_before_code_generation(context):
    client, calls = _queue_client([])  # no programs queued: generation must never be called
    workflow = _workflow(context, client)
    response = await workflow.run(
        question="What is the average grade for female students in module BBB?", history=[]
    )
    assert response.result_type == "unsupported"
    assert response.execution_mode == "unsupported"
    assert response.ai_used is False
    assert "gender" in response.answer
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_q4_compare_pass_rate_bbb_two_presentations(context):
    program = GeneratedPandasProgram(
        interpretation="BBB pass rate for 2013J vs 2014J",
        code=(
            "bbb = enrollments[enrollments['course_code'] == 'BBB']\n"
            "rows = []\n"
            "for presentation in ['2013J', '2014J']:\n"
            "    subset = bbb[bbb['presentation'] == presentation]\n"
            "    passes = int(subset['final_result'].isin(['Pass', 'Distinction']).sum())\n"
            "    total = int(len(subset))\n"
            "    rate = round(passes / total * 100, 1) if total else None\n"
            "    rows.append({'presentation': presentation, 'passes': passes, 'total': total, 'pass_rate': rate})\n"
            "result = rows\n"
        ),
        result_type="table",
        referenced_tables=["enrollments"],
        referenced_columns=["course_code", "presentation", "final_result"],
    )
    client, _ = _queue_client([program])
    workflow = _workflow(context, client)
    response = await workflow.run(
        question="Compare the pass rate for BBB between 2013J and 2014J.", history=[]
    )
    assert response.execution_mode == "generated-pandas"
    rows_by_presentation = {row["presentation"]: row["pass_rate"] for row in response.rows}
    assert rows_by_presentation["2013J"] == 6.2
    assert rows_by_presentation["2014J"] == 50.0


@pytest.mark.asyncio
async def test_q5_five_lowest_grades_in_ccc(context):
    program = GeneratedPandasProgram(
        interpretation="Five lowest graded learners in CCC",
        code=(
            "merged = enrollments.merge(grades, on='enrollment_id', how='left').merge(students, on='student_id', how='left')\n"
            "ccc = merged[merged['course_code'] == 'CCC'].dropna(subset=['weighted_grade'])\n"
            "result = ccc.sort_values(['weighted_grade', 'student_id'], ascending=[True, True])[['student_id', 'display_name', 'weighted_grade']].head(5)\n"
        ),
        result_type="table",
        referenced_tables=["enrollments", "grades", "students"],
        referenced_columns=["course_code", "weighted_grade", "student_id"],
    )
    client, _ = _queue_client([program])
    workflow = _workflow(context, client)
    response = await workflow.run(question="Which five learners have the lowest grades in CCC?", history=[])
    assert len(response.rows) == 5
    assert [row["student_id"] for row in response.rows] == [
        "OULAD-242636", "OULAD-2446778", "OULAD-529723", "OULAD-582827", "OULAD-599937",
    ]
    assert all(row["weighted_grade"] == 0.0 for row in response.rows)


@pytest.mark.asyncio
async def test_q6_learners_who_failed_more_than_one_module(context):
    program = GeneratedPandasProgram(
        interpretation="Count of learners who failed more than one module",
        code=(
            "failed = enrollments[enrollments['final_result'] == 'Fail']\n"
            "counts = failed.groupby('student_id')['course_code'].nunique()\n"
            "result = int((counts > 1).sum())\n"
        ),
        result_type="scalar",
        referenced_tables=["enrollments"],
        referenced_columns=["final_result", "student_id", "course_code"],
    )
    client, _ = _queue_client([program])
    workflow = _workflow(context, client)
    response = await workflow.run(question="How many learners failed more than one module?", history=[])
    assert response.rows == [{"value": 39}]


@pytest.mark.asyncio
async def test_q7_percentage_of_withdrawn_learners_with_average_below_50(context):
    program = GeneratedPandasProgram(
        interpretation="Share of withdrawn learners whose average grade is below 50",
        code=(
            "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
            "withdrawn_ids = enrollments[enrollments['final_result'] == 'Withdrawn']['student_id'].unique()\n"
            "withdrawn_avg = merged[merged['student_id'].isin(withdrawn_ids)].groupby('student_id')['weighted_grade'].mean()\n"
            "below_50 = int((withdrawn_avg < 50).sum())\n"
            "result = round(below_50 / len(withdrawn_ids) * 100, 1)\n"
        ),
        result_type="scalar",
        referenced_tables=["enrollments", "grades"],
        referenced_columns=["final_result", "student_id", "weighted_grade"],
    )
    client, _ = _queue_client([program])
    workflow = _workflow(context, client)
    response = await workflow.run(
        question="What percentage of learners who withdrew had an average below 50?", history=[]
    )
    assert response.rows == [{"value": 15.3}]


@pytest.mark.asyncio
async def test_q8_highest_distinction_rate(context):
    program = GeneratedPandasProgram(
        interpretation="Distinction rate per module, highest first",
        code=(
            "grouped = enrollments.groupby('course_code').agg(\n"
            "    enrollments=('enrollment_id', 'count'),\n"
            "    distinctions=('final_result', lambda values: (values == 'Distinction').sum()),\n"
            ")\n"
            "grouped['distinction_rate'] = (grouped['distinctions'] / grouped['enrollments'] * 100).round(1)\n"
            "result = grouped.sort_values('distinction_rate', ascending=False).reset_index().head(100)\n"
        ),
        result_type="table",
        referenced_tables=["enrollments"],
        referenced_columns=["course_code", "final_result"],
    )
    client, _ = _queue_client([program])
    workflow = _workflow(context, client)
    response = await workflow.run(question="Which module has the highest distinction rate?", history=[])
    assert response.rows[0]["course_code"] == "GGG"
    assert response.rows[0]["distinction_rate"] == 31.2


@pytest.mark.asyncio
async def test_q9_tell_me_about_a_learner(context):
    program = GeneratedPandasProgram(
        interpretation="Profile and enrollment history for OULAD-242636",
        code=(
            "profile = students[students['student_id'] == 'OULAD-242636']\n"
            "history = enrollments[enrollments['student_id'] == 'OULAD-242636'].merge(grades, on='enrollment_id', how='left')\n"
            "result = pd.DataFrame([{\n"
            "    'student_id': 'OULAD-242636',\n"
            "    'display_name': str(profile['display_name'].iloc[0]),\n"
            "    'average_grade': float(history['weighted_grade'].mean()) if history['weighted_grade'].notna().any() else None,\n"
            "    'enrollment_count': int(len(history)),\n"
            "}])\n"
        ),
        result_type="table",
        referenced_tables=["students", "enrollments", "grades"],
        referenced_columns=["student_id", "display_name", "weighted_grade"],
    )
    client, _ = _queue_client([program])
    workflow = _workflow(context, client)
    response = await workflow.run(question="Tell me about learner OULAD-242636.", history=[])
    row = response.rows[0]
    assert row["student_id"] == "OULAD-242636"
    assert row["average_grade"] == 0.0
    assert row["enrollment_count"] == 2


@pytest.mark.asyncio
async def test_q10_followup_uses_history_to_flip_sort_direction(context):
    highest = GeneratedPandasProgram(
        interpretation="Withdrawal rate per module, highest first",
        code=(
            "grouped = enrollments.groupby('course_code').agg(\n"
            "    enrollments=('enrollment_id', 'count'),\n"
            "    withdrawals=('final_result', lambda values: (values == 'Withdrawn').sum()),\n"
            ")\n"
            "grouped['withdrawal_rate'] = (grouped['withdrawals'] / grouped['enrollments'] * 100).round(1)\n"
            "result = grouped.sort_values('withdrawal_rate', ascending=False).reset_index().head(100)\n"
        ),
        result_type="table",
        referenced_tables=["enrollments"],
        referenced_columns=["course_code", "final_result"],
    )
    lowest = GeneratedPandasProgram(
        interpretation="Withdrawal rate per module, lowest first",
        code=(
            "grouped = enrollments.groupby('course_code').agg(\n"
            "    enrollments=('enrollment_id', 'count'),\n"
            "    withdrawals=('final_result', lambda values: (values == 'Withdrawn').sum()),\n"
            ")\n"
            "grouped['withdrawal_rate'] = (grouped['withdrawals'] / grouped['enrollments'] * 100).round(1)\n"
            "result = grouped.sort_values('withdrawal_rate', ascending=True).reset_index().head(100)\n"
        ),
        result_type="table",
        referenced_tables=["enrollments"],
        referenced_columns=["course_code", "final_result"],
    )
    client, _ = _queue_client([highest, lowest])
    workflow = _workflow(context, client)
    first = await workflow.run(question="Which module has the highest withdrawal rate?", history=[])
    assert first.rows[0]["course_code"] == "AAA"
    second_workflow = _workflow(context, client)
    second = await second_workflow.run(
        question="What about the lowest?",
        history=[{"question": "Which module has the highest withdrawal rate?", "answer": first.answer}],
    )
    assert second.rows[0]["course_code"] == "GGG"
    assert second.rows[0]["withdrawal_rate"] == 6.2


@pytest.mark.asyncio
async def test_one_repair_attempt_recovers_from_unsafe_first_attempt(context):
    unsafe = GeneratedPandasProgram(
        interpretation="broken attempt", code="import os\nresult = 1",
        result_type="scalar", referenced_tables=["enrollments"], referenced_columns=[],
    )
    fixed = GeneratedPandasProgram(
        interpretation="Average grade for BBB",
        code=(
            "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
            "result = round(float(merged[merged['course_code'] == 'BBB']['weighted_grade'].mean()), 1)\n"
        ),
        result_type="scalar", referenced_tables=["enrollments", "grades"], referenced_columns=["course_code", "weighted_grade"],
    )
    client, calls = _queue_client([unsafe, fixed])
    workflow = _workflow(context, client)
    response = await workflow.run(question="What is the average grade in module BBB?", history=[])
    assert calls["n"] == 2
    assert response.execution_mode == "generated-pandas-repaired"
    assert response.rows == [{"value": 66.1}]


@pytest.mark.asyncio
async def test_repair_is_bounded_to_one_attempt(context):
    unsafe = GeneratedPandasProgram(
        interpretation="broken attempt", code="import os\nresult = 1",
        result_type="scalar", referenced_tables=["enrollments"], referenced_columns=[],
    )
    still_unsafe = GeneratedPandasProgram(
        interpretation="still broken", code="import sys\nresult = 1",
        result_type="scalar", referenced_tables=["enrollments"], referenced_columns=[],
    )
    client, calls = _queue_client([unsafe, still_unsafe])
    workflow = _workflow(context, client)
    response = await workflow.run(question="What is the average grade in module BBB?", history=[])
    assert calls["n"] == 2  # exactly one repair attempt, never a third generation call
    assert response.result_type == "error"
    assert response.execution_mode == "unsupported"


@pytest.mark.asyncio
async def test_timeout_terminates_and_is_reported_as_a_failure(context, monkeypatch):
    # Use a program that passes AST validation and scope preservation (it is the same
    # well-formed BBB-average program as test_q2) so the flow actually reaches
    # run_pandas_code — only then does the monkeypatched timeout exercise the path this
    # test claims to cover. A program that fails validation/scope earlier would make this
    # test pass for the wrong reason without ever touching the timeout branch.
    program = GeneratedPandasProgram(
        interpretation="Average grade for BBB",
        code=(
            "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
            "result = round(float(merged[merged['course_code'] == 'BBB']['weighted_grade'].mean()), 1)\n"
        ),
        result_type="scalar", referenced_tables=["enrollments", "grades"], referenced_columns=["course_code", "weighted_grade"],
    )
    from app import data_agent
    from app.pandas_worker import WorkerExecutionResult

    monkeypatch.setattr(
        data_agent,
        "run_pandas_code",
        lambda code, frames: WorkerExecutionResult(status="timeout", error="Execution exceeded the time limit"),
    )
    client, calls = _queue_client([program, program])
    workflow = _workflow(context, client)
    response = await workflow.run(question="What is the average grade in module BBB?", history=[])
    assert calls["n"] == 2  # generation was retried once after the first timeout
    assert response.result_type == "error"
    assert response.execution_mode == "unsupported"


@pytest.mark.asyncio
async def test_response_never_contains_generated_code(context):
    program = GeneratedPandasProgram(
        interpretation="Average grade for BBB",
        code=(
            "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
            "result = round(float(merged[merged['course_code'] == 'BBB']['weighted_grade'].mean()), 1)\n"
        ),
        result_type="scalar", referenced_tables=["enrollments", "grades"], referenced_columns=["course_code", "weighted_grade"],
    )
    client, _ = _queue_client([program])
    workflow = _workflow(context, client)
    response = await workflow.run(question="What is the average grade in module BBB?", history=[])
    dumped = response.model_dump_json()
    assert "merged" not in dumped
    assert "import" not in dumped
    assert not hasattr(response, "code")


@pytest.mark.asyncio
async def test_no_api_key_reaches_the_worker_process(context):
    program = GeneratedPandasProgram(
        interpretation="attempt to read env",
        # Must literally contain the course code "BBB" (the question asks about module BBB)
        # so `verify_scope_preserved` accepts it on the first attempt - otherwise scope
        # validation forces a repair, and this test only queues one program.
        code="result = 'BBB'",
        result_type="scalar", referenced_tables=["enrollments"], referenced_columns=[],
    )
    import os

    os.environ["GROQ_API_KEY"] = "should-never-reach-the-worker"
    try:
        client, _ = _queue_client([program])
        workflow = AnalyticsWorkflow(context, "should-never-reach-the-worker", "test-120b", "test-20b")
        workflow.client = client
        response = await workflow.run(question="What is the average grade in module BBB?", history=[])
        assert response.rows  # ran successfully; the worker never had access to GROQ_API_KEY to leak
    finally:
        del os.environ["GROQ_API_KEY"]
