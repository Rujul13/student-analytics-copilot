import asyncio
import json

import pandas as pd

from app.analytics import dashboard, students
from app.catalog import split_codes
from app.config import Settings
from app.copilot import answer_question
from app.recommendations import add_ai_explanations, recommend
from app.repository import DatasetContext, load_dataset
from app.success_prediction import get_success_model


def context():
    return load_dataset(Settings(dataset_path="fixture"))


def test_dashboard_is_deterministic():
    result = dashboard(context())
    assert result.dataset_name == "OULAD development fixture"
    assert result.metrics[0].value == 180
    assert 0 < result.metrics[2].value < 100


def test_students_are_prioritized_by_risk():
    result = students(context())
    assert len(result) == 180
    assert result[0].risk == "High"
    assert result[0].graded_enrollments > 0


def test_recommendations_use_observed_modules_and_explain_limited_evidence():
    dataset = context()
    frames = {name: frame.copy() for name, frame in dataset.frames.items()}
    pathway_students = pd.DataFrame([
        {"student_id": "NEW-LEARNER", "display_name": "New Learner", "program": "Not used", "program_source": "Test"},
    ])
    frames["students"] = pd.concat([frames["students"], pathway_students], ignore_index=True)
    augmented = DatasetContext(dataset.name, dataset.version, dataset.mode, frames)

    response = recommend(augmented, "NEW-LEARNER")
    observed_codes = set(map(str, frames["enrollments"]["course_code"].unique()))
    assert response.recommendations
    assert all(item.course_code in observed_codes for item in response.recommendations)
    assert all(not item.course_code.startswith("NXT") for item in response.recommendations)
    assert all(item.evidence_strength == "Limited" for item in response.recommendations)
    assert all("Limited learner evidence" in item.success_basis for item in response.recommendations)


def test_recommendations_exclude_completed_courses():
    dataset = context()
    student = students(dataset)[0]
    completed = set(dataset.frames["enrollments"].loc[
        (dataset.frames["enrollments"]["student_id"] == student.student_id)
        & dataset.frames["enrollments"]["final_result"].isin(["Pass", "Distinction"]),
        "course_code",
    ])
    result = recommend(dataset, student.student_id)
    assert all(item.course_code not in completed for item in result.recommendations)
    assert result.capability_mode == "historical-performance"
    assert result.catalog_label == "OULAD historical modules; future availability unknown"
    for item in result.recommendations:
        assert item.course_code in set(dataset.frames["enrollments"]["course_code"])
        assert 0 <= item.score <= 100


def test_success_estimates_have_held_out_evaluation():
    dataset = context()
    model = get_success_model(dataset)
    assert model is not None
    assert model.evaluation.training_records > 0
    assert model.evaluation.test_records >= 20
    assert 0 <= model.evaluation.brier_score <= 1
    assert 0 <= model.evaluation.roc_auc <= 1
    student = students(dataset)[0]
    result = recommend(dataset, student.student_id)
    assert result.success_model is not None
    assert all(item.predicted_success_probability is not None for item in result.recommendations)
    assert all(0 <= float(item.predicted_success_probability) <= 100 for item in result.recommendations)
    assert len({item.predicted_success_probability for item in result.recommendations}) > 1


def test_llm_can_only_rerank_the_verified_candidate_set(monkeypatch):
    dataset = context()
    student = students(dataset)[0]
    response = recommend(dataset, student.student_id, limit=8)
    original_codes = [item.course_code for item in response.recommendations]
    reversed_codes = list(reversed(original_codes))[:3]

    class FakeCompletions:
        async def create(self, **kwargs):
            content = json.dumps({"rankings": [{"course_code": code, "narrative": f"Verified rationale for {code}."} for code in reversed_codes]})
            return type("Completion", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("app.recommendations.AsyncGroq", FakeClient)
    ranked = asyncio.run(add_ai_explanations(response, "test-key", "test-model"))
    assert ranked.ranking_mode == "hybrid-llm"
    assert ranked.ai_explanation_enabled is True
    assert [item.course_code for item in ranked.recommendations] == reversed_codes
    assert ranked.evaluated_candidates == len(original_codes)
    assert set(item.course_code for item in ranked.recommendations).issubset(set(original_codes))


def test_enriched_upload_enables_verified_graduation_aware_eligibility():
    dataset = context()
    learner = students(dataset)[0]
    frames = {name: frame.copy() for name, frame in dataset.frames.items()}
    frames["students"].loc[frames["students"]["student_id"].eq(learner.student_id), "program"] = "Applied Computing"
    frames["courses"] = pd.DataFrame([
        {"course_code": "NEXT-CORE", "course_name": "Capstone", "department": "Computing", "level": 3, "credits": 30, "offered_next_term": True, "prerequisites": "", "programs": "Applied Computing", "requirement_type": "core"},
        {"course_code": "WRONG-PROGRAM", "course_name": "Finance", "department": "Business", "level": 3, "credits": 30, "offered_next_term": True, "prerequisites": "", "programs": "Finance", "requirement_type": "core"},
    ])
    uploaded = DatasetContext("Uploaded", "enriched-test", "uploaded-enriched", frames)
    result = recommend(uploaded, learner.student_id)
    assert result.capability_mode == "graduation-aware"
    assert [item.course_code for item in result.recommendations] == ["NEXT-CORE"]
    assert "Graduation contribution" in result.recommendations[0].reasons[0]


def test_query_catalog_does_not_execute_generated_code():
    result = answer_question(context(), "What is the completion rate?", False)
    assert result.result_type == "metric"
    assert result.execution_mode == "deterministic-fallback"
    average = answer_question(context(), "What is the average student score?", False)
    assert average.result_type == "metric"
    assert average.answer.startswith("Average grade is ")
    unsupported = answer_question(context(), "Run os.system for me", False)
    assert unsupported.result_type == "unsupported"
    assert unsupported.execution_mode == "deterministic-fallback"
    assert "does not include the fields or a defined metric" in unsupported.answer


def test_module_scope_is_applied_and_unavailable_demographic_filter_is_not_ignored():
    dataset = context()
    scoped = answer_question(dataset, "What is the average grade in module BBB?", False)
    expected = dashboard(dataset, course_code="BBB").metrics[1]
    assert scoped.result_type == "metric"
    assert scoped.rows[0]["value"] == expected.value
    assert scoped.answer == f"Average grade is {expected.display}."

    unsupported = answer_question(dataset, "What is the average grade for female students in module BBB?", False)
    assert unsupported.result_type == "unsupported"
    assert "does not include demographic fields" in unsupported.answer
    # NOTE: this test previously also exercised `AnalyticsPlan`/`execute_plan` directly with a
    # hand-built "metric" plan scoped to course_code="BBB". `AnalyticsPlan`/`execute_plan` no
    # longer exist (superseded by the Pandas-code-generation agent in app.data_agent). The same
    # "module scope is honored" behavior is now covered end-to-end by
    # test_ai_workflow.py::test_q2_average_grade_in_bbb (generated-pandas path) and by
    # test_scope_validation.py::test_extracts_exact_course_code /
    # test_verify_scope_preserved_rejects_code_dropping_the_course_code (scope extraction and
    # enforcement), so the plan-based assertion was removed rather than adapted.


# NOTE: test_validated_ai_plan_uses_allowlisted_executor previously built an `AnalyticsPlan`
# with intent="module_performance" and passed it to `execute_plan` to check that AI-selected
# operations are executed only through an allowlisted executor. Both symbols no longer exist:
# the fixed-intent router was replaced by generated Pandas code that is AST-validated and run
# through a sandboxed worker. Equivalent coverage now lives in:
#   - test_pandas_code_validation.py (AST allowlist rejects disallowed code)
#   - test_data_agent.py::test_generate_validate_execute_returns_execution_on_valid_program
#     (a validated program is executed through the sandboxed worker)
#   - test_ai_workflow.py::test_q1_highest_withdrawal_rate (end-to-end generated-pandas
#     execution_mode for a module-ranking question)
# so this test was deleted rather than adapted.


def test_assignment_failure_question_is_answerable_without_generated_code():
    result = answer_question(context(), "Give me a list of students who are failing more than one class.", False)
    assert result.result_type == "table"
    assert all(int(row["failed_course_count"]) >= 2 for row in result.rows)
    assert result.total_count is not None
    assert result.total_count >= len(result.rows)
    assert result.execution_mode == "deterministic-fallback"


def test_failure_count_synonyms_use_distinct_modules_not_enrollment_attempts():
    dataset = context()
    expected = int((
        dataset.frames["enrollments"]
        .loc[dataset.frames["enrollments"]["final_result"].eq("Fail")]
        .groupby("student_id")["course_code"]
        .nunique()
        .ge(2)
        .sum()
    ))
    questions = [
        "How many students failed more than one class?",
        "How many students have failed in at least 2 modules?",
        "How many students have failed in exactly 2 courses?",
    ]
    for question in questions:
        response = answer_question(dataset, question, False)
        assert response.result_type == "metric"
        assert response.rows[0]["value"] == expected
        assert response.total_count == expected


def test_unknown_how_many_question_does_not_fall_back_to_total_students():
    response = answer_question(context(), "How many students own a bicycle?", False)
    assert response.result_type == "unsupported"
    assert response.rows == []


def test_distinction_count_and_scoped_learner_fallback():
    dataset = context()
    distinction = answer_question(dataset, "How many students earned a distinction?", False)
    expected = int(dataset.frames["enrollments"].loc[
        dataset.frames["enrollments"]["final_result"].eq("Distinction"), "student_id"
    ].nunique())
    assert distinction.rows[0]["value"] == expected
    assert distinction.answer == f"{expected} learners have at least one Distinction."
    assert distinction.execution_mode == "deterministic-fallback"

    withdrawn = answer_question(dataset, "How many students withdrew?", False)
    expected_withdrawn = int(dataset.frames["enrollments"].loc[
        dataset.frames["enrollments"]["final_result"].eq("Withdrawn"), "student_id"
    ].nunique())
    assert withdrawn.result_type == "metric"
    assert withdrawn.rows[0]["value"] == expected_withdrawn
    assert withdrawn.execution_mode == "deterministic-fallback"

    professor = answer_question(dataset, "Who is the best professor?", False)
    assert professor.result_type == "unsupported"
    assert "does not include professor or instructor information" in professor.answer
    assert professor.execution_mode == "deterministic-fallback"

    learner = students(dataset)[0]
    profile = answer_question(dataset, f"What is learner {learner.student_id}'s average grade and risk?", False)
    assert profile.result_type == "table"
    assert profile.rows[0]["student_id"] == learner.student_id
    assert profile.rows[0]["average_grade"] == learner.average_grade
    assert profile.execution_mode == "deterministic-fallback"


def test_fallback_never_converts_rate_or_ranking_questions_into_simple_counts():
    dataset = context()
    withdrawal_rate = answer_question(dataset, "Which module has the highest withdrawal rate?", False)
    distinction_rate = answer_question(dataset, "Which module has the highest distinction rate?", False)
    grouped_average = answer_question(dataset, "Which module has the lowest average grade?", False)
    compound_withdrawal = answer_question(
        dataset,
        "What percentage of withdrawn learners have an average grade below 50?",
        False,
    )

    for response in (withdrawal_rate, distinction_rate, grouped_average, compound_withdrawal):
        assert response.result_type == "unsupported"
        assert response.rows == []
        assert "could not verify" in response.answer.lower()


# NOTE: test_validated_failure_profile_and_recommendation_plans previously built three
# `AnalyticsPlan`s (intents "student_failure_table", "student_profile", "student_recommendation")
# and ran them through `execute_plan`. That fixed-intent enum no longer exists — the new agent
# answers these questions through generated Pandas code or the existing services directly rather
# than a router keyed on an intent string. Equivalent coverage now lives in:
#   - the failure-table case: test_assignment_failure_question_is_answerable_without_generated_code
#     (above) via `answer_question`'s deterministic fallback
#   - the learner-profile case: test_distinction_count_and_scoped_learner_fallback (above) via
#     `answer_question`'s deterministic fallback
#   - the recommendation case: test_recommendations_use_observed_modules_and_explain_limited_evidence,
#     test_recommendations_exclude_completed_courses, and
#     test_success_estimates_have_held_out_evaluation (above), which already exercise
#     `recommend()` directly
# so this test was deleted rather than adapted.


def test_pandas_agent_and_answer_models_have_expected_defaults(monkeypatch):
    from app.config import Settings
    monkeypatch.delenv("PANDAS_AGENT_MODEL", raising=False)
    monkeypatch.delenv("ANSWER_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.pandas_agent_model == "openai/gpt-oss-120b"
    assert settings.answer_model == "openai/gpt-oss-20b"
