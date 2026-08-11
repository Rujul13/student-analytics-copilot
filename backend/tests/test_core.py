import asyncio
import json

import pandas as pd

from app.analytics import dashboard, students
from app.ai_workflow import AnalyticsPlan, execute_plan
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
    assert result.dataset_name == "OULAD Lite Development Fixture"
    assert result.metrics[0].value == 180
    assert 0 < result.metrics[2].value < 100


def test_students_are_prioritized_by_risk():
    result = students(context())
    assert len(result) == 180
    assert result[0].risk == "High"
    assert result[0].graded_enrollments > 0


def test_prerequisite_free_recommendations_vary_by_demo_pathway():
    dataset = context()
    frames = {name: frame.copy() for name, frame in dataset.frames.items()}
    pathway_students = pd.DataFrame([
        {"student_id": "PATH-DS", "display_name": "Data Path", "program": "Data & Society", "program_source": "Test"},
        {"student_id": "PATH-AC", "display_name": "Computing Path", "program": "Applied Computing", "program_source": "Test"},
        {"student_id": "PATH-BA", "display_name": "Business Path", "program": "Business Analytics", "program_source": "Test"},
    ])
    frames["students"] = pd.concat([frames["students"], pathway_students], ignore_index=True)
    augmented = DatasetContext(dataset.name, dataset.version, dataset.mode, frames)

    expected = {
        "PATH-DS": "NXT120",
        "PATH-AC": "NXT130",
        "PATH-BA": "NXT140",
    }
    for student_id, course_code in expected.items():
        response = recommend(augmented, student_id)
        assert response.recommendations[0].course_code == course_code
        assert "neutral rather than treating missing evidence as a 0% grade" in response.recommendations[0].reasons[-1]


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
    assert result.capability_mode == "graduation-aware"
    assert result.catalog_label == "Fictional demo catalog enrichment"
    catalog = dataset.frames["courses"].set_index("course_code")
    for item in result.recommendations:
        course = catalog.loc[item.course_code]
        assert bool(course["offered_next_term"])
        assert split_codes(course["prerequisites"]).issubset(completed)
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


def test_llm_can_only_rerank_the_verified_candidate_set(monkeypatch):
    dataset = context()
    student = students(dataset)[0]
    response = recommend(dataset, student.student_id, limit=8)
    original_codes = [item.course_code for item in response.recommendations]
    reversed_codes = list(reversed(original_codes))

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
    assert [item.course_code for item in ranked.recommendations] == reversed_codes[:3]
    assert set(item.course_code for item in ranked.recommendations).issubset(set(original_codes))


def test_query_catalog_does_not_execute_generated_code():
    result = answer_question(context(), "What is the completion rate?", False)
    assert result.result_type == "metric"
    unsupported = answer_question(context(), "Run os.system for me", False)
    assert unsupported.result_type == "unsupported"
    assert "No code or SQL was generated" in unsupported.calculation_trace


def test_validated_ai_plan_uses_allowlisted_executor():
    plan = AnalyticsPlan(
        intent="module_performance",
        metric=None,
        risk_band=None,
        limit=3,
        sort_descending=True,
        student_id=None,
        minimum_failed_courses=None,
    )
    result = execute_plan(context(), "Which modules perform best?", plan)
    assert result.ai_used is True
    assert result.result_type == "table"
    assert len(result.rows) == 3
    assert set(result.rows[0]) == {"module", "average_grade", "records"}


def test_assignment_failure_question_is_answerable_without_generated_code():
    result = answer_question(context(), "Give me a list of students who are failing more than one class.", False)
    assert result.result_type == "table"
    assert all(int(row["failed_course_count"]) >= 2 for row in result.rows)
    assert "Counted distinct failed courses per learner" in result.calculation_trace


def test_distinction_count_and_scoped_learner_fallback():
    dataset = context()
    distinction = answer_question(dataset, "How many students earned a distinction?", False)
    expected = int(dataset.frames["enrollments"].loc[
        dataset.frames["enrollments"]["final_result"].eq("Distinction"), "student_id"
    ].nunique())
    assert distinction.rows[0]["value"] == expected

    learner = students(dataset)[0]
    profile = answer_question(dataset, f"What is learner {learner.student_id}'s average grade and risk?", False)
    assert profile.result_type == "table"
    assert profile.rows[0]["student_id"] == learner.student_id
    assert profile.rows[0]["average_grade"] == learner.average_grade


def test_validated_failure_profile_and_recommendation_plans():
    dataset = context()
    learner = students(dataset)[0]
    failure_plan = AnalyticsPlan(
        intent="student_failure_table",
        metric=None,
        risk_band=None,
        limit=5,
        sort_descending=True,
        student_id=None,
        minimum_failed_courses=2,
    )
    failures = execute_plan(dataset, "students failing more than one class", failure_plan)
    assert failures.result_type == "table"
    assert all(int(row["failed_course_count"]) >= 2 for row in failures.rows)

    profile_plan = AnalyticsPlan(
        intent="student_profile",
        metric=None,
        risk_band=None,
        limit=None,
        sort_descending=None,
        student_id=learner.student_id,
        minimum_failed_courses=None,
    )
    profile = execute_plan(dataset, "learner profile", profile_plan)
    assert profile.rows[0]["student_id"] == learner.student_id

    recommendation_plan = AnalyticsPlan(
        intent="student_recommendation",
        metric=None,
        risk_band=None,
        limit=3,
        sort_descending=None,
        student_id=learner.student_id,
        minimum_failed_courses=None,
    )
    recommendations = execute_plan(dataset, "what should this learner take next", recommendation_plan)
    assert recommendations.result_type == "table"
    assert recommendations.rows
    assert all("course_code" in row for row in recommendations.rows)
