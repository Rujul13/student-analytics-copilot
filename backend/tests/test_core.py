from app.analytics import dashboard, students
from app.ai_workflow import AnalyticsPlan, execute_plan
from app.catalog import split_codes
from app.config import Settings
from app.copilot import answer_question
from app.recommendations import recommend
from app.repository import load_dataset


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
    )
    result = execute_plan(context(), "Which modules perform best?", plan)
    assert result.ai_used is True
    assert result.result_type == "table"
    assert len(result.rows) == 3
    assert set(result.rows[0]) == {"module", "average_grade", "records"}
