import pandas as pd
import pytest

from app.repository import DatasetContext
from app.scope_validation import extract_scope, missing_field_answer, verify_scope_preserved


@pytest.fixture
def context():
    frames = {
        "students": pd.DataFrame({"student_id": ["OULAD-242636", "S2"], "display_name": ["A", "B"], "program": ["P", "P"]}),
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


def test_extracts_exact_course_code(context):
    scope = extract_scope("What is the average grade in module BBB?", context)
    assert scope.course_codes == ["BBB"]


def test_extracts_exact_presentation(context):
    scope = extract_scope("Compare the pass rate for BBB between 2013J and 2014J.", context)
    assert set(scope.presentations) == {"2013J", "2014J"}
    assert scope.course_codes == ["BBB"]


def test_extracts_exact_learner_id(context):
    scope = extract_scope("Tell me about learner OULAD-242636.", context)
    assert scope.student_ids == ["OULAD-242636"]


def test_detects_highest_and_lowest(context):
    assert extract_scope("Which module has the highest withdrawal rate?", context).sort_direction == "highest"
    assert extract_scope("What about the lowest?", context).sort_direction == "lowest"


def test_detects_requested_count(context):
    scope = extract_scope("Which five learners have the lowest grades in CCC?", context)
    assert scope.requested_count == 5
    assert scope.course_codes == ["CCC"]


def test_detects_group_by_module(context):
    scope = extract_scope("Break down the completion rate by module.", context)
    assert scope.group_by_module is True


def test_detects_rate_language(context):
    scope = extract_scope("What is the withdrawal rate for CCC?", context)
    assert scope.wants_rate is True


def test_detects_missing_demographic_field(context):
    scope = extract_scope("What is the average grade for female students in module BBB?", context)
    assert scope.missing_fields == ["gender"]
    assert scope.course_codes == ["BBB"]


def test_missing_field_answer_names_the_field():
    answer = missing_field_answer(["gender"])
    assert "gender" in answer
    assert "does not include" in answer


def test_verify_scope_preserved_accepts_code_containing_the_course_code(context):
    scope = extract_scope("What is the average grade in module BBB?", context)
    code = "result = float(enrollments[enrollments['course_code'] == 'BBB']['weighted_grade'].mean())"
    assert verify_scope_preserved(scope, code, ["course_code", "weighted_grade"]) is None


def test_verify_scope_preserved_rejects_code_dropping_the_course_code(context):
    scope = extract_scope("What is the average grade in module BBB?", context)
    code = "result = float(enrollments['weighted_grade'].mean())"
    error = verify_scope_preserved(scope, code, ["weighted_grade"])
    assert error is not None
    assert "BBB" in error


def test_verify_scope_preserved_rejects_missing_learner_id(context):
    scope = extract_scope("Tell me about learner OULAD-242636.", context)
    code = "result = students.to_dict('records')"
    error = verify_scope_preserved(scope, code, [])
    assert error is not None
    assert "OULAD-242636" in error


def test_verify_scope_preserved_rejects_count_only_rate_question(context):
    scope = extract_scope("What is the withdrawal rate for CCC?", context)
    code = "result = int(enrollments[(enrollments['course_code']=='CCC') & (enrollments['final_result']=='Withdrawn')].shape[0])"
    error = verify_scope_preserved(scope, code, ["course_code", "final_result"])
    assert error is not None
