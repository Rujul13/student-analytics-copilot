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


def test_resolves_a_unique_numeric_learner_alias_to_the_canonical_id(context):
    scope = extract_scope("What is learner 242636's average grade and risk?", context)
    assert scope.student_ids == ["OULAD-242636"]
    assert scope.wants_risk is True


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


def test_verify_scope_preserved_accepts_boolean_mean_as_a_rate_computation(context):
    scope = extract_scope("What is the withdrawal rate for CCC?", context)
    code = "subset = enrollments[enrollments['course_code'] == 'CCC']\nresult = float((subset['final_result'] == 'Withdrawn').mean())"
    assert verify_scope_preserved(scope, code, ["course_code", "final_result"]) is None


def test_wants_rate_is_not_triggered_by_the_word_generate(context):
    scope = extract_scope("Can you generate a report of the average grade in BBB?", context)
    assert scope.wants_rate is False
    code = "result = float(enrollments.merge(grades, on='enrollment_id')[enrollments['course_code']=='BBB']['weighted_grade'].mean())"
    assert verify_scope_preserved(scope, code, ["course_code", "weighted_grade"]) is None


def test_outcome_is_not_extracted_from_an_unrelated_word_containing_it(context):
    scope = extract_scope("Give me a summary encompassing all learners in BBB", context)
    assert scope.outcomes == []


def test_verify_scope_preserved_rejects_code_dropping_the_requested_outcome(context):
    scope = extract_scope("How many students passed BBB?", context)
    assert scope.outcomes == ["Pass"]
    code = "result = int(enrollments[enrollments['course_code'] == 'BBB'].shape[0])"
    error = verify_scope_preserved(scope, code, ["course_code"])
    assert error is not None
    assert "Pass" in error


def test_verify_scope_preserved_accepts_code_that_applies_the_requested_outcome(context):
    scope = extract_scope("How many students passed BBB?", context)
    code = "result = int(enrollments.loc[(enrollments['course_code'] == 'BBB') & (enrollments['final_result'] == 'Pass'), 'student_id'].nunique())"
    assert verify_scope_preserved(scope, code, ["course_code", "final_result", "student_id"]) is None


def test_verify_scope_preserved_rejects_code_ignoring_the_requested_count(context):
    scope = extract_scope("Which five learners have the lowest grades in CCC?", context)
    assert scope.requested_count == 5
    code = (
        "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
        "result = merged[merged['course_code'] == 'CCC'].sort_values('weighted_grade').head(20)\n"
    )
    error = verify_scope_preserved(scope, code, ["course_code", "weighted_grade"])
    assert error is not None
    assert "5" in error


def test_verify_scope_preserved_accepts_code_that_applies_the_requested_count(context):
    scope = extract_scope("Which five learners have the lowest grades in CCC?", context)
    code = (
        "merged = enrollments.merge(grades, on='enrollment_id', how='left')\n"
        "result = merged[merged['course_code'] == 'CCC'].sort_values('weighted_grade').head(5)\n"
    )
    assert verify_scope_preserved(scope, code, ["course_code", "weighted_grade"]) is None


def test_rejects_enrollment_row_count_for_a_distinct_student_question(context):
    scope = extract_scope("How many students withdrew?", context)
    assert scope.wants_distinct_learners is True
    code = "result = int(enrollments[enrollments['final_result'] == 'Withdrawn'].shape[0])"
    error = verify_scope_preserved(scope, code, ["student_id", "final_result"])
    assert error is not None
    assert "distinct" in error


def test_accepts_distinct_student_count_for_a_student_question(context):
    scope = extract_scope("How many students withdrew?", context)
    code = "result = int(enrollments.loc[enrollments['final_result'] == 'Withdrawn', 'student_id'].nunique())"
    assert verify_scope_preserved(scope, code, ["student_id", "final_result"]) is None


def test_risk_question_requires_the_defined_grade_and_withdrawal_calculation(context):
    scope = extract_scope("What is learner 242636's average grade and risk?", context)
    wrong_code = "result = students[students['student_id'] == 'OULAD-242636']"
    error = verify_scope_preserved(scope, wrong_code, ["student_id"])
    assert error is not None
    assert "risk" in error.lower()

    valid_code = (
        "joined = enrollments.merge(grades, on='enrollment_id', how='left')\n"
        "learner = joined[joined['student_id'] == 'OULAD-242636']\n"
        "average_grade = float(learner['weighted_grade'].mean())\n"
        "withdrawals = int((learner['final_result'] == 'Withdrawn').sum())\n"
        "risk = 'High' if average_grade < 50 or withdrawals >= 2 else ('Medium' if average_grade < 65 or withdrawals == 1 else 'Low')\n"
        "result = {'student_id': 'OULAD-242636', 'average_grade': average_grade, 'risk': risk}\n"
    )
    assert verify_scope_preserved(scope, valid_code, ["student_id", "weighted_grade", "final_result"]) is None


def test_extract_scope_does_not_confuse_a_learner_id_with_its_numeric_prefix():
    # OULAD IDs are numeric, so a shorter ID can be a literal prefix of a longer one in the
    # dataset (e.g. OULAD-11391 is a prefix of OULAD-113910). A bare substring check would
    # incorrectly extract both when only the longer one is named in the question; the `\b`
    # word-boundary regex must not match between two digits, so it must extract only the one
    # actually present.
    frames = {
        "students": pd.DataFrame(
            {"student_id": ["OULAD-11391", "OULAD-113910"], "display_name": ["A", "B"], "program": ["P", "P"]}
        ),
        "courses": pd.DataFrame({"course_code": ["BBB"], "course_name": ["X"]}),
        "enrollments": pd.DataFrame(
            {
                "enrollment_id": ["E1", "E2"],
                "student_id": ["OULAD-11391", "OULAD-113910"],
                "course_code": ["BBB", "BBB"],
                "presentation": ["2014J", "2014J"],
                "final_result": ["Pass", "Pass"],
            }
        ),
        "grades": pd.DataFrame({"enrollment_id": ["E1", "E2"], "weighted_grade": [70.0, 80.0]}),
    }
    id_collision_context = DatasetContext("Test", "v1", "test", frames)
    scope = extract_scope("Tell me about learner OULAD-113910.", id_collision_context)
    assert scope.student_ids == ["OULAD-113910"]
