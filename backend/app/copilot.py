from __future__ import annotations

import re

from .analytics import dashboard, students
from .models import QueryResponse
from .repository import DatasetContext
from .scope_validation import extract_scope


def data_availability_answer(question: str) -> str:
    normalized = question.lower()
    if any(term in normalized for term in ["professor", "instructor", "faculty", "teacher"]):
        return "The dataset does not include professor or instructor information, so I cannot determine the best professor objectively."
    if "weather" in normalized:
        return "The dataset does not include weather information, so I cannot answer that from the available academic data."
    return "The available data does not include the fields or a defined metric needed to answer that question objectively."


def unsupported_fallback_answer() -> QueryResponse:
    return QueryResponse(
        answer="I could not verify that calculation in fallback mode. Please retry when AI analytics is available.",
        result_type="unsupported",
        execution_mode="deterministic-fallback",
        ai_used=False,
    )


def answer_question(context: DatasetContext, question: str, ai_enabled: bool) -> QueryResponse:
    """Maps supported questions to verified Pandas-backed operations."""
    normalized = re.sub(r"\s+", " ", question.lower().strip())
    known_codes = set(map(str, context.frames["enrollments"]["course_code"].dropna().unique()))
    mentioned_codes = [code for code in sorted(known_codes) if re.search(rf"\b{re.escape(code.lower())}\b", normalized)]
    semantic_outcome_terms = {
        "dropout": set(context.semantic.failure_outcomes) | set(context.semantic.withdrawal_outcomes),
        "dropped out": set(context.semantic.failure_outcomes) | set(context.semantic.withdrawal_outcomes),
        "graduate": set(context.semantic.success_outcomes),
        "graduated": set(context.semantic.success_outcomes),
        "graduation": set(context.semantic.success_outcomes),
        "enrolled": {"Enrolled"},
    }
    for term, outcomes in semantic_outcome_terms.items() if not context.semantic.capabilities.individual_course_history else []:
        if term in normalized and any(phrase in normalized for phrase in ["how many", "number of", "count"]):
            enrollments = context.frames["enrollments"]
            count = int(enrollments.loc[enrollments["final_result"].isin(outcomes), "student_id"].nunique())
            label = "dropped out" if "drop" in term else "graduated" if "graduat" in term else "are enrolled"
            learner_word = "learner" if count == 1 else "learners"
            return QueryResponse(
                answer=f"{count:,} {learner_word} {label}.",
                result_type="metric",
                rows=[{"metric": f"Learners who {label}", "value": count}],
                execution_mode="deterministic-fallback",
                ai_used=False,
            )
    if not context.semantic.capabilities.individual_course_history and "average" in normalized:
        semester_columns = {
            "first semester grade": "semester_1_grade",
            "1st semester grade": "semester_1_grade",
            "semester 1 grade": "semester_1_grade",
            "second semester grade": "semester_2_grade",
            "2nd semester grade": "semester_2_grade",
            "semester 2 grade": "semester_2_grade",
        }
        for phrase, column in semester_columns.items():
            if phrase in normalized and column in context.frames["enrollments"].columns:
                value = float(context.frames["enrollments"][column].dropna().mean())
                label = "First semester average grade" if column == "semester_1_grade" else "Second semester average grade"
                return QueryResponse(
                    answer=f"{label} is {value:.1f}%.",
                    result_type="metric",
                    rows=[{"metric": label, "value": round(value, 1)}],
                    execution_mode="deterministic-fallback",
                    ai_used=False,
                )
        if any(phrase in normalized for phrase in ("by degree program", "per degree program", "each degree program")):
            enrollments = context.frames["enrollments"]
            grades = context.frames["grades"]
            courses = context.frames["courses"][["course_code", "course_name"]]
            grouped = (
                enrollments.merge(grades, on="enrollment_id", how="left")
                .groupby("course_code", as_index=False)["weighted_grade"]
                .mean()
                .merge(courses, on="course_code", how="left")
                .sort_values("course_name")
            )
            rows = [
                {"degree_program": row.course_name, "average_grade": round(float(row.weighted_grade), 1)}
                for row in grouped.itertuples()
            ]
            return QueryResponse(
                answer=f"Average grades were calculated for {len(rows)} degree programs.",
                result_type="table",
                rows=rows,
                total_count=len(rows),
                execution_mode="deterministic-fallback",
                ai_used=False,
            )
    if any(term in normalized for term in ["female", " male", "gender", "region", "disability", "age band"]):
        return QueryResponse(
            answer="The curated application dataset does not include demographic fields, so I cannot calculate that filtered result.",
            result_type="unsupported",
            execution_mode="deterministic-fallback",
            ai_used=False,
        )
    course_code = mentioned_codes[0] if len(mentioned_codes) == 1 else None
    summary = dashboard(context, course_code=course_code)
    learner_match = re.search(r"(?:learner|student)\s+((?:[a-z]+-)?[a-z]*\d[a-z0-9-]*)", normalized)
    if learner_match:
        learner_id = learner_match.group(1)
        learner = next((item for item in students(context) if item.student_id.lower() == learner_id), None)
        if learner and any(phrase in normalized for phrase in ["average", "grade", "risk", "profile"]):
            return QueryResponse(
                answer=f"{learner.display_name} has a {learner.average_grade:.1f}% average and a {learner.risk.lower()} academic-support priority.",
                result_type="table",
                rows=[learner.model_dump()],
                execution_mode="deterministic-fallback",
                ai_used=False,
            )
        return QueryResponse(
            answer="I recognized a learner-specific question, but the requested learner operation is not available in fallback mode.",
            result_type="unsupported",
            execution_mode="deterministic-fallback",
            ai_used=False,
        )
    if "distinction" in normalized:
        if any(term in normalized for term in ["rate", "percent", "highest", "lowest", "best", "worst", "by module", "per module"]):
            return unsupported_fallback_answer()
        enrollments = context.frames["enrollments"]
        count = int(enrollments.loc[enrollments["final_result"].eq("Distinction"), "student_id"].nunique())
        return QueryResponse(
            answer=f"{count} learners have at least one Distinction.",
            result_type="metric",
            rows=[{"metric": "Learners with a Distinction", "value": count}],
            execution_mode="deterministic-fallback",
            ai_used=False,
        )
    if "withdraw" in normalized or "withdrew" in normalized:
        if any(term in normalized for term in [
            "rate", "percent", "highest", "lowest", "best", "worst", "by module", "per module",
            "average", "below", "above", "compare", "comparison",
        ]):
            return unsupported_fallback_answer()
        enrollments = context.frames["enrollments"]
        count = int(enrollments.loc[enrollments["final_result"].eq("Withdrawn"), "student_id"].nunique())
        return QueryResponse(
            answer=f"{count} learners withdrew from at least one course.",
            result_type="metric",
            rows=[{"metric": "Learners with a withdrawal", "value": count}],
            execution_mode="deterministic-fallback",
            ai_used=False,
        )
    failure_scope = extract_scope(question, context)
    if "fail" in normalized and failure_scope.failed_course_threshold is not None:
        failed = context.frames["enrollments"]
        failed = failed[failed["final_result"].eq("Fail")]
        grouped = failed.groupby("student_id")["course_code"].agg(lambda values: sorted(set(map(str, values))))
        learner_map = {item.student_id: item for item in students(context)}
        rows = []
        for student_id, course_codes in grouped.items():
            threshold = failure_scope.failed_course_threshold
            comparison = failure_scope.failed_course_comparison
            matches = (
                len(course_codes) > threshold if comparison == "more_than"
                else len(course_codes) >= threshold if comparison == "at_least"
                else len(course_codes) == threshold
            )
            if not matches or student_id not in learner_map:
                continue
            learner = learner_map[student_id]
            rows.append({
                "student_id": learner.student_id,
                "display_name": learner.display_name,
                "failed_course_count": len(course_codes),
                "failed_courses": ", ".join(course_codes),
                "average_grade": learner.average_grade,
                "risk": learner.risk,
            })
        rows.sort(key=lambda row: (-int(row["failed_course_count"]), float(row["average_grade"])))
        comparison_text = {
            "more_than": f"more than {threshold}",
            "at_least": f"at least {threshold}",
            "exactly": f"exactly {threshold}",
        }[comparison or "exactly"]
        asks_for_list = bool(re.search(r"\b(?:which|who|list|show|give me)\b", normalized))
        if not asks_for_list:
            return QueryResponse(
                answer=f"{len(rows):,} learners failed {comparison_text} distinct course modules.",
                result_type="metric",
                rows=[{"metric": f"Learners who failed {comparison_text} distinct modules", "value": len(rows)}],
                total_count=len(rows),
                execution_mode="deterministic-fallback",
                ai_used=False,
            )
        preview = rows[:20]
        return QueryResponse(
            answer=f"I found {len(rows):,} learners who failed {comparison_text} distinct course modules.",
            result_type="table",
            rows=preview,
            total_count=len(rows),
            rows_truncated=len(preview) < len(rows),
            execution_mode="deterministic-fallback",
            ai_used=False,
        )
    metric_map = {
        "average grade": summary.metrics[1],
        "average student score": summary.metrics[1],
        "average score": summary.metrics[1],
        "completion rate": summary.metrics[2],
        "student count": summary.metrics[0],
        "total students": summary.metrics[0],
        "total learners": summary.metrics[0],
        "high-risk": summary.metrics[3],
        "high risk": summary.metrics[3],
    }
    for phrase, metric in metric_map.items():
        if phrase in normalized:
            if course_code is None and any(term in normalized for term in [
                "by module", "per module", "each module", "highest", "lowest", "best", "worst", "top", "bottom",
                "compare", "comparison", "below", "above",
            ]):
                return unsupported_fallback_answer()
            return QueryResponse(
                answer=f"{metric.label} is {metric.display}.",
                result_type="metric",
                rows=[{"metric": metric.label, "value": metric.value}],
                execution_mode="deterministic-fallback",
                ai_used=False,
            )
    if any(phrase in normalized for phrase in ["at risk", "at-risk", "lowest", "struggling"]):
        rows = [student.model_dump() for student in students(context)[:8]]
        return QueryResponse(
            answer=f"I found {sum(student['risk'] == 'High' for student in rows)} high-priority learners in the first eight priority records.",
            result_type="table",
            rows=rows,
            execution_mode="deterministic-fallback",
            ai_used=False,
        )
    return QueryResponse(
        answer=data_availability_answer(question),
        result_type="unsupported",
        execution_mode="deterministic-fallback",
        ai_used=False,
    )
