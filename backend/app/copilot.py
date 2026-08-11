from __future__ import annotations

import re

from .analytics import dashboard, students
from .models import QueryResponse
from .repository import DatasetContext


def answer_question(context: DatasetContext, question: str, ai_enabled: bool) -> QueryResponse:
    """Safe MVP planner: maps supported intents to allowlisted Pandas-backed operations."""
    normalized = re.sub(r"\s+", " ", question.lower().strip())
    summary = dashboard(context)
    learner_match = re.search(r"(?:learner|student)\s+([a-z0-9-]+)", normalized)
    if learner_match:
        learner_id = learner_match.group(1)
        learner = next((item for item in students(context) if item.student_id.lower() == learner_id), None)
        if learner and any(phrase in normalized for phrase in ["average", "grade", "risk", "profile"]):
            return QueryResponse(
                answer=f"{learner.display_name} has a {learner.average_grade:.1f}% average and is classified as {learner.risk} risk.",
                result_type="table",
                rows=[learner.model_dump()],
                calculation_trace=["Matched a scoped learner-profile intent", "Executed an allowlisted learner lookup", f"Dataset version: {context.version}"],
                ai_used=False,
            )
        return QueryResponse(
            answer="I recognized a learner-specific question, but the requested learner operation is not available in fallback mode.",
            result_type="unsupported",
            calculation_trace=["Preserved the learner scope", "No cohort metric was substituted", "No code or SQL was generated"],
            ai_used=False,
        )
    if "distinction" in normalized:
        enrollments = context.frames["enrollments"]
        count = int(enrollments.loc[enrollments["final_result"].eq("Distinction"), "student_id"].nunique())
        return QueryResponse(
            answer=f"{count} learners have at least one Distinction outcome in the active dataset.",
            result_type="metric",
            rows=[{"metric": "Learners with a Distinction", "value": count}],
            calculation_trace=["Matched the distinction metric", "Counted distinct learners with a Distinction outcome", f"Dataset version: {context.version}"],
            ai_used=False,
        )
    if "fail" in normalized and any(phrase in normalized for phrase in ["more than one", "multiple", "at least two"]):
        failed = context.frames["enrollments"]
        failed = failed[failed["final_result"].eq("Fail")]
        grouped = failed.groupby("student_id")["course_code"].agg(lambda values: sorted(set(map(str, values))))
        learner_map = {item.student_id: item for item in students(context)}
        rows = []
        for student_id, course_codes in grouped.items():
            if len(course_codes) < 2 or student_id not in learner_map:
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
        return QueryResponse(
            answer=f"I found {len(rows)} learners who failed more than one course.",
            result_type="table",
            rows=rows[:20],
            calculation_trace=["Matched the multi-course failure intent", "Counted distinct failed courses per learner", f"Dataset version: {context.version}"],
            ai_used=False,
        )
    metric_map = {
        "average grade": summary.metrics[1],
        "completion rate": summary.metrics[2],
        "how many students": summary.metrics[0],
        "student count": summary.metrics[0],
        "high-risk": summary.metrics[3],
        "high risk": summary.metrics[3],
    }
    for phrase, metric in metric_map.items():
        if phrase in normalized:
            return QueryResponse(
                answer=f"{metric.label} is {metric.display} for the active dataset.",
                result_type="metric",
                rows=[{"metric": metric.label, "value": metric.value}],
                calculation_trace=["Matched a supported metric intent", "Executed an allowlisted aggregate", f"Dataset version: {context.version}"],
                ai_used=False,
            )
    if any(phrase in normalized for phrase in ["at risk", "at-risk", "lowest", "struggling"]):
        rows = [student.model_dump() for student in students(context)[:8]]
        return QueryResponse(
            answer=f"I found {sum(student['risk'] == 'High' for student in rows)} high-risk learners in the first eight priority records.",
            result_type="table",
            rows=rows,
            calculation_trace=["Matched the risk-ranking intent", "Calculated grade and withdrawal risk", "Sorted by risk band and grade", f"Dataset version: {context.version}"],
            ai_used=False,
        )
    return QueryResponse(
        answer="That question is outside the current safe query catalog. Try asking about average grade, completion rate, student count, or at-risk learners.",
        result_type="unsupported",
        calculation_trace=["No allowlisted intent matched", "No code or SQL was generated"],
        ai_used=False,
    )
