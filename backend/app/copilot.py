from __future__ import annotations

import re

from .analytics import dashboard, students
from .models import QueryResponse
from .repository import DatasetContext


def data_availability_answer(question: str) -> str:
    normalized = question.lower()
    if any(term in normalized for term in ["professor", "instructor", "faculty", "teacher"]):
        return "The dataset does not include professor or instructor information, so I cannot determine the best professor objectively."
    if "weather" in normalized:
        return "The dataset does not include weather information, so I cannot answer that from the available academic data."
    return "The available data does not include the fields or a defined metric needed to answer that question objectively."


def answer_question(context: DatasetContext, question: str, ai_enabled: bool) -> QueryResponse:
    """Maps supported questions to verified Pandas-backed operations."""
    normalized = re.sub(r"\s+", " ", question.lower().strip())
    known_codes = set(map(str, context.frames["enrollments"]["course_code"].dropna().unique()))
    mentioned_codes = [code for code in sorted(known_codes) if re.search(rf"\b{re.escape(code.lower())}\b", normalized)]
    if any(term in normalized for term in ["female", " male", "gender", "region", "disability", "age band"]):
        return QueryResponse(
            answer="The curated application dataset does not include demographic fields, so I cannot calculate that filtered result.",
            result_type="unsupported",
            calculation_trace=["Detected a requested filter that is not present in the application dataset"],
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
            answer=f"{count} learners have at least one Distinction.",
            result_type="metric",
            rows=[{"metric": "Learners with a Distinction", "value": count}],
            calculation_trace=["Matched the distinction metric", "Counted distinct learners with a Distinction outcome", f"Dataset version: {context.version}"],
            ai_used=False,
        )
    if "withdraw" in normalized or "withdrew" in normalized:
        enrollments = context.frames["enrollments"]
        count = int(enrollments.loc[enrollments["final_result"].eq("Withdrawn"), "student_id"].nunique())
        return QueryResponse(
            answer=f"{count} learners withdrew from at least one course.",
            result_type="metric",
            rows=[{"metric": "Learners with a withdrawal", "value": count}],
            calculation_trace=["Matched the withdrawal metric", "Counted distinct learners with at least one Withdrawn outcome", f"Dataset version: {context.version}"],
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
        "average student score": summary.metrics[1],
        "average score": summary.metrics[1],
        "completion rate": summary.metrics[2],
        "how many students": summary.metrics[0],
        "student count": summary.metrics[0],
        "high-risk": summary.metrics[3],
        "high risk": summary.metrics[3],
    }
    for phrase, metric in metric_map.items():
        if phrase in normalized:
            return QueryResponse(
                answer=f"{metric.label} is {metric.display}.",
                result_type="metric",
                rows=[{"metric": metric.label, "value": metric.value}],
                calculation_trace=["Matched a supported metric intent", "Executed an allowlisted aggregate", f"Dataset version: {context.version}"],
                ai_used=False,
            )
    if any(phrase in normalized for phrase in ["at risk", "at-risk", "lowest", "struggling"]):
        rows = [student.model_dump() for student in students(context)[:8]]
        return QueryResponse(
            answer=f"I found {sum(student['risk'] == 'High' for student in rows)} high-priority learners in the first eight priority records.",
            result_type="table",
            rows=rows,
            calculation_trace=["Matched the risk-ranking intent", "Calculated grade and withdrawal risk", "Sorted by risk band and grade", f"Dataset version: {context.version}"],
            ai_used=False,
        )
    return QueryResponse(
        answer=data_availability_answer(question),
        result_type="unsupported",
        calculation_trace=["No matching metric or available data field was found"],
        ai_used=False,
    )
