from __future__ import annotations

import re

from .analytics import dashboard, students
from .models import QueryResponse
from .repository import DatasetContext


def answer_question(context: DatasetContext, question: str, ai_enabled: bool) -> QueryResponse:
    """Safe MVP planner: maps supported intents to allowlisted Pandas-backed operations."""
    normalized = re.sub(r"\s+", " ", question.lower().strip())
    summary = dashboard(context)
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

