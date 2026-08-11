from __future__ import annotations

import json

from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict

from .analytics import students
from .catalog import split_codes
from .models import Recommendation, RecommendationResponse
from .repository import DatasetContext


class NarrativeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    course_code: str
    narrative: str


class NarrativeBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanations: list[NarrativeItem]


def _completed_and_current(context: DatasetContext, student_id: str) -> tuple[set[str], set[str]]:
    enrollments = context.frames["enrollments"]
    learner_rows = enrollments[enrollments["student_id"] == student_id]
    completed = set(learner_rows.loc[learner_rows["final_result"].isin(["Pass", "Distinction"]), "course_code"])
    current = set(learner_rows.loc[learner_rows["status"].eq("Active"), "course_code"])
    return completed, current


def _score(student_average: float, credits_earned: int, level: int, requirement_type: str, in_program: bool) -> tuple[int, int, int, int]:
    if in_program and requirement_type == "core":
        requirement_fit = 100
    elif in_program and requirement_type == "elective":
        requirement_fit = 82
    else:
        requirement_fit = 42

    target_grade = 58 + level * 5
    performance_fit = max(35, min(100, round(100 - abs(student_average - target_grade) * 1.35)))
    expected_level = max(1, min(3, credits_earned // 60 + 1))
    progression_fit = max(35, 100 - abs(level - expected_level) * 28)
    total = round(requirement_fit * 0.45 + performance_fit * 0.30 + progression_fit * 0.25)
    return total, requirement_fit, performance_fit, progression_fit


def recommend(context: DatasetContext, student_id: str) -> RecommendationResponse:
    student_map = {student.student_id: student for student in students(context)}
    if student_id not in student_map:
        raise KeyError(student_id)
    student = student_map[student_id]
    completed, current = _completed_and_current(context, student_id)
    catalog = context.frames["courses"]
    candidates: list[Recommendation] = []

    for row in catalog.itertuples():
        prerequisites = split_codes(row.prerequisites)
        programs = split_codes(row.programs)
        eligible = (
            bool(row.offered_next_term)
            and row.course_code not in completed
            and row.course_code not in current
            and prerequisites.issubset(completed)
        )
        if not eligible:
            continue
        in_program = student.program in programs
        score, requirement_fit, performance_fit, progression_fit = _score(
            student.average_grade,
            student.credits_earned,
            int(row.level),
            row.requirement_type,
            in_program,
        )
        requirement_reason = (
            f"Counts as a {row.requirement_type} option for the {student.program} demo pathway"
            if in_program
            else "Available as an out-of-pathway elective"
        )
        prerequisite_reason = (
            f"Completed prerequisites: {', '.join(sorted(prerequisites))}"
            if prerequisites
            else "No prerequisite course is required"
        )
        candidates.append(
            Recommendation(
                course_code=row.course_code,
                course_name=row.course_name,
                score=score,
                confidence="High" if score >= 82 else "Medium" if score >= 66 else "Low",
                reasons=[
                    requirement_reason,
                    prerequisite_reason,
                    f"Level {row.level} fit is based on {student.credits_earned} earned credits",
                    f"Performance fit uses the recorded {student.average_grade:.1f}% average",
                ],
                requirement_fit=requirement_fit,
                performance_fit=performance_fit,
                progression_fit=progression_fit,
                requirement_type=row.requirement_type,
                prerequisites_met=sorted(prerequisites),
            )
        )

    candidates.sort(key=lambda item: (-item.score, item.course_code))
    graduation_aware = context.mode != "uploaded-canonical"
    return RecommendationResponse(
        student=student,
        capability_mode="graduation-aware" if graduation_aware else "performance-only",
        recommendations=candidates[:3],
        ai_explanation_enabled=False,
        catalog_label="Fictional demo catalog enrichment" if context.mode.startswith("canonical") or context.mode == "development-fixture" else "User-provided catalog",
    )


async def add_ai_explanations(response: RecommendationResponse, api_key: str | None, model: str) -> RecommendationResponse:
    if not api_key or not response.recommendations:
        return response
    payload = {
        "student": {
            "program": response.student.program,
            "average_grade": response.student.average_grade,
            "credits_earned": response.student.credits_earned,
        },
        "recommendations": [
            {
                "course_code": item.course_code,
                "course_name": item.course_name,
                "score": item.score,
                "reasons": item.reasons,
            }
            for item in response.recommendations
        ],
    }
    try:
        client = AsyncGroq(api_key=api_key, timeout=12, max_retries=1)
        completion = await client.chat.completions.create(
            model=model,
            temperature=0.2,
            reasoning_effort="low",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Explain each already-ranked course recommendation in one concise sentence. "
                        "Do not change ranking, eligibility, scores, or claim the fictional demo pathway is an official degree program. "
                        "Use only the supplied academic evidence and do not infer demographics."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "recommendation_explanations",
                    "strict": True,
                    "schema": NarrativeBundle.model_json_schema(),
                },
            },
        )
        content = completion.choices[0].message.content
        bundle = NarrativeBundle.model_validate_json(content or "{}")
        allowed = {item.course_code for item in response.recommendations}
        narratives = {item.course_code: item.narrative for item in bundle.explanations if item.course_code in allowed}
        for recommendation in response.recommendations:
            recommendation.narrative = narratives.get(recommendation.course_code)
        response.ai_explanation_enabled = bool(narratives)
    except Exception:
        response.ai_explanation_enabled = False
    return response
