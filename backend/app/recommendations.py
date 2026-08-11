from __future__ import annotations

import json

from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict

from .analytics import students
from .catalog import split_codes
from .models import Recommendation, RecommendationResponse, SuccessModelSummary
from .repository import DatasetContext
from .success_prediction import candidate_features, get_success_model


class NarrativeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    course_code: str
    narrative: str


class NarrativeBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanations: list[NarrativeItem]


class RankedNarrativeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    course_code: str
    narrative: str


class RankedNarrativeBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rankings: list[RankedNarrativeItem]


def _completed_and_current(context: DatasetContext, student_id: str) -> tuple[set[str], set[str]]:
    enrollments = context.frames["enrollments"]
    learner_rows = enrollments[enrollments["student_id"] == student_id]
    completed = set(learner_rows.loc[learner_rows["final_result"].isin(["Pass", "Distinction"]), "course_code"])
    current = set(learner_rows.loc[learner_rows["status"].eq("Active"), "course_code"])
    return completed, current


def _score(
    student_average: float,
    credits_earned: int,
    graded_enrollments: int,
    withdrawals: int,
    level: int,
    requirement_type: str,
    in_program: bool,
    department: str,
) -> tuple[int, int, int, int]:
    if in_program and requirement_type == "core":
        requirement_fit = 100
    elif in_program and requirement_type == "elective":
        requirement_fit = 76
    else:
        requirement_fit = 42

    if graded_enrollments == 0:
        performance_fit = 70 if department != "Student Success" else 78
    elif department == "Student Success":
        performance_fit = min(100, 62 + withdrawals * 12 + max(0, round((55 - student_average) * 0.8)))
    else:
        target_grade = 58 + level * 5
        performance_fit = max(55, min(100, round(100 - abs(student_average - target_grade) * 1.35)))
    expected_level = max(1, min(3, credits_earned // 60 + 1))
    progression_fit = max(35, 100 - abs(level - expected_level) * 28)
    total = round(requirement_fit * 0.45 + performance_fit * 0.30 + progression_fit * 0.25)
    return total, requirement_fit, performance_fit, progression_fit


def recommend(context: DatasetContext, student_id: str, limit: int = 3) -> RecommendationResponse:
    student_map = {student.student_id: student for student in students(context)}
    if student_id not in student_map:
        raise KeyError(student_id)
    student = student_map[student_id]
    completed, current = _completed_and_current(context, student_id)
    catalog = context.frames["courses"]
    candidates: list[Recommendation] = []
    success_model = get_success_model(context)

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
            student.graded_enrollments,
            student.withdrawals,
            int(row.level),
            row.requirement_type,
            in_program,
            row.department,
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
        evidence_reason = (
            f"Performance fit uses {student.graded_enrollments} graded module record(s) and the recorded {student.average_grade:.1f}% average"
            if student.graded_enrollments
            else "No graded assessment is recorded, so performance fit is neutral rather than treating missing evidence as a 0% grade"
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
                    evidence_reason,
                ],
                requirement_fit=requirement_fit,
                performance_fit=performance_fit,
                progression_fit=progression_fit,
                requirement_type=row.requirement_type,
                prerequisites_met=sorted(prerequisites),
                predicted_success_probability=(
                    success_model.predict(candidate_features(
                        context,
                        student.student_id,
                        student.average_grade,
                        student.withdrawals,
                        student.credits_earned,
                        int(row.level),
                    ))
                    if success_model else None
                ),
            )
        )

    candidates.sort(key=lambda item: (-item.score, item.course_code))
    graduation_aware = context.mode != "uploaded-canonical"
    return RecommendationResponse(
        student=student,
        capability_mode="graduation-aware" if graduation_aware else "performance-only",
        recommendations=candidates[:limit],
        ai_explanation_enabled=False,
        catalog_label="Fictional demo catalog enrichment" if context.mode.startswith("canonical") or context.mode == "development-fixture" else "User-provided catalog",
        ranking_mode="deterministic",
        success_model=SuccessModelSummary(**success_model.evaluation.__dict__) if success_model else None,
    )


async def add_ai_explanations(response: RecommendationResponse, api_key: str | None, model: str) -> RecommendationResponse:
    final_limit = min(3, len(response.recommendations))
    if not api_key or not response.recommendations:
        response.recommendations = response.recommendations[:final_limit]
        return response
    payload = {
        "student": {
            "program": response.student.program,
            "average_grade": response.student.average_grade,
            "credits_earned": response.student.credits_earned,
            "graded_enrollments": response.student.graded_enrollments,
            "withdrawals": response.student.withdrawals,
        },
        "recommendations": [
            {
                "course_code": item.course_code,
                "course_name": item.course_name,
                "score": item.score,
                "predicted_success_probability": item.predicted_success_probability,
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
                        "Rank every supplied eligible course from best to worst for this learner, then explain each in one concise sentence. "
                        "You may change order but must include every supplied course exactly once. Never add a course, change eligibility or scores, "
                        "or claim the fictional demo pathway is official. Use only supplied academic evidence and do not infer demographics. "
                        "Treat predicted success as an evaluated baseline estimate, not a guarantee."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "recommendation_rankings",
                    "strict": True,
                    "schema": RankedNarrativeBundle.model_json_schema(),
                },
            },
        )
        content = completion.choices[0].message.content
        bundle = RankedNarrativeBundle.model_validate_json(content or "{}")
        allowed = {item.course_code for item in response.recommendations}
        returned = [item.course_code for item in bundle.rankings]
        if len(returned) != len(set(returned)) or set(returned) != allowed:
            raise ValueError("LLM ranking must contain every eligible candidate exactly once")
        by_code = {item.course_code: item for item in response.recommendations}
        response.recommendations = [by_code[item.course_code] for item in bundle.rankings]
        for item in bundle.rankings:
            by_code[item.course_code].narrative = item.narrative
        response.recommendations = response.recommendations[:final_limit]
        response.ai_explanation_enabled = True
        response.ranking_mode = "hybrid-llm"
    except Exception:
        response.ai_explanation_enabled = False
        response.ranking_mode = "deterministic"
        response.recommendations = response.recommendations[:final_limit]
    return response
