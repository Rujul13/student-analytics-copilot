from __future__ import annotations

import json

from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict

from .analytics import students
from .models import Recommendation, RecommendationResponse, SuccessModelSummary
from .repository import DatasetContext
from .success_prediction import candidate_features, get_success_model, module_empirical_stats


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


def _evidence_strength(graded_enrollments: int) -> str:
    if graded_enrollments <= 1:
        return "Limited"
    if graded_enrollments <= 3:
        return "Moderate"
    return "Strong"


def recommend(context: DatasetContext, student_id: str, limit: int = 3) -> RecommendationResponse:
    student_map = {student.student_id: student for student in students(context)}
    if student_id not in student_map:
        raise KeyError(student_id)
    student = student_map[student_id]
    completed, current = _completed_and_current(context, student_id)
    catalog = context.frames["courses"]
    historical_codes = set(map(str, context.frames["enrollments"]["course_code"].dropna().unique()))
    candidates: list[Recommendation] = []
    success_model = get_success_model(context)

    for row in catalog.itertuples():
        # OULAD contains historical module outcomes, not a future-offering or degree-audit feed.
        # Recommend only authentic module codes with observed outcomes and label availability as unknown.
        eligible = row.course_code in historical_codes and row.course_code not in completed and row.course_code not in current
        if not eligible:
            continue
        module = module_empirical_stats(context, str(row.course_code))
        probability = (
            success_model.predict(candidate_features(
                context,
                student.student_id,
                student.average_grade,
                student.withdrawals,
                student.credits_earned,
                str(row.course_code),
            ), str(row.course_code))
            if success_model else round(float(module["pass_rate"]) * 100, 1)
        )
        course_history_fit = round(float(module["pass_rate"]) * 100)
        learner_fit = round(probability)
        evidence_fit = min(100, 35 + min(student.graded_enrollments, 5) * 10 + min(int(module["records"]), 200) // 4)
        score = round(learner_fit * 0.60 + course_history_fit * 0.25 + evidence_fit * 0.15)
        strength = _evidence_strength(student.graded_enrollments)
        success_basis = (
            f"Limited learner evidence: the estimate relies mainly on {int(module['records'])} historical {row.course_code} records."
            if student.graded_enrollments <= 1
            else f"Combines {student.graded_enrollments} graded learner records with {int(module['records'])} historical {row.course_code} records."
        )
        candidates.append(
            Recommendation(
                course_code=row.course_code,
                course_name=row.course_name,
                score=score,
                evidence_strength=strength,
                reasons=[
                    f"Historical pass rate: {float(module['pass_rate']) * 100:.1f}% across {int(module['records'])} learner records",
                    f"Historical average grade: {float(module['average_grade']):.1f}%",
                    f"Historical withdrawal rate: {float(module['withdrawal_rate']) * 100:.1f}%",
                    success_basis,
                    "OULAD does not provide future availability or prerequisite data; verify those before enrollment.",
                ],
                requirement_fit=course_history_fit,
                performance_fit=learner_fit,
                progression_fit=evidence_fit,
                course_pass_rate=round(float(module["pass_rate"]) * 100, 1),
                course_withdrawal_rate=round(float(module["withdrawal_rate"]) * 100, 1),
                course_average_grade=round(float(module["average_grade"]), 1),
                historical_records=int(module["records"]),
                success_basis=success_basis,
                predicted_success_probability=probability,
            )
        )

    candidates.sort(key=lambda item: (-item.score, item.course_code))
    return RecommendationResponse(
        student=student,
        capability_mode="historical-performance",
        recommendations=candidates[:limit],
        ai_explanation_enabled=False,
        catalog_label="OULAD historical modules; future availability unknown",
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
                        "or claim a module is available in a future term. Use only supplied academic and historical module evidence. "
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
