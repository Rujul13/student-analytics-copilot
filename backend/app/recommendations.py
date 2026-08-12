from __future__ import annotations

import json

from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict

from .analytics import students
from .catalog import split_codes
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


def recommend(context: DatasetContext, student_id: str, limit: int | None = None) -> RecommendationResponse:
    if not context.semantic.capabilities.historical_recommendations:
        raise ValueError("The active dataset does not support individual course recommendations")
    student_map = {student.student_id: student for student in students(context)}
    if student_id not in student_map:
        raise KeyError(student_id)
    student = student_map[student_id]
    completed, current = _completed_and_current(context, student_id)
    catalog = context.frames["courses"]
    historical_codes = set(map(str, context.frames["enrollments"]["course_code"].dropna().unique()))
    candidates: list[Recommendation] = []
    success_model = get_success_model(context)
    # Preserve compatibility for callers constructing an enriched DatasetContext
    # directly while making the semantic manifest authoritative for imported data.
    graduation_aware = context.semantic.capabilities.graduation_aware_recommendations or context.mode == "uploaded-enriched"

    for row in catalog.itertuples():
        prerequisites = split_codes(getattr(row, "prerequisites", ""))
        programs = split_codes(getattr(row, "programs", ""))
        program_match = not programs or student.program in programs
        prerequisites_met = prerequisites.issubset(completed)
        if graduation_aware:
            eligible = bool(getattr(row, "offered_next_term", False)) and program_match and prerequisites_met and row.course_code not in completed and row.course_code not in current
        else:
            # OULAD contains historical outcomes, not a future-offering or degree-audit feed.
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
        course_history_fit = (
            100 if graduation_aware and str(getattr(row, "requirement_type", "elective")).lower() == "core"
            else 75 if graduation_aware else round(float(module["pass_rate"]) * 100)
        )
        learner_fit = round(probability)
        evidence_fit = min(100, 35 + min(student.graded_enrollments, 5) * 10 + min(int(module["records"]), 200) // 4)
        score = round(learner_fit * 0.55 + course_history_fit * 0.30 + evidence_fit * 0.15) if graduation_aware else round(learner_fit * 0.60 + course_history_fit * 0.25 + evidence_fit * 0.15)
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
                    *([f"Graduation contribution: {str(getattr(row, 'requirement_type', 'elective')).title()} for {student.program}", "All listed prerequisites are completed", "Listed as offered next term in the uploaded catalog"] if graduation_aware else []),
                    f"Historical pass rate: {float(module['pass_rate']) * 100:.1f}% across {int(module['records'])} learner records",
                    f"Historical average grade: {float(module['average_grade']):.1f}%",
                    f"Historical withdrawal rate: {float(module['withdrawal_rate']) * 100:.1f}%",
                    success_basis,
                    *( [] if graduation_aware else ["OULAD does not provide future availability or prerequisite data; verify those before enrollment."] ),
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
        capability_mode="graduation-aware" if graduation_aware else "historical-performance",
        recommendations=candidates if limit is None else candidates[:limit],
        ai_explanation_enabled=False,
        catalog_label="Uploaded next-term catalog with program and requirement metadata" if graduation_aware else "OULAD historical modules; future availability unknown",
        ranking_mode="deterministic",
        evaluated_candidates=len(candidates),
        selection_summary=f"All {len(candidates)} eligible modules were evaluated from verified learner and module evidence.",
        success_model=SuccessModelSummary(**success_model.evaluation.__dict__) if success_model else None,
    )


async def add_ai_explanations(response: RecommendationResponse, api_key: str | None, model: str) -> RecommendationResponse:
    final_limit = min(3, len(response.recommendations))
    if not api_key or not response.recommendations:
        response.recommendations = response.recommendations[:final_limit]
        return response
    payload = {
        "capability_mode": response.capability_mode,
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
                        f"Select exactly {final_limit} courses from the complete supplied eligible set, rank them from best to worst for this learner, "
                        "and explain each in one concise sentence. Balance predicted success, the learner's academic history, module outcomes, and evidence strength. "
                        "Never add a course, change eligibility or scores, "
                        "or make any availability, prerequisite, or graduation claim not present in the supplied evidence. Use only supplied academic and module evidence. "
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
        if len(returned) != final_limit or len(returned) != len(set(returned)) or not set(returned).issubset(allowed):
            raise ValueError("LLM ranking must contain the required number of distinct eligible candidates")
        by_code = {item.course_code: item for item in response.recommendations}
        response.recommendations = [by_code[item.course_code] for item in bundle.rankings]
        for item in bundle.rankings:
            by_code[item.course_code].narrative = item.narrative
        response.recommendations = response.recommendations[:final_limit]
        response.ai_explanation_enabled = True
        response.ranking_mode = "hybrid-llm"
        response.selection_summary = f"AI selected and ranked 3 from all {response.evaluated_candidates} verified eligible modules."
    except Exception:
        response.ai_explanation_enabled = False
        response.ranking_mode = "deterministic"
        response.selection_summary = f"The top 3 were selected deterministically from all {response.evaluated_candidates} verified eligible modules because live AI selection was unavailable."
        response.recommendations = response.recommendations[:final_limit]
    return response
