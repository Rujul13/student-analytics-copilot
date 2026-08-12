from __future__ import annotations

import re
from dataclasses import dataclass, field

from .repository import DatasetContext

DEMOGRAPHIC_TERMS: dict[str, str] = {
    "female": "gender",
    "male": "gender",
    "gender": "gender",
    "region": "region",
    "disability": "disability",
    "disabled": "disability",
    "age band": "age band",
    "age group": "age band",
    "ethnicity": "ethnicity",
}

_HIGHEST_TERMS = ("highest", "most", "greatest", "best", "top")
_LOWEST_TERMS = ("lowest", "least", "worst", "bottom")
_COUNT_CONTEXT_TERMS = ("lowest", "highest", "learners", "students", "top", "five", "learner")
_WORD_TO_NUMBER = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


@dataclass
class ScopeFilters:
    course_codes: list[str] = field(default_factory=list)
    presentations: list[str] = field(default_factory=list)
    student_ids: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    sort_direction: str | None = None
    requested_count: int | None = None
    group_by_module: bool = False
    wants_rate: bool = False
    missing_fields: list[str] = field(default_factory=list)


def extract_scope(question: str, context: DatasetContext) -> ScopeFilters:
    normalized = question.lower()
    known_codes = sorted(set(map(str, context.frames["enrollments"]["course_code"].dropna().unique())))
    known_presentations = sorted(set(map(str, context.frames["enrollments"]["presentation"].dropna().unique())))
    known_student_ids = sorted(set(map(str, context.frames["students"]["student_id"].unique())))
    known_outcomes = sorted(set(map(str, context.frames["enrollments"]["final_result"].dropna().unique())))

    scope = ScopeFilters()
    scope.course_codes = [code for code in known_codes if re.search(rf"\b{re.escape(code.lower())}\b", normalized)]
    scope.presentations = [pres for pres in known_presentations if pres.lower() in normalized]
    scope.student_ids = [sid for sid in known_student_ids if sid.lower() in normalized]
    scope.outcomes = [outcome for outcome in known_outcomes if outcome.lower() in normalized]

    if any(re.search(rf"\b{term}\b", normalized) for term in _HIGHEST_TERMS):
        scope.sort_direction = "highest"
    elif any(re.search(rf"\b{term}\b", normalized) for term in _LOWEST_TERMS):
        scope.sort_direction = "lowest"

    has_count_context = any(term in normalized for term in _COUNT_CONTEXT_TERMS)
    if has_count_context:
        digit_match = re.search(r"\b(\d{1,2})\b", normalized)
        if digit_match:
            scope.requested_count = int(digit_match.group(1))
        else:
            for word, number in _WORD_TO_NUMBER.items():
                if re.search(rf"\b{word}\b", normalized):
                    scope.requested_count = number
                    break

    scope.group_by_module = any(
        phrase in normalized for phrase in ("by module", "by course", "per module", "per course", "each module", "each course")
    )
    scope.wants_rate = any(term in normalized for term in ("rate", "percentage", "percent"))
    scope.missing_fields = sorted(
        {canonical for term, canonical in DEMOGRAPHIC_TERMS.items() if re.search(rf"\b{re.escape(term)}\b", normalized)}
    )
    return scope


def verify_scope_preserved(scope: ScopeFilters, code: str, referenced_columns: list[str]) -> str | None:
    for course_code in scope.course_codes:
        if course_code not in code:
            return f"The generated code did not apply the requested course module {course_code}."
    for presentation in scope.presentations:
        if presentation not in code:
            return f"The generated code did not apply the requested presentation {presentation}."
    for student_id in scope.student_ids:
        if student_id not in code:
            return f"The generated code did not apply the requested learner identifier {student_id}."
    if scope.group_by_module and "course_code" not in code and "course_code" not in referenced_columns:
        return "The generated code did not group results by module as requested."
    if scope.sort_direction == "highest":
        descending_markers = ("ascending=False", "idxmax", ".max(", "nlargest")
        if not any(marker in code for marker in descending_markers):
            return "The generated code did not apply a highest/descending ordering as requested."
    if scope.sort_direction == "lowest":
        ascending_markers = ("ascending=True", "idxmin", ".min(", "nsmallest")
        default_ascending_sort = "sort_values" in code and "ascending=False" not in code and (".head(" in code or "idxmin" in code)
        if not (any(marker in code for marker in ascending_markers) or default_ascending_sort):
            return "The generated code did not apply a lowest/ascending ordering as requested."
    if scope.wants_rate and "/" not in code and "rate" not in code.lower():
        return "The generated code returned a count rather than a calculated rate."
    return None


def missing_field_answer(missing_fields: list[str]) -> str:
    fields = ", ".join(missing_fields)
    return f"The current application dataset does not include {fields}, so I cannot calculate that filtered result."
