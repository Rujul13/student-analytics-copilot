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
    "professor": "professor or instructor information",
    "instructor": "professor or instructor information",
    "faculty": "professor or instructor information",
    "teacher": "professor or instructor information",
}

# Maps question-text terms (including common inflections) to the canonical `final_result`
# value they refer to. A plain substring/word-boundary check against `known_outcomes`
# directly ("pass" from the dataset value "Pass") either misses grammatical variants like
# "passed"/"withdrawal" (if word-boundary matching only the literal value) or false-positives
# on unrelated words like "encompassing" containing "pass" (if using bare substring
# matching without boundaries) - an explicit term list with `\b`-bounded matching avoids both.
OUTCOME_TERMS: dict[str, str] = {
    "pass": "Pass", "passed": "Pass", "passing": "Pass",
    "distinction": "Distinction",
    "fail": "Fail", "failed": "Fail", "failing": "Fail",
    "withdrew": "Withdrawn", "withdrawn": "Withdrawn", "withdrawal": "Withdrawn", "withdraw": "Withdrawn",
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
    wants_distinct_learners: bool = False
    wants_risk: bool = False
    missing_fields: list[str] = field(default_factory=list)


def extract_scope(question: str, context: DatasetContext) -> ScopeFilters:
    normalized = question.lower()
    known_codes = sorted(set(map(str, context.frames["enrollments"]["course_code"].dropna().unique())))
    known_presentations = sorted(set(map(str, context.frames["enrollments"]["presentation"].dropna().unique())))
    known_student_ids = sorted(set(map(str, context.frames["students"]["student_id"].unique())))
    known_outcomes = sorted(set(map(str, context.frames["enrollments"]["final_result"].dropna().unique())))

    scope = ScopeFilters()
    scope.course_codes = [code for code in known_codes if re.search(rf"\b{re.escape(code.lower())}\b", normalized)]
    # Word-boundary matching (not bare substring `in`) for presentations/student_ids/outcomes
    # too: without it, "encompassing" silently extracts the outcome "Pass" (it contains
    # "pass"), and a shorter learner ID that is numerically a prefix of a longer one in the
    # dataset (e.g. OULAD-11391 vs OULAD-113910 - realistic, since OULAD IDs are numeric)
    # gets extracted alongside the one actually named in the question. `\b` does not match
    # between two `\w` characters (digits count), so `\bOULAD-11391\b` correctly fails to
    # match inside "OULAD-113910" - the boundary the course_codes line above already relies on.
    scope.presentations = [pres for pres in known_presentations if re.search(rf"\b{re.escape(pres.lower())}\b", normalized)]
    scope.student_ids = [sid for sid in known_student_ids if re.search(rf"\b{re.escape(sid.lower())}\b", normalized)]
    # Administrators naturally say "learner 242636" even though the canonical ID stored in
    # the OULAD cohort is "OULAD-242636". Resolve a numeric/short alias only when it maps to
    # exactly one canonical learner, then require the generated code to use that full ID.
    learner_alias = re.search(r"\b(?:learner|student)\s+([a-z0-9-]+)\b", normalized)
    if not scope.student_ids and learner_alias:
        alias = learner_alias.group(1)
        matches = [sid for sid in known_student_ids if sid.lower() == alias or sid.lower().endswith(f"-{alias}")]
        if len(matches) == 1:
            scope.student_ids = matches
    scope.outcomes = sorted(
        {
            canonical
            for term, canonical in OUTCOME_TERMS.items()
            if canonical in known_outcomes and re.search(rf"\b{term}\b", normalized)
        }
    )

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
    # Word-boundary matching for "rate" specifically - a bare substring check makes
    # "generate", "separate", "moderate", "operate", "accelerate", etc. all falsely set
    # wants_rate, which then wrongly demands rate-shaped code from an otherwise-correct
    # program (e.g. "Can you generate a report of the average grade in BBB?").
    scope.wants_rate = any(re.search(rf"\b{term}\b", normalized) for term in ("rate", "percentage", "percent"))
    scope.wants_distinct_learners = bool(
        re.search(r"\bhow many\s+(?:students|learners)\b", normalized)
        or re.search(r"\bnumber of\s+(?:students|learners)\b", normalized)
    )
    scope.wants_risk = bool(re.search(r"\b(?:risk|academic-support priority|support priority)\b", normalized))
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
    for outcome in scope.outcomes:
        if outcome not in code:
            return f"The generated code did not apply the requested outcome filter '{outcome}'."
    if scope.requested_count is not None and str(scope.requested_count) not in code:
        return f"The generated code did not apply the requested result count of {scope.requested_count}."
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
    if scope.wants_rate and "/" not in code and "rate" not in code.lower() and ".mean(" not in code:
        # `.mean()` on a boolean condition (`(series == value).mean()`) is an idiomatic,
        # correct way to compute a rate/proportion in Pandas without an explicit `/` or the
        # literal word "rate" appearing anywhere in the code - accept it as evidence too.
        return "The generated code returned a count rather than a calculated rate."
    if scope.wants_distinct_learners:
        unique_markers = ("nunique(", ".unique(", "drop_duplicates(")
        grouped_learner_count = "groupby(" in code and "student_id" in code and any(
            marker in code for marker in ("len(", ".size", ".shape[0]")
        )
        if "student_id" not in code or not (any(marker in code for marker in unique_markers) or grouped_learner_count):
            return "The question asks for learners, but the generated code counted enrollment records instead of distinct student_id values."
    if scope.wants_risk:
        required_markers = ("risk", "weighted_grade", "withdraw")
        if not all(marker in code.lower() for marker in required_markers):
            return "The generated code did not calculate the requested academic-support risk from average grade and withdrawals."
    return None


def missing_field_answer(missing_fields: list[str]) -> str:
    fields = ", ".join(missing_fields)
    return f"The current application dataset does not include {fields}, so I cannot calculate that filtered result."
