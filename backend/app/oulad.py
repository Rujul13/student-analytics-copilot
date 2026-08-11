from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FILES = {
    "studentInfo.csv",
    "studentRegistration.csv",
    "courses.csv",
    "assessments.csv",
    "studentAssessment.csv",
}


class OuladValidationError(ValueError):
    pass


def validate_source_directory(source_dir: Path) -> None:
    missing = sorted(name for name in REQUIRED_FILES if not (source_dir / name).exists())
    if missing:
        raise OuladValidationError(f"Missing OULAD files: {', '.join(missing)}")


def _select_students(info: pd.DataFrame, limit_students: int) -> pd.Index:
    """Choose a fixed, stratified cohort with at least two academic histories."""
    history_counts = info.groupby("id_student").size()
    eligible = info[info["id_student"].isin(history_counts[history_counts >= 2].index)].copy()
    if len(history_counts) <= limit_students:
        return pd.Index(sorted(eligible["id_student"].unique()))

    # One deterministic stratum per learner, taken from their latest recorded history.
    profiles = (
        eligible.sort_values(["id_student", "code_presentation", "code_module"])
        .drop_duplicates("id_student", keep="last")
        [["id_student", "code_module", "final_result"]]
    )
    profiles["stratum"] = profiles["code_module"].astype(str) + "|" + profiles["final_result"].astype(str)
    shuffled = profiles.sample(frac=1, random_state=42)
    group_sizes = shuffled["stratum"].value_counts()
    quotas = (group_sizes / len(shuffled) * limit_students).round().clip(lower=1).astype(int)
    selected: list[int] = []
    for stratum, group in shuffled.groupby("stratum", sort=True):
        selected.extend(group.head(int(quotas[stratum]))["id_student"].tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < limit_students:
        remainder = shuffled[~shuffled["id_student"].isin(selected)]["id_student"].tolist()
        selected.extend(remainder[: limit_students - len(selected)])
    return pd.Index(sorted(selected[:limit_students]))


def transform_oulad(source_dir: Path, limit_students: int = 750) -> dict[str, pd.DataFrame]:
    """Transform canonical OULAD tables into the application's four-table model."""
    validate_source_directory(source_dir)
    info = pd.read_csv(source_dir / "studentInfo.csv")
    registration = pd.read_csv(source_dir / "studentRegistration.csv")
    source_courses = pd.read_csv(source_dir / "courses.csv")
    assessments = pd.read_csv(source_dir / "assessments.csv")
    results = pd.read_csv(source_dir / "studentAssessment.csv")

    eligible = _select_students(info, limit_students)
    info = info[info["id_student"].isin(eligible)].copy()
    registration = registration[registration["id_student"].isin(eligible)].copy()

    students = (
        info[["id_student"]]
        .drop_duplicates()
        .assign(
            student_id=lambda frame: "OULAD-" + frame["id_student"].astype(str),
            display_name=lambda frame: "Learner " + frame["id_student"].astype(str),
            program="Open University",
        )[["student_id", "display_name", "program"]]
    )

    course_names = {code: f"Module {code}" for code in sorted(source_courses["code_module"].unique())}
    courses = (
        source_courses.drop_duplicates("code_module")
        .assign(
            course_code=lambda frame: frame["code_module"],
            course_name=lambda frame: frame["code_module"].map(course_names),
            department="Open University",
            level=1,
            credits=30,
            offered_next_term=True,
            catalog_source="OULAD-derived",
        )[["course_code", "course_name", "department", "level", "credits", "offered_next_term"]]
    )

    base = info.merge(
        registration,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    enrollments = base.assign(
        student_id=lambda frame: "OULAD-" + frame["id_student"].astype(str),
        course_code=lambda frame: frame["code_module"],
        presentation=lambda frame: frame["code_presentation"],
        enrollment_id=lambda frame: frame["student_id"] + "-" + frame["course_code"] + "-" + frame["presentation"],
        status=lambda frame: frame["final_result"].map({"Withdrawn": "Withdrawn"}).fillna("Completed"),
        credits=lambda frame: frame["final_result"].isin(["Pass", "Distinction"]).astype(int) * 30,
    )[["enrollment_id", "student_id", "course_code", "presentation", "status", "final_result", "credits"]]

    assessment_scores = results.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "weight"]],
        on="id_assessment",
        how="inner",
    )
    assessment_scores = assessment_scores[assessment_scores["id_student"].isin(eligible)].copy()
    assessment_scores["score"] = pd.to_numeric(assessment_scores["score"], errors="coerce").fillna(0)
    assessment_scores["weight"] = pd.to_numeric(assessment_scores["weight"], errors="coerce").fillna(0)
    assessment_scores["weighted"] = assessment_scores["score"] * assessment_scores["weight"]
    grouped = assessment_scores.groupby(["id_student", "code_module", "code_presentation"], as_index=False).agg(
        weighted_sum=("weighted", "sum"), weight_sum=("weight", "sum"), score_mean=("score", "mean")
    )
    denominator = grouped["weight_sum"].astype(float).where(grouped["weight_sum"].ne(0))
    weighted_average = grouped["weighted_sum"].div(denominator)
    # GGG's continuous assessments carry zero official weight and its exam scores
    # are not present in studentAssessment.csv. Use the recorded assessment mean
    # instead of fabricating a zero grade when no positive weights are available.
    grouped["weighted_grade"] = weighted_average.fillna(grouped["score_mean"]).fillna(0).round(1)
    grades = grouped.assign(
        student_id=lambda frame: "OULAD-" + frame["id_student"].astype(str),
        enrollment_id=lambda frame: frame["student_id"] + "-" + frame["code_module"] + "-" + frame["code_presentation"],
    )[["enrollment_id", "weighted_grade"]]
    return {"students": students, "courses": courses, "enrollments": enrollments, "grades": grades}
