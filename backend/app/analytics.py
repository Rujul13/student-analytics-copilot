from __future__ import annotations

import pandas as pd

from .models import DashboardFilterOptions, DashboardResponse, DashboardSpecificationModel, DistributionPoint, Metric, StudentPage, StudentSummary
from .repository import DatasetContext


def _joined(context: DatasetContext) -> pd.DataFrame:
    return context.frames["enrollments"].merge(context.frames["grades"], on="enrollment_id", how="left")


def risk_label(average_grade: float, withdrawals: int = 0) -> str:
    if average_grade < 50 or withdrawals >= 2:
        return "High"
    if average_grade < 65 or withdrawals == 1:
        return "Medium"
    return "Low"


def dashboard(
    context: DatasetContext,
    course_code: str | None = None,
    presentation: str | None = None,
    final_result: str | None = None,
) -> DashboardResponse:
    source_enrollments = context.frames["enrollments"]
    enrollments = source_enrollments
    if course_code:
        enrollments = enrollments[enrollments["course_code"].eq(course_code)]
    if presentation:
        enrollments = enrollments[enrollments["presentation"].eq(presentation)]
    if final_result:
        enrollments = enrollments[enrollments["final_result"].eq(final_result)]
    joined = enrollments.merge(context.frames["grades"], on="enrollment_id", how="left")
    success_outcomes = set(context.semantic.success_outcomes)
    withdrawal_outcomes = set(context.semantic.withdrawal_outcomes)
    student_count = int(enrollments["student_id"].nunique()) if any([course_code, presentation, final_result]) else len(context.frames["students"])
    average_grade = float(joined["weighted_grade"].mean()) if len(joined) else 0.0
    completion_rate = float(joined["final_result"].isin(success_outcomes).mean() * 100) if len(joined) else 0.0
    withdrawal_rate = float(joined["final_result"].isin(withdrawal_outcomes).mean() * 100) if len(joined) else 0.0
    student_rollup = joined.groupby("student_id").agg(
        average_grade=("weighted_grade", "mean"), withdrawals=("final_result", lambda values: int(values.isin(withdrawal_outcomes).sum()))
    ) if len(joined) else pd.DataFrame(columns=["average_grade", "withdrawals"])
    if len(student_rollup):
        student_rollup["risk"] = student_rollup.apply(lambda row: risk_label(row.average_grade, row.withdrawals), axis=1)
    else:
        student_rollup["risk"] = pd.Series(dtype="object")
    high_risk = int(student_rollup["risk"].eq("High").sum())

    outcomes = joined["final_result"].value_counts()
    module_rollup = joined.groupby("course_code")["weighted_grade"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    risks = student_rollup["risk"].value_counts()
    program_outcome_mode = not context.semantic.capabilities.individual_course_history
    metrics = [
        Metric(label="Students", value=student_count, display=f"{student_count:,}", delta="Current cohort", direction="neutral"),
        Metric(label="Average grade", value=round(average_grade, 1), display=f"{average_grade:.1f}%", delta="Across recorded assessments", direction="up"),
        Metric(label="Graduation rate" if program_outcome_mode else "Completion rate", value=round(completion_rate, 1), display=f"{completion_rate:.1f}%", delta="Graduate" if program_outcome_mode else "Pass or distinction", direction="up"),
        Metric(label="Dropout rate" if program_outcome_mode else "High-priority learners", value=round(withdrawal_rate, 1) if program_outcome_mode else high_risk, display=f"{withdrawal_rate:.1f}%" if program_outcome_mode else str(high_risk), delta="Recorded dropout outcome" if program_outcome_mode else f"{withdrawal_rate:.1f}% withdrawal rate", direction="down"),
    ]
    raw_course_labels = context.frames["courses"].set_index("course_code")["course_name"].astype(str).to_dict()
    course_labels = (
        raw_course_labels
        if not context.semantic.capabilities.individual_course_history
        else {str(code): str(code) for code in raw_course_labels}
    )
    return DashboardResponse(
        dataset_name=context.name,
        dataset_version=context.version,
        mode=context.mode,
        metrics=metrics,
        outcomes=[DistributionPoint(label=str(label), value=round(count / len(joined) * 100, 1), count=int(count)) for label, count in outcomes.items()] if len(joined) else [],
        modules=[DistributionPoint(label=course_labels.get(code, str(code)), key=str(code), value=round(float(row["mean"]), 1), count=int(row["count"])) for code, row in module_rollup.iterrows()],
        risk_bands=[DistributionPoint(label=label, value=round(count / max(student_count, 1) * 100, 1), count=int(count)) for label, count in risks.items()],
        filter_options=DashboardFilterOptions(
            courses=sorted(map(str, source_enrollments["course_code"].dropna().unique())),
            presentations=sorted(map(str, source_enrollments["presentation"].dropna().unique())),
            outcomes=sorted(map(str, source_enrollments["final_result"].dropna().unique())),
            course_labels={str(code): label for code, label in course_labels.items()},
        ),
        specification=DashboardSpecificationModel(**{
            **context.semantic.dashboard.__dict__,
            "enabled_filters": list(context.semantic.dashboard.enabled_filters),
        }),
    )


def _student_rollup(context: DatasetContext) -> pd.DataFrame:
    joined = _joined(context)
    rollup = joined.groupby("student_id").agg(
        average_grade=("weighted_grade", "mean"),
        credits_earned=("credits", "sum"),
        graded_enrollments=("weighted_grade", "count"),
        withdrawals=("final_result", lambda values: int(values.isin(set(context.semantic.withdrawal_outcomes)).sum())),
    ).reset_index()
    merged = context.frames["students"].merge(rollup, on="student_id", how="left")
    for column in ["average_grade", "credits_earned", "graded_enrollments", "withdrawals"]:
        merged[column] = merged[column].fillna(0)
    merged["risk"] = merged.apply(lambda row: risk_label(float(row.average_grade), int(row.withdrawals)), axis=1)
    merged["status"] = merged["graded_enrollments"].map(lambda value: "Graded evidence available" if int(value) else "No graded assessment")
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    merged["risk_order"] = merged["risk"].map(risk_order)
    merged["no_graded_assessment"] = merged["graded_enrollments"].eq(0)
    return merged.sort_values(
        ["risk_order", "no_graded_assessment", "average_grade", "student_id"],
        ascending=[True, True, True, True],
    )


def _student_models(frame: pd.DataFrame) -> list[StudentSummary]:
    result: list[StudentSummary] = []
    for row in frame.itertuples():
        result.append(StudentSummary(
            student_id=row.student_id,
            display_name=row.display_name,
            program=row.program,
            average_grade=round(float(row.average_grade), 1),
            credits_earned=int(row.credits_earned),
            graded_enrollments=int(row.graded_enrollments),
            withdrawals=int(row.withdrawals),
            risk=row.risk,
            status=row.status,
        ))
    return result


def students(context: DatasetContext) -> list[StudentSummary]:
    return _student_models(_student_rollup(context))


def student_page(
    context: DatasetContext,
    search: str | None = None,
    risk: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> StudentPage:
    frame = _student_rollup(context)
    if risk and risk != "All":
        frame = frame[frame["risk"].eq(risk)]
    if search and search.strip():
        needle = search.strip().casefold()
        haystack = (frame["display_name"].astype(str) + " " + frame["student_id"].astype(str)).str.casefold()
        frame = frame[haystack.str.contains(needle, regex=False)]
    total = len(frame)
    page = frame.iloc[offset:offset + limit]
    return StudentPage(items=_student_models(page), total=total, offset=offset, limit=limit)
