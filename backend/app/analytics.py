from __future__ import annotations

import pandas as pd

from .models import DashboardResponse, DistributionPoint, Metric, StudentSummary
from .repository import DatasetContext


def _joined(context: DatasetContext) -> pd.DataFrame:
    return context.frames["enrollments"].merge(context.frames["grades"], on="enrollment_id", how="left")


def risk_label(average_grade: float, withdrawals: int = 0) -> str:
    if average_grade < 50 or withdrawals >= 2:
        return "High"
    if average_grade < 65 or withdrawals == 1:
        return "Medium"
    return "Low"


def dashboard(context: DatasetContext) -> DashboardResponse:
    joined = _joined(context)
    student_count = len(context.frames["students"])
    average_grade = float(joined["weighted_grade"].mean())
    completion_rate = float(joined["final_result"].isin(["Pass", "Distinction"]).mean() * 100)
    withdrawal_rate = float(joined["final_result"].eq("Withdrawn").mean() * 100)
    student_rollup = joined.groupby("student_id").agg(
        average_grade=("weighted_grade", "mean"), withdrawals=("final_result", lambda values: int((values == "Withdrawn").sum()))
    )
    student_rollup["risk"] = student_rollup.apply(lambda row: risk_label(row.average_grade, row.withdrawals), axis=1)
    high_risk = int(student_rollup["risk"].eq("High").sum())

    outcomes = joined["final_result"].value_counts()
    module_rollup = joined.groupby("course_code")["weighted_grade"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    risks = student_rollup["risk"].value_counts()
    return DashboardResponse(
        dataset_name=context.name,
        dataset_version=context.version,
        mode=context.mode,
        metrics=[
            Metric(label="Students", value=student_count, display=f"{student_count:,}", delta="Active dataset", direction="neutral"),
            Metric(label="Average grade", value=round(average_grade, 1), display=f"{average_grade:.1f}%", delta="Across recorded assessments", direction="up"),
            Metric(label="Completion rate", value=round(completion_rate, 1), display=f"{completion_rate:.1f}%", delta="Pass or distinction", direction="up"),
            Metric(label="High-risk learners", value=high_risk, display=str(high_risk), delta=f"{withdrawal_rate:.1f}% withdrawal rate", direction="down"),
        ],
        outcomes=[DistributionPoint(label=str(label), value=round(count / len(joined) * 100, 1), count=int(count)) for label, count in outcomes.items()],
        modules=[DistributionPoint(label=str(code), value=round(float(row["mean"]), 1), count=int(row["count"])) for code, row in module_rollup.iterrows()],
        risk_bands=[DistributionPoint(label=label, value=round(count / student_count * 100, 1), count=int(count)) for label, count in risks.items()],
    )


def students(context: DatasetContext) -> list[StudentSummary]:
    joined = _joined(context)
    rollup = joined.groupby("student_id").agg(
        average_grade=("weighted_grade", "mean"),
        credits_earned=("credits", "sum"),
        withdrawals=("final_result", lambda values: int((values == "Withdrawn").sum())),
    ).reset_index()
    merged = context.frames["students"].merge(rollup, on="student_id", how="left").fillna(0)
    result = []
    for row in merged.itertuples():
        result.append(StudentSummary(
            student_id=row.student_id,
            display_name=row.display_name,
            program=row.program,
            average_grade=round(float(row.average_grade), 1),
            credits_earned=int(row.credits_earned),
            risk=risk_label(float(row.average_grade), int(row.withdrawals)),
            status="Active",
        ))
    return sorted(result, key=lambda student: (student.risk != "High", student.average_grade))

