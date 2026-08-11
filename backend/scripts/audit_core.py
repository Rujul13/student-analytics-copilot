from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pandas as pd


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.ai_workflow import AnalyticsPlan, execute_plan  # noqa: E402
from app.analytics import dashboard, risk_label, students  # noqa: E402
from app.catalog import split_codes  # noqa: E402
from app.config import Settings  # noqa: E402
from app.recommendations import recommend  # noqa: E402
from app.repository import load_dataset  # noqa: E402


def main() -> None:
    context = load_dataset(Settings())
    joined = context.frames["enrollments"].merge(context.frames["grades"], on="enrollment_id", how="left")
    summary = dashboard(context)

    assert context.name == "OULAD Lite"
    assert len(context.frames["students"]) == 750
    assert len(context.frames["enrollments"]) == 1548
    assert len(context.frames["grades"]) == 1182
    assert len(context.frames["courses"]) == 12
    assert summary.metrics[0].value == len(context.frames["students"])
    assert summary.metrics[1].value == round(float(joined["weighted_grade"].mean()), 1)
    expected_completion = round(float(joined["final_result"].isin(["Pass", "Distinction"]).mean() * 100), 1)
    assert summary.metrics[2].value == expected_completion
    assert all(0 <= point.value <= 100 for point in summary.modules)
    assert {point.label for point in summary.modules} == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}

    rollup = joined.groupby("student_id").agg(
        average_grade=("weighted_grade", "mean"),
        withdrawals=("final_result", lambda values: int((values == "Withdrawn").sum())),
    )
    expected_high_risk = int(rollup.apply(lambda row: risk_label(row.average_grade, row.withdrawals), axis=1).eq("High").sum())
    assert summary.metrics[3].value == expected_high_risk

    learner_summaries = students(context)
    representatives = {}
    for learner in learner_summaries:
        representatives.setdefault((learner.program, learner.risk), learner)
    catalog = context.frames["courses"].set_index("course_code")
    eligibility_checks = 0
    for learner in representatives.values():
        result = recommend(context, learner.student_id)
        rows = context.frames["enrollments"]
        learner_rows = rows[rows["student_id"] == learner.student_id]
        completed = set(learner_rows.loc[learner_rows["final_result"].isin(["Pass", "Distinction"]), "course_code"])
        current = set(learner_rows.loc[learner_rows["status"].eq("Active"), "course_code"])
        assert [item.score for item in result.recommendations] == sorted([item.score for item in result.recommendations], reverse=True)
        for item in result.recommendations:
            course = catalog.loc[item.course_code]
            assert bool(course["offered_next_term"])
            assert item.course_code not in completed | current
            assert split_codes(course["prerequisites"]).issubset(completed)
            assert 0 <= item.score <= 100
            eligibility_checks += 1

    source = inspect.getsource(recommend).lower()
    assert not any(field in source for field in ["gender", "region", "disability", "deprivation", "age_band"])

    plans = [
        AnalyticsPlan(intent="metric", metric="completion_rate", risk_band=None, limit=None, sort_descending=None),
        AnalyticsPlan(intent="student_risk_table", metric=None, risk_band="High", limit=5, sort_descending=False),
        AnalyticsPlan(intent="module_performance", metric=None, risk_band=None, limit=3, sort_descending=True),
    ]
    plan_results = [execute_plan(context, "audit", plan) for plan in plans]
    assert [result.result_type for result in plan_results] == ["metric", "table", "table"]
    assert len(plan_results[1].rows) == 5
    assert len(plan_results[2].rows) == 3

    report = {
        "dataset": context.name,
        "version": context.version,
        "canonical_rows": {name: len(frame) for name, frame in context.frames.items()},
        "dashboard": {
            "average_grade": summary.metrics[1].display,
            "completion_rate": summary.metrics[2].display,
            "high_risk_learners": summary.metrics[3].value,
        },
        "recommendation_profiles_checked": len(representatives),
        "recommendation_items_checked": eligibility_checks,
        "query_executor_contracts_checked": len(plan_results),
        "protected_attributes_used_in_recommendation_code": False,
        "status": "pass",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

