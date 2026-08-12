from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .repository import DatasetContext


FEATURE_NAMES = (
    "prior_average",
    "prior_success_rate",
    "prior_withdrawals",
    "prior_credits",
    "previous_attempts",
    "current_study_load",
    "education_level",
    "prior_vle_clicks",
    "prior_vle_active_days",
    "module_pass_rate",
    "module_average_grade",
    "module_withdrawal_rate",
)


@dataclass(frozen=True)
class SuccessEvaluation:
    model_name: str
    training_records: int
    test_records: int
    accuracy: float
    brier_score: float
    roc_auc: float
    dataset_version: str


@dataclass
class SuccessModel:
    weights: np.ndarray
    evaluation: SuccessEvaluation
    module_codes: tuple[str, ...]

    def predict(self, features: list[float], course_code: str) -> float:
        module_flags = [1.0 if code == course_code else 0.0 for code in self.module_codes]
        vector = np.asarray([1.0, *features, *module_flags], dtype=float)
        probability = 1.0 / (1.0 + np.exp(-float(vector @ self.weights)))
        return round(float(np.clip(probability, 0.01, 0.99)) * 100, 1)


_MODEL_CACHE: dict[str, SuccessModel | None] = {}


def _packaged_model(dataset_version: str) -> SuccessModel | None:
    path = Path(__file__).resolve().parents[1] / "data" / "full_processed" / "success_model.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation", {}).get("dataset_version") != dataset_version:
        return None
    return SuccessModel(
        weights=np.asarray(payload["weights"], dtype=float),
        evaluation=SuccessEvaluation(**payload["evaluation"]),
        module_codes=tuple(payload["module_codes"]),
    )


def save_success_model(model: SuccessModel, path: Path) -> None:
    path.write_text(json.dumps({
        "weights": model.weights.tolist(),
        "module_codes": list(model.module_codes),
        "feature_names": list(FEATURE_NAMES),
        "evaluation": model.evaluation.__dict__,
    }, indent=2) + "\n", encoding="utf-8")


EDUCATION_LEVELS = {
    "No Formal quals": 0.0,
    "Lower Than A Level": 0.2,
    "A Level or Equivalent": 0.4,
    "HE Qualification": 0.65,
    "Post Graduate Qualification": 1.0,
}


def _student_bucket(student_id: str) -> int:
    digest = hashlib.sha256(student_id.encode()).digest()
    return int.from_bytes(digest[:2], "big") % 5


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    wins = sum(float(value > other) + 0.5 * float(value == other) for value in positives for other in negatives)
    return wins / (len(positives) * len(negatives))


def _feature_rows(context: DatasetContext) -> pd.DataFrame:
    enrollments = context.frames["enrollments"].copy()
    grades = context.frames["grades"][["enrollment_id", "weighted_grade"]]
    joined = enrollments.merge(grades, on="enrollment_id", how="left")
    joined["presentation_order"] = joined["presentation"].astype(str)
    rows: list[dict[str, float | int | str]] = []
    student_state: dict[str, dict[str, float]] = {}
    module_state: dict[str, dict[str, float]] = {}
    for presentation, current_term in joined.sort_values("presentation_order").groupby("presentation_order", sort=True):
        for current in current_term.itertuples():
            student_id = str(current.student_id)
            course_code = str(current.course_code)
            learner = student_state.get(student_id, {"records": 0, "grade_sum": 0, "grade_count": 0, "successes": 0, "withdrawals": 0, "credits": 0, "vle_clicks": 0, "vle_days": 0})
            module = module_state.get(course_code, {"records": 0, "grade_sum": 0, "grade_count": 0, "successes": 0, "withdrawals": 0})
            rows.append({
                "student_id": student_id,
                "course_code": course_code,
                "presentation": str(presentation),
                "prior_average": learner["grade_sum"] / learner["grade_count"] / 100 if learner["grade_count"] else 0.5,
                "prior_success_rate": learner["successes"] / learner["records"] if learner["records"] else 0.5,
                "prior_withdrawals": min(learner["withdrawals"], 3) / 3,
                "prior_credits": min(learner["credits"], 180) / 180,
                "previous_attempts": min(float(getattr(current, "previous_attempts", 0) or 0), 3) / 3,
                "current_study_load": min(float(getattr(current, "studied_credits", 60) or 60), 240) / 240,
                "education_level": EDUCATION_LEVELS.get(str(getattr(current, "highest_education", "")), 0.4),
                "prior_vle_clicks": min(learner["vle_clicks"] / max(learner["records"], 1), 10000) / 10000,
                "prior_vle_active_days": min(learner["vle_days"] / max(learner["records"], 1), 250) / 250,
                "module_pass_rate": module["successes"] / module["records"] if module["records"] else 0.5,
                "module_average_grade": module["grade_sum"] / module["grade_count"] / 100 if module["grade_count"] else 0.5,
                "module_withdrawal_rate": module["withdrawals"] / module["records"] if module["records"] else 0.25,
                "success": int(current.final_result in {"Pass", "Distinction"}),
            })
        # Update state only after the complete presentation, preventing same-term leakage.
        for current in current_term.itertuples():
            student_id = str(current.student_id)
            course_code = str(current.course_code)
            learner = student_state.setdefault(student_id, {"records": 0, "grade_sum": 0, "grade_count": 0, "successes": 0, "withdrawals": 0, "credits": 0, "vle_clicks": 0, "vle_days": 0})
            module = module_state.setdefault(course_code, {"records": 0, "grade_sum": 0, "grade_count": 0, "successes": 0, "withdrawals": 0})
            grade = getattr(current, "weighted_grade", np.nan)
            success = int(current.final_result in {"Pass", "Distinction"})
            withdrawn = int(current.final_result == "Withdrawn")
            for state in (learner, module):
                state["records"] += 1
                state["successes"] += success
                state["withdrawals"] += withdrawn
                if pd.notna(grade):
                    state["grade_sum"] += float(grade)
                    state["grade_count"] += 1
            learner["credits"] += float(current.credits) if success else 0
            learner["vle_clicks"] += float(getattr(current, "vle_total_clicks", 0) or 0)
            learner["vle_days"] += float(getattr(current, "vle_active_days", 0) or 0)
    return pd.DataFrame(rows)


def get_success_model(context: DatasetContext) -> SuccessModel | None:
    if context.version in _MODEL_CACHE:
        return _MODEL_CACHE[context.version]
    packaged = _packaged_model(context.version)
    if packaged is not None:
        _MODEL_CACHE[context.version] = packaged
        return packaged
    rows = _feature_rows(context)
    if len(rows) < 100 or rows["success"].nunique() < 2:
        _MODEL_CACHE[context.version] = None
        return None
    # The latest presentation is a true temporal holdout; no outcome from it is used
    # when constructing its learner or module features.
    test_mask = (
        rows["presentation"].eq(rows["presentation"].max())
        if rows["presentation"].nunique() > 1
        else rows["student_id"].map(_student_bucket).eq(0)
    )
    train = rows[~test_mask]
    test = rows[test_mask]
    if len(test) < 20 or train["success"].nunique() < 2 or test["success"].nunique() < 2:
        _MODEL_CACHE[context.version] = None
        return None

    module_codes = tuple(sorted(map(str, rows["course_code"].unique())))

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        base = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
        flags = np.column_stack([frame["course_code"].eq(code).to_numpy(dtype=float) for code in module_codes])
        return np.column_stack([base, flags])

    x_train = matrix(train)
    y_train = train["success"].to_numpy(dtype=float)
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    weights = np.zeros(x_train.shape[1], dtype=float)
    for _ in range(1200):
        predictions = 1 / (1 + np.exp(-np.clip(x_train @ weights, -20, 20)))
        gradient = (x_train.T @ (predictions - y_train)) / len(x_train)
        regularization = np.r_[0.0, weights[1:]] * 0.01
        weights -= 0.08 * (gradient + regularization)

    x_test = np.column_stack([np.ones(len(test)), matrix(test)])
    y_test = test["success"].to_numpy(dtype=float)
    probabilities = 1 / (1 + np.exp(-np.clip(x_test @ weights, -20, 20)))
    evaluation = SuccessEvaluation(
        model_name="Temporal learner-and-module logistic baseline",
        training_records=len(train),
        test_records=len(test),
        accuracy=round(float(((probabilities >= 0.5) == y_test).mean()), 3),
        brier_score=round(float(np.mean((probabilities - y_test) ** 2)), 3),
        roc_auc=round(float(_auc(y_test, probabilities)), 3),
        dataset_version=context.version,
    )
    model = SuccessModel(weights=weights, evaluation=evaluation, module_codes=module_codes)
    _MODEL_CACHE[context.version] = model
    return model


def module_empirical_stats(context: DatasetContext, course_code: str) -> dict[str, float | int]:
    enrollments = context.frames["enrollments"]
    grades = context.frames["grades"][["enrollment_id", "weighted_grade"]]
    records = enrollments[enrollments["course_code"].eq(course_code)].merge(grades, on="enrollment_id", how="left")
    recorded_grades = records["weighted_grade"].dropna()
    return {
        "records": int(len(records)),
        "pass_rate": float(records["final_result"].isin(["Pass", "Distinction"]).mean()) if len(records) else 0.5,
        "average_grade": float(recorded_grades.mean()) if len(recorded_grades) else 50.0,
        "withdrawal_rate": float(records["final_result"].eq("Withdrawn").mean()) if len(records) else 0.25,
    }


def candidate_features(
    context: DatasetContext,
    student_id: str,
    average_grade: float,
    withdrawals: int,
    credits_earned: int,
    course_code: str,
) -> list[float]:
    enrollments = context.frames["enrollments"]
    history = enrollments[enrollments["student_id"].eq(student_id)]
    success_rate = float(history["final_result"].isin(["Pass", "Distinction"]).mean()) if len(history) else 0.5
    module = module_empirical_stats(context, course_code)
    latest = history.sort_values("presentation").tail(1)
    previous_attempts = float(latest["previous_attempts"].iloc[0]) if len(latest) and "previous_attempts" in latest else 0.0
    study_load = float(latest["studied_credits"].iloc[0]) if len(latest) and "studied_credits" in latest else 60.0
    education = str(latest["highest_education"].iloc[0]) if len(latest) and "highest_education" in latest else ""
    mean_clicks = float(history["vle_total_clicks"].mean()) if len(history) and "vle_total_clicks" in history else 0.0
    mean_days = float(history["vle_active_days"].mean()) if len(history) and "vle_active_days" in history else 0.0
    return [
        average_grade / 100 if history["final_result"].ne("Withdrawn").any() else 0.5,
        success_rate,
        min(withdrawals, 3) / 3,
        min(credits_earned, 180) / 180,
        min(previous_attempts, 3) / 3,
        min(study_load, 240) / 240,
        EDUCATION_LEVELS.get(education, 0.4),
        min(mean_clicks, 10000) / 10000,
        min(mean_days, 250) / 250,
        float(module["pass_rate"]),
        float(module["average_grade"]) / 100,
        float(module["withdrawal_rate"]),
    ]
