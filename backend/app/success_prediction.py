from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .repository import DatasetContext


FEATURE_NAMES = (
    "prior_average",
    "prior_success_rate",
    "prior_withdrawals",
    "prior_credits",
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
    rows = []
    for student_id, history in joined.groupby("student_id"):
        history = history.sort_values(["presentation_order", "course_code", "enrollment_id"])
        for current in history.itertuples():
            prior = history[history["presentation_order"] < current.presentation_order]
            prior_module = joined[
                joined["course_code"].eq(current.course_code)
                & (joined["presentation_order"] < current.presentation_order)
            ]
            prior_grades = prior["weighted_grade"].dropna()
            module_grades = prior_module["weighted_grade"].dropna()
            prior_average = float(prior_grades.mean()) / 100 if len(prior_grades) else 0.5
            prior_success = float(prior["final_result"].isin(["Pass", "Distinction"]).mean()) if len(prior) else 0.5
            prior_withdrawals = min(int(prior["final_result"].eq("Withdrawn").sum()), 3) / 3
            prior_credits = min(float(prior.loc[prior["final_result"].isin(["Pass", "Distinction"]), "credits"].sum()), 180) / 180
            module_pass_rate = float(prior_module["final_result"].isin(["Pass", "Distinction"]).mean()) if len(prior_module) else 0.5
            module_average = float(module_grades.mean()) / 100 if len(module_grades) else 0.5
            module_withdrawal_rate = float(prior_module["final_result"].eq("Withdrawn").mean()) if len(prior_module) else 0.25
            rows.append({
                "student_id": str(student_id),
                "course_code": str(current.course_code),
                "prior_average": prior_average,
                "prior_success_rate": prior_success,
                "prior_withdrawals": prior_withdrawals,
                "prior_credits": prior_credits,
                "module_pass_rate": module_pass_rate,
                "module_average_grade": module_average,
                "module_withdrawal_rate": module_withdrawal_rate,
                "success": int(current.final_result in {"Pass", "Distinction"}),
            })
    return pd.DataFrame(rows)


def get_success_model(context: DatasetContext) -> SuccessModel | None:
    if context.version in _MODEL_CACHE:
        return _MODEL_CACHE[context.version]
    rows = _feature_rows(context)
    if len(rows) < 100 or rows["success"].nunique() < 2:
        _MODEL_CACHE[context.version] = None
        return None
    test_mask = rows["student_id"].map(_student_bucket).eq(0)
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
    return [
        average_grade / 100 if history["final_result"].ne("Withdrawn").any() else 0.5,
        success_rate,
        min(withdrawals, 3) / 3,
        min(credits_earned, 180) / 180,
        float(module["pass_rate"]),
        float(module["average_grade"]) / 100,
        float(module["withdrawal_rate"]),
    ]
