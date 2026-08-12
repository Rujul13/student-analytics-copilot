from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .semantic import DashboardSpecification, DatasetCapabilities, SemanticMetadata, TableProfile, profile_frame


def header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


@dataclass(frozen=True)
class AdapterMatch:
    adapter_id: str
    confidence: float
    reason: str


@dataclass
class AdapterResult:
    frames: dict[str, pd.DataFrame]
    metadata: SemanticMetadata
    warnings: list[str]
    dataset_name: str


class KaggleAcademicSuccessAdapter:
    adapter_id = "kaggle-academic-success"
    target_aliases = {"target", "academicoutcome", "studentoutcome"}
    program_aliases = {"course", "degreeprogram", "programme", "program"}
    first_grade_aliases = {"curricularunits1stsemgrade", "firstsemestergrade", "semester1grade"}
    second_grade_aliases = {"curricularunits2ndsemgrade", "secondsemestergrade", "semester2grade"}
    first_approved_aliases = {"curricularunits1stsemapproved", "firstsemesterapproved", "semester1approved"}
    second_approved_aliases = {"curricularunits2ndsemapproved", "secondsemesterapproved", "semester2approved"}

    def match(self, frames: list[tuple[str, pd.DataFrame]]) -> AdapterMatch | None:
        if len(frames) != 1:
            return None
        keys = {header_key(str(column)) for column in frames[0][1].columns}
        required = [self.target_aliases, self.program_aliases, self.first_grade_aliases, self.second_grade_aliases]
        if all(keys & group for group in required):
            return AdapterMatch(self.adapter_id, 0.99, "Recognized the higher-education academic-success schema")
        return None

    @staticmethod
    def _column(frame: pd.DataFrame, aliases: set[str]) -> str:
        matches = [str(column) for column in frame.columns if header_key(str(column)) in aliases]
        if len(matches) != 1:
            raise ValueError(f"Expected one column matching {sorted(aliases)}")
        return matches[0]

    def transform(self, frames: list[tuple[str, pd.DataFrame]]) -> AdapterResult:
        source = frames[0][1].copy()
        target_column = self._column(source, self.target_aliases)
        program_column = self._column(source, self.program_aliases)
        first_grade = self._column(source, self.first_grade_aliases)
        second_grade = self._column(source, self.second_grade_aliases)
        row_number = pd.Series(range(1, len(source) + 1), index=source.index)
        student_id = row_number.map(lambda value: f"KAS-{value:06d}")
        program_value = source[program_column].fillna("Unknown").astype(str).str.strip()
        program_label = program_value.map(lambda value: f"Degree program {value}")
        outcomes = source[target_column].fillna("Unknown").astype(str).str.strip().str.title()
        grade_frame = pd.DataFrame({
            "semester_1_grade": pd.to_numeric(source[first_grade], errors="coerce"),
            "semester_2_grade": pd.to_numeric(source[second_grade], errors="coerce"),
        })
        keyed_columns = {header_key(str(column)): str(column) for column in source.columns}
        first_approved = next((keyed_columns[key] for key in self.first_approved_aliases if key in keyed_columns), None)
        second_approved = next((keyed_columns[key] for key in self.second_approved_aliases if key in keyed_columns), None)
        if first_approved:
            grade_frame.loc[pd.to_numeric(source[first_approved], errors="coerce").fillna(0).eq(0), "semester_1_grade"] = pd.NA
        if second_approved:
            grade_frame.loc[pd.to_numeric(source[second_approved], errors="coerce").fillna(0).eq(0), "semester_2_grade"] = pd.NA
        grade_scale_warning = None
        if grade_frame.max(skipna=True).max() <= 20:
            grade_frame = grade_frame * 5
            grade_scale_warning = "Semester grades were converted from the source 0-20 scale to percentages."
        average_grade = grade_frame.mean(axis=1, skipna=True)

        students = pd.DataFrame({
            "student_id": student_id,
            "display_name": student_id.map(lambda value: f"Learner {value.removeprefix('KAS-')}"),
            "program": program_label,
        })
        # Preserve profile columns for natural-language analytics while keeping their source names explicit.
        for column in source.columns:
            key = header_key(str(column))
            if key not in self.target_aliases | self.program_aliases | self.first_grade_aliases | self.second_grade_aliases:
                safe_name = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
                if safe_name and safe_name not in students.columns:
                    students[safe_name] = source[column].values

        courses = pd.DataFrame({
            "course_code": "PROGRAM-" + program_value,
            "course_name": program_label,
            "department": "Not provided",
            "level": 0,
            "credits": 0,
            "offered_next_term": False,
            "prerequisites": "",
            "programs": program_label,
            "requirement_type": "program",
            "catalog_source": "Kaggle academic-success dataset",
        }).drop_duplicates("course_code")
        enrollments = pd.DataFrame({
            "enrollment_id": row_number.map(lambda value: f"KAS-R{value:06d}"),
            "student_id": student_id,
            "course_code": "PROGRAM-" + program_value,
            "presentation": "Two-semester academic record",
            "status": outcomes.map(lambda value: "Active" if value == "Enrolled" else "Completed"),
            "final_result": outcomes,
            "credits": 0,
            "semester_1_grade": grade_frame["semester_1_grade"],
            "semester_2_grade": grade_frame["semester_2_grade"],
        })
        grades = pd.DataFrame({"enrollment_id": enrollments["enrollment_id"], "weighted_grade": average_grade}).dropna(subset=["weighted_grade"])
        metadata = SemanticMetadata(
            adapter_id=self.adapter_id,
            record_grain="One row per learner with aggregated first- and second-semester results",
            dimension_semantics="Degree program, not an individual course",
            capabilities=DatasetCapabilities(
                degree_programs=True,
                individual_course_history=False,
                course_catalog=False,
                historical_recommendations=False,
                learner_risk=False,
            ),
            dashboard=DashboardSpecification(
                dimension_label="Degree program",
                period_label="Academic record",
                performance_title="Program performance",
                performance_eyebrow="Academic success",
                outcome_title="Graduation outcomes",
                priority_enabled=False,
                enabled_filters=("course_code", "final_result"),
            ),
            success_outcomes=("Graduate",),
            withdrawal_outcomes=("Dropout",),
            failure_outcomes=("Dropout",),
            table_descriptions={
                "students": "One row per anonymized learner with enrollment and demographic profile fields.",
                "courses": "One row per degree program. Despite the internal course_code key, these are programs, not individual classes.",
                "enrollments": "One aggregate two-semester academic record per learner, linked to their degree program.",
                "grades": "The mean of the supplied first- and second-semester aggregate grade fields per learner.",
            },
            metric_definitions=(
                "graduation = final_result == 'Graduate'",
                "dropout = final_result == 'Dropout'",
                "program means degree program; this dataset contains no individual course/module history",
            ),
            source="Kaggle mirror of the UCI Machine Learning Repository",
            doi="10.24432/C5MC89",
            license="CC BY 4.0",
        )
        return AdapterResult(
            frames={"students": students, "courses": courses, "enrollments": enrollments, "grades": grades},
            metadata=metadata,
            warnings=[
                "Recognized the Kaggle/UCI academic-success schema; Course was mapped to degree program, not individual class.",
                "Stable session learner IDs were generated from source row order because the dataset has no learner identifier.",
                "Course recommendations are unavailable because this dataset has no individual course history or course catalog.",
                *([grade_scale_warning] if grade_scale_warning else []),
            ],
            dataset_name="Kaggle Student Dropout and Academic Success",
        )


ADAPTERS = (KaggleAcademicSuccessAdapter(),)


def select_adapter(frames: list[tuple[str, pd.DataFrame]]) -> tuple[object, AdapterMatch] | None:
    matches = [(adapter, match) for adapter in ADAPTERS if (match := adapter.match(frames)) is not None]
    return max(matches, key=lambda item: item[1].confidence) if matches else None


def profile_sources(frames: list[tuple[str, pd.DataFrame]]) -> list[TableProfile]:
    return [profile_frame(filename, frame) for filename, frame in frames]
