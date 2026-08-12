from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
import pandas as pd


@dataclass(frozen=True)
class DatasetCapabilities:
    learner_identity: bool = True
    numeric_grades: bool = True
    academic_outcomes: bool = True
    terms_or_semesters: bool = True
    degree_programs: bool = False
    individual_course_history: bool = True
    course_catalog: bool = True
    prerequisites: bool = False
    graduation_requirements: bool = False
    natural_language_analytics: bool = True
    historical_recommendations: bool = True
    graduation_aware_recommendations: bool = False
    learner_risk: bool = True


@dataclass(frozen=True)
class DashboardSpecification:
    dimension_label: str = "Course module"
    period_label: str = "Term"
    outcome_label: str = "Outcome"
    performance_title: str = "Module pulse"
    performance_eyebrow: str = "Academic performance"
    performance_tag: str = "Average grade"
    outcome_title: str = "Result mix"
    priority_enabled: bool = True
    enabled_filters: tuple[str, ...] = ("course_code", "presentation", "final_result")


@dataclass(frozen=True)
class SemanticMetadata:
    adapter_id: str = "canonical-academic-history"
    record_grain: str = "One row per learner-course registration"
    dimension_semantics: str = "Individual course or module"
    capabilities: DatasetCapabilities = field(default_factory=DatasetCapabilities)
    dashboard: DashboardSpecification = field(default_factory=DashboardSpecification)
    success_outcomes: tuple[str, ...] = ("Pass", "Distinction")
    withdrawal_outcomes: tuple[str, ...] = ("Withdrawn",)
    failure_outcomes: tuple[str, ...] = ("Fail",)
    table_descriptions: dict[str, str] = field(default_factory=dict)
    metric_definitions: tuple[str, ...] = ()
    source: str | None = None
    doi: str | None = None
    license: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_metadata(*, graduation_aware: bool = False, adapter_id: str = "canonical-academic-history") -> SemanticMetadata:
    capabilities = DatasetCapabilities(
        degree_programs=graduation_aware,
        prerequisites=graduation_aware,
        graduation_requirements=graduation_aware,
        graduation_aware_recommendations=graduation_aware,
    )
    return SemanticMetadata(adapter_id=adapter_id, capabilities=capabilities)


def oulad_metadata() -> SemanticMetadata:
    return SemanticMetadata(
        adapter_id="oulad",
        record_grain="One row per learner-module presentation",
        dimension_semantics="OULAD module",
        capabilities=DatasetCapabilities(course_catalog=False),
        dashboard=DashboardSpecification(period_label="Term (OULAD presentation)"),
        source="UCI Machine Learning Repository",
        doi="10.24432/C5KK69",
        license="CC BY 4.0",
    )


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    null_rate: float
    distinct_count: int
    examples: tuple[str, ...]
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class TableProfile:
    filename: str
    row_count: int
    columns: tuple[ColumnProfile, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_frame(filename: str, frame: pd.DataFrame) -> TableProfile:
    """Profile an uploaded frame through DuckDB without changing source values."""
    connection = duckdb.connect()
    connection.register("source", frame)
    try:
        columns: list[ColumnProfile] = []
        row_count = len(frame)
        for column in frame.columns:
            quoted = '"' + str(column).replace('"', '""') + '"'
            stats = connection.execute(
                f"SELECT COUNT(DISTINCT {quoted}), COUNT(*) FILTER (WHERE {quoted} IS NULL) FROM source"
            ).fetchone()
            series = frame[column]
            numeric = pd.to_numeric(series, errors="coerce")
            numeric_values = numeric.dropna()
            examples = tuple(map(str, series.dropna().astype(str).drop_duplicates().head(8).tolist()))
            columns.append(ColumnProfile(
                name=str(column),
                dtype=str(series.dtype),
                null_rate=round((int(stats[1]) / max(row_count, 1)) * 100, 2),
                distinct_count=int(stats[0]),
                examples=examples,
                minimum=float(numeric_values.min()) if len(numeric_values) else None,
                maximum=float(numeric_values.max()) if len(numeric_values) else None,
            ))
        return TableProfile(filename=filename, row_count=row_count, columns=tuple(columns))
    finally:
        connection.unregister("source")
        connection.close()
