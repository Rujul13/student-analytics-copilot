from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import pandas as pd
from fastapi import UploadFile

from .repository import DatasetContext


MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 15 * 1024 * 1024
ROLE_HEADERS = {
    "students": ["student_id", "display_name", "program"],
    "courses": ["course_code", "course_name", "department", "level", "credits", "offered_next_term", "prerequisites", "programs", "requirement_type"],
    "enrollments": ["enrollment_id", "student_id", "course_code", "presentation", "status", "final_result", "credits"],
    "grades": ["enrollment_id", "weighted_grade"],
}
ROLE_COLUMNS = {
    "students": set(ROLE_HEADERS["students"]),
    "courses": {"course_code", "course_name", "department", "level", "credits", "offered_next_term"},
    "enrollments": set(ROLE_HEADERS["enrollments"]),
    "grades": set(ROLE_HEADERS["grades"]),
}
TEMPLATE_ROWS = {
    "students": ["SAMPLE-1", "Example Student", "Applied Computing"],
    "courses": ["SAMPLE-101", "Example Course", "Computing", "1", "30", "true", "", "Applied Computing", "core"],
    "enrollments": ["SAMPLE-E1", "SAMPLE-1", "SAMPLE-101", "2026J", "Completed", "Pass", "30"],
    "grades": ["SAMPLE-E1", "82"],
}
ROW_LIMITS = {"students": 5000, "courses": 500, "enrollments": 50000, "grades": 200000}


class ImportValidationError(ValueError):
    pass


@dataclass
class ParsedUpload:
    filename: str
    role: str | None
    frame: pd.DataFrame
    missing: list[str]


def _detect_role(columns: set[str]) -> tuple[str | None, list[str]]:
    exact = [role for role, required in ROLE_COLUMNS.items() if required.issubset(columns)]
    if len(exact) == 1:
        return exact[0], []
    best_role = max(ROLE_COLUMNS, key=lambda role: len(ROLE_COLUMNS[role] & columns))
    return None, sorted(ROLE_COLUMNS[best_role] - columns)


async def parse_uploads(files: list[UploadFile]) -> list[ParsedUpload]:
    if not 1 <= len(files) <= 4:
        raise ImportValidationError("Upload exactly four canonical CSV files")
    parsed: list[ParsedUpload] = []
    total_bytes = 0
    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".csv"):
            raise ImportValidationError("Only .csv files are accepted")
        content = await upload.read(MAX_FILE_BYTES + 1)
        total_bytes += len(content)
        if len(content) > MAX_FILE_BYTES or total_bytes > MAX_TOTAL_BYTES:
            raise ImportValidationError("Upload exceeds the configured size limit")
        try:
            frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        except Exception as error:
            raise ImportValidationError(f"{upload.filename} is not a readable UTF-8 CSV") from error
        frame.columns = [str(column).strip() for column in frame.columns]
        role, missing = _detect_role(set(frame.columns))
        parsed.append(ParsedUpload(upload.filename, role, frame, missing))
    return parsed


def _normalize(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], bool]:
    normalized = {name: frame.copy() for name, frame in frames.items()}
    for frame in normalized.values():
        frame.dropna(how="all", inplace=True)

    for column in ["level", "credits"]:
        normalized["courses"][column] = pd.to_numeric(normalized["courses"][column], errors="raise")
    normalized["courses"]["offered_next_term"] = normalized["courses"]["offered_next_term"].astype(str).str.lower().isin(["true", "1", "yes"])
    normalized["enrollments"]["credits"] = pd.to_numeric(normalized["enrollments"]["credits"], errors="raise")
    normalized["grades"]["weighted_grade"] = pd.to_numeric(normalized["grades"]["weighted_grade"], errors="raise")

    enrichment_columns = {"prerequisites", "programs", "requirement_type"}
    enriched = enrichment_columns.issubset(normalized["courses"].columns)
    defaults = {
        "prerequisites": "",
        "programs": "",
        "requirement_type": "elective",
        "catalog_source": "User upload",
    }
    for column, value in defaults.items():
        if column not in normalized["courses"].columns:
            normalized["courses"][column] = value
        normalized["courses"][column] = normalized["courses"][column].fillna(value).astype(str)
    return normalized, enriched


def validate_and_build(parsed: list[ParsedUpload]) -> tuple[DatasetContext, list[str]]:
    roles = [item.role for item in parsed if item.role]
    if len(set(roles)) != len(roles):
        raise ImportValidationError("Two files were detected as the same canonical table")
    missing_roles = sorted(set(ROLE_COLUMNS) - set(roles))
    if missing_roles:
        raise ImportValidationError(f"Missing canonical tables: {', '.join(missing_roles)}")
    frames, enriched = _normalize({item.role: item.frame for item in parsed if item.role})

    for role, frame in frames.items():
        if len(frame) == 0:
            raise ImportValidationError(f"{role}.csv contains no records")
        if len(frame) > ROW_LIMITS[role]:
            raise ImportValidationError(f"{role}.csv exceeds the {ROW_LIMITS[role]:,}-row limit")

    keys = {"students": "student_id", "courses": "course_code", "enrollments": "enrollment_id"}
    for role, key in keys.items():
        if frames[role][key].isna().any() or frames[role][key].astype(str).str.strip().eq("").any():
            raise ImportValidationError(f"{role}.{key} contains blank values")
        if frames[role][key].duplicated().any():
            raise ImportValidationError(f"{role}.{key} must be unique")
    if not frames["enrollments"]["student_id"].isin(frames["students"]["student_id"]).all():
        raise ImportValidationError("enrollments.csv references an unknown student_id")
    if not frames["enrollments"]["course_code"].isin(frames["courses"]["course_code"]).all():
        raise ImportValidationError("enrollments.csv references an unknown course_code")
    if not frames["grades"]["enrollment_id"].isin(frames["enrollments"]["enrollment_id"]).all():
        raise ImportValidationError("grades.csv references an unknown enrollment_id")
    if not frames["grades"]["weighted_grade"].between(0, 100).all():
        raise ImportValidationError("grades.weighted_grade must be between 0 and 100")

    digest = hashlib.sha256()
    for role in sorted(frames):
        digest.update(pd.util.hash_pandas_object(frames[role], index=True).values.tobytes())
    version = digest.hexdigest()[:12]
    mode = "uploaded-enriched" if enriched else "uploaded-canonical"
    warnings = [] if enriched else ["No prerequisite/program enrichment was detected; recommendations will use performance-only mode."]
    return DatasetContext("Uploaded Canonical Dataset", version, mode, frames), warnings


def preview_payload(parsed: list[ParsedUpload], token: str, context: DatasetContext, warnings: list[str]) -> dict:
    return {
        "token": token,
        "valid": True,
        "dataset_name": context.name,
        "dataset_version": context.version,
        "mode": context.mode,
        "warnings": warnings,
        "files": [
            {
                "filename": item.filename,
                "role": item.role,
                "rows": len(item.frame),
                "columns": list(item.frame.columns),
                "missing": item.missing,
            }
            for item in parsed
        ],
    }
