from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from fastapi import UploadFile
from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict

from .repository import DatasetContext
from .oulad import REQUIRED_FILES as OULAD_REQUIRED_FILES, transform_oulad_dataframes
from .analytical_store import stage_csv_frames
from .dataset_adapters import profile_sources, select_adapter
from .semantic import canonical_metadata


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
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
ROW_LIMITS = {"students": 50000, "courses": 5000, "enrollments": 100000, "grades": 250000}

HEADER_ALIASES = {
    "students": {
        "student_id": {"studentid", "student_id", "id_student", "learnerid", "learner_id"},
        "display_name": {"displayname", "display_name", "studentname", "student_name", "name"},
        "program": {"program", "programme", "major", "degree", "pathway"},
    },
    "courses": {
        "course_code": {"coursecode", "course_code", "courseid", "course_id", "code_module", "modulecode"},
        "course_name": {"coursename", "course_name", "title", "module_name", "subject"},
        "department": {"department", "dept", "school", "faculty"},
        "level": {"level", "courselevel", "course_level", "year_level"},
        "credits": {"credits", "credit", "credit_hours"},
        "offered_next_term": {"offerednextterm", "offered_next_term", "available_next_term", "is_offered"},
        "prerequisites": {"prerequisites", "prerequisite", "prereqs"},
        "programs": {"programs", "program", "programmes", "majors"},
        "requirement_type": {"requirementtype", "requirement_type", "course_type"},
    },
    "enrollments": {
        "enrollment_id": {"enrollmentid", "enrollment_id", "registration_id", "id_registration"},
        "student_id": {"studentid", "student_id", "id_student", "learner_id"},
        "course_code": {"coursecode", "course_code", "courseid", "course_id", "code_module"},
        "presentation": {"presentation", "term", "semester", "code_presentation"},
        "status": {"status", "enrollment_status", "registration_status"},
        "final_result": {"finalresult", "final_result", "result", "outcome"},
        "credits": {"credits", "credit", "studied_credits"},
    },
    "grades": {
        "enrollment_id": {"enrollmentid", "enrollment_id", "registration_id", "id_registration"},
        "weighted_grade": {"weightedgrade", "weighted_grade", "grade", "score", "numeric_grade", "final_grade"},
    },
}


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower().strip())


class ColumnMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str


class FileMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    role: Literal["students", "courses", "enrollments", "grades"]
    columns: list[ColumnMapping]


class MappingBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mappings: list[FileMapping]


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
    if not 1 <= len(files) <= 8:
        raise ImportValidationError("Upload between one and eight CSV files")
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
            # Academic CSV exports commonly use comma, semicolon, or tab delimiters.
            # Python's CSV sniffer (via sep=None) detects the boundary before semantic profiling.
            frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", sep=None, engine="python")
        except Exception as error:
            raise ImportValidationError(f"{upload.filename} is not a readable UTF-8 CSV") from error
        frame.columns = [str(column).strip() for column in frame.columns]
        role, missing = _detect_role(set(frame.columns))
        parsed.append(ParsedUpload(upload.filename, role, frame, missing))
    return parsed


def _deterministic_mapping(item: ParsedUpload) -> FileMapping:
    columns = list(map(str, item.frame.columns))
    scored: list[tuple[int, str, list[ColumnMapping]]] = []
    for role, aliases in HEADER_ALIASES.items():
        matches: list[ColumnMapping] = []
        used_targets: set[str] = set()
        for source in columns:
            key = _header_key(source)
            for target, candidates in aliases.items():
                if target not in used_targets and key in {_header_key(value) for value in candidates}:
                    matches.append(ColumnMapping(source=source, target=target))
                    used_targets.add(target)
                    break
        scored.append((len(ROLE_COLUMNS[role] & used_targets), role, matches))
    _, role, matches = max(scored, key=lambda value: (value[0], len(value[2])))
    return FileMapping(filename=item.filename, role=role, columns=matches)


def _mapping_payload(bundle: MappingBundle, parsed: list[ParsedUpload], ai_used: bool) -> dict:
    parsed_by_name = {item.filename: item for item in parsed}
    mappings = []
    for mapping in bundle.mappings:
        parsed_file = parsed_by_name.get(mapping.filename)
        source_columns = set(map(str, parsed_file.frame.columns)) if parsed_file else set()
        valid_targets = set(ROLE_HEADERS[mapping.role])
        valid_columns = [item for item in mapping.columns if item.source in source_columns and item.target in valid_targets]
        targets = {item.target for item in valid_columns}
        missing = sorted(ROLE_COLUMNS[mapping.role] - targets)
        mappings.append({
            "filename": mapping.filename,
            "role": mapping.role,
            "columns": [item.model_dump() for item in valid_columns],
            "missing": missing,
        })
    unique_roles = len({item["role"] for item in mappings}) == len(mappings)
    safe = len(mappings) == 4 and unique_roles and not any(item["missing"] for item in mappings)
    if not safe:
        flexible = _flexible_profile(parsed)
        if flexible["safe_to_apply"]:
            return flexible | {"ai_used": ai_used}
    return {"mappings": mappings, "ai_used": ai_used, "safe_to_apply": safe, "ingestion_mode": "canonical"}


def _global_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply unambiguous academic aliases without assigning the file a single role."""
    renamed = frame.copy()
    targets: dict[str, str] = {}
    for source in map(str, renamed.columns):
        key = _header_key(source)
        matches = {
            target
            for aliases in HEADER_ALIASES.values()
            for target, candidates in aliases.items()
            if key in {_header_key(value) for value in candidates}
        }
        # Shared identifiers resolve to the same target. Ambiguous generic fields
        # such as "program" or "credits" already have their canonical spelling.
        if len(matches) == 1:
            targets[source] = next(iter(matches))
        elif key in {"program", "credits"}:
            targets[source] = key
    return renamed.rename(columns=targets)


def _flexible_profile(parsed: list[ParsedUpload]) -> dict:
    sources = [(item.filename, item.frame) for item in parsed]
    selected = select_adapter(sources)
    if selected:
        _, match = selected
        return {
            "mappings": [{
                "filename": item.filename,
                "role": "semantic_source",
                "columns": [],
                "missing": [],
            } for item in parsed],
            "safe_to_apply": True,
            "ingestion_mode": "semantic-adapter",
            "adapter_id": match.adapter_id,
            "adapter_confidence": match.confidence,
            "note": match.reason + ". A dedicated deterministic adapter will normalize it after confirmation.",
            "profiles": [profile.to_dict() for profile in profile_sources(sources)],
        }
    profiled = [_global_aliases(item.frame) for item in parsed]
    enrollment_index = max(
        range(len(profiled)),
        key=lambda index: len({"student_id", "course_code", "final_result", "weighted_grade"} & set(profiled[index].columns)),
    )
    enrollment = profiled[enrollment_index]
    has_identity = {"student_id", "course_code"}.issubset(enrollment.columns)
    has_outcome = "final_result" in enrollment.columns or "weighted_grade" in enrollment.columns
    mappings = []
    for item, frame in zip(parsed, profiled):
        columns = [
            {"source": str(source), "target": str(target)}
            for source, target in zip(item.frame.columns, frame.columns)
            if str(source) != str(target)
        ]
        mappings.append({
            "filename": item.filename,
            "role": "flat_academic_history" if item is parsed[enrollment_index] else (item.role or "supporting_table"),
            "columns": columns,
            "missing": sorted(({"student_id", "course_code"} - set(frame.columns)) if item is parsed[enrollment_index] else set()),
        })
    return {
        "mappings": mappings,
        "safe_to_apply": bool(has_identity and has_outcome),
        "ingestion_mode": "flexible",
        "note": "Files will be normalized into students, courses, enrollments, and grades before activation.",
    }


async def suggest_upload_mappings(parsed: list[ParsedUpload], api_key: str | None, model: str) -> dict:
    deterministic = MappingBundle(mappings=[_deterministic_mapping(item) for item in parsed])
    if not api_key:
        return _mapping_payload(deterministic, parsed, False)
    schemas = {role: sorted(columns) for role, columns in ROLE_COLUMNS.items()}
    files = [{"filename": item.filename, "columns": list(map(str, item.frame.columns))} for item in parsed]
    try:
        client = AsyncGroq(api_key=api_key, timeout=12, max_retries=1)
        completion = await client.chat.completions.create(
            model=model,
            temperature=0,
            reasoning_effort="low",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Map CSV headers to four canonical academic tables. Use only supplied filenames and headers. "
                        "Never invent a source column. Assign each file one unique role and each source column at most once. "
                        "Map only when meaning is clear and omit uncertain mappings. Complete data values are intentionally unavailable."
                    ),
                },
                {"role": "user", "content": json.dumps({"canonical_required_columns": schemas, "files": files})},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "csv_mapping_bundle", "strict": True, "schema": MappingBundle.model_json_schema()},
            },
        )
        bundle = MappingBundle.model_validate_json(completion.choices[0].message.content or "{}")
        if len({item.filename for item in bundle.mappings}) != len(parsed) or len({item.role for item in bundle.mappings}) != len(bundle.mappings):
            raise ValueError("AI mapping did not assign unique files and roles")
        return _mapping_payload(bundle, parsed, True)
    except Exception:
        return _mapping_payload(deterministic, parsed, False)


def apply_upload_mappings(parsed: list[ParsedUpload], mapping_json: str | None) -> list[ParsedUpload]:
    if not mapping_json:
        return parsed
    raw = json.loads(mapping_json)
    if raw.get("ingestion_mode") in {"flexible", "semantic-adapter"}:
        return parsed
    bundle = MappingBundle.model_validate({
        "mappings": [
            {key: value for key, value in item.items() if key in {"filename", "role", "columns"}}
            for item in raw.get("mappings", [])
        ]
    })
    by_filename = {item.filename: item for item in bundle.mappings}
    result = []
    for item in parsed:
        mapping = by_filename.get(item.filename)
        if mapping is None:
            raise ImportValidationError(f"No confirmed mapping was supplied for {item.filename}")
        rename = {column.source: column.target for column in mapping.columns}
        invalid_targets = sorted(set(rename.values()) - set(ROLE_HEADERS[mapping.role]))
        if invalid_targets:
            raise ImportValidationError(f"{item.filename} contains invalid canonical targets: {', '.join(invalid_targets)}")
        if len(rename.values()) != len(set(rename.values())):
            raise ImportValidationError(f"{item.filename} maps two columns to the same canonical field")
        frame = item.frame.rename(columns=rename)
        missing = sorted(ROLE_COLUMNS[mapping.role] - set(frame.columns))
        result.append(ParsedUpload(item.filename, mapping.role if not missing else None, frame, missing))
    return result


def _flexible_frames(parsed: list[ParsedUpload]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames = [_global_aliases(item.frame) for item in parsed]
    staging = stage_csv_frames(frames)
    try:
        frames = [staging.execute(f"SELECT * FROM upload_{index}").fetchdf() for index in range(len(frames))]
    finally:
        staging.close()
    by_role = {item.role: frame for item, frame in zip(parsed, frames) if item.role}
    warnings = ["Input files were normalized into the four canonical academic tables."]

    enrollment = by_role.get("enrollments")
    if enrollment is None:
        enrollment = max(
            frames,
            key=lambda frame: len({"student_id", "course_code", "final_result", "weighted_grade", "presentation"} & set(frame.columns)),
        ).copy()
    else:
        enrollment = enrollment.copy()
    if not {"student_id", "course_code"}.issubset(enrollment.columns):
        raise ImportValidationError("The uploaded data needs student and course identifiers to construct academic history")

    separate_grades = by_role.get("grades")
    if "enrollment_id" not in enrollment.columns:
        enrollment["enrollment_id"] = [f"IMPORT-E{index + 1}" for index in range(len(enrollment))]
        warnings.append("Enrollment IDs were generated from row order.")
    if enrollment["enrollment_id"].duplicated().any():
        raise ImportValidationError("The academic-history rows do not produce unique enrollment IDs")

    if "weighted_grade" not in enrollment.columns and separate_grades is not None:
        enrollment = enrollment.merge(separate_grades[["enrollment_id", "weighted_grade"]], on="enrollment_id", how="left")
    if "final_result" not in enrollment.columns:
        if "weighted_grade" not in enrollment.columns:
            raise ImportValidationError("Academic history needs either final_result or a numeric grade column")
        numeric = pd.to_numeric(enrollment["weighted_grade"], errors="coerce")
        enrollment["final_result"] = pd.cut(
            numeric,
            bins=[-float("inf"), 39.999, 84.999, float("inf")],
            labels=["Fail", "Pass", "Distinction"],
        ).astype("object")
        warnings.append("Final outcomes were derived from grades: below 40 Fail, 40–84.9 Pass, and 85+ Distinction.")
    if "weighted_grade" not in enrollment.columns:
        raise ImportValidationError("A numeric grade column is required for analytics and success recommendations")

    enrollment["presentation"] = enrollment.get("presentation", "Uploaded")
    enrollment["status"] = enrollment.get("status", enrollment["final_result"].map(lambda value: "Withdrawn" if value == "Withdrawn" else "Completed"))

    course_source = by_role.get("courses")
    if course_source is None:
        course_columns = [column for column in ROLE_HEADERS["courses"] if column in enrollment.columns]
        course_source = enrollment[["course_code", *[column for column in course_columns if column != "course_code"]]].drop_duplicates("course_code")
        warnings.append("The course lookup table was derived from academic-history rows.")
    courses = course_source.copy().drop_duplicates("course_code")
    course_defaults = {"course_name": None, "department": "Not provided", "level": 0, "credits": 0, "offered_next_term": False}
    for column, default in course_defaults.items():
        if column not in courses.columns:
            courses[column] = courses["course_code"].astype(str).map(lambda code: f"Course {code}") if column == "course_name" else default

    student_source = by_role.get("students")
    if student_source is None:
        student_columns = [column for column in ROLE_HEADERS["students"] if column in enrollment.columns]
        student_source = enrollment[["student_id", *[column for column in student_columns if column != "student_id"]]].drop_duplicates("student_id")
        warnings.append("The student lookup table was derived from academic-history rows.")
    students = student_source.copy().drop_duplicates("student_id")
    if "display_name" not in students.columns:
        students["display_name"] = students["student_id"].astype(str).map(lambda value: f"Learner {value}")
    if "program" not in students.columns:
        students["program"] = "Not provided"

    course_credits = courses.set_index("course_code")["credits"]
    if "credits" not in enrollment.columns:
        attempted = enrollment["course_code"].map(course_credits).fillna(0)
        enrollment["credits"] = attempted.where(enrollment["final_result"].isin(["Pass", "Distinction"]), 0)
    grades = enrollment[["enrollment_id", "weighted_grade"]].dropna(subset=["weighted_grade"]).copy()
    canonical_enrollments = enrollment[["enrollment_id", "student_id", "course_code", "presentation", "status", "final_result", "credits"]].copy()
    return {"students": students, "courses": courses, "enrollments": canonical_enrollments, "grades": grades}, warnings


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
    if enriched:
        enriched = bool(
            normalized["courses"]["programs"].fillna("").astype(str).str.strip().ne("").any()
            and normalized["courses"]["requirement_type"].fillna("").astype(str).str.strip().ne("").any()
        )
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
    uploaded_by_name = {item.filename: item.frame for item in parsed}
    selected_adapter = select_adapter([(item.filename, item.frame) for item in parsed])
    adapter_metadata = None
    dataset_name = "Uploaded Canonical Dataset"
    if selected_adapter:
        adapter, _ = selected_adapter
        adapted = adapter.transform([(item.filename, item.frame) for item in parsed])
        raw_frames = adapted.frames
        flexible_warnings = adapted.warnings
        adapter_metadata = adapted.metadata
        dataset_name = adapted.dataset_name
        roles = []
        missing_roles = []
    elif OULAD_REQUIRED_FILES.issubset(uploaded_by_name):
        raw_frames = transform_oulad_dataframes(
            uploaded_by_name["studentInfo.csv"],
            uploaded_by_name["studentRegistration.csv"],
            uploaded_by_name["courses.csv"],
            uploaded_by_name["assessments.csv"],
            uploaded_by_name["studentAssessment.csv"],
            limit_students=None,
        )
        roles = []
        missing_roles = []
        flexible_warnings = ["The official OULAD package was recognized and normalized into the four canonical tables."]
    else:
        raw_frames = None
    if not selected_adapter:
        roles = [item.role for item in parsed if item.role]
        if len(set(roles)) != len(roles):
            raise ImportValidationError("Two files were detected as the same canonical table")
        missing_roles = sorted(set(ROLE_COLUMNS) - set(roles))
        flexible_warnings = flexible_warnings if raw_frames is not None else []
        if raw_frames is not None:
            pass
        elif missing_roles:
            raw_frames, flexible_warnings = _flexible_frames(parsed)
        else:
            raw_frames = {item.role: item.frame for item in parsed if item.role}
    frames, enriched = _normalize(raw_frames)

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
    mode = "uploaded-semantic" if adapter_metadata else ("uploaded-enriched" if enriched else "uploaded-canonical")
    warnings = flexible_warnings + ([] if enriched or adapter_metadata else ["No prerequisite/program enrichment was detected; recommendations will use historical-performance mode."])
    semantic = adapter_metadata or canonical_metadata(graduation_aware=enriched, adapter_id="uploaded-canonical")
    return DatasetContext(dataset_name, version, mode, frames, semantic), warnings


def preview_payload(parsed: list[ParsedUpload], token: str, context: DatasetContext, warnings: list[str]) -> dict:
    return {
        "token": token,
        "valid": True,
        "dataset_name": context.name,
        "dataset_version": context.version,
        "mode": context.mode,
        "warnings": warnings,
        "capabilities": {
            "dashboard": True,
            "natural_language_analytics": context.semantic.capabilities.natural_language_analytics,
            "historical_recommendations": context.semantic.capabilities.historical_recommendations,
            "graduation_aware_recommendations": context.semantic.capabilities.graduation_aware_recommendations,
        },
        "adapter": context.semantic.to_dict(),
        "files": [
            {
                "filename": item.filename,
                "role": item.role or "source",
                "rows": len(item.frame),
                "columns": list(item.frame.columns),
                "missing": item.missing,
            }
            for item in parsed
        ],
    }
