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
    return {"mappings": mappings, "ai_used": ai_used, "safe_to_apply": safe}


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
