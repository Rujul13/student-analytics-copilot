from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Settings
from .catalog import apply_catalog_enrichment
from .fixture import build_development_fixture
from .oulad import transform_oulad
from .analytical_store import has_parquet_dataset, read_parquet_dataset


@dataclass(frozen=True)
class DatasetContext:
    name: str
    version: str
    mode: str
    frames: dict[str, pd.DataFrame]


def dataset_fingerprint(frames: dict[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for name, frame in sorted(frames.items()):
        digest.update(name.encode())
        digest.update(frame.to_csv(index=False, lineterminator="\n").encode())
    return digest.hexdigest()[:12]


def load_dataset(settings: Settings) -> DatasetContext:
    if settings.dataset_path == "fixture":
        frames = apply_catalog_enrichment(build_development_fixture())
        return DatasetContext("OULAD development fixture", "fixture-v1-seed-42", "development-fixture", frames)
    bundled_source = Path(__file__).resolve().parents[1] / "data" / "oulad"
    full_processed_source = Path(__file__).resolve().parents[1] / "data" / "full_processed"
    processed_source = Path(__file__).resolve().parents[1] / "data" / "processed"
    source = Path(settings.dataset_path) if settings.dataset_path else bundled_source
    if not settings.dataset_path and has_parquet_dataset(full_processed_source):
        frames = apply_catalog_enrichment(read_parquet_dataset(full_processed_source))
        manifest = json.loads((full_processed_source / "manifest.json").read_text(encoding="utf-8"))
        return DatasetContext(manifest["dataset_name"], manifest["dataset_version"], "full-parquet", frames)
    if source.exists() and all((source / name).exists() for name in ["studentInfo.csv", "studentRegistration.csv", "courses.csv", "assessments.csv", "studentAssessment.csv"]):
        frames = apply_catalog_enrichment(transform_oulad(source, limit_students=None))
        fingerprint = dataset_fingerprint(frames)
        return DatasetContext("OULAD (full academic cohort)", fingerprint, "full-source", frames)
    canonical_files = {name: processed_source / f"{name}.csv" for name in ["students", "courses", "enrollments", "grades"]}
    if all(path.exists() for path in canonical_files.values()):
        frames = apply_catalog_enrichment({name: pd.read_csv(path) for name, path in canonical_files.items()})
        manifest_path = processed_source / "manifest.json"
        if manifest_path.exists():
            fingerprint = json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_version"]
        else:
            fingerprint = dataset_fingerprint(frames)
        return DatasetContext("OULAD (curated 750-learner cohort)", fingerprint, "canonical-processed", frames)
    frames = build_development_fixture()
    return DatasetContext("OULAD development fixture", "fixture-v1-seed-42", "development-fixture", frames)
