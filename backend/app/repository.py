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
        return DatasetContext("OULAD Lite Development Fixture", "fixture-v1-seed-42", "development-fixture", frames)
    bundled_source = Path(__file__).resolve().parents[1] / "data" / "oulad"
    processed_source = Path(__file__).resolve().parents[1] / "data" / "processed"
    source = Path(settings.dataset_path) if settings.dataset_path else bundled_source
    if source.exists() and all((source / name).exists() for name in ["studentInfo.csv", "studentRegistration.csv", "courses.csv", "assessments.csv", "studentAssessment.csv"]):
        frames = apply_catalog_enrichment(transform_oulad(source))
        fingerprint = dataset_fingerprint(frames)
        return DatasetContext("OULAD Lite", fingerprint, "canonical-source", frames)
    canonical_files = {name: processed_source / f"{name}.csv" for name in ["students", "courses", "enrollments", "grades"]}
    if all(path.exists() for path in canonical_files.values()):
        frames = apply_catalog_enrichment({name: pd.read_csv(path) for name, path in canonical_files.items()})
        manifest_path = processed_source / "manifest.json"
        if manifest_path.exists():
            fingerprint = json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_version"]
        else:
            fingerprint = dataset_fingerprint(frames)
        return DatasetContext("OULAD Lite", fingerprint, "canonical-processed", frames)
    frames = build_development_fixture()
    return DatasetContext("OULAD Lite Development Fixture", "fixture-v1-seed-42", "development-fixture", frames)
