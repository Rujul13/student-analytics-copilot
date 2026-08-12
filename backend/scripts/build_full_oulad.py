from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from app.analytical_store import add_oulad_vle_aggregates, write_parquet_dataset
from app.oulad import transform_oulad
from app.repository import dataset_fingerprint
from app.repository import DatasetContext
from app.success_prediction import get_success_model, save_success_model


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "oulad.zip"
SOURCE = ROOT / "data" / "oulad"
TARGET = ROOT / "data" / "full_processed"


def ensure_source() -> None:
    required = {"studentInfo.csv", "studentRegistration.csv", "courses.csv", "assessments.csv", "studentAssessment.csv", "studentVle.csv", "vle.csv"}
    if all((SOURCE / name).exists() for name in required):
        return
    SOURCE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in required:
            archive.extract(name, SOURCE)


def main() -> None:
    ensure_source()
    frames = transform_oulad(SOURCE, limit_students=None)
    frames = add_oulad_vle_aggregates(frames, SOURCE)
    write_parquet_dataset(frames, TARGET)
    version = dataset_fingerprint(frames)
    manifest = {
        "dataset_name": "OULAD (full academic cohort)",
        "dataset_version": version,
        "selection_policy": "all anonymized learners in the official OULAD core academic tables",
        "source_archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest().upper(),
        "source_doi": "10.24432/C5KK69",
        "license": "CC BY 4.0",
        "storage": "DuckDB-generated Zstandard-compressed Parquet",
        "tables": {name: len(frame) for name, frame in frames.items()},
        "excluded_runtime_tables": ["studentVle.csv", "vle.csv"],
        "offline_features": ["per-history VLE total clicks", "per-history VLE active days"],
        "note": "Raw VLE rows are aggregated offline, then excluded from deployment and request-time scans.",
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    model = get_success_model(DatasetContext(manifest["dataset_name"], version, "full-parquet", frames))
    if model is None:
        raise RuntimeError("The full OULAD cohort did not produce an evaluable success model")
    save_success_model(model, TARGET / "success_model.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
