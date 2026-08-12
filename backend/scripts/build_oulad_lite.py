from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.oulad import transform_oulad  # noqa: E402
from app.catalog import apply_catalog_enrichment  # noqa: E402
from app.repository import dataset_fingerprint  # noqa: E402


SOURCE = BACKEND / "data" / "oulad"
TARGET = BACKEND / "data" / "processed"
ARCHIVE_SHA256 = "F2ED1902616C1FE8D2824D872C0B7D2D72BE435BF0124D077044FE4BE2C6D3E4"


def main() -> None:
    frames = apply_catalog_enrichment(transform_oulad(SOURCE, limit_students=750))
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(TARGET / f"{name}.csv", index=False)
    dataset_version = dataset_fingerprint(frames)
    manifest = {
        "dataset_name": "OULAD (curated 750-learner cohort)",
        "dataset_version": dataset_version,
        "selection_seed": 42,
        "selection_policy": "stratified learners with at least two module-presentation histories",
        "source_archive_sha256": ARCHIVE_SHA256,
        "source_doi": "10.24432/C5KK69",
        "license": "CC BY 4.0",
        "tables": {name: len(frame) for name, frame in frames.items()},
        "excluded_source_tables": ["studentVle.csv", "vle.csv"],
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
