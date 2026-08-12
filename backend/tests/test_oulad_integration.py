import json
from pathlib import Path

import pytest

from app.oulad import REQUIRED_FILES, transform_oulad
from app.config import Settings
from app.repository import load_dataset


SOURCE = Path(__file__).resolve().parents[1] / "data" / "oulad"


@pytest.mark.skipif(not all((SOURCE / name).exists() for name in REQUIRED_FILES), reason="Local OULAD source is not installed")
def test_real_oulad_transformation_contract():
    frames = transform_oulad(SOURCE)
    assert len(frames["students"]) == 750
    assert set(frames["courses"]["course_code"]) == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}
    assert frames["enrollments"]["student_id"].isin(frames["students"]["student_id"]).all()
    assert frames["grades"]["enrollment_id"].isin(frames["enrollments"]["enrollment_id"]).all()
    assert frames["grades"]["weighted_grade"].between(0, 100).all()


def test_deployment_can_load_packaged_canonical_cohort():
    context = load_dataset(Settings(dataset_path="intentionally-missing-source"))
    assert context.name == "OULAD (curated 750-learner cohort)"
    assert context.mode == "canonical-processed"
    manifest = json.loads((SOURCE.parent / "processed" / "manifest.json").read_text(encoding="utf-8"))
    assert context.version == manifest["dataset_version"]
    assert len(context.frames["students"]) == 750
    assert len(context.frames["courses"]) == 7
    assert set(context.frames["courses"]["course_code"]) == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}
