import io
import json
import zipfile

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.repository import load_dataset
from app.sessions import SessionStore


def canonical_files(unknown_student: bool = False):
    enrollment_student = "MISSING" if unknown_student else "S1"
    return [
        ("files", ("students.csv", "student_id,display_name,program\nS1,Student One,Applied Computing\nS2,Student Two,Business Analytics\n", "text/csv")),
        ("files", ("courses.csv", "course_code,course_name,department,level,credits,offered_next_term\nC101,Intro Computing,Computing,1,30,true\nC201,Applied Systems,Systems,2,30,true\n", "text/csv")),
        ("files", ("enrollments.csv", f"enrollment_id,student_id,course_code,presentation,status,final_result,credits\nE1,{enrollment_student},C101,2026J,Completed,Pass,30\nE2,S2,C101,2026J,Completed,Fail,0\n", "text/csv")),
        ("files", ("grades.csv", "enrollment_id,weighted_grade\nE1,82\nE2,44\n", "text/csv")),
    ]


def aliased_files():
    return [
        ("files", ("people.csv", "Student ID,Name,Major\nS1,Student One,Applied Computing\nS2,Student Two,Business Analytics\n", "text/csv")),
        ("files", ("modules.csv", "Course ID,Title,Dept,Course Level,Credit Hours,Available Next Term\nC101,Intro Computing,Computing,1,30,true\nC201,Applied Systems,Systems,2,30,true\n", "text/csv")),
        ("files", ("registrations.csv", "Registration ID,Student ID,Course ID,Semester,Enrollment Status,Outcome,Studied Credits\nE1,S1,C101,2026J,Completed,Pass,30\nE2,S2,C101,2026J,Completed,Fail,0\n", "text/csv")),
        ("files", ("scores.csv", "Registration ID,Final Grade\nE1,82\nE2,44\n", "text/csv")),
    ]


def test_preview_commit_reset_and_browser_session_isolation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with TestClient(app) as first, TestClient(app) as second:
        assert first.get("/api/dashboard").json()["metrics"][0]["value"] == 28785
        assert second.get("/api/dashboard").json()["metrics"][0]["value"] == 28785

        preview = first.post("/api/import/preview", files=canonical_files())
        assert preview.status_code == 200
        body = preview.json()
        assert body["mode"] == "uploaded-canonical"
        assert {item["role"] for item in body["files"]} == {"students", "courses", "enrollments", "grades"}
        assert first.get("/api/dashboard").json()["metrics"][0]["value"] == 28785

        commit = first.post("/api/import/commit", json={"token": body["token"]})
        assert commit.status_code == 200
        assert first.get("/api/dashboard").json()["metrics"][0]["value"] == 2
        assert second.get("/api/dashboard").json()["metrics"][0]["value"] == 28785

        invalid = first.post("/api/import/preview", files=canonical_files(unknown_student=True))
        assert invalid.status_code == 400
        assert first.get("/api/dashboard").json()["metrics"][0]["value"] == 2

        reset = first.post("/api/dataset/reset")
        assert reset.status_code == 200
        assert first.get("/api/dashboard").json()["metrics"][0]["value"] == 28785


def test_single_flat_file_is_normalized_into_canonical_tables(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    flat = [("files", ("history.csv", "Student ID,Name,Course ID,Title,Semester,Final Grade\nS1,Ada,C101,Analytics,2026J,82\nS2,Lin,C101,Analytics,2026J,38\n", "text/csv"))]
    with TestClient(app) as client:
        suggestion = client.post("/api/import/mapping-suggestions", files=flat)
        assert suggestion.status_code == 200
        mapping = suggestion.json()
        assert mapping["safe_to_apply"] is True
        assert mapping["ingestion_mode"] == "flexible"
        preview = client.post("/api/import/preview", files=flat, data={"mapping_json": json.dumps(mapping)})
        assert preview.status_code == 200, preview.text
        body = preview.json()
        assert body["capabilities"]["historical_recommendations"] is True
        assert body["capabilities"]["graduation_aware_recommendations"] is False


def test_template_download_and_lru_capacity(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with TestClient(app) as client:
        response = client.get("/api/import/templates")
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert set(archive.namelist()) == {"students.csv", "courses.csv", "enrollments.csv", "grades.csv"}

    default = load_dataset(Settings(dataset_path="fixture"))
    store = SessionStore(default, max_sessions=2)
    store.get("a" * 32)
    store.get("b" * 32)
    store.get("c" * 32)
    assert len(store._states) == 2
    assert "a" * 32 not in store._states
    for _ in range(20):
        assert store.allow_query("c" * 32)
    assert not store.allow_query("c" * 32)


def test_header_mapping_is_suggested_and_requires_confirmation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with TestClient(app) as client:
        suggestion = client.post("/api/import/mapping-suggestions", files=aliased_files())
        assert suggestion.status_code == 200
        mapping = suggestion.json()
        assert mapping["ai_used"] is False
        assert mapping["safe_to_apply"] is True
        assert {item["role"] for item in mapping["mappings"]} == {"students", "courses", "enrollments", "grades"}

        preview = client.post(
            "/api/import/preview",
            files=aliased_files(),
            data={"mapping_json": json.dumps(mapping)},
        )
        assert preview.status_code == 200
        assert preview.json()["valid"] is True
