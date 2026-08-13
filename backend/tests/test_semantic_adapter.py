import json

from fastapi.testclient import TestClient

from app.config import get_settings
from app.data_agent import build_schema_context
from app.main import app
from app.scope_validation import extract_scope
from app.copilot import answer_question


def academic_success_file():
    csv = """Marital status,Course,Curricular units 1st sem (grade),Curricular units 2nd sem (grade),Gender,Target
1,33,12,14,1,Graduate
1,33,8,10,0,Dropout
2,171,16,18,0,Enrolled
"""
    return [("files", ("academic_success.csv", csv, "text/csv"))]


def test_kaggle_academic_success_adapter_is_recognized_and_capability_gated(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with TestClient(app) as client:
        suggestion = client.post("/api/import/mapping-suggestions", files=academic_success_file())
        assert suggestion.status_code == 200, suggestion.text
        mapping = suggestion.json()
        assert mapping["ingestion_mode"] == "semantic-adapter"
        assert mapping["adapter_id"] == "kaggle-academic-success"
        assert mapping["safe_to_apply"] is True
        assert mapping["profiles"][0]["row_count"] == 3

        preview = client.post(
            "/api/import/preview",
            files=academic_success_file(),
            data={"mapping_json": json.dumps(mapping)},
        )
        assert preview.status_code == 200, preview.text
        payload = preview.json()
        assert payload["dataset_name"] == "Kaggle Student Dropout and Academic Success"
        assert payload["capabilities"]["natural_language_analytics"] is True
        assert payload["capabilities"]["historical_recommendations"] is False
        assert any("0-20 scale" in warning for warning in payload["warnings"])

        committed = client.post("/api/import/commit", json={"token": payload["token"]})
        assert committed.status_code == 200
        info = committed.json()
        assert info["semantic"]["adapter_id"] == "kaggle-academic-success"
        assert info["capabilities"]["individual_course_history"] is False
        assert info["capabilities"]["learner_risk"] is False
        assert info["source"] == "Kaggle mirror of the UCI Machine Learning Repository"
        assert info["doi"] == "10.24432/C5MC89"

        dashboard = client.get("/api/dashboard").json()
        assert dashboard["metrics"][0]["value"] == 3
        assert dashboard["metrics"][1]["value"] == 65.0
        assert dashboard["metrics"][2]["label"] == "Graduation rate"
        assert dashboard["metrics"][2]["value"] == 33.3
        assert dashboard["metrics"][3]["label"] == "Dropout rate"
        assert dashboard["metrics"][3]["value"] == 33.3
        assert dashboard["specification"]["dimension_label"] == "Degree program"
        assert dashboard["specification"]["priority_enabled"] is False
        assert {point["label"] for point in dashboard["modules"]} == {"Degree program 33", "Degree program 171"}

        learners = client.get("/api/students").json()["items"]
        recommendations = client.get(f"/api/students/{learners[0]['student_id']}/recommendations")
        assert recommendations.status_code == 409
        assert "no individual course history" in recommendations.json()["detail"]


def test_semantic_contract_is_available_to_the_pandas_agent(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with TestClient(app) as client:
        mapping = client.post("/api/import/mapping-suggestions", files=academic_success_file()).json()
        preview = client.post(
            "/api/import/preview", files=academic_success_file(), data={"mapping_json": json.dumps(mapping)}
        ).json()
        client.post("/api/import/commit", json={"token": preview["token"]})
        context = app.state.session_store.get(client.cookies.get("analytics_session"))
        schema = build_schema_context(context)
        assert schema["semantic_contract"]["dimension_semantics"] == "Degree program, not an individual course"
        assert schema["semantic_contract"]["capabilities"]["historical_recommendations"] is False
        assert not any(item.startswith("academic-support risk") for item in schema["metric_definitions"])
        assert "programs, not individual classes" in schema["tables"]["courses"]["description"]
        assert extract_scope("What is the average grade for female students?", context).missing_fields == []
        assert extract_scope("How many students dropped out?", context).outcomes == ["Dropout"]
        assert extract_scope("Which students are at risk?", context).missing_fields == ["a validated learner-risk metric"]
        fallback = answer_question(context, "How many students dropped out?", ai_enabled=False)
        assert fallback.answer == "1 learner dropped out."
        assert fallback.rows[0]["value"] == 1


def test_semicolon_delimited_source_and_ungraded_semesters_are_handled(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    content = (
        "Course;Curricular units 1st sem (approved);Curricular units 1st sem (grade);"
        "Curricular units 2nd sem (approved);Curricular units 2nd sem (grade);Target\n"
        "33;0;0;0;0;Dropout\n33;3;12;4;14;Graduate\n"
    )
    files = [("files", ("data.csv", content, "text/csv"))]
    with TestClient(app) as client:
        mapping = client.post("/api/import/mapping-suggestions", files=files).json()
        assert mapping["adapter_id"] == "kaggle-academic-success"
        preview = client.post("/api/import/preview", files=files, data={"mapping_json": json.dumps(mapping)}).json()
        client.post("/api/import/commit", json={"token": preview["token"]})
        dashboard = client.get("/api/dashboard").json()
        assert dashboard["metrics"][1]["value"] == 65.0
