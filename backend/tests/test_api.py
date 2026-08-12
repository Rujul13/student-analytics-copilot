from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_public_api_vertical_slice(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        dashboard = client.get("/api/dashboard")
        assert dashboard.status_code == 200
        assert dashboard.json()["dataset_name"] == "OULAD (curated 750-learner cohort)"
        assert dashboard.json()["metrics"][0]["value"] == 750

        filtered_dashboard = client.get("/api/dashboard", params={"course_code": "CCC"})
        assert filtered_dashboard.status_code == 200
        assert filtered_dashboard.json()["metrics"][0]["value"] < 750
        assert [point["label"] for point in filtered_dashboard.json()["modules"]] == ["CCC"]

        student = client.get("/api/students").json()[0]
        recommendations = client.get(f"/api/students/{student['student_id']}/recommendations")
        assert recommendations.status_code == 200
        assert recommendations.json()["capability_mode"] == "historical-performance"
        assert recommendations.json()["catalog_label"] == "OULAD historical modules; future availability unknown"

        query = client.post("/api/query", json={"question": "What is the average grade?"})
        assert query.status_code == 200
        body = query.json()
        assert body["result_type"] == "metric"
        assert body["execution_mode"] == "deterministic-fallback"
        assert "calculation_trace" not in body
