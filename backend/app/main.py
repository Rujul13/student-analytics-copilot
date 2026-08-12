from __future__ import annotations

import uuid
import csv
import io
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .ai_workflow import AnalyticsWorkflow, run_copilot
from .analytics import dashboard, students
from .config import get_settings
from .imports import ImportValidationError, ROLE_HEADERS, TEMPLATE_ROWS, apply_upload_mappings, parse_uploads, preview_payload, suggest_upload_mappings, validate_and_build
from .models import (
    DashboardResponse,
    ImportCommitRequest,
    QueryRequest,
    QueryResponse,
    RecommendationResponse,
    StudentSummary,
)
from .recommendations import add_ai_explanations, recommend
from .repository import DatasetContext, load_dataset
from .sessions import SessionStore, valid_session_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    default_dataset = load_dataset(settings)
    app.state.settings = settings
    app.state.dataset = default_dataset
    app.state.session_store = SessionStore(default_dataset, ttl_seconds=1800, max_sessions=64)
    yield


app = FastAPI(title="Student Analytics Copilot API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def session_and_security(request: Request, call_next):
    supplied = request.cookies.get("analytics_session")
    session_id = supplied if valid_session_id(supplied) else uuid.uuid4().hex
    request.state.session_id = session_id
    store = getattr(request.app.state, "session_store", None)
    if store:
        store.get(session_id)
    response = await call_next(request)
    settings = getattr(request.app.state, "settings", get_settings())
    response.set_cookie(
        "analytics_session",
        session_id,
        max_age=1800,
        httponly=True,
        samesite="lax",
        secure=settings.environment == "production",
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; img-src 'self' data:"
    response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    return response


def _context(request: Request) -> DatasetContext:
    return request.app.state.session_store.get(request.state.session_id)


def _dataset_payload(context: DatasetContext) -> dict:
    uploaded = context.mode.startswith("uploaded")
    future_courses = 0
    return {
        "name": context.name,
        "version": context.version,
        "mode": context.mode,
        "tables": {name: len(frame) for name, frame in context.frames.items()},
        "source": "User-provided CSV upload" if uploaded else "UCI Machine Learning Repository",
        "doi": None if uploaded else "10.24432/C5KK69",
        "license": "User-managed" if uploaded else "CC BY 4.0",
        "excluded": [] if uploaded else ["studentVle.csv", "vle.csv"],
        "enrichment": {
            "label": "User-provided catalog" if uploaded else "Authentic OULAD module history only",
            "future_courses": future_courses,
            "program_assignment": "Provided by upload" if uploaded else "Not provided by OULAD and not used for recommendations",
        },
    }


@app.get("/api/health")
def health(request: Request) -> dict:
    return {"status": "ok", "dataset": _context(request).name}


@app.get("/api/config")
def public_config(request: Request) -> dict:
    settings = request.app.state.settings
    return {"ai_enabled": bool(settings.groq_api_key), "model": settings.llm_model if settings.groq_api_key else None}


@app.get("/api/dataset")
def dataset_info(request: Request) -> dict:
    return _dataset_payload(_context(request))


@app.post("/api/dataset/reset")
def dataset_reset(request: Request) -> dict:
    context = request.app.state.session_store.reset(request.state.session_id)
    return _dataset_payload(context)


@app.post("/api/import/preview")
async def import_preview(request: Request, files: list[UploadFile] = File(...), mapping_json: str | None = Form(None)) -> dict:
    try:
        parsed = await parse_uploads(files)
        parsed = apply_upload_mappings(parsed, mapping_json)
        context, warnings = validate_and_build(parsed)
        token = request.app.state.session_store.stage(request.state.session_id, context)
        return preview_payload(parsed, token, context, warnings)
    except ImportValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"CSV normalization failed: {error}") from error


@app.post("/api/import/mapping-suggestions")
async def import_mapping_suggestions(request: Request, files: list[UploadFile] = File(...)) -> dict:
    try:
        parsed = await parse_uploads(files)
        settings = request.app.state.settings
        return await suggest_upload_mappings(parsed, settings.groq_api_key, settings.llm_model)
    except ImportValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/import/templates")
def import_templates() -> Response:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for role, headers in ROLE_HEADERS.items():
            text = io.StringIO(newline="")
            writer = csv.writer(text)
            writer.writerow(headers)
            writer.writerow(TEMPLATE_ROWS[role])
            archive.writestr(f"{role}.csv", text.getvalue())
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="student-analytics-csv-templates.zip"'},
    )


@app.post("/api/import/commit")
def import_commit(payload: ImportCommitRequest, request: Request) -> dict:
    try:
        context = request.app.state.session_store.commit(request.state.session_id, payload.token)
        return _dataset_payload(context)
    except KeyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/dashboard", response_model=DashboardResponse)
def dashboard_endpoint(
    request: Request,
    course_code: str | None = None,
    presentation: str | None = None,
    final_result: str | None = None,
) -> DashboardResponse:
    return dashboard(_context(request), course_code=course_code, presentation=presentation, final_result=final_result)


@app.get("/api/students", response_model=list[StudentSummary])
def students_endpoint(request: Request) -> list[StudentSummary]:
    return students(_context(request))


@app.get("/api/students/{student_id}/recommendations", response_model=RecommendationResponse)
async def recommendations_endpoint(student_id: str, request: Request) -> RecommendationResponse:
    try:
        ranked = recommend(_context(request), student_id, limit=3)
        return await add_ai_explanations(
            ranked,
            request.app.state.settings.groq_api_key,
            request.app.state.settings.llm_model,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Student not found") from error


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest, request: Request) -> QueryResponse:
    if not request.app.state.session_store.allow_query(request.state.session_id):
        raise HTTPException(status_code=429, detail="Query rate limit exceeded; retry in one minute")
    context = _context(request)
    settings = request.app.state.settings
    workflow = (
        AnalyticsWorkflow(context, settings.groq_api_key, settings.pandas_agent_model, settings.answer_model)
        if settings.groq_api_key
        else None
    )
    return await run_copilot(context, payload.question, workflow, [turn.model_dump() for turn in payload.history])


static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
