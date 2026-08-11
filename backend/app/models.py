from typing import Literal

from pydantic import BaseModel, Field


class Metric(BaseModel):
    label: str
    value: float | int
    display: str
    delta: str
    direction: Literal["up", "down", "neutral"]


class DistributionPoint(BaseModel):
    label: str
    value: float
    count: int


class DashboardResponse(BaseModel):
    dataset_name: str
    dataset_version: str
    mode: str
    metrics: list[Metric]
    outcomes: list[DistributionPoint]
    modules: list[DistributionPoint]
    risk_bands: list[DistributionPoint]


class StudentSummary(BaseModel):
    student_id: str
    display_name: str
    program: str
    average_grade: float
    credits_earned: int
    risk: Literal["Low", "Medium", "High"]
    status: str


class Recommendation(BaseModel):
    course_code: str
    course_name: str
    score: int = Field(ge=0, le=100)
    confidence: Literal["High", "Medium", "Low"]
    reasons: list[str]
    requirement_fit: int
    performance_fit: int
    progression_fit: int
    requirement_type: str
    prerequisites_met: list[str]
    narrative: str | None = None


class RecommendationResponse(BaseModel):
    student: StudentSummary
    capability_mode: Literal["performance-only", "graduation-aware"]
    recommendations: list[Recommendation]
    ai_explanation_enabled: bool
    catalog_label: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class ImportCommitRequest(BaseModel):
    token: str = Field(min_length=32, max_length=32)


class QueryResponse(BaseModel):
    answer: str
    result_type: Literal["metric", "table", "unsupported"]
    rows: list[dict[str, str | int | float]] = Field(default_factory=list)
    calculation_trace: list[str]
    ai_used: bool
