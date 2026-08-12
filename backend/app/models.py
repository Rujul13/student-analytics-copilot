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
    key: str | None = None


class DashboardSpecificationModel(BaseModel):
    dimension_label: str
    period_label: str
    outcome_label: str
    performance_title: str
    performance_eyebrow: str
    performance_tag: str
    outcome_title: str
    priority_enabled: bool
    enabled_filters: list[str]


class DashboardResponse(BaseModel):
    dataset_name: str
    dataset_version: str
    mode: str
    metrics: list[Metric]
    outcomes: list[DistributionPoint]
    modules: list[DistributionPoint]
    risk_bands: list[DistributionPoint]
    filter_options: "DashboardFilterOptions"
    specification: DashboardSpecificationModel


class DashboardFilterOptions(BaseModel):
    courses: list[str]
    presentations: list[str]
    outcomes: list[str]
    course_labels: dict[str, str] = Field(default_factory=dict)


class StudentSummary(BaseModel):
    student_id: str
    display_name: str
    program: str
    average_grade: float
    credits_earned: int
    graded_enrollments: int
    withdrawals: int
    risk: Literal["Low", "Medium", "High"]
    status: str


class Recommendation(BaseModel):
    course_code: str
    course_name: str
    score: int = Field(ge=0, le=100)
    evidence_strength: Literal["Limited", "Moderate", "Strong"]
    reasons: list[str]
    requirement_fit: int
    performance_fit: int
    progression_fit: int
    course_pass_rate: float
    course_withdrawal_rate: float
    course_average_grade: float
    historical_records: int
    success_basis: str
    narrative: str | None = None
    predicted_success_probability: float | None = Field(default=None, ge=0, le=100)


class SuccessModelSummary(BaseModel):
    model_name: str
    training_records: int
    test_records: int
    accuracy: float
    brier_score: float
    roc_auc: float
    dataset_version: str


class RecommendationResponse(BaseModel):
    student: StudentSummary
    capability_mode: Literal["performance-only", "historical-performance", "graduation-aware"]
    recommendations: list[Recommendation]
    ai_explanation_enabled: bool
    catalog_label: str
    ranking_mode: Literal["deterministic", "hybrid-llm"] = "deterministic"
    evaluated_candidates: int = 0
    selection_summary: str = ""
    success_model: SuccessModelSummary | None = None


class ConversationTurn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    answer: str = Field(min_length=1, max_length=1500)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=6)


class ImportCommitRequest(BaseModel):
    token: str = Field(min_length=32, max_length=32)


JSONScalar = str | int | float | bool | None


class QueryResponse(BaseModel):
    answer: str
    result_type: Literal["metric", "table", "unsupported", "error"]
    rows: list[dict[str, JSONScalar]] = Field(default_factory=list)
    execution_mode: Literal[
        "generated-pandas",
        "generated-pandas-repaired",
        "deterministic-fallback",
        "unsupported",
    ]
    ai_used: bool
