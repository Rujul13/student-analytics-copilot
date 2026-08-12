export type Direction = "up" | "down" | "neutral";

export interface Metric {
  label: string;
  value: number;
  display: string;
  delta: string;
  direction: Direction;
}

export interface DistributionPoint {
  label: string;
  value: number;
  count: number;
  key?: string | null;
}

export interface DashboardData {
  dataset_name: string;
  dataset_version: string;
  mode: string;
  metrics: Metric[];
  outcomes: DistributionPoint[];
  modules: DistributionPoint[];
  risk_bands: DistributionPoint[];
  filter_options: {
    courses: string[];
    presentations: string[];
    outcomes: string[];
    course_labels: Record<string, string>;
  };
  specification: {
    dimension_label: string;
    period_label: string;
    outcome_label: string;
    performance_title: string;
    performance_eyebrow: string;
    performance_tag: string;
    outcome_title: string;
    priority_enabled: boolean;
    enabled_filters: string[];
  };
}

export interface DashboardFilters {
  course_code?: string;
  presentation?: string;
  final_result?: string;
}

export interface DatasetInfo {
  name: string;
  version: string;
  mode: string;
  tables: Record<string, number>;
  source: string;
  doi: string | null;
  license: string;
  excluded: string[];
  enrichment: {
    label: string;
    future_courses: number;
    program_assignment: string;
  };
  capabilities: {
    learner_identity: boolean;
    numeric_grades: boolean;
    academic_outcomes: boolean;
    terms_or_semesters: boolean;
    degree_programs: boolean;
    individual_course_history: boolean;
    course_catalog: boolean;
    prerequisites: boolean;
    graduation_requirements: boolean;
    natural_language_analytics: boolean;
    historical_recommendations: boolean;
    graduation_aware_recommendations: boolean;
    learner_risk: boolean;
  };
  semantic: {
    adapter_id: string;
    record_grain: string;
    dimension_semantics: string;
  };
}

export interface ImportFilePreview {
  filename: string;
  role: string;
  rows: number;
  columns: string[];
  missing: string[];
}

export interface ImportPreview {
  token: string;
  valid: boolean;
  dataset_name: string;
  dataset_version: string;
  mode: string;
  warnings: string[];
  capabilities: {
    dashboard: boolean;
    natural_language_analytics: boolean;
    historical_recommendations: boolean;
    graduation_aware_recommendations: boolean;
  };
  files: ImportFilePreview[];
  adapter?: Record<string, unknown>;
}

export interface ImportMappingSuggestion {
  mappings: {
    filename: string;
    role: string;
    columns: { source: string; target: string }[];
    missing: string[];
  }[];
  ai_used: boolean;
  safe_to_apply: boolean;
  ingestion_mode: "canonical" | "flexible" | "semantic-adapter";
  note?: string;
  adapter_id?: string;
  adapter_confidence?: number;
  profiles?: unknown[];
}

export interface Student {
  student_id: string;
  display_name: string;
  program: string;
  average_grade: number;
  credits_earned: number;
  graded_enrollments: number;
  withdrawals: number;
  risk: "Low" | "Medium" | "High";
  status: string;
}

export interface Recommendation {
  course_code: string;
  course_name: string;
  score: number;
  evidence_strength: "Limited" | "Moderate" | "Strong";
  reasons: string[];
  requirement_fit: number;
  performance_fit: number;
  progression_fit: number;
  course_pass_rate: number;
  course_withdrawal_rate: number;
  course_average_grade: number;
  historical_records: number;
  success_basis: string;
  narrative: string | null;
  predicted_success_probability: number | null;
}

export interface RecommendationResponse {
  student: Student;
  capability_mode: string;
  recommendations: Recommendation[];
  ai_explanation_enabled: boolean;
  catalog_label: string;
  ranking_mode: "deterministic" | "hybrid-llm";
  evaluated_candidates: number;
  selection_summary: string;
  success_model: {
    model_name: string;
    training_records: number;
    test_records: number;
    accuracy: number;
    brier_score: number;
    roc_auc: number;
    dataset_version: string;
  } | null;
}

export interface QueryResponse {
  answer: string;
  result_type: "metric" | "table" | "unsupported" | "error";
  rows: Record<string, string | number | boolean | null>[];
  execution_mode: "generated-pandas" | "generated-pandas-repaired" | "deterministic-fallback" | "unsupported";
  ai_used: boolean;
}

export interface ConversationTurn {
  question: string;
  answer: string;
}
