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
}

export interface DashboardData {
  dataset_name: string;
  dataset_version: string;
  mode: string;
  metrics: Metric[];
  outcomes: DistributionPoint[];
  modules: DistributionPoint[];
  risk_bands: DistributionPoint[];
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
  files: ImportFilePreview[];
}

export interface Student {
  student_id: string;
  display_name: string;
  program: string;
  average_grade: number;
  credits_earned: number;
  risk: "Low" | "Medium" | "High";
  status: string;
}

export interface Recommendation {
  course_code: string;
  course_name: string;
  score: number;
  confidence: string;
  reasons: string[];
  requirement_fit: number;
  performance_fit: number;
  progression_fit: number;
  requirement_type: string;
  prerequisites_met: string[];
  narrative: string | null;
}

export interface RecommendationResponse {
  student: Student;
  capability_mode: string;
  recommendations: Recommendation[];
  ai_explanation_enabled: boolean;
  catalog_label: string;
}

export interface QueryResponse {
  answer: string;
  result_type: "metric" | "table" | "unsupported";
  rows: Record<string, string | number>[];
  calculation_trace: string[];
  ai_used: boolean;
}
