import type { ConversationTurn, DashboardData, DashboardFilters, DatasetInfo, ImportMappingSuggestion, ImportPreview, QueryResponse, RecommendationResponse, Student } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const error = await response.json() as { detail?: string };
      if (error.detail) message = error.detail;
    } catch { /* response had no JSON error payload */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  dashboard: (filters: DashboardFilters = {}) => {
    const params = new URLSearchParams(Object.entries(filters).filter(([, value]) => Boolean(value)) as [string, string][]);
    return request<DashboardData>(`/api/dashboard${params.size ? `?${params}` : ""}`);
  },
  dataset: () => request<DatasetInfo>("/api/dataset"),
  students: () => request<Student[]>("/api/students"),
  recommendations: (studentId: string) =>
    request<RecommendationResponse>(`/api/students/${encodeURIComponent(studentId)}/recommendations`),
  query: (question: string, history: ConversationTurn[] = []) =>
    request<QueryResponse>("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: history.slice(-6) }),
    }),
  config: () => request<{ ai_enabled: boolean; model: string | null }>("/api/config"),
  suggestImportMapping: (files: File[]) => {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    return request<ImportMappingSuggestion>("/api/import/mapping-suggestions", { method: "POST", body });
  },
  previewImport: (files: File[], mapping?: ImportMappingSuggestion) => {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    if (mapping) body.append("mapping_json", JSON.stringify(mapping));
    return request<ImportPreview>("/api/import/preview", { method: "POST", body });
  },
  commitImport: (token: string) => request<DatasetInfo>("/api/import/commit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  }),
  resetDataset: () => request<DatasetInfo>("/api/dataset/reset", { method: "POST" }),
};
