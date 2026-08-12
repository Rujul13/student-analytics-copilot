import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  BookOpenCheck,
  Bot,
  ChevronRight,
  CircleAlert,
  Database,
  Gauge,
  GraduationCap,
  LayoutDashboard,
  FileCheck2,
  FileSpreadsheet,
  RotateCcw,
  Search,
  Send,
  Sparkles,
  TrendingUp,
  UploadCloud,
  Users,
} from "lucide-react";
import { api } from "./api";
import type { DashboardData, DashboardFilters, DatasetInfo, ImportMappingSuggestion, ImportPreview, QueryResponse, RecommendationResponse, Student } from "./types";

type View = "overview" | "copilot" | "recommendations" | "import";

const navItems: { id: View; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "copilot", label: "Ask your data", icon: Bot },
  { id: "recommendations", label: "Course planning", icon: BookOpenCheck },
  { id: "import", label: "Import data", icon: UploadCloud },
];

function Loading() {
  return <div className="loading"><span /><span /><span /></div>;
}

function presentationLabel(value: string) {
  const match = value.match(/^(\d{4})([BJ])$/);
  if (!match) return value;
  return `${value} · ${match[2] === "B" ? "February" : "October"} ${match[1]}`;
}

function BarList({ points, suffix = "%", onSelect }: { points: DashboardData["outcomes"]; suffix?: string; onSelect?: (key: string) => void }) {
  const max = Math.max(...points.map((point) => point.value), 1);
  return (
    <div className="bar-list">
      {points.map((point) => (
        <button className={`bar-row ${onSelect ? "selectable" : ""}`} key={point.key ?? point.label} onClick={() => onSelect?.(point.key ?? point.label)} disabled={!onSelect}>
          <div className="bar-meta"><span>{point.label}</span><strong>{point.value}{suffix}</strong></div>
          <div className="bar-track"><span style={{ width: `${(point.value / max) * 100}%` }} /></div>
          <small>{point.count} records</small>
        </button>
      ))}
    </div>
  );
}

function DonutChart({ points, onSelect }: { points: DashboardData["outcomes"]; onSelect: (label: string) => void }) {
  const colors = ["#3758f9", "#22c7a9", "#ffb75e", "#ef6f8f", "#8b6cf6"];
  let cursor = 0;
  const segments = points.map((point, index) => {
    const start = cursor; cursor += point.value;
    return `${colors[index % colors.length]} ${start}% ${cursor}%`;
  });
  return <div className="donut-layout">
    <div className="donut" style={{ background: `conic-gradient(${segments.join(", ")})` }}><span><strong>{points.reduce((sum, point) => sum + point.count, 0)}</strong><small>records</small></span></div>
    <div className="donut-legend">{points.map((point, index) => <button onClick={() => onSelect(point.label)} key={point.label}><i style={{ background: colors[index % colors.length] }} /><span>{point.label}<small>{point.value}%</small></span></button>)}</div>
  </div>;
}

function Overview({ data, dataset, filters, onFilter, onNavigate, onOpenPriority }: { data: DashboardData; dataset: DatasetInfo | null; filters: DashboardFilters; onFilter: (filters: DashboardFilters) => void; onNavigate: (view: View) => void; onOpenPriority: () => void }) {
  const icons = [Users, Gauge, TrendingUp, CircleAlert];
  const highRiskCount = data.risk_bands.find((point) => point.label === "High")?.count ?? 0;
  return (
    <>
      <section className="hero-grid">
        <div>
          <p className="eyebrow">Academic intelligence workspace</p>
          <h1>See the story behind every student outcome.</h1>
          <p className="lede">A grounded view of performance, completion, and risk—calculated from your active academic dataset.</p>
        </div>
        <button className="hero-action" onClick={() => onNavigate("copilot")}>
          <Sparkles size={18} />
          <span><small>Start with a question</small>What is driving withdrawals?</span>
          <ArrowRight size={20} />
        </button>
      </section>

      <section className="metric-grid" aria-label="Headline metrics">
        {data.metrics.map((metric, index) => {
          const Icon = icons[index];
          return <article className="metric-card" key={metric.label}>
            <div className="metric-icon"><Icon size={20} /></div>
            <div><p>{metric.label}</p><strong>{metric.display}</strong><small>{metric.delta}</small></div>
          </article>;
        })}
      </section>

      <section className="dashboard-filters" aria-label="Dashboard filters">
        {data.specification.enabled_filters.includes("course_code") && <div><label htmlFor="course-filter">{data.specification.dimension_label}</label><select id="course-filter" value={filters.course_code ?? ""} onChange={(event) => onFilter({ ...filters, course_code: event.target.value || undefined })}><option value="">All {data.specification.dimension_label.toLowerCase()}s</option>{data.filter_options.courses.map((item) => <option key={item} value={item}>{data.filter_options.course_labels[item] ?? item}</option>)}</select></div>}
        {data.specification.enabled_filters.includes("presentation") && <div><label htmlFor="presentation-filter">{data.specification.period_label}</label><select id="presentation-filter" value={filters.presentation ?? ""} onChange={(event) => onFilter({ ...filters, presentation: event.target.value || undefined })}><option value="">All terms</option>{data.filter_options.presentations.map((item) => <option key={item} value={item}>{presentationLabel(item)}</option>)}</select></div>}
        {data.specification.enabled_filters.includes("final_result") && <div><label htmlFor="outcome-filter">{data.specification.outcome_label}</label><select id="outcome-filter" value={filters.final_result ?? ""} onChange={(event) => onFilter({ ...filters, final_result: event.target.value || undefined })}><option value="">All outcomes</option>{data.filter_options.outcomes.map((item) => <option key={item}>{item}</option>)}</select></div>}
        {Object.values(filters).some(Boolean) && <button onClick={() => onFilter({})}><RotateCcw size={14} /> Clear filters</button>}
      </section>

      {dataset && <section className="provenance-strip" aria-label="Dataset provenance">
        <div><Database size={19} /><span><small>Verified source</small><strong>{dataset.source}</strong></span></div>
        <div><span><small>Canonical cohort</small><strong>{dataset.tables.students.toLocaleString()} learners · {dataset.tables.enrollments.toLocaleString()} histories</strong></span></div>
        <div><span><small>License & citation</small><strong>{dataset.doi ? `${dataset.license} · DOI ${dataset.doi}` : dataset.license}</strong></span></div>
      </section>}

      <section className="content-grid">
        <article className="panel wide">
          <div className="panel-heading"><div><p className="eyebrow">{data.specification.performance_eyebrow}</p><h2>{data.specification.performance_title}</h2></div><span className="tag">{data.specification.performance_tag}</span></div>
          <BarList points={data.modules} onSelect={(course_code) => onFilter({ ...filters, course_code })} />
        </article>
        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">Learner outcomes</p><h2>{data.specification.outcome_title}</h2></div></div>
          <DonutChart points={data.outcomes} onSelect={(final_result) => onFilter({ ...filters, final_result })} />
        </article>
        {data.specification.priority_enabled && <article className="panel dark-panel">
          <p className="eyebrow">Priority signal</p>
          <h2>{highRiskCount} {highRiskCount === 1 ? "learner needs" : "learners need"} attention</h2>
          <p>Academic-support priority highlights learners with low recorded performance or repeated withdrawals. It is a triage signal, not a judgment.</p>
          <button onClick={onOpenPriority}>Review high-priority learners <ChevronRight size={17} /></button>
        </article>}
      </section>
    </>
  );
}

function Copilot({ aiEnabled, dataset }: { aiEnabled: boolean; dataset: DatasetInfo | null }) {
  const courseHistoryAvailable = dataset?.capabilities.individual_course_history !== false;
  const suggestions = courseHistoryAvailable ? [
    "How many learners earned a distinction?",
    "Which modules have the lowest average grades?",
    "Which students failed more than one class?",
  ] : [
    "How many students dropped out?",
    "Which degree programs have the lowest average grades?",
    "How many students graduated?",
  ];
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<{ id: number; question: string; result: QueryResponse }[]>([]);
  const [busy, setBusy] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  async function submit(event?: FormEvent, preset?: string) {
    event?.preventDefault();
    const value = preset ?? question;
    if (!value.trim()) return;
    setQuestion(""); setBusy(true); setChatError(null);
    try {
      const history = messages.slice(-6).map((message) => ({ question: message.question, answer: message.result.answer }));
      const result = await api.query(value, history);
      setMessages((current) => [...current, { id: Date.now(), question: value, result }]);
    } catch (caught) {
      setChatError(caught instanceof Error ? caught.message : "The question could not be answered.");
    } finally { setBusy(false); }
  }

  function rowTitle(row: QueryResponse["rows"][number], index: number) {
    return String(row.display_name ?? row.course_name ?? row.module ?? row.metric ?? row.student_id ?? row.course_code ?? `Result ${index + 1}`);
  }

  function rowEvidence(row: QueryResponse["rows"][number]) {
    const hidden = new Set(["display_name", "course_name", "module", "metric"]);
    return Object.entries(row).filter(([key]) => !hidden.has(key));
  }

  function executionStatusLabel(mode: QueryResponse["execution_mode"]): string {
    switch (mode) {
      case "generated-pandas":
      case "generated-pandas-repaired":
        return "Calculated from the active dataset using Pandas";
      case "deterministic-fallback":
        return "Verified fallback calculation";
      default:
        return "";
    }
  }

  return <section className="copilot-layout">
    <div className="copilot-intro">
      <div className="orb"><Bot size={34} /></div>
      <p className="eyebrow">Natural-language analytics</p>
      <h1>Ask the dataset.<br />Get the evidence.</h1>
      <p>Northstar turns your question into a verified calculation, then explains the result in plain language.</p>
      <div className={`status-pill ${aiEnabled ? "online" : "offline"}`}><span />{aiEnabled ? "AI-assisted analytics enabled" : "Verified analytics · API key pending"}</div>
    </div>
    <div className="conversation-card" aria-label="Analytics conversation">
      <div className="conversation-tools">
        <div className="suggestions">
        {suggestions.map((item) => <button key={item} onClick={() => void submit(undefined, item)}>{item}</button>)}
        </div>
        {messages.length > 0 && <button className="clear-chat" onClick={() => setMessages([])}><RotateCcw size={14} /> New conversation</button>}
      </div>
      <div className="conversation-stream" aria-live="polite">
        {messages.length === 0 && !busy && <div className="empty-answer"><BarChart3 size={32} /><h2>Start a conversation with your data</h2><p>Ask about outcomes, distinctions, failed courses, learner profiles, risk, modules, or recommendations.</p></div>}
        {messages.map((message) => <article className="chat-turn" key={message.id}>
          <div className="user-message"><span>You</span><p>{message.question}</p></div>
          <div className="assistant-message">
            <div className="answer-label"><Sparkles size={16} /> Northstar</div>
            <h2>{message.result.answer}</h2>
            {message.result.rows.length > 0 && <div className="evidence-table">
              <div className="evidence-heading"><strong>Verified evidence</strong><span>{message.result.rows.length} result{message.result.rows.length === 1 ? "" : "s"}</span></div>
              {message.result.rows.map((row, index) => <div className="evidence-row" key={`${message.id}-${index}`}>
                <strong>{rowTitle(row, index)}</strong>
                <div>{rowEvidence(row).map(([key, value]) => <span key={key}><small>{key.replaceAll("_", " ")}</small>{String(value)}</span>)}</div>
              </div>)}
            </div>}
            {executionStatusLabel(message.result.execution_mode) && (
              <div className="execution-status">{executionStatusLabel(message.result.execution_mode)}</div>
            )}
          </div>
        </article>)}
        {busy && <Loading />}
        {chatError && <div className="error-banner"><CircleAlert size={18} />{chatError}</div>}
      </div>
      <form className="query-box" onSubmit={submit}>
        <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about performance, completion, or risk…" aria-label="Ask the dataset" />
        <button disabled={busy || question.trim().length < 3} aria-label="Send question"><Send size={18} /></button>
      </form>
    </div>
  </section>;
}

function Recommendations({ students, initialRisk }: { students: Student[]; initialRisk: Student["risk"] | "All" }) {
  const LEARNER_PAGE_SIZE = 50;
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState<Student["risk"] | "All">(initialRisk);
  const [visibleLearners, setVisibleLearners] = useState(LEARNER_PAGE_SIZE);
  const [selected, setSelected] = useState<Student | null>(students.find((student) => student.graded_enrollments > 0) ?? students[0] ?? null);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const filtered = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return students.filter((student) =>
      (riskFilter === "All" || student.risk === riskFilter)
      && (!normalizedSearch || `${student.display_name} ${student.student_id}`.toLowerCase().includes(normalizedSearch))
    );
  }, [students, search, riskFilter]);
  const displayed = useMemo(() => filtered.slice(0, visibleLearners), [filtered, visibleLearners]);

  useEffect(() => { setRiskFilter(initialRisk); }, [initialRisk]);

  useEffect(() => { setVisibleLearners(LEARNER_PAGE_SIZE); }, [search, riskFilter]);

  useEffect(() => {
    if (!filtered.length) {
      setSelected(null);
    } else if (!filtered.some((student) => student.student_id === selected?.student_id)) {
      setSelected(filtered[0]);
    }
  }, [filtered, selected]);

  useEffect(() => {
    setResult(null);
    if (!selected) return;
    void api.recommendations(selected.student_id).then(setResult);
  }, [selected]);

  return <section>
    <div className="section-title"><div><p className="eyebrow">Course recommendation engine</p><h1>Explore the next best module.</h1></div><p>Historical outcomes and learner evidence first. AI ranking and explanation last.</p></div>
    <div className="catalog-disclosure"><CircleAlert size={17} /><span><strong>OULAD evidence boundary:</strong> recommendations use authentic learner histories and module outcomes. OULAD does not contain future availability, prerequisites, degree requirements, or official module titles, so administrators must verify those before acting.</span></div>
    <div className="student-layout">
      <aside className="student-list panel">
        <label className="search-box"><Search size={17} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a learner" /></label>
        <label className="priority-filter"><span>Academic-support priority</span><select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value as Student["risk"] | "All")}><option>All</option><option>High</option><option>Medium</option><option>Low</option></select></label>
        <p className="priority-help">Support priority, not course suitability. High: average below 50% or 2+ withdrawals. Medium: below 65% or 1 withdrawal. Low: otherwise.</p>
        <p className="learner-result-count" aria-live="polite">{filtered.length.toLocaleString()} matching learner{filtered.length === 1 ? "" : "s"}</p>
        <div className="student-items">
          {displayed.map((student) => <button type="button" className={selected?.student_id === student.student_id ? "selected" : ""} onClick={() => setSelected(student)} key={student.student_id}>
            <span className="avatar">{student.display_name.split(" ").at(-1)?.slice(-2)}</span>
            <span><strong>{student.display_name}</strong><small>{student.student_id} · {student.graded_enrollments ? `${student.average_grade}%` : "No recorded grade"}</small></span>
            <i className={`risk ${student.risk.toLowerCase()}`} title={`${student.risk} academic-support priority`}>{student.risk} priority</i>
          </button>)}
          {!filtered.length && <p className="no-learners">No learners match this search and priority.</p>}
          {displayed.length < filtered.length && <button type="button" className="load-more-learners" onClick={() => setVisibleLearners((count) => count + LEARNER_PAGE_SIZE)}>
            Show more learners ({(filtered.length - displayed.length).toLocaleString()} remaining)
          </button>}
        </div>
      </aside>
      <div className="recommendation-space">
        {selected && <div className="student-banner">
          <div><p className="eyebrow">Selected learner</p><h2>{selected.display_name}</h2><span>{selected.credits_earned} credits earned · {selected.graded_enrollments} graded module{selected.graded_enrollments === 1 ? "" : "s"} · {selected.withdrawals} withdrawal{selected.withdrawals === 1 ? "" : "s"} · {selected.risk} support priority</span></div>
          <div className="grade-ring"><strong>{selected.graded_enrollments ? selected.average_grade : "—"}</strong><small>{selected.graded_enrollments ? "avg." : "no grade"}</small></div>
        </div>}
        {!result && <Loading />}
        {result && <>
          <div className="mode-note"><Database size={16} /><span><strong>{result.capability_mode === "graduation-aware" ? "Graduation-aware recommendation" : "Historical-performance recommendation"}</strong> · {result.catalog_label} · {result.selection_summary}</span></div>
          {result.success_model && <div className="model-evaluation"><TrendingUp size={16} /><span><strong>Evaluated success baseline</strong><small>{result.success_model.model_name} · held-out n={result.success_model.test_records} · accuracy {(result.success_model.accuracy * 100).toFixed(1)}% · ROC AUC {result.success_model.roc_auc.toFixed(3)} · Brier {result.success_model.brier_score.toFixed(3)}</small></span></div>}
          <div className="recommendation-grid">
            {result.recommendations.map((item, index) => <article className="recommendation-card" key={item.course_code}>
              <div className="course-rank">0{index + 1}</div>
              <div className="course-top"><span>{item.course_code}</span><div><strong>{item.score}</strong><small>/100</small></div></div>
              <h3>{item.course_name}</h3>
              <p>{item.narrative ?? item.reasons[0]}</p>
              {item.predicted_success_probability !== null && <div className="success-estimate"><span>Estimated success in {item.course_code}</span><strong>{item.predicted_success_probability}%</strong><small>{item.evidence_strength} evidence · estimate, not a guarantee</small></div>}
              <p className="success-basis">{item.success_basis}</p>
              <div className="module-evidence"><span><small>Historical pass</small>{item.course_pass_rate}%</span><span><small>Withdrawal</small>{item.course_withdrawal_rate}%</span><span><small>Avg. grade</small>{item.course_average_grade}%</span></div>
              <div className="score-bars">
                {[['Module history', item.requirement_fit], ['Learner fit', item.performance_fit], ['Evidence', item.progression_fit]].map(([label, score]) => <div key={label as string}><span>{label}</span><i><b style={{ width: `${score}%` }} /></i></div>)}
              </div>
              <details className="rationale"><summary>View verified rationale <ChevronRight size={16} /></summary><ul>{item.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></details>
            </article>)}
          </div>
        </>}
      </div>
    </div>
  </section>;
}

function ImportData({ dataset, onActivated }: { dataset: DatasetInfo | null; onActivated: () => Promise<void> }) {
  const [files, setFiles] = useState<File[]>([]);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [mapping, setMapping] = useState<ImportMappingSuggestion | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activated, setActivated] = useState(false);
  const expected = ["students", "courses", "enrollments", "grades"];

  async function analyzeMapping() {
    setBusy(true); setError(null); setPreview(null); setActivated(false);
    try { setMapping(await api.suggestImportMapping(files)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "The file headers could not be analyzed."); }
    finally { setBusy(false); }
  }

  async function inspect() {
    setBusy(true); setError(null); setPreview(null); setActivated(false);
    try { setPreview(await api.previewImport(files, mapping ?? undefined)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "The files could not be validated."); }
    finally { setBusy(false); }
  }

  async function activate() {
    if (!preview) return;
    setBusy(true); setError(null);
    try { await api.commitImport(preview.token); await onActivated(); setActivated(true); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "The dataset could not be activated."); }
    finally { setBusy(false); }
  }

  async function reset() {
    setBusy(true); setError(null);
    try { await api.resetDataset(); await onActivated(); setFiles([]); setMapping(null); setPreview(null); setActivated(false); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "The dataset could not be reset."); }
    finally { setBusy(false); }
  }

  return <section>
    <div className="section-title"><div><p className="eyebrow">Dataset workspace</p><h1>Bring your own academic data.</h1></div><p>Files are validated together, staged privately for this browser session, and activated only after every relationship passes.</p></div>
    <div className="import-status">
      <div><Database size={19} /><span><small>Active dataset</small><strong>{dataset?.name ?? "Loading"}</strong></span></div>
      <div><span><small>Session boundary</small><strong>Private · expires after 30 minutes</strong></span></div>
      <button onClick={() => void reset()} disabled={busy || !dataset?.mode.startsWith("uploaded")}><RotateCcw size={15} /> Reset to OULAD</button>
    </div>
    <div className="import-layout">
      <aside className="import-steps panel">
        {["Select 1–8 files", "Normalize & validate", "Activate atomically"].map((label, index) => {
          const complete = index === 0 ? files.length > 0 : index === 1 ? Boolean(preview) : activated;
          const current = index === 0 ? files.length === 0 : index === 1 ? files.length > 0 && !preview : Boolean(preview) && !activated;
          return <div className={`${complete ? "complete" : ""} ${current ? "current" : ""}`} key={label}><span>{complete ? <FileCheck2 size={16} /> : index + 1}</span><div><strong>{label}</strong><small>{index === 0 ? "Canonical CSV tables" : index === 1 ? "Keys, types, and relationships" : "No partial replacements"}</small></div></div>;
        })}
      </aside>
      <div className="import-workspace panel">
        <label className="drop-zone">
          <input type="file" accept=".csv,text/csv" multiple onChange={(event) => { setFiles(Array.from(event.target.files ?? []).slice(0, 8)); setMapping(null); setPreview(null); setActivated(false); setError(null); }} />
          <div className="upload-icon"><UploadCloud size={28} /></div>
          <h2>Choose one or more academic CSV files</h2>
          <p>Flat history, partial package, or four canonical tables · up to 8 files</p>
          <span>Browse files</span>
        </label>
        <div className="role-grid">
          {expected.map((role) => {
            const match = preview?.files.find((item) => item.role === role);
            const selected = files.find((file) => file.name.toLowerCase().includes(role.replace("enrollments", "enrollment").replace("students", "student")));
            return <div className={match ? "ready" : ""} key={role}><FileSpreadsheet size={18} /><span><strong>{role}.csv</strong><small>{match ? `${match.rows.toLocaleString()} rows verified` : selected?.name ?? "Waiting for file"}</small></span>{match && <FileCheck2 size={17} />}</div>;
          })}
        </div>
        {error && <div className="error-banner"><CircleAlert size={19} />{error}</div>}
        {preview?.warnings.map((warning) => <div className="warning-banner" key={warning}><CircleAlert size={18} />{warning}</div>)}
        {mapping && <div className="mapping-review">
          <div className="mapping-head"><div><Sparkles size={17} /><span><strong>{mapping.ai_used ? "AI-assisted normalization plan" : "Deterministic normalization plan"}</strong><small>{mapping.note ?? "Only filenames and column headers were analyzed. Review before applying."}</small></span></div><i>{mapping.safe_to_apply ? "Ready for confirmation" : "Missing academic identity or outcomes"}</i></div>
          {mapping.mappings.map((file) => <div className="mapping-file" key={file.filename}><div><strong>{file.filename}</strong><span>{file.role}</span></div><p>{file.columns.map((column) => `${column.source} → ${column.target}`).join(" · ") || (mapping.ingestion_mode === "semantic-adapter" ? `Dedicated ${mapping.adapter_id} transformation` : "No confident mappings")}</p>{file.missing.length > 0 && <small>Missing: {file.missing.join(", ")}</small>}</div>)}
        </div>}
        {activated && <div className="success-banner"><FileCheck2 size={19} />Dataset activated for this browser session. Dashboard and Copilot now use version {preview?.dataset_version}{preview?.capabilities.historical_recommendations ? "; course planning is also available." : "; course planning is hidden because individual course history is unavailable."}</div>}
        {preview && !activated && <div className="preview-table">
          <div className="preview-head"><strong>Validation report</strong><span>{preview.mode}</span></div>
          {preview.files.map((file) => <div key={file.filename}><span>{file.filename}</span><strong>{file.role}</strong><small>{file.rows.toLocaleString()} rows · {file.columns.length} columns</small></div>)}
        </div>}
        {preview && <div className="capability-report"><strong>Available after activation</strong><span>Dashboard ✓</span><span>Natural-language analytics {preview.capabilities.natural_language_analytics ? "✓" : "—"}</span><span>Historical recommendations {preview.capabilities.historical_recommendations ? "✓" : "— no individual course history"}</span><span>Graduation-aware planning {preview.capabilities.graduation_aware_recommendations ? "✓" : "— needs catalog and requirements data"}</span></div>}
        <div className="import-actions">
          <p>{preview ? activated ? "This dataset is active only for the current browser session." : "Preview is valid for 10 minutes. Your active dataset has not changed yet." : <>Uploads remain in memory only and are never sent to Groq. <a href="/api/import/templates">Download starter templates</a></>}</p>
          {!mapping && !preview && <button className="primary" disabled={busy || files.length === 0} onClick={() => void analyzeMapping()}>{busy ? "Analyzing…" : "Analyze & normalize"}<Sparkles size={16} /></button>}
          {mapping && !preview && <button className="primary" disabled={busy || !mapping.safe_to_apply} onClick={() => void inspect()}>{busy ? "Validating…" : "Confirm mapping & validate"}<ArrowRight size={16} /></button>}
          {preview && !activated && <button className="primary" disabled={busy} onClick={() => void activate()}>{busy ? "Activating…" : "Activate dataset"}<ArrowRight size={16} /></button>}
        </div>
      </div>
    </div>
  </section>;
}

export default function App() {
  const [view, setView] = useState<View>("overview");
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [students, setStudents] = useState<Student[]>([]);
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [aiEnabled, setAiEnabled] = useState(false);
  const [dashboardFilters, setDashboardFilters] = useState<DashboardFilters>({});
  const [recommendationRiskFilter, setRecommendationRiskFilter] = useState<Student["risk"] | "All">("All");
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      const [dashboardData, studentData, config, datasetInfo] = await Promise.all([api.dashboard(dashboardFilters), api.students(), api.config(), api.dataset()]);
      setDashboard(dashboardData); setStudents(studentData); setAiEnabled(config.ai_enabled); setDataset(datasetInfo); setError(null);
    } catch { setError("The analytics service is unavailable. Start the API and refresh this page."); }
  }, [dashboardFilters]);

  useEffect(() => { void loadData(); }, [loadData]);

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div><GraduationCap size={23} /></div><span>Northstar<small>Student intelligence</small></span></div>
      <nav aria-label="Primary navigation">
        {navItems.filter((item) => item.id !== "recommendations" || dataset?.capabilities.historical_recommendations !== false).map(({ id, label, icon: Icon }) => <button aria-label={label} className={view === id ? "active" : ""} onClick={() => { if (id === "recommendations") setRecommendationRiskFilter("All"); setView(id); }} key={id}><Icon size={19} /><span>{label}</span></button>)}
      </nav>
      <div className="sidebar-foot"><Database size={18} /><span><strong>{dashboard?.dataset_name ?? "Connecting…"}</strong><small>{dashboard?.mode ?? "Loading dataset"}</small></span></div>
    </aside>
    <main>
      <header className="topbar"><div className="mobile-brand"><GraduationCap size={22} /> Northstar</div><div className="dataset-chip"><span className="live-dot" />{dashboard?.dataset_name ?? "Loading dataset"}<small>{dashboard?.dataset_version}</small></div><div className="admin-avatar">RA</div></header>
      <div className="page-content">
        {error && <div className="error-banner"><CircleAlert size={20} />{error}</div>}
        {!error && !dashboard && <Loading />}
        {dashboard && view === "overview" && <Overview data={dashboard} dataset={dataset} filters={dashboardFilters} onFilter={setDashboardFilters} onNavigate={setView} onOpenPriority={() => { setRecommendationRiskFilter("High"); setView("recommendations"); }} />}
        {dashboard && view === "copilot" && <Copilot aiEnabled={aiEnabled} dataset={dataset} />}
        {dashboard && view === "recommendations" && <Recommendations students={students} initialRisk={recommendationRiskFilter} />}
        {dashboard && view === "import" && <ImportData dataset={dataset} onActivated={loadData} />}
      </div>
    </main>
  </div>;
}
