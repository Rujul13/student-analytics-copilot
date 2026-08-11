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
import type { DashboardData, DatasetInfo, ImportPreview, QueryResponse, RecommendationResponse, Student } from "./types";

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

function BarList({ points, suffix = "%" }: { points: DashboardData["outcomes"]; suffix?: string }) {
  const max = Math.max(...points.map((point) => point.value), 1);
  return (
    <div className="bar-list">
      {points.map((point) => (
        <div className="bar-row" key={point.label}>
          <div className="bar-meta"><span>{point.label}</span><strong>{point.value}{suffix}</strong></div>
          <div className="bar-track"><span style={{ width: `${(point.value / max) * 100}%` }} /></div>
          <small>{point.count} records</small>
        </div>
      ))}
    </div>
  );
}

function Overview({ data, dataset, onNavigate }: { data: DashboardData; dataset: DatasetInfo | null; onNavigate: (view: View) => void }) {
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

      {dataset && <section className="provenance-strip" aria-label="Dataset provenance">
        <div><Database size={19} /><span><small>Verified source</small><strong>{dataset.source}</strong></span></div>
        <div><span><small>Canonical cohort</small><strong>{dataset.tables.students.toLocaleString()} learners · {dataset.tables.enrollments.toLocaleString()} histories</strong></span></div>
        <div><span><small>License & citation</small><strong>{dataset.doi ? `${dataset.license} · DOI ${dataset.doi}` : dataset.license}</strong></span></div>
      </section>}

      <section className="content-grid">
        <article className="panel wide">
          <div className="panel-heading"><div><p className="eyebrow">Academic performance</p><h2>Module pulse</h2></div><span className="tag">Average grade</span></div>
          <BarList points={data.modules} />
        </article>
        <article className="panel">
          <div className="panel-heading"><div><p className="eyebrow">Learner outcomes</p><h2>Result mix</h2></div></div>
          <BarList points={data.outcomes} />
        </article>
        <article className="panel dark-panel">
          <p className="eyebrow">Priority signal</p>
          <h2>{highRiskCount} {highRiskCount === 1 ? "learner needs" : "learners need"} attention</h2>
          <p>Risk is calculated from assessment performance and withdrawal history—not protected demographic attributes.</p>
          <button onClick={() => onNavigate("recommendations")}>Open student explorer <ChevronRight size={17} /></button>
        </article>
      </section>
    </>
  );
}

function Copilot({ aiEnabled }: { aiEnabled: boolean }) {
  const suggestions = ["What is the completion rate?", "Show me at-risk learners", "What is the average grade?"];
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event?: FormEvent, preset?: string) {
    event?.preventDefault();
    const value = preset ?? question;
    if (!value.trim()) return;
    setQuestion(value); setBusy(true);
    try { setResult(await api.query(value)); } finally { setBusy(false); }
  }

  return <section className="copilot-layout">
    <div className="copilot-intro">
      <div className="orb"><Bot size={34} /></div>
      <p className="eyebrow">Natural-language analytics</p>
      <h1>Ask the dataset.<br />Get the evidence.</h1>
      <p>Every response is constrained to approved calculations and includes a trace you can audit.</p>
      <div className={`status-pill ${aiEnabled ? "online" : "offline"}`}><span />{aiEnabled ? "AI explanations enabled" : "Safe analytics mode · API key pending"}</div>
    </div>
    <div className="conversation-card">
      <div className="suggestions">
        {suggestions.map((item) => <button key={item} onClick={() => void submit(undefined, item)}>{item}</button>)}
      </div>
      <div className="answer-area" aria-live="polite">
        {!result && !busy && <div className="empty-answer"><BarChart3 size={32} /><h2>Your answer will appear here</h2><p>Try one of the suggested questions or write your own.</p></div>}
        {busy && <Loading />}
        {result && !busy && <>
          <div className="answer-label"><Sparkles size={16} /> Answer</div>
          <h2>{result.answer}</h2>
          {result.rows.length > 1 && <div className="mini-table">
            {result.rows.slice(0, 6).map((row, index) => <div key={index}><strong>{String(row.display_name ?? row.metric ?? row.module ?? `Result ${index + 1}`)}</strong><span>{row.risk ? `${row.risk} risk` : row.average_grade !== undefined ? `${row.average_grade}%` : String(row.value ?? "")}</span></div>)}
          </div>}
          <details><summary>Calculation trace</summary><ol>{result.calculation_trace.map((step) => <li key={step}>{step}</li>)}</ol></details>
        </>}
      </div>
      <form className="query-box" onSubmit={submit}>
        <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about performance, completion, or risk…" aria-label="Ask the dataset" />
        <button disabled={busy || question.trim().length < 3} aria-label="Send question"><Send size={18} /></button>
      </form>
    </div>
  </section>;
}

function Recommendations({ students }: { students: Student[] }) {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Student | null>(students[0] ?? null);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const filtered = useMemo(() => students.filter((student) => `${student.display_name} ${student.student_id}`.toLowerCase().includes(search.toLowerCase())).slice(0, 12), [students, search]);

  useEffect(() => {
    if (!selected) return;
    setResult(null);
    void api.recommendations(selected.student_id).then(setResult);
  }, [selected]);

  return <section>
    <div className="section-title"><div><p className="eyebrow">Course recommendation engine</p><h1>Plan the next best step.</h1></div><p>Eligibility first. Transparent scoring second. AI explanation last.</p></div>
    <div className="catalog-disclosure"><CircleAlert size={17} /><span><strong>Data boundary:</strong> learner history, grades, and completed module codes are authentic OULAD records. Program pathways, future courses, names, prerequisites, and offerings are fictional demo enrichment.</span></div>
    <div className="student-layout">
      <aside className="student-list panel">
        <label className="search-box"><Search size={17} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find a learner" /></label>
        <div className="student-items">
          {filtered.map((student) => <button className={selected?.student_id === student.student_id ? "selected" : ""} onClick={() => setSelected(student)} key={student.student_id}>
            <span className="avatar">{student.display_name.split(" ").at(-1)?.slice(-2)}</span>
            <span><strong>{student.display_name}</strong><small>{student.student_id} · {student.average_grade}%</small></span>
            <i className={`risk ${student.risk.toLowerCase()}`}>{student.risk}</i>
          </button>)}
        </div>
      </aside>
      <div className="recommendation-space">
        {selected && <div className="student-banner">
          <div><p className="eyebrow">Selected learner</p><h2>{selected.display_name}</h2><span>{selected.program} · {selected.credits_earned} credits earned</span></div>
          <div className="grade-ring"><strong>{selected.average_grade}</strong><small>avg.</small></div>
        </div>}
        {!result && <Loading />}
        {result && <>
          <div className="mode-note"><Database size={16} /><span><strong>{result.capability_mode}</strong> recommendation mode · {result.catalog_label} · {result.ai_explanation_enabled ? "AI explanations on" : "deterministic explanations"}</span></div>
          <div className="recommendation-grid">
            {result.recommendations.map((item, index) => <article className="recommendation-card" key={item.course_code}>
              <div className="course-rank">0{index + 1}</div>
              <div className="course-top"><span>{item.course_code}</span><div><strong>{item.score}</strong><small>/100</small></div></div>
              <h3>{item.course_name}</h3>
              <p>{item.narrative ?? item.reasons[0]}</p>
              <div className="score-bars">
                {[['Requirement', item.requirement_fit], ['Performance', item.performance_fit], ['Progression', item.progression_fit]].map(([label, score]) => <div key={label as string}><span>{label}</span><i><b style={{ width: `${score}%` }} /></i></div>)}
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
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activated, setActivated] = useState(false);
  const expected = ["students", "courses", "enrollments", "grades"];

  async function inspect() {
    setBusy(true); setError(null); setPreview(null); setActivated(false);
    try { setPreview(await api.previewImport(files)); }
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
    try { await api.resetDataset(); await onActivated(); setFiles([]); setPreview(null); setActivated(false); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "The dataset could not be reset."); }
    finally { setBusy(false); }
  }

  return <section>
    <div className="section-title"><div><p className="eyebrow">Dataset workspace</p><h1>Bring your own academic data.</h1></div><p>Files are validated together, staged privately for this browser session, and activated only after every relationship passes.</p></div>
    <div className="import-status">
      <div><Database size={19} /><span><small>Active dataset</small><strong>{dataset?.name ?? "Loading"}</strong></span></div>
      <div><span><small>Session boundary</small><strong>Private · expires after 30 minutes</strong></span></div>
      <button onClick={() => void reset()} disabled={busy || dataset?.mode.startsWith("canonical")}><RotateCcw size={15} /> Reset to OULAD</button>
    </div>
    <div className="import-layout">
      <aside className="import-steps panel">
        {["Select four files", "Validate schema", "Activate atomically"].map((label, index) => {
          const complete = index === 0 ? files.length === 4 : index === 1 ? Boolean(preview) : activated;
          const current = index === 0 ? files.length !== 4 : index === 1 ? files.length === 4 && !preview : Boolean(preview) && !activated;
          return <div className={`${complete ? "complete" : ""} ${current ? "current" : ""}`} key={label}><span>{complete ? <FileCheck2 size={16} /> : index + 1}</span><div><strong>{label}</strong><small>{index === 0 ? "Canonical CSV tables" : index === 1 ? "Keys, types, and relationships" : "No partial replacements"}</small></div></div>;
        })}
      </aside>
      <div className="import-workspace panel">
        <label className="drop-zone">
          <input type="file" accept=".csv,text/csv" multiple onChange={(event) => { setFiles(Array.from(event.target.files ?? []).slice(0, 4)); setPreview(null); setActivated(false); setError(null); }} />
          <div className="upload-icon"><UploadCloud size={28} /></div>
          <h2>Choose four canonical CSV files</h2>
          <p>students, courses, enrollments, and grades · 5 MB per file</p>
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
        {activated && <div className="success-banner"><FileCheck2 size={19} />Dataset activated for this browser session. Dashboard, Copilot, and recommendations now use version {preview?.dataset_version}.</div>}
        {preview && !activated && <div className="preview-table">
          <div className="preview-head"><strong>Validation report</strong><span>{preview.mode}</span></div>
          {preview.files.map((file) => <div key={file.filename}><span>{file.filename}</span><strong>{file.role}</strong><small>{file.rows.toLocaleString()} rows · {file.columns.length} columns</small></div>)}
        </div>}
        <div className="import-actions">
          <p>{preview ? activated ? "This dataset is active only for the current browser session." : "Preview is valid for 10 minutes. Your active dataset has not changed yet." : <>Uploads remain in memory only and are never sent to Groq. <a href="/api/import/templates">Download starter templates</a></>}</p>
          {!preview && <button className="primary" disabled={busy || files.length !== 4} onClick={() => void inspect()}>{busy ? "Validating…" : "Validate files"}<ArrowRight size={16} /></button>}
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
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      const [dashboardData, studentData, config, datasetInfo] = await Promise.all([api.dashboard(), api.students(), api.config(), api.dataset()]);
      setDashboard(dashboardData); setStudents(studentData); setAiEnabled(config.ai_enabled); setDataset(datasetInfo); setError(null);
    } catch { setError("The analytics service is unavailable. Start the API and refresh this page."); }
  }, []);

  useEffect(() => { void loadData(); }, [loadData]);

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div><GraduationCap size={23} /></div><span>Northstar<small>Student intelligence</small></span></div>
      <nav aria-label="Primary navigation">
        {navItems.map(({ id, label, icon: Icon }) => <button aria-label={label} className={view === id ? "active" : ""} onClick={() => setView(id)} key={id}><Icon size={19} /><span>{label}</span></button>)}
      </nav>
      <div className="sidebar-foot"><Database size={18} /><span><strong>OULAD Lite</strong><small>{dashboard?.mode ?? "Connecting…"}</small></span></div>
    </aside>
    <main>
      <header className="topbar"><div className="mobile-brand"><GraduationCap size={22} /> Northstar</div><div className="dataset-chip"><span className="live-dot" />{dashboard?.dataset_name ?? "Loading dataset"}<small>{dashboard?.dataset_version}</small></div><div className="admin-avatar">RA</div></header>
      <div className="page-content">
        {error && <div className="error-banner"><CircleAlert size={20} />{error}</div>}
        {!error && !dashboard && <Loading />}
        {dashboard && view === "overview" && <Overview data={dashboard} dataset={dataset} onNavigate={setView} />}
        {dashboard && view === "copilot" && <Copilot aiEnabled={aiEnabled} />}
        {dashboard && view === "recommendations" && <Recommendations students={students} />}
        {dashboard && view === "import" && <ImportData dataset={dataset} onActivated={loadData} />}
      </div>
    </main>
  </div>;
}
