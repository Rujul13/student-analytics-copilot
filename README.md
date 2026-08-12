# Student Analytics Copilot

An evidence-first full-OULAD analytics application with three core experiences:

- deterministic academic dashboard;
- verified natural-language data querying;
- transparent course recommendations.

The living rationale for the system design, implemented trade-offs, evaluated alternatives, and improvement roadmap is maintained in [DECISIONS.md](DECISIONS.md).

The recommendation engine considers every eligible authentic OULAD module code. A packaged, temporally evaluated learner-and-module model combines prior grades, outcomes, credits, attempts, study load, prior education, historical VLE engagement, and module outcome evidence. GPT-OSS 120B selects and ranks three from the complete verified eligible set and explains the order; it cannot invent a module or alter calculated evidence. If live AI is unavailable, the top three are selected deterministically. Because OULAD does not contain future offerings, prerequisites, or degree requirements, administrators are told to verify those before acting.

## Enhanced assignment coverage

- Dashboard filters and clickable drill-downs cover module, presentation, and outcome, with an outcome donut visualization.
- The analytics copilot keeps bounded conversation history, asks GPT-OSS 120B to write Pandas against the four canonical dataframes, validates the program structurally and semantically, executes it in an isolated local process, and asks GPT-OSS 20B to phrase only the computed result.
- Assignment-level questions for distinctions and learners failing multiple courses are supported, along with scoped learner profiles and recommendation questions.
- CSV imports include an AI-assisted header-mapping review step. Only filenames and headers are sent for mapping; users must confirm before deterministic validation and atomic activation.
- Recommendation eligibility and evidence remain deterministic. GPT-OSS 120B selects three from the complete eligible set through a strict schema, and the UI reports whether AI selection or deterministic fallback produced the result.
- A temporal learner-and-module logistic baseline provides course-specific held-out success estimates with accuracy, ROC AUC, and Brier score disclosed in the UI. Evidence strength is shown as limited, moderate, or strong; estimates are explicitly not guarantees.

## Current build status

The deployed build loads the complete official OULAD academic cohort from compact Parquet artifacts generated through DuckDB. It contains 28,785 anonymized learners, 32,593 module enrollments, and 25,843 aggregated graded histories. Raw VLE interactions are aggregated offline into historical click and active-day features and are not scanned at request time.

## Local setup

1. Copy `.env.example` to `.env` and leave `GROQ_API_KEY` blank until needed.
2. Create a Python environment and install `backend/requirements.txt`.
3. Start the API from `backend` with `uvicorn app.main:app --reload`.
4. Install frontend packages in `frontend` and run `npm run dev`.
5. Open `http://127.0.0.1:5173`.

## Activating OULAD

Download the OULAD CSV package and place these files in a private local directory:

- `studentInfo.csv`
- `studentRegistration.csv`
- `courses.csv`
- `assessments.csv`
- `studentAssessment.csv`

The current workspace already contains these files in the ignored local data directory. Run `python backend/scripts/build_full_oulad.py` to regenerate the deployable Parquet feature store and evaluated model from the full source archive. The service still exposes the canonical `students`, `courses`, `enrollments`, and `grades` DataFrames required by the Pandas agent. Raw data is intentionally ignored by Git.

The reproducible canonical cohort is stored in `backend/data/processed` with its manifest and is included in the deployment image. Rebuild it after changing the selection logic with `python backend/scripts/build_oulad_lite.py`.

## AI workflow behavior

With `GROQ_API_KEY`, a bounded LlamaIndex Workflow sends schema metadata—not full dataframe rows—to `openai/gpt-oss-120b`. The model returns a strict, schema-validated Pandas program for the active session's `students`, `courses`, `enrollments`, and `grades` dataframes. The service performs AST validation and deterministic scope checks, runs accepted code locally in a short-lived `multiprocessing` child with a five-second timeout, and normalizes the result to a bounded evidence payload. `openai/gpt-oss-20b` then turns that verified payload into a concise answer. One repair attempt is permitted; there is no open-ended agent loop. Generated code, prompts, and raw exceptions are never returned to the browser.

Without the key—or when the live agent is unavailable—the dashboard, deterministic recommendations, and a limited deterministic analytics fallback continue to work. Query responses identify generated Pandas, repaired generated Pandas, deterministic fallback, or unsupported execution. The UI shows only a short provenance status, not generated code or hidden reasoning.

## Importing another dataset

Open **Import data** and select one to eight CSV files. The importer accepts a single flat academic-history file, partial related tables, the four canonical tables, or the five core official OULAD files. AI-assisted header analysis proposes a normalization plan; deterministic Pandas/DuckDB transformations construct the four canonical tables. The preview reports available capabilities and validates types, ranges, keys, and cross-table relationships before issuing a ten-minute activation token.

Uploaded data is held only in server memory, expires after 30 minutes of inactivity, and is isolated by an opaque HTTP-only session cookie. The data agent sends Groq only schema metadata, bounded categorical examples, row counts, and metric definitions—not full uploaded rows. Resetting returns only that browser session to the bundled full OULAD cohort.

## Deployment controls

The production container runs as a non-root user, excludes `.env` and raw OULAD files from the build context, exposes a health check, restricts CORS, adds browser security headers, limits uploads, caps in-memory sessions, and rate-limits natural-language queries per session.

## GitHub and Render deployment

The repository includes a GitHub Actions workflow that runs the backend tests, builds the frontend, builds the production Docker image, and smoke-tests the container. Render is configured through the root `render.yaml` Blueprint and deploys only after the linked branch's checks pass.

1. Push the repository to GitHub.
2. In Render, create a new Blueprint and select the repository.
3. Enter `GROQ_API_KEY` when Render prompts for the `sync: false` secret. Never commit this value.
4. Deploy the Blueprint and confirm `/api/health` returns `{"status":"ok", ...}`.

The default `free` Render instance can spin down when idle. Uploaded CSV datasets are intentionally held in memory only and are lost when a process restarts; the bundled full OULAD Parquet cohort remains available from the immutable container image.
