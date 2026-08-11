# Student Analytics Copilot

An evidence-first OULAD analytics application with three core experiences:

- deterministic academic dashboard;
- safe natural-language data querying;
- transparent course recommendations.

The recommendation engine uses a clearly labeled fictional demo catalog because OULAD does not contain prerequisites, degree requirements, or future offerings. Eligibility and ranking are deterministic; Groq may explain an existing recommendation but cannot create, remove, reorder, or rescore one.

## Current build status

The first vertical slice is implemented and the local build automatically loads the selected official OULAD tables from `backend/data/oulad`. If those ignored local files are unavailable, it falls back to a deterministic fixture that is clearly labeled in the UI.

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

The current workspace already contains these files in the ignored local data directory. For another machine, set `DATASET_PATH` to the downloaded directory. The service validates the inputs and transforms a fixed, stratified cohort of 750 learners into the canonical `students`, `courses`, `enrollments`, and `grades` tables at startup. Raw data is intentionally ignored by Git.

The reproducible canonical cohort is stored in `backend/data/processed` with its manifest and is included in the deployment image. Rebuild it after changing the selection logic with `python backend/scripts/build_oulad_lite.py`.

## AI workflow behavior

With `GROQ_API_KEY`, a bounded LlamaIndex Workflow retrieves relevant analytics capabilities with BM25 and asks Groq for a strict, schema-validated `AnalyticsPlan`. The plan can select only allowlisted Pandas executors; it cannot contain Python, SQL, expressions, column names, or function references. The result includes an audit trace.

Without the key—or when the live planner times out or fails validation—the dashboard, deterministic rankings, and safe fallback query catalog continue to work.

## Importing another dataset

Open **Import data** in the application and select the four canonical CSV files together. Downloadable starter templates are provided in the wizard. The service validates file size, required columns, numeric ranges, primary keys, and cross-table foreign keys before issuing a ten-minute preview token. Activating the token swaps all four tables atomically for that browser session.

Uploaded data is held only in server memory, expires after 30 minutes of inactivity, and is isolated by an opaque HTTP-only session cookie. It is never included in the Groq planning prompt. Resetting returns only that browser session to the bundled OULAD Lite cohort.

## Deployment controls

The production container runs as a non-root user, excludes `.env` and raw OULAD files from the build context, exposes a health check, restricts CORS, adds browser security headers, limits uploads, caps in-memory sessions, and rate-limits natural-language queries per session.

## GitHub and Render deployment

The repository includes a GitHub Actions workflow that runs the backend tests, builds the frontend, builds the production Docker image, and smoke-tests the container. Render is configured through the root `render.yaml` Blueprint and deploys only after the linked branch's checks pass.

1. Push the repository to GitHub.
2. In Render, create a new Blueprint and select the repository.
3. Enter `GROQ_API_KEY` when Render prompts for the `sync: false` secret. Never commit this value.
4. Deploy the Blueprint and confirm `/api/health` returns `{"status":"ok", ...}`.

The default `free` Render instance can spin down when idle. Uploaded CSV datasets are intentionally held in memory only and are lost when a process restarts; the bundled OULAD Lite cohort remains available from the immutable container image.
