# CLAUDE.md

Guidance for AI assistants (Claude Code and others) working in this repository.

## What this repo is

This is a **multi-project monorepo** for **ESTV**, a crypto/token venture. It is
not a single application — it is a collection of loosely-related sub-projects
(a token simulator, a real-time translator, desktop/mobile wrappers, a marketing
dashboard, a kids' reward chart) plus a large set of Korean-language business,
strategy, and tokenomics documents. Each sub-project has its own toolchain,
dependencies, and deployment target; they are grouped here mostly because they
belong to the same organization, not because they share code.

**Most user-facing documentation and business content is in Korean.** Code,
identifiers, and comments are usually in English; UI strings, docs, and commit
context are often Korean. Match the language of whatever you are editing.

## Sub-projects at a glance

| Path | Stack | What it is | Deploys to |
| :--- | :--- | :--- | :--- |
| `app.py` (root) | Python / Streamlit | ESTV Strategic AI Advisor — Monte-Carlo token price simulator with an OpenAI-powered strategy report + PDF export | Dev container / Streamlit (manual) |
| `gemini_live_translator/` | Python / FastAPI + vanilla JS | Real-time audio-to-audio interpreter on the Gemini Live API | Render (Docker) |
| `estv_desktop/` | Node / Electron | Windows desktop shell around the deployed translator ("invisible" captions for Meet/Zoom) | GitHub Release (Windows installer) |
| `marketing_dashboard/` | React 18 + Vite + Recharts | Marketing analytics dashboard | (build only) |
| `praise_grape/` | Static HTML + localStorage | 칭찬 포도 — a kids' reward-chart web app, no build step | GitHub Pages |
| `assets/fonts/` | — | NanumGothic Korean fonts, used for Korean PDF generation | — |
| `misc_data/` | — | ~200 opaque short scratch files; treat as junk/temporary unless told otherwise | — |
| Root `*.md` / `*.json` | — | Business strategy, tokenomics, contracts (MOA), analysis data — mostly Korean | — |

There is no top-level `README.md`; this file is the entry point.

## Sub-project details

### `app.py` — ESTV Strategic AI Advisor (Streamlit)

- Single-file Streamlit app. Key pieces: `TokenSimulationEngine` (daily
  buy/sell/liquidity price-walk simulation), `generate_ai_strategy_report`
  (OpenAI GPT-4 narrative), `generate_strategy_pdf` (FPDF, uses the bundled
  NanumGothic font for Korean), `scenario_text_to_inputs` (natural-language →
  simulator inputs via GPT-4), and `main()` (the UI).
- Dependencies: root `requirements.txt` (`streamlit pandas numpy matplotlib
  plotly fpdf openai python-dotenv`).
- Config: reads `OPENAI_API_KEY` from the environment / `.env` (via
  `python-dotenv`). Tokenomics vesting lives in `tokenomics_config.json`.
- Run locally: `pip install -r requirements.txt && streamlit run app.py`
  (serves on `:8501`). The dev container auto-runs this on attach.
- Korean fonts for PDF are expected at `assets/fonts/NanumGothic.ttf` and
  `assets/fonts/NanumGothic-Bold.ttf`; PDF falls back to Arial (breaks Korean)
  if missing.

### `gemini_live_translator/` — Gemini Live Translator (FastAPI)

- The most substantial code project. See its own `README.md` and `ANDROID.md`
  (Korean) for full detail.
- Backend: `main.py` (FastAPI) serves the static frontend and a WebSocket at
  `/api/stream` that bridges browser mic audio ⇄ Gemini Live. Business logic is
  in `services/` (`gemini_live.py`, `rooms.py`, `exporters.py`).
- Frontend: `static/` — plain `index.html` + `app.js` + `audio-processor.js`
  (no bundler). To add a language, extend `LANGUAGES` in `static/app.js` **and**
  `SUPPORTED_LANGUAGES` in `services/gemini_live.py` (keep them in sync).
- Non-realtime helpers run on separate text models and are fired off the
  realtime path (on `turn_complete`) so they never add latency: Risk Guard,
  Clarify, Summarize, Ask, Pronounce, Export (Markdown/Word/PDF).
- Config: copy `.env.example` → `.env`. Key vars: `GEMINI_API_KEY`,
  `GEMINI_LIVE_MODEL`, `RISK_MODEL`/`CLARIFY_MODEL`/`SUMMARY_MODEL`,
  `GEMINI_VOICE`, and `ACCESS_TOKEN` (gates the WebSocket — always set it for
  public deploys; the client connects with `?token=<value>`).
- Dependencies: `gemini_live_translator/requirements.txt` (FastAPI, uvicorn,
  `google-genai`, `python-docx`, `fpdf2`).
- Run locally: `cd gemini_live_translator && pip install -r requirements.txt &&
  uvicorn main:app --reload` (default `:8000`).
- Android: `package.json` scripts wrap Capacitor. `npm run android:build` does
  `cap sync` → `patch-android.mjs` (injects mic permissions) → Gradle
  `assembleDebug`. The app is a thin client; the API key stays on the backend.

### `estv_desktop/` — ESTV Interpreter (Electron)

- Electron shell that loads the **deployed** translator URL (`ESTV_URL`, default
  the Render server) rather than bundling the web code. `main.js` +
  `preload.js` only.
- Its purpose is desktop-only powers: `setContentProtection` (captions invisible
  to screen-share/recording), always-on-top overlay, pre-granted mic/display
  permissions. Toggles: Ctrl+Shift+H (hide from capture), Ctrl+Shift+T
  (always-on-top).
- Build: `cd estv_desktop && npm install && npm run dist:win` (electron-builder,
  NSIS installer into `dist/`). `npm start` runs it locally.

### `marketing_dashboard/` — React + Vite

- Standard Vite React app. `App.jsx` / `MarketingDashboard.jsx`, charts via
  Recharts, routing via react-router-dom. Data in
  `marketing_dashboard/marketing_data.json` (a copy also sits at repo root).
- `npm install && npm run dev` (`:5173`) / `npm run build` / `npm run preview`.

### `praise_grape/` — 칭찬 포도 (static)

- No build, no dependencies. `index.html` (app) + `promo.html` (landing).
  State persists in browser `localStorage`. Just open the file, or view the
  Pages deployment.

## Build, run & deploy commands

Each sub-project is independent — always `cd` into its directory first.

```bash
# Root Streamlit simulator
pip install -r requirements.txt && streamlit run app.py           # :8501

# Gemini translator backend
cd gemini_live_translator && pip install -r requirements.txt
uvicorn main:app --reload                                         # :8000

# Gemini translator → Android APK
cd gemini_live_translator && npm install && npm run android:build

# Desktop (Windows, run on Windows)
cd estv_desktop && npm install && npm run dist:win

# Marketing dashboard
cd marketing_dashboard && npm install && npm run dev             # :5173
```

There is **no test suite, linter, or formatter configured** in any sub-project.
Verify changes by running the relevant app manually.

## CI/CD (GitHub Actions)

All workflows live in `.github/workflows/` and are **path-filtered** — each fires
only when its sub-project changes on `main` (or via manual `workflow_dispatch`):

- `render-deploy.yml` — on changes to `gemini_live_translator/**` or
  `render.yaml`, POSTs the `RENDER_DEPLOY_HOOK` secret to trigger a Render
  redeploy. `render.yaml` (repo root) is the Render Blueprint pointing at
  `gemini_live_translator/Dockerfile`.
- `android-build.yml` — builds the translator's debug APK (JDK 21 + Node 22 +
  Capacitor + Gradle) and, on `main`, publishes it to the `apk-latest` release.
- `desktop-build.yml` — builds the Windows installer on `windows-latest` and
  publishes to the `desktop-latest` release.
- `praise-grape-pages.yml` — deploys `praise_grape/` to GitHub Pages.

Secrets used: `RENDER_DEPLOY_HOOK` (Render deploy), `GITHUB_TOKEN` (release
publishing). App runtime secrets (`GEMINI_API_KEY`, `ACCESS_TOKEN`) live in the
Render service, not in CI.

## Conventions & things to watch for

- **Independence:** don't assume shared code, config, or dependency versions
  across sub-projects. A change in one rarely affects another. Keep edits scoped
  to the sub-project you're asked about.
- **Language:** keep Korean UI/doc strings in Korean; keep code and comments in
  the style already present in the file you're editing.
- **Secrets:** never commit API keys. `.env` is git-ignored (root `.gitignore`
  and per-project ones). Use `.env` locally and platform env vars in production.
  Note: `gemini_live_translator/ci-debug.keystore` is a debug signing keystore
  checked in for CI APK builds — that is intentional and not a secret leak.
- **Fonts:** Korean PDF export in both `app.py` and the translator depends on the
  NanumGothic `.ttf` files. Don't remove them.
- **`misc_data/`** and the root business `*.md`/`*.json` files are content/data,
  not code. Don't refactor, rename, or delete them unless explicitly asked.
- **Model IDs** in the translator's `.env.example` (e.g.
  `gemini-3.5-live-translate-preview`, `gemini-3.5-flash`) are deliberately
  current for that project — don't "correct" them to older names.

## Git & branching

- Follow whatever branch you've been asked to develop on; commit with clear,
  descriptive messages and push that branch. Do **not** push to `main` or open a
  pull request unless explicitly asked.
- Default branch is `main`. The desktop app's `package.json` declares the repo
  as `github.com/makewayyourself/estv`.
