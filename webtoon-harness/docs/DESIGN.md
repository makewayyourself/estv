# Webtoon Harness v2 — Design Contract

This document is the **single source of truth** for how every skill, agent persona,
and script in this harness behaves. All contributors (human or agent) MUST follow it
so the pieces compose. It also records *why* v2 differs from the v1 harness it is
modeled on (`revfactory/webtoon-harness`).

---

## 0. What this is

A Claude Code harness that turns a one-line request ("make a webtoon episode") into a
finished vertical-scroll webtoon chapter: trend research → high-tension, dialogue-led,
twist-every-episode scenario → reference-locked character art → 50+ rendered panels →
QA'd vertical-scroll viewer. It runs as a team of specialized agent personas, driven
by a **deterministic Workflow script** (primary) with an **agent-team fallback**.

## 1. The v1 → v2 upgrade thesis

v2 keeps v1's strongest ideas (reference-first consistency, validate-regenerate loop,
phase teams, workspace audit trail) and fixes its five documented weak points.

| # | v1 weakness | v2 upgrade | Where it lives |
|---|-------------|-----------|----------------|
| U1 | **Korean text baked into images breaks** (v1's #1 failure mode) | **Hybrid lettering**: text is rendered as a real HTML/SVG overlay using bundled fonts (NanumGothic) by default; baking is opt-in per panel. Guaranteed-correct Hangul typography. | `skills/webtoon-lettering`, `scripts/lettering/overlay.mjs`, `viewer/template.html` |
| U2 | **Non-deterministic team coordination** (TeamCreate + SendMessage) is fragile and unrepeatable | **Deterministic Workflow orchestration**: a single JS script defines fan-out/pipeline, with resume after failure and a token budget ceiling. Team mode kept as fallback. | `workflows/webtoon-episode.js`, `skills/webtoon-orchestrator` |
| U3 | **Hard-locked to codex CLI** (ChatGPT OAuth, global 5-session cap = structural bottleneck) | **Pluggable image backend**: one adapter contract, swappable backends (codex / gpt-image / gemini / local SD). Concurrency is a config knob, not a law. | `scripts/imagegen/` |
| U4 | **Continuity tracked only as prose** → drifts silently | **Structured continuity ledger + story bible (JSON)**, schema-validated, machine-checkable across episodes. | `schemas/`, `continuity-manager` |
| U5 | **Character consistency judged by eye**; only md5 catches dupes | **Objective C1 gate**: perceptual-hash / embedding similarity vs the reference sheet, plus md5 dedupe and 0-byte/corruption checks. | `scripts/validate/panel_check.mjs`, `panel-validator` |
| U6 | **No render cache** → re-runs redraw everything | **Content-addressed panel cache**: identical (prompt+refs+seed) → cache hit, skipped. Composes with Workflow resume. | `scripts/cache/panel_cache.mjs` |

If a proposed change is not at least one of U1–U6 (or a correctness fix), it is "just
similar" and should not be added — keep the surface small.

## 2. Directory layout (contract)

```
webtoon-harness/
├── README.md                     # human entry point
├── docs/DESIGN.md                # this file
├── .claude/
│   ├── skills/<skill>/SKILL.md   # 7 skills
│   ├── agents/<persona>.md       # 27 personas
│   └── workflows/webtoon-episode.js
├── schemas/*.schema.json         # story-bible, continuity-ledger, panel-manifest
├── scripts/
│   ├── imagegen/{adapter.md,render.sh,backends/*.sh}
│   ├── lettering/overlay.mjs
│   ├── validate/panel_check.mjs
│   └── cache/panel_cache.mjs
└── viewer/template.html
```

Runtime artifacts live in the **consuming project's** `_workspace/` (never committed
by the harness). The harness itself is copy-into-your-project, exactly like v1.

## 3. The `_workspace/` artifact layout (contract)

Every persona reads its inputs from and writes its outputs to these fixed paths.
`{NN}` is the zero-padded episode number.

```
_workspace/
├── 00_input/brief.md                         # request, episode #, constraints
├── 01_research/{trend-scout,platform-ranker,audience-analyst,hook-analyst}.md
│   └── trend-brief.md
├── 02_story/{concept,world,characters,series-arc,twist-plan,tension-curve}.md
│   ├── story-bible.json                       # ← schema: story-bible.schema.json (U4)
│   └── continuity-ledger.json                 # ← schema: continuity-ledger.schema.json (U4)
├── 03_episode/ep{NN}_{beatsheet,script,script_final}.md
├── 04_visual/
│   ├── style-bible.md  character-sheets.md
│   ├── refs/*.png  refs/INDEX.md              # reference sheets (rendered first)
│   ├── ep{NN}_{shotlist,lettering,prompts,validation}.md
│   └── ep{NN}_panels.json                     # ← schema: panel-manifest.schema.json
├── 05_panels/ep{NN}/panel_*.png              # validated renders (art only; text via overlay)
├── 06_assembly/ep{NN}/{index.html,qa_report.md}
│   └── continuity.md
└── RELEASE/ep{NN}/                            # signed-off package
```

## 4. The team (27 personas, 4 phase-teams)

Sequential phase-teams; one active team at a time. Outputs persist in `_workspace/`
so the next team reads them. Personas map 1:1 to v1 names (so v1 users transfer
cleanly) — the upgrades are in *how* they validate and hand off, not the roster.

- **Research (5):** trend-scout, platform-ranker, audience-analyst, hook-analyst, trend-synthesizer
- **Scenario (9):** concept-architect, worldbuilder, character-designer, series-plotter, twist-master, tension-engineer, episode-outliner, dialogue-writer, script-editor
- **Visual (9):** art-director, ref-sheet-artist, panel-director, letterer, prompt-smith, panel-artist-a, panel-artist-b, panel-artist-c, panel-validator
- **Assembly (4):** episode-compositor, quality-reviewer, continuity-manager, showrunner

### Persona file contract (`.claude/agents/<name>.md`)
Every persona file has YAML frontmatter then a body:
```markdown
---
name: <persona-id>
description: <one line: what it does, when to use it>
tools: Read, Write, Edit, Glob, Grep, Bash   # least privilege; only what the role needs
model: opus
---
# <Persona Name> — <role>
## Mission            (1–2 sentences)
## Inputs             (exact _workspace/ paths it reads)
## Outputs            (exact _workspace/ paths it writes)
## Method             (numbered steps)
## Definition of done (checklist; the gate before handoff)
## Upgrade hooks       (which of U1–U6 this persona is responsible for, if any)
```
Personas write **files**, not chat prose. Image-rendering personas call the imagegen
adapter (§6), never a backend directly. Validators emit machine-readable verdicts.

## 5. Story principles (non-negotiable creative gates)

1. **Dialogue-led**: scripts are conversation-forward; minimize narration boxes.
2. **High tension**: every episode rises to a cliffhanger (tension-engineer owns the curve).
3. **A twist every episode**: twist-master writes it into `twist-plan.md`; script-editor
   verifies the twist *lands clearly* in `script_final`.
4. **50+ panels**: episode-outliner splits beats until ≥50; panel-director never drops below.
5. **Series continuity**: world/style/character/refs are reused across episodes, never
   re-derived. continuity-manager reconciles `continuity-ledger.json` after each episode.

## 6. Image generation contract (U3)

All rendering goes through one adapter. Personas call:
```
scripts/imagegen/render.sh --backend "$WT_IMG_BACKEND" --prompt-file <md> \
  --out <png> --ref <ref.png> [--seed N] [--concurrency K]
```
- `render.sh` resolves the backend from `--backend` / `$WT_IMG_BACKEND` (default `codex`),
  enforces `--concurrency` (default 5, overridable — **not** a structural cap), and
  routes to `backends/<name>.sh`.
- Each backend implements the same CLI contract (see `scripts/imagegen/adapter.md`):
  inputs `--prompt-file --out --ref --seed`, exit 0 on a valid PNG.
- **Prompt rule:** because text is overlaid (U1), default prompts request *empty speech
  balloons / clean art with negative space for text*, with negative `no English/gibberish/
  misspelled text`. Only `--bake` panels request in-image Hangul.

## 7. Lettering contract (U1)

- Default = **overlay**. `letterer` emits `ep{NN}_lettering.md` + entries in
  `ep{NN}_panels.json` (`balloons[]`: text, speaker, box xywh %, tail dir, style).
- `scripts/lettering/overlay.mjs` + `viewer/template.html` render balloons as DOM/SVG
  over the art PNG using bundled fonts → perfect Hangul, editable, re-typesettable.
- **bake** mode is per-panel opt-in (e.g. SFX integrated into art); when baked, the
  panel-validator's C3 text check applies; when overlaid, C3 is N/A (text is always correct).

## 8. Validation contract (U5) — `panel_check.mjs` + panel-validator

6 axes; each panel gets ACCEPT / REGEN(reason) / ACCEPT-FLAG.
- **C1 character/reference** — perceptual-hash + embedding similarity vs `refs/`; below
  threshold → REGEN. *(objective, not eyeballed)*
- **C2 background/location continuity** — scene's `location` token consistent within a scene.
- **C3 balloon/Hangul text** — only for `--bake` panels (overlaid text is exact).
- **C4 prompt fidelity** — panel depicts its shotlist beat.
- **C5 dialogue flow** — reading order/scene sense across the batch.
- **C6 integrity** — non-zero, valid PNG header, **md5 dedupe** (distinct panels ≠ same image).
- Loop: REGEN → prompt-smith strengthens prompt → owning panel-artist re-renders *that
  panel only* → re-check. **Max 3 tries/panel**, then ACCEPT-FLAG with a logged limitation.

## 9. Orchestration contract (U2)

- **Primary:** `.claude/workflows/webtoon-episode.js`, invoked via the Workflow tool.
  It runs phases as `pipeline()`/`parallel()` stages, spawns personas with
  `agent(prompt, {agentType:'<persona>', model:'opus'})`, and is resumable.
- **Phase 0 routing** (run before anything): inspect `_workspace/` + request →
  initial / next-episode / partial-rerun / new-project. Reuse world/style/refs on
  next-episode; only re-run affected downstream stages on partial-rerun.
- **Fallback:** agent-team mode (TeamCreate/TaskCreate/SendMessage/TeamDelete) for
  environments without the Workflow tool. The orchestrator SKILL documents both.
- **Budget/limits:** panel regen ≤3×; rework loop ≤2×; never delete conflicting data
  (annotate sources instead).

## 10. Caching contract (U6)

`panel_cache.mjs` keys on `sha256(prompt + ref-ids + seed + backend)`. Before a render,
check cache → hit copies the PNG and skips the backend. After an ACCEPTed render, store.
Cache lives under `_workspace/.cache/panels/`. Workflow resume + cache means a re-run
only redraws what actually changed.

## 11. House style for generated files

- Korean for creative/user-facing copy (scripts, briefs); English allowed in code/JSON keys.
- Markdown outputs are concrete and parseable (tables, fixed headings) — downstream
  personas parse them.
- Scripts: POSIX `sh` for backends, Node ESM (`.mjs`) for JSON/image utilities, no
  external npm deps beyond what's vendored. Fail loudly with non-zero exit + stderr.
