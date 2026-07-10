export const meta = {
  name: 'webtoon-episode',
  description: 'Deterministic end-to-end webtoon episode pipeline: research → scenario → reference-locked visuals → 50+ validated panels → overlay-lettered vertical-scroll viewer (U2).',
  whenToUse: 'Run a full or partial webtoon episode build with repeatable, resumable orchestration instead of ad-hoc agent-team coordination.',
  phases: [
    { title: 'Route',    detail: 'Phase 0: inspect _workspace + request to pick run mode' },
    { title: 'Prep',     detail: 'record brief, ensure workspace dirs' },
    { title: 'Research', detail: '5 personas → trend-brief' },
    { title: 'Scenario', detail: '9 personas → script_final (+ story-bible.json)' },
    { title: 'Visual',   detail: 'refs first → shotlist/lettering/prompts → render → validate loop' },
    { title: 'Assembly', detail: 'overlay viewer → QA → continuity → release' },
  ],
};

// ---------------------------------------------------------------------------
// args: { episode?: number, request?: string, mode?: string, only?: string,
//         backend?: string, concurrency?: number }
// mode (optional, else inferred by the Route persona): initial | next | partial | new
// ---------------------------------------------------------------------------
const A = (args && typeof args === 'object') ? args : {};
const EP = A.episode || 1;
const NN = String(EP).padStart(2, '0');
const REQUEST = A.request || 'Create the next webtoon episode reflecting current trends.';
const BACKEND = A.backend || 'codex';
const CONC = A.concurrency || 5;
const M = 'opus';

const ctx = `Episode {NN}=${NN} (EP=${EP}). Backend=${BACKEND}, concurrency=${CONC}.
User request: """${REQUEST}"""
Read your inputs from and write your outputs to _workspace/ exactly as defined in
docs/DESIGN.md §3. Write FILES, not chat prose. Return a one-line status.`;

const ask = (persona, task) =>
  agent(`${task}\n\n${ctx}`, { agentType: persona, label: persona, model: M });

// ── Phase 0: route ─────────────────────────────────────────────────────────
phase('Route');
const ROUTE_SCHEMA = {
  type: 'object',
  required: ['mode', 'reason', 'run_research', 'run_scenario', 'run_visual', 'run_assembly'],
  properties: {
    mode: { type: 'string', enum: ['initial', 'next', 'partial', 'new'] },
    reason: { type: 'string' },
    run_research: { type: 'boolean' },
    run_scenario: { type: 'boolean' },
    run_visual: { type: 'boolean' },
    run_assembly: { type: 'boolean' },
    reuse: { type: 'array', items: { type: 'string' } },
  },
};
const route = A.mode
  ? { mode: A.mode, reason: 'forced via args.mode', run_research: A.mode === 'initial' || A.mode === 'new',
      run_scenario: A.mode !== 'partial' || /story|script|twist|scenario/i.test(A.only || ''),
      run_visual: true, run_assembly: true, reuse: [] }
  : await agent(
      `Decide the run mode (DESIGN §9 Phase 0). Inspect whether _workspace/ exists and what it
contains (ls -R _workspace 2>/dev/null), and read the user request. Choose:
- initial: no _workspace → run everything.
- next: _workspace exists + "next episode" → reuse 02_story/world/style/refs/continuity; run scenario(from beatsheet)→visual→assembly for the new ${NN}.
- partial: "redo only X" → run just the affected stage(s) downstream. only="${A.only || ''}".
- new: brand-new premise → archive _workspace to _workspace_archive and run everything.
Return which stages to run. Do NOT delete data; archive instead.\n\n${ctx}`,
      { agentType: 'showrunner', label: 'route', model: M, schema: ROUTE_SCHEMA });
log(`mode=${route.mode} — ${route.reason}`);

// ── Phase 1: prep ───────────────────────────────────────────────────────────
phase('Prep');
await ask('showrunner',
  `PREP ONLY: write _workspace/00_input/brief.md (request, episode ${EP}, constraints) and
ensure dirs: mkdir -p _workspace/{00_input,01_research,02_story,03_episode,04_visual/refs,05_panels/ep${NN},06_assembly/ep${NN},RELEASE/ep${NN},.cache/panels}.`);

// ── Phase 2: research (fan-out → synthesize) ────────────────────────────────
if (route.run_research) {
  phase('Research');
  await parallel([
    () => ask('trend-scout', 'Produce 01_research/trend-scout.md.'),
    () => ask('platform-ranker', 'Produce 01_research/platform-ranker.md.'),
    () => ask('audience-analyst', 'Produce 01_research/audience-analyst.md.'),
    () => ask('hook-analyst', 'Produce 01_research/hook-analyst.md.'),
  ]);
  await ask('trend-synthesizer',
    'Read the 4 research files and synthesize 01_research/trend-brief.md (planning brief).');
} else {
  log('skip Research (reusing existing trend-brief).');
}

// ── Phase 3: scenario (dependency pipeline + fan-out) ───────────────────────
if (route.run_scenario) {
  phase('Scenario');
  if (route.mode === 'initial' || route.mode === 'new') {
    await ask('concept-architect', 'From trend-brief, write 02_story/concept.md.');
    await ask('worldbuilder', 'Write 02_story/world.md and seed 02_story/story-bible.json (locations/rules).');
    await ask('character-designer', 'Write 02_story/characters.md and the characters[] of 02_story/story-bible.json (schema-valid).');
    await ask('series-plotter', 'Write 02_story/series-arc.md (arc + episode map).');
    await parallel([
      () => ask('twist-master', 'Write 02_story/twist-plan.md — a distinct twist for EVERY episode.'),
      () => ask('tension-engineer', 'Write 02_story/tension-curve.md (rising tension + cliffhangers).'),
    ]);
  } else {
    log('reuse 02_story/* (series bible) for next/partial episode.');
  }
  await ask('episode-outliner', `Write 03_episode/ep${NN}_beatsheet.md — split beats until ≥50 panels.`);
  await ask('dialogue-writer', `Write 03_episode/ep${NN}_script.md — dialogue-led.`);
  await ask('script-editor', `Write 03_episode/ep${NN}_script_final.md — verify the twist LANDS clearly; ≥50 panels of material.`);
} else {
  log('skip Scenario.');
}

// ── Phase 4: visual (refs first → breakdown → render → validate loop) ────────
if (route.run_visual) {
  phase('Visual');
  await ask('art-director',
    'Write 04_visual/style-bible.md (incl. LOC_* tokens + balloon convention) and character-sheets.md.');
  if (route.mode === 'initial' || route.mode === 'new') {
    await ask('ref-sheet-artist',
      `Render character reference sheets FIRST via scripts/imagegen/render.sh (backend=${BACKEND}, concurrency=${CONC}) into 04_visual/refs/ and write refs/INDEX.md. Reused across episodes.`);
  } else {
    log('reuse 04_visual/refs/ (reference SSOT) — no re-render.');
  }
  await parallel([
    () => ask('panel-director', `Write 04_visual/ep${NN}_shotlist.md — ≥50 panels, each scene_id + LOC_* token.`),
    () => ask('letterer', `Write 04_visual/ep${NN}_lettering.md and balloon entries in 04_visual/ep${NN}_panels.json (overlay mode default; bake opt-in).`),
  ]);
  await ask('prompt-smith',
    `Write 04_visual/ep${NN}_prompts.md (style+LOC+character/ref anchor+empty-balloon convention) and assign panels to scene groups A/B/C in ep${NN}_panels.json.`);

  // render → validate → regen loop, up to 3 passes (DESIGN §8)
  let pass = 0, done = false;
  while (!done && pass < 3) {
    pass++;
    log(`render/validate pass ${pass}`);
    await parallel([
      () => ask('panel-artist-a', `Render group A panels (cache-check then scripts/imagegen/render.sh, concurrency≤${CONC}) into 05_panels/ep${NN}/. Pass ${pass}: only render PENDING/REGEN panels.`),
      () => ask('panel-artist-b', `Render group B panels (same contract). Pass ${pass}: only PENDING/REGEN.`),
      () => ask('panel-artist-c', `Render group C panels (same contract). Pass ${pass}: only PENDING/REGEN.`),
    ]);
    const v = await agent(
      `Run scripts/validate/panel_check.mjs over 05_panels/ep${NN}/ (refs 04_visual/refs), judge C1–C6,
write 04_visual/ep${NN}_validation.md and update each panel's validation in ep${NN}_panels.json.
Mark REGEN panels for re-render (max 3 tries → ACCEPT-FLAG). Report whether all panels are ACCEPT/ACCEPT-FLAG.\n\n${ctx}`,
      { agentType: 'panel-validator', label: 'panel-validator', model: M, schema: {
        type: 'object', required: ['all_resolved', 'regen_count'],
        properties: { all_resolved: { type: 'boolean' }, regen_count: { type: 'integer' }, notes: { type: 'string' } } } });
    log(`validation: ${v.regen_count} REGEN remaining`);
    done = v.all_resolved || v.regen_count === 0;
    if (!done) await ask('prompt-smith', `Strengthen prompts for REGEN panels listed in ep${NN}_validation.md (LOC tokens, ref anchors, shorter/bolder).`);
  }
  if (!done) log('panel loop hit max passes — proceeding with ACCEPT-FLAG panels.');
} else {
  log('skip Visual.');
}

// ── Phase 5: assembly (overlay viewer → QA → continuity → release) ──────────
phase('Assembly');
const QA_SCHEMA = { type: 'object', required: ['verdict'], properties: {
  verdict: { type: 'string', enum: ['PASS', 'FIX', 'REDO'] }, issues: { type: 'array', items: { type: 'string' } } } };
let qaPass = 0, qa = { verdict: 'FIX' };
while (qa.verdict !== 'PASS' && qaPass < 2) {
  qaPass++;
  await ask('episode-compositor',
    `Assemble 06_assembly/ep${NN}/index.html via scripts/lettering/overlay.mjs (--manifest 04_visual/ep${NN}_panels.json --panels-dir 05_panels/ep${NN} --template viewer/template.html). Art-only panels + overlay balloons (U1).`);
  qa = await agent(
    `QA the episode (DESIGN §8 gates): ≥50 panels, ref-consistent appearance, background continuity,
overlay text correctness/legibility, dialogue flow, twist delivery, no corrupt/dup images.
Read 04_visual/ep${NN}_validation.md first. Write 06_assembly/ep${NN}/qa_report.md and return verdict.\n\n${ctx}`,
    { agentType: 'quality-reviewer', label: 'quality-reviewer', model: M, schema: QA_SCHEMA });
  log(`QA pass ${qaPass}: ${qa.verdict}`);
  if (qa.verdict === 'REDO') { log('QA REDO → routing fixes back to visual/scenario is required; stopping auto-loop for human review.'); break; }
}
await ask('continuity-manager',
  `Update 06_assembly/continuity.md and reconcile 02_story/continuity-ledger.json (character state, open threads, revealed twists) for episode ${EP}.`);
const release = await ask('showrunner',
  `Sign off and package RELEASE/ep${NN}/ (copy viewer + assets). Propose the next-episode seed (cliffhanger handoff). Summarize: title, logline, the one-line twist, panel count, viewer path.`);

return {
  episode: EP, mode: route.mode,
  viewer: `_workspace/RELEASE/ep${NN}/index.html`,
  qa: qa.verdict,
  summary: release,
};
