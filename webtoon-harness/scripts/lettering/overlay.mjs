#!/usr/bin/env node
// overlay.mjs — Hybrid lettering, overlay path (U1).
//
// v1's #1 failure mode was Korean text baked into images coming out broken. v2 renders
// dialogue as a real DOM/SVG layer positioned over art-only panels, using a bundled
// font, so Hangul is always correct, legible, editable, and re-typesettable.
//
// Reads a panel manifest (schemas/panel-manifest.schema.json) and emits a self-contained
// vertical-scroll index.html (art <img> + absolutely-positioned balloons per panel).
//
// Usage:
//   node overlay.mjs --manifest <ep_panels.json> --panels-dir <dir> --out <index.html>
//                    [--title "..."] [--font <path-to-ttf>] [--template <template.html>]
//
// Panels with lettering_mode:"bake" keep their baked text and get no overlay.

import { readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

function arg(name, def = undefined) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

const manifestPath = arg("manifest");
const panelsDir = arg("panels-dir");
const outPath = arg("out", "index.html");
const title = arg("title", "Webtoon");
const fontPath = arg("font", resolve(HERE, "../../assets/fonts/NanumGothic.ttf"));
const templatePath = arg("template", resolve(HERE, "../../viewer/template.html"));

if (!manifestPath || !panelsDir) {
  console.error("overlay.mjs: --manifest and --panels-dir are required");
  process.exit(1);
}

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const panels = (manifest.panels || []).slice().sort((a, b) => a.id.localeCompare(b.id));
const outDir = dirname(resolve(outPath));

// Font as a relative URL so the viewer is portable.
let fontHref;
try {
  fontHref = relative(outDir, resolve(fontPath)) || basename(fontPath);
} catch {
  fontHref = fontPath;
}

const KIND_CLASS = {
  speech: "balloon speech",
  thought: "balloon thought",
  narration: "balloon narration",
  sfx: "balloon sfx",
  shout: "balloon shout",
};

function renderBalloon(b) {
  const { x = 0, y = 0, w = 30, h = 12 } = b.box || {};
  const cls = KIND_CLASS[b.kind || "speech"] || KIND_CLASS.speech;
  const style = [
    `left:${x}%`, `top:${y}%`, `width:${w}%`, `min-height:${h}%`,
  ].join(";");
  const tail = b.tail && b.tail !== "none" ? `<span class="tail tail-${esc(b.tail)}"></span>` : "";
  const extra = b.style ? ` ${esc(b.style)}` : "";
  return `      <div class="${cls}${extra}" style="${style}">${esc(b.text)}${tail}</div>`;
}

function renderPanel(p) {
  const art = relative(outDir, resolve(panelsDir, basename(p.art_path || `${p.id}.png`)));
  const baked = (p.lettering_mode || "overlay") === "bake";
  const balloons = baked ? "" : (p.balloons || []).map(renderBalloon).join("\n");
  return `    <figure class="panel" id="${esc(p.id)}" data-scene="${esc(p.scene_id || "")}" data-loc="${esc(p.location || "")}">
      <img src="${esc(art)}" alt="${esc(p.id)}" loading="lazy">
${balloons}
    </figure>`;
}

const panelsHtml = panels.map(renderPanel).join("\n");

let template;
try {
  template = readFileSync(templatePath, "utf8");
} catch {
  // Minimal fallback so the tool works even if the template is missing.
  template = `<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>{{TITLE}}</title><style>{{FONT_FACE}}
body{margin:0;background:#111;font-family:'WTKR',sans-serif}
.viewer{max-width:800px;margin:0 auto}
.panel{position:relative;margin:0;line-height:0}
.panel img{width:100%;display:block}
.balloon{position:absolute;line-height:1.3;background:#fff;color:#111;border-radius:14px;
  padding:2.2% 2.8%;font-size:clamp(13px,2.6vw,20px);box-shadow:0 1px 4px rgba(0,0,0,.3)}
.thought{border-radius:50%}.narration{background:#000;color:#fff;border-radius:4px}
.shout{font-weight:800}.sfx{background:transparent;color:#fff;font-weight:800;text-shadow:0 0 4px #000}
</style></head><body><main class="viewer">
{{PANELS}}
</main></body></html>`;
}

const fontFace = `@font-face{font-family:'WTKR';src:url('${esc(fontHref)}') format('truetype');font-display:swap}`;

const html = template
  .replace(/\{\{\s*TITLE\s*\}\}/g, esc(title))
  .replace(/\{\{\s*FONT_FACE\s*\}\}/g, fontFace)
  .replace(/\{\{\s*PANELS\s*\}\}/g, panelsHtml);

writeFileSync(outPath, html);
const overlaid = panels.filter((p) => (p.lettering_mode || "overlay") !== "bake").length;
console.error(`overlay.mjs: wrote ${outPath} — ${panels.length} panels (${overlaid} overlaid).`);
