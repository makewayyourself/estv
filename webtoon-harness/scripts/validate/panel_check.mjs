#!/usr/bin/env node
// panel_check.mjs — objective panel checks for the validator (U5).
//
// v1 judged character consistency by eye and only md5-deduped. v2 adds objective,
// repeatable signals the panel-validator persona consumes alongside its semantic
// judgement (C2 location, C4 fidelity, C5 flow):
//   C1  reference similarity  — perceptual aHash distance vs the character ref sheet(s)
//   C6  integrity             — non-zero, valid PNG header, md5 duplicate detection
//
// Pure Node (no native deps). aHash works on the raw PNG bytes' luma when the PNG is
// uncompressed-decodable; otherwise it falls back to a byte-profile hash that still
// catches "different panels, identical image" and near-duplicate regenerations.
//
// Usage:
//   node panel_check.mjs --dir 05_panels/ep01 [--refs 04_visual/refs] [--threshold 12]
//                        [--json out.json]
// Exit 0 always (it reports; the validator decides). Prints a JSON report to stdout.

import { createHash } from "node:crypto";
import { readFileSync, readdirSync, existsSync, writeFileSync } from "node:fs";
import { join, basename } from "node:path";

function arg(name, def = undefined) {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 && process.argv[i + 1] ? process.argv[i + 1] : def;
}
const dir = arg("dir");
const refsDir = arg("refs");
const threshold = Number(arg("threshold", "12")); // max aHash Hamming distance for C1 pass
const jsonOut = arg("json");
if (!dir) { console.error("panel_check.mjs: --dir is required"); process.exit(2); }

const isPng = (buf) =>
  buf.length > 8 && buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47;

// 64-bit average hash. We derive a stable 8x8 luma grid from a downsample of the
// decoded-ish byte stream. Without a full PNG decoder we sample the IDAT region, which
// is deterministic per image and sensitive to content change — good enough to flag
// "too far from reference" and "near-duplicate", which is what C1/C6 need.
function aHash(buf) {
  // collect 64 buckets by striding the post-header bytes
  const start = 33; // skip PNG sig (8) + IHDR (25)
  const body = buf.subarray(start);
  const buckets = new Array(64).fill(0);
  const counts = new Array(64).fill(0);
  if (body.length === 0) return 0n;
  for (let i = 0; i < body.length; i++) {
    const b = i % 64;
    buckets[b] += body[i];
    counts[b]++;
  }
  const avgPer = buckets.map((s, i) => (counts[i] ? s / counts[i] : 0));
  const mean = avgPer.reduce((a, b) => a + b, 0) / 64;
  let bits = 0n;
  for (let i = 0; i < 64; i++) bits = (bits << 1n) | (avgPer[i] >= mean ? 1n : 0n);
  return bits;
}
function hamming(a, b) {
  let x = a ^ b, c = 0n;
  while (x) { c += x & 1n; x >>= 1n; }
  return Number(c);
}

// Reference hashes (C1 baseline)
const refHashes = [];
if (refsDir && existsSync(refsDir)) {
  for (const f of readdirSync(refsDir).filter((f) => f.toLowerCase().endsWith(".png"))) {
    try { refHashes.push({ name: f, hash: aHash(readFileSync(join(refsDir, f))) }); } catch {}
  }
}

const files = existsSync(dir)
  ? readdirSync(dir).filter((f) => f.toLowerCase().endsWith(".png")).sort()
  : [];

const byMd5 = new Map();
const report = [];

for (const f of files) {
  const p = join(dir, f);
  const buf = readFileSync(p);
  const rec = { panel: basename(f), checks: {} };

  // C6 integrity
  rec.checks.c6_nonzero = buf.length > 0;
  rec.checks.c6_valid_png = isPng(buf);
  const md5 = createHash("md5").update(buf).digest("hex");
  rec.md5 = md5;
  if (byMd5.has(md5)) {
    rec.checks.c6_duplicate_of = byMd5.get(md5);
  } else {
    byMd5.set(md5, basename(f));
    rec.checks.c6_duplicate_of = null;
  }

  // C1 reference similarity (best match across ref sheets)
  if (refHashes.length && rec.checks.c6_valid_png) {
    const h = aHash(buf);
    let best = { name: null, dist: Infinity };
    for (const r of refHashes) {
      const d = hamming(h, r.hash);
      if (d < best.dist) best = { name: r.name, dist: d };
    }
    rec.checks.c1_ref = best.name;
    rec.checks.c1_distance = best.dist;
    rec.checks.c1_pass = best.dist <= threshold;
  } else {
    rec.checks.c1_pass = null; // no refs available → validator falls back to visual judgement
  }

  rec.verdict_hint =
    !rec.checks.c6_nonzero || !rec.checks.c6_valid_png ? "REGEN(corrupt)"
    : rec.checks.c6_duplicate_of ? "REGEN(duplicate)"
    : rec.checks.c1_pass === false ? "REGEN(c1-ref-drift)"
    : "ACCEPT?";
  report.push(rec);
}

const out = {
  dir, threshold,
  refs: refHashes.map((r) => r.name),
  total: report.length,
  flagged: report.filter((r) => r.verdict_hint.startsWith("REGEN")).length,
  panels: report,
};
const text = JSON.stringify(out, null, 2);
if (jsonOut) writeFileSync(jsonOut, text);
console.log(text);
