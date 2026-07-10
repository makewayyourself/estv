#!/usr/bin/env sh
# codex backend — renders via the codex CLI image_generation tool (v1-compatible).
# Requires: codex CLI logged in (ChatGPT OAuth). Honors the adapter contract.
set -eu

command -v codex >/dev/null 2>&1 || { echo "codex CLI not found on PATH" >&2; exit 1; }

PROMPT=$(cat "$WT_PROMPT_FILE")
SEED_ARG=""
[ -n "${WT_SEED:-}" ] && SEED_ARG="seed=$WT_SEED"

# Build a reference clause from WT_REFS (colon-separated paths).
REF_CLAUSE=""
if [ -n "${WT_REFS:-}" ]; then
  REF_CLAUSE=" Use these reference images for character/style consistency: $(printf '%s' "$WT_REFS" | tr ':' ' ')."
fi

BAKE_CLAUSE=" Draw empty speech balloons with clear negative space for text; no English/gibberish/misspelled text."
[ "${WT_BAKE:-0}" = "1" ] && BAKE_CLAUSE=" Render the specified Korean text cleanly integrated into the art."

# codex exec drives the image_generation tool non-interactively and writes to WT_OUT.
codex exec --skip-git-repo-check \
  "Generate a single webtoon panel image and save it to ${WT_OUT}. ${SEED_ARG} ${PROMPT}${REF_CLAUSE}${BAKE_CLAUSE}" \
  >/dev/null 2>&1 || { echo "codex exec failed" >&2; exit 1; }

[ -s "$WT_OUT" ] || { echo "codex produced no output at $WT_OUT" >&2; exit 1; }
