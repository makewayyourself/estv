#!/usr/bin/env sh
# Backend template (U3). Copy to <name>.sh and implement the API/CLI call.
# Contract (see ../adapter.md):
#   inputs  : WT_PROMPT_FILE  WT_OUT  WT_REFS(:-sep)  WT_SEED  WT_BAKE(0|1)
#   success : exit 0 AND WT_OUT is a valid non-zero PNG
#   failure : non-zero exit + one-line reason on stderr; nothing on stdout
set -eu

PROMPT=$(cat "$WT_PROMPT_FILE")

# Reference images, if any (newline-separated -> handle as your API expects):
REFS_NL=$(printf '%s' "${WT_REFS:-}" | tr ':' '\n')

# If text must be baked into the art (SFX etc.), WT_BAKE=1; otherwise prompt should
# already ask for empty balloons / negative space (U1 overlay handles dialogue).

# TODO: call your image model, writing the PNG to "$WT_OUT".
echo "_template.sh is a stub — copy and implement a real backend" >&2
exit 1
