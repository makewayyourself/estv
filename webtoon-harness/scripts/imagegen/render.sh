#!/usr/bin/env sh
# render.sh — front door for all panel/reference rendering (U3).
# Resolves a pluggable backend, enforces a concurrency semaphore (a knob, not a cap),
# and guarantees the output is a valid non-zero PNG. See scripts/imagegen/adapter.md.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

BACKEND="${WT_IMG_BACKEND:-codex}"
PROMPT_FILE=""
OUT=""
SEED=""
CONCURRENCY="${WT_CONCURRENCY:-5}"
BAKE=0
REFS=""

die() { echo "render.sh: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --backend)     BACKEND="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --ref)         REFS="${REFS:+$REFS:}$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --bake)        BAKE=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$PROMPT_FILE" ] || die "--prompt-file is required"
[ -n "$OUT" ] || die "--out is required"
[ -f "$PROMPT_FILE" ] || die "prompt file not found: $PROMPT_FILE"

BACKEND_SH="$HERE/backends/$BACKEND.sh"
[ -f "$BACKEND_SH" ] || die "unknown backend '$BACKEND' (expected $BACKEND_SH)"

mkdir -p "$(dirname -- "$OUT")"

# --- concurrency semaphore -------------------------------------------------
# A lock dir caps simultaneous backend execs across all panel-artists. Default 5
# matches codex; raise it for backends without a session cap. This is the v2 fix
# for v1's structural 5-session bottleneck: the limit is configurable here.
SEM_DIR="${WT_SEM_DIR:-${TMPDIR:-/tmp}/wt_render_sem}"
mkdir -p "$SEM_DIR"
acquire() {
  i=0
  while : ; do
    n=$(find "$SEM_DIR" -maxdepth 1 -name 'slot.*' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -lt "$CONCURRENCY" ]; then
      slot="$SEM_DIR/slot.$$.$i"
      if mkdir "$slot" 2>/dev/null; then echo "$slot"; return 0; fi
    fi
    i=$((i+1))
    sleep 1
  done
}
SLOT=$(acquire)
# shellcheck disable=SC2064
trap "rmdir '$SLOT' 2>/dev/null || true" EXIT INT TERM

# --- dispatch --------------------------------------------------------------
WT_PROMPT_FILE="$PROMPT_FILE" \
WT_OUT="$OUT" \
WT_REFS="$REFS" \
WT_SEED="$SEED" \
WT_BAKE="$BAKE" \
  sh "$BACKEND_SH" || die "backend '$BACKEND' failed for $OUT"

# --- validate output -------------------------------------------------------
[ -s "$OUT" ] || die "backend produced empty/missing PNG: $OUT"
# PNG magic: \x89PNG
head -c 4 "$OUT" | od -An -tx1 2>/dev/null | grep -qi '89 50 4e 47' \
  || die "output is not a valid PNG: $OUT"

echo "$OUT"
