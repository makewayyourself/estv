#!/usr/bin/env sh
# gemini backend — Google Gemini image generation. Honors the adapter contract.
# Requires: GEMINI_API_KEY, curl, python3.
set -eu

: "${GEMINI_API_KEY:?GEMINI_API_KEY is required for the gemini backend}"
command -v curl >/dev/null 2>&1 || { echo "curl not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

MODEL="${WT_GEMINI_MODEL:-gemini-2.5-flash-image}"
PROMPT=$(cat "$WT_PROMPT_FILE")
if [ "${WT_BAKE:-0}" = "1" ]; then
  PROMPT="$PROMPT  Render the specified Korean text cleanly integrated into the art."
else
  PROMPT="$PROMPT  Draw empty speech balloons with negative space for text; no English/gibberish/misspelled text."
fi

URL="https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${GEMINI_API_KEY}"
RESP=$(curl -sS "$URL" -H "Content-Type: application/json" \
  -d "$(python3 - "$PROMPT" <<'PY'
import json, sys
print(json.dumps({"contents":[{"parts":[{"text":sys.argv[1]}]}]}))
PY
)") || { echo "gemini request failed" >&2; exit 1; }

printf '%s' "$RESP" | python3 - "$WT_OUT" <<'PY' || { echo "gemini: failed to decode image" >&2; exit 1; }
import base64, json, sys
out = sys.argv[1]
data = json.load(sys.stdin)
try:
    parts = data["candidates"][0]["content"]["parts"]
    b64 = next(p["inlineData"]["data"] for p in parts if "inlineData" in p)
except Exception:
    sys.stderr.write("gemini: no image in response: %s\n" % str(data)[:200]); sys.exit(1)
open(out, "wb").write(base64.b64decode(b64))
PY
