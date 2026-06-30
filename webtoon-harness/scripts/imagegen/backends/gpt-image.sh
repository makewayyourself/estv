#!/usr/bin/env sh
# gpt-image backend — OpenAI Images API (gpt-image-1). Honors the adapter contract.
# Requires: OPENAI_API_KEY, curl, and a JSON tool (python3) to decode base64.
set -eu

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required for the gpt-image backend}"
command -v curl >/dev/null 2>&1 || { echo "curl not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

PROMPT=$(cat "$WT_PROMPT_FILE")
if [ "${WT_BAKE:-0}" = "1" ]; then
  PROMPT="$PROMPT  Render the specified Korean text cleanly integrated into the art."
else
  PROMPT="$PROMPT  Draw empty speech balloons with negative space for text; no English/gibberish/misspelled text."
fi

# Note: gpt-image-1 has no public seed param; WT_SEED is recorded by the cache layer
# for keying but not sent here. References are described in-prompt by the caller.
RESP=$(curl -sS https://api.openai.com/v1/images/generations \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$(python3 - "$PROMPT" <<'PY'
import json, sys
print(json.dumps({"model":"gpt-image-1","prompt":sys.argv[1],"size":"1024x1536","n":1}))
PY
)") || { echo "gpt-image request failed" >&2; exit 1; }

printf '%s' "$RESP" | python3 - "$WT_OUT" <<'PY' || { echo "gpt-image: failed to decode image" >&2; exit 1; }
import base64, json, sys
out = sys.argv[1]
data = json.load(sys.stdin)
try:
    b64 = data["data"][0]["b64_json"]
except Exception:
    sys.stderr.write("gpt-image: unexpected response: %s\n" % str(data)[:200]); sys.exit(1)
open(out, "wb").write(base64.b64decode(b64))
PY
