#!/usr/bin/env sh
# local-sd backend — local Stable Diffusion via an AUTOMATIC1111-compatible API.
# No session cap, so you can raise --concurrency well above 5 (U3 payoff).
# Requires: a running SD WebUI at WT_SD_URL (default 127.0.0.1:7860), curl, python3.
set -eu

SD_URL="${WT_SD_URL:-http://127.0.0.1:7860}"
command -v curl >/dev/null 2>&1 || { echo "curl not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

PROMPT=$(cat "$WT_PROMPT_FILE")
NEG="text, english text, gibberish, misspelled text, watermark, signature"
[ "${WT_BAKE:-0}" = "1" ] && NEG="english text, gibberish, watermark, signature"
SEED="${WT_SEED:--1}"

RESP=$(curl -sS "$SD_URL/sdapi/v1/txt2img" -H "Content-Type: application/json" \
  -d "$(python3 - "$PROMPT" "$NEG" "$SEED" <<'PY'
import json, sys
prompt, neg, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({
  "prompt": prompt, "negative_prompt": neg, "seed": seed,
  "width": 1024, "height": 1536, "steps": 30, "cfg_scale": 7, "sampler_name": "DPM++ 2M Karras"
}))
PY
)") || { echo "local-sd request failed (is the WebUI running at $SD_URL?)" >&2; exit 1; }

printf '%s' "$RESP" | python3 - "$WT_OUT" <<'PY' || { echo "local-sd: failed to decode image" >&2; exit 1; }
import base64, json, sys
out = sys.argv[1]
data = json.load(sys.stdin)
try:
    b64 = data["images"][0]
except Exception:
    sys.stderr.write("local-sd: no image in response: %s\n" % str(data)[:200]); sys.exit(1)
open(out, "wb").write(base64.b64decode(b64.split(",",1)[-1]))
PY
