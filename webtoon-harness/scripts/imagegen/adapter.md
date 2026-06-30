# Image Backend Adapter Contract (U3)

v1 was hard-locked to the codex CLI, so its global 5-session cap became a *structural*
bottleneck. v2 puts every render behind one adapter so the backend is swappable and
concurrency is a config knob, not a law.

## Front door

All rendering personas call **one** command — never a backend directly:

```sh
scripts/imagegen/render.sh \
  --backend  "${WT_IMG_BACKEND:-codex}" \   # codex | gpt-image | gemini | local-sd
  --prompt-file <path.md|.txt> \            # the panel prompt (required)
  --out        <path.png> \                 # output PNG (required)
  --ref        <ref.png> \                  # reference sheet anchor (repeatable, optional)
  --seed       <int> \                      # optional; enables reproducibility + caching
  --concurrency <K> \                       # optional; default 5 (knob, not a cap)
  --bake                                    # optional flag: request in-image Hangul text
```

`render.sh` resolves the backend, enforces concurrency with a semaphore, then execs
`backends/<backend>.sh` with the normalized arguments below. It exits non-zero if the
backend is unknown or the produced PNG is missing/0-byte.

## Backend contract

Every `backends/<name>.sh` MUST accept exactly these env-normalized arguments and
behave identically from the caller's point of view:

| arg            | env var passed       | meaning                                  |
|----------------|----------------------|------------------------------------------|
| prompt file    | `WT_PROMPT_FILE`     | path to the text prompt                  |
| output path    | `WT_OUT`             | where to write the PNG                   |
| reference list | `WT_REFS` (`:`-sep)  | reference image paths (may be empty)     |
| seed           | `WT_SEED`            | integer or empty                         |
| bake flag      | `WT_BAKE`            | `1` to bake text, else `0`               |

**Exit code:** `0` only if `WT_OUT` is a valid, non-zero PNG. Any other condition →
non-zero with a one-line reason on stderr. Backends MUST NOT print anything else to
stdout (callers may capture it).

## Adding a backend

1. Copy `backends/_template.sh` to `backends/<name>.sh`.
2. Implement the call to your image API/CLI using the env vars above.
3. `chmod +x` it. Select it with `--backend <name>` or `export WT_IMG_BACKEND=<name>`.

No other file needs to change — that is the whole point of U3.

## Prompt convention (works with U1 lettering)

Because text is overlaid as a real font layer by default (U1), prompts SHOULD request
**clean art with empty speech balloons and negative space for text**, and carry the
negative `no English/gibberish/misspelled text`. Only when `--bake` is set should the
prompt ask the model to render Hangul into the image (e.g. integrated SFX).
