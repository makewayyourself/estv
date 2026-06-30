#!/usr/bin/env sh
# placeholder backend — credential-free, fully offline panel renderer (demo/test/CI).
# It does NOT call any AI model; it draws a deterministic, art-only stand-in panel
# (gradient by location/seed, simple staged figures, negative space for overlay text)
# so the full render → validate → overlay → viewer chain can run without keys.
# Honors the adapter contract. Use --backend placeholder. Requires python3 + Pillow.
set -eu

command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }
python3 - "$WT_PROMPT_FILE" "$WT_OUT" "${WT_SEED:-0}" "${WT_REFS:-}" "${WT_BAKE:-0}" <<'PY' || { echo "placeholder render failed" >&2; exit 1; }
import sys, hashlib, re
from PIL import Image, ImageDraw, ImageFont

prompt_file, out, seed, refs, bake = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
prompt = open(prompt_file, encoding="utf-8").read().strip()

# Deterministic palette from prompt+seed (so the same panel re-renders identically -> cache-friendly).
h = hashlib.sha256((prompt + "|" + str(seed)).encode()).hexdigest()
def hx(i): return int(h[i:i+2], 16)
W, Hh = 1024, 1536
top = (hx(0)//2+40, hx(2)//2+40, hx(4)//2+60)
bot = (hx(6)//3+20, hx(8)//3+20, hx(10)//3+30)

img = Image.new("RGB", (W, Hh), top)
px = img.load()
for y in range(Hh):                      # vertical gradient
    t = y / Hh
    c = tuple(int(top[i]*(1-t) + bot[i]*t) for i in range(3))
    for x in range(W):
        px[x, y] = c
d = ImageDraw.Draw(img)

# Pull a few hint tokens (LOC_*, CHAR_*) and a short scene line from the prompt.
locs = re.findall(r"LOC_[A-Z0-9_]+", prompt)
chars = re.findall(r"CHAR_[A-Z0-9_]+", prompt)
scene = (re.sub(r"\s+", " ", prompt))[:90]

# Simple staged "figures" so panels read as different shots (no dialogue — overlaid later).
n = (hx(12) % 3) + 1
for i in range(n):
    cx = int(W*(0.25 + 0.5*i/max(1,n-1))) if n > 1 else W//2
    fw, fh = 150, 360
    fill = (235-hx(14+i)//4, 235-hx(16+i)//4, 240)
    d.rounded_rectangle([cx-fw//2, Hh-520, cx+fw//2, Hh-160], radius=40, fill=fill, outline=(30,30,40), width=4)
    d.ellipse([cx-55, Hh-620, cx+55, Hh-510], fill=fill, outline=(30,30,40), width=4)  # head

def font(sz):
    try: return ImageFont.truetype("/home/user/estv/webtoon-harness/assets/fonts/NanumGothic.ttf", sz)
    except Exception:
        try: return ImageFont.truetype("assets/fonts/NanumGothic.ttf", sz)
        except Exception: return ImageFont.load_default()

# Corner metadata band (art-only marker; NOT dialogue).
d.rectangle([0, 0, W, 96], fill=(0,0,0))
tag = (" / ".join(locs[:1] + chars[:2])) or "SCENE"
d.text((28, 28), f"[placeholder]  {tag}", font=font(34), fill=(255,255,255))
d.text((28, Hh-90), scene, font=font(28), fill=(255,255,255))
if bake == "1":
    d.text((W//2-120, 120), "(baked-text panel)", font=font(30), fill=(255,230,120))

img.save(out, "PNG")
PY
[ -s "$WT_OUT" ] || { echo "placeholder produced no output" >&2; exit 1; }
