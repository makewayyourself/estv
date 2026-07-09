# ESTV Interpreter — Desktop (Windows)

An Electron shell around the ESTV web interpreter that adds what a browser tab
can't:

- **Invisible to screen-share** — captions are hidden from Zoom/Meet/Teams
  screen sharing and screen recording (`setContentProtection`). Toggle with
  **Ctrl+Shift+H**.
- **Always-on-top overlay** — floats over Google Meet. Toggle **Ctrl+Shift+T**.
- **No permission friction** — mic and tab/screen audio capture are pre-granted.

The window loads the live web app (`ESTV_URL`, default the Render server), so it
always runs the latest UI.

## Build / run

```bash
cd estv_desktop
npm install
npm start                # run locally
npm run dist:win         # build the Windows installer into dist/
```

CI (`.github/workflows/desktop-build.yml`) builds the installer on every push to
`main` and publishes it to the `desktop-latest` GitHub release.

## Google Meet — inbound (understand the other party) ✅ works now

1. Join the Meet in Chrome.
2. Open ESTV Interpreter, pick **🎯 Precision** mode, set your languages.
3. Input source → **🖥️ Screen/tab audio** → Start → share the **Meet tab**
   with **"Also share tab audio"** ticked.
4. The other party's speech appears as live translated captions (read-aloud
   optional). Ctrl+Shift+H keeps the captions off your shared screen.

## Google Meet — outbound (they hear your translated voice) — Phase 2b

The app already produces your translated voice (Precision + mic + read-aloud).
To send it into Meet's microphone it must be routed through a virtual audio
device. Phase 2b automates the routing; the one-time pieces that Windows/Meet
require from the user:

1. Install **VB-CABLE** (free virtual audio device).
2. In Google Meet settings, set **Microphone = "CABLE Output"**.
3. In ESTV, enable outbound voice — the app plays the translated audio into
   **"CABLE Input"**, so Meet transmits only the translation.

> A virtual audio **driver** can't be silently installed by an app (it's a
> signed kernel driver), and Meet's mic selection lives in Meet — so those two
> steps stay manual. Everything the app controls (which device the voice plays
> to, when) is automated.
