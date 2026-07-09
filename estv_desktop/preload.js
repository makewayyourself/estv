// Bridge for desktop-only capabilities the web app can use when running inside
// the Electron shell. Kept minimal and safe (contextIsolation on).
//
// Phase 2b will expose audio-output-device routing here so the app's translated
// voice can be sent to a virtual mic (e.g. VB-CABLE) and thus into Google Meet.
// For now it just advertises that we're the desktop host and lists output
// devices, which the web app can feature-detect.

const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("estvDesktop", {
  isDesktop: true,
  platform: process.platform,
  // Enumerate audio OUTPUT devices (for future TTS routing to a virtual mic).
  async listOutputs() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter((d) => d.kind === "audiooutput")
        .map((d) => ({ id: d.deviceId, label: d.label }));
    } catch (_) {
      return [];
    }
  },
});
