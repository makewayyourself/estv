/**
 * Build-time configuration for the packaged app.
 *
 * Set DEFAULT_SERVER_URL to your deployed cloud backend BEFORE building the
 * Android APK so the app ships pre-pointed at your server, e.g.:
 *
 *   window.DEFAULT_SERVER_URL = "https://gemini-live-translator.onrender.com";
 *
 * If left empty:
 *   - On the web it falls back to the same origin that served the page.
 *   - In the Android app the user must type the URL into the "Server URL" box
 *     (it is then remembered on the device).
 */
window.DEFAULT_SERVER_URL = "";
