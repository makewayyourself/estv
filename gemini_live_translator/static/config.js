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

/**
 * Optional: the server access token (must match ACCESS_TOKEN on the backend).
 * You normally leave this blank and type the token into the app's "Access
 * Token" field once — it is then remembered on the device. Do NOT commit a
 * real token to a public repository.
 */
window.DEFAULT_ACCESS_TOKEN = "";
