// ESTV desktop interpreter (Electron shell).
//
// Wraps the existing ESTV web app in a desktop window and adds the things a
// browser tab can't do:
//   * setContentProtection — captions are invisible to screen-share/recording
//     (Windows WDA_EXCLUDEFROMCAPTURE / macOS NSWindowSharingNone).
//   * always-on-top frameless overlay so captions float over Google Meet.
//   * auto-granted mic + display-capture permissions (no repeated prompts).
//
// Phase 2b (outbound voice → Meet mic) is layered on top via preload once this
// shell is verified on a real machine.

const { app, BrowserWindow, session, globalShortcut, Menu } = require("electron");

// The deployed web UI. Override with ESTV_URL to point at a staging server.
const APP_URL = process.env.ESTV_URL || "https://interpretation-pbf4.onrender.com/";

let win = null;
let contentProtected = true;

function createWindow() {
  win = new BrowserWindow({
    width: 460,
    height: 720,
    minWidth: 360,
    minHeight: 480,
    alwaysOnTop: true,
    frame: true,
    title: "ESTV Interpreter",
    backgroundColor: "#0f172a",
    webPreferences: {
      preload: require("path").join(__dirname, "preload.js"),
      // The renderer only loads our own trusted web app.
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Hide from screen sharing / recording by default (the core "invisible" trick).
  win.setContentProtection(contentProtected);
  win.setAlwaysOnTop(true, "screen-saver");

  // Auto-grant microphone + screen/tab audio capture so getUserMedia and
  // getDisplayMedia work without the WebView blocking them.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, cb) => {
    cb(["media", "display-capture", "audioCapture", "videoCapture"].includes(permission));
  });

  win.loadURL(APP_URL);

  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
}

function toggleProtection() {
  contentProtected = !contentProtected;
  if (win) win.setContentProtection(contentProtected);
}

function toggleOnTop() {
  if (!win) return;
  const next = !win.isAlwaysOnTop();
  win.setAlwaysOnTop(next, "screen-saver");
}

app.whenReady().then(() => {
  createWindow();

  // Shortcuts: Ctrl/Cmd+Shift+H = toggle invisibility, +T = toggle on-top.
  globalShortcut.register("CommandOrControl+Shift+H", toggleProtection);
  globalShortcut.register("CommandOrControl+Shift+T", toggleOnTop);

  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {
      label: "View",
      submenu: [
        { label: "Hide from screen-share (toggle)", accelerator: "CommandOrControl+Shift+H", click: toggleProtection },
        { label: "Always on top (toggle)", accelerator: "CommandOrControl+Shift+T", click: toggleOnTop },
        { type: "separator" },
        { role: "reload" },
        { role: "toggleDevTools" },
        { role: "quit" },
      ],
    },
  ]));

  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on("will-quit", () => globalShortcut.unregisterAll());
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
