/**
 * Injects the runtime permissions the translator needs into the generated
 * Android project. Run after `npx cap add android` (and safe to re-run).
 *
 * Capacitor's WebView grants the in-page getUserMedia() microphone request
 * automatically *as long as* RECORD_AUDIO is declared in the manifest — its
 * BridgeWebChromeClient maps RESOURCE_AUDIO_CAPTURE to the Android runtime
 * permission and prompts the user on first use.
 */
import { readFileSync, writeFileSync, existsSync, copyFileSync } from "node:fs";
import { resolve } from "node:path";

// --- Stable debug signing -------------------------------------------------
// Gradle auto-generates a *random* debug key each build, so a new APK can't
// install over a previously-installed one (signature mismatch → "App not
// installed"). Pin the debug signingConfig to a committed keystore so every
// build shares one signature and updates install in place.
const KEYSTORE_SRC = resolve("ci-debug.keystore");
const gradlePath = resolve("android", "app", "build.gradle");
if (existsSync(KEYSTORE_SRC) && existsSync(gradlePath)) {
  copyFileSync(KEYSTORE_SRC, resolve("android", "app", "ci-debug.keystore"));
  let gradle = readFileSync(gradlePath, "utf8");
  if (!gradle.includes("ci-debug.keystore")) {
    const signing = `
    signingConfigs {
        debug {
            storeFile file('ci-debug.keystore')
            storePassword 'android'
            keyAlias 'androiddebugkey'
            keyPassword 'android'
        }
    }`;
    // Inject as the first block inside `android { ... }`.
    gradle = gradle.replace(/android\s*\{/, (m) => `${m}\n${signing}`);
    writeFileSync(gradlePath, gradle, "utf8");
    console.log("Pinned debug signingConfig to ci-debug.keystore.");
  } else {
    console.log("Debug signingConfig already pinned — nothing to do.");
  }
} else {
  console.warn("ci-debug.keystore or app/build.gradle missing — skipping signing pin.");
}

const manifestPath = resolve(
  "android",
  "app",
  "src",
  "main",
  "AndroidManifest.xml"
);

if (!existsSync(manifestPath)) {
  console.error(
    `AndroidManifest.xml not found at ${manifestPath}.\n` +
      `Run "npx cap add android" first.`
  );
  process.exit(1);
}

let xml = readFileSync(manifestPath, "utf8");

const permissions = [
  "android.permission.INTERNET",
  "android.permission.RECORD_AUDIO",
  "android.permission.MODIFY_AUDIO_SETTINGS",
  // Keep translating with the screen off / another app in front: a
  // microphone-type foreground service is the only way Android lets a
  // backgrounded app keep capturing audio.
  "android.permission.FOREGROUND_SERVICE",
  "android.permission.FOREGROUND_SERVICE_MICROPHONE",
  "android.permission.POST_NOTIFICATIONS",
  "android.permission.WAKE_LOCK",
];

// --- Background keep-alive (foreground service + Capacitor plugin) ----------
// Android suspends mic capture the moment the app leaves the foreground or the
// screen locks. Inject a small microphone-type foreground service (the same
// mechanism voice recorders use) plus a Capacitor plugin so the web app can
// start/stop it around capture sessions.
const pkgDir = resolve("android", "app", "src", "main", "java", "com", "estv", "geminitranslator");
if (existsSync(pkgDir)) {
  writeFileSync(resolve(pkgDir, "KeepAliveService.java"), `package com.estv.geminitranslator;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;

public class KeepAliveService extends Service {
  private static final String CHANNEL_ID = "estv_keepalive";
  private static final int NOTIF_ID = 1001;

  @Override
  public int onStartCommand(Intent intent, int flags, int startId) {
    NotificationManager nm = getSystemService(NotificationManager.class);
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      nm.createNotificationChannel(new NotificationChannel(
          CHANNEL_ID, "Live interpreting", NotificationManager.IMPORTANCE_LOW));
    }
    Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
        ? new Notification.Builder(this, CHANNEL_ID)
        : new Notification.Builder(this);
    Notification n = b.setContentTitle("통역 진행 중 / Interpreting")
        .setContentText("화면이 꺼져도 계속 번역합니다")
        .setSmallIcon(getApplicationInfo().icon)
        .setOngoing(true)
        .build();
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE);
    } else {
      startForeground(NOTIF_ID, n);
    }
    return START_STICKY;
  }

  @Override
  public IBinder onBind(Intent intent) { return null; }
}
`, "utf8");

  writeFileSync(resolve(pkgDir, "KeepAlivePlugin.java"), `package com.estv.geminitranslator;

import android.content.Intent;
import androidx.core.content.ContextCompat;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

@CapacitorPlugin(name = "KeepAlive")
public class KeepAlivePlugin extends Plugin {
  @PluginMethod
  public void start(PluginCall call) {
    ContextCompat.startForegroundService(
        getContext(), new Intent(getContext(), KeepAliveService.class));
    call.resolve();
  }

  @PluginMethod
  public void stop(PluginCall call) {
    getContext().stopService(new Intent(getContext(), KeepAliveService.class));
    call.resolve();
  }
}
`, "utf8");

  const mainPath = resolve(pkgDir, "MainActivity.java");
  if (existsSync(mainPath)) {
    let main = readFileSync(mainPath, "utf8");
    if (!main.includes("KeepAlivePlugin")) {
      main = main.replace(
        /public class MainActivity extends BridgeActivity \{\}/,
        `public class MainActivity extends BridgeActivity {
  @Override
  public void onCreate(android.os.Bundle savedInstanceState) {
    registerPlugin(KeepAlivePlugin.class);
    super.onCreate(savedInstanceState);
  }
}`
      );
      writeFileSync(mainPath, main, "utf8");
      console.log("Registered KeepAlivePlugin in MainActivity.");
    }
  }

  if (!xml.includes("KeepAliveService")) {
    xml = xml.replace(
      /(\n\s*)(<\/application>)/,
      `$1    <service android:name=".KeepAliveService" android:exported="false" android:foregroundServiceType="microphone" />$1$2`
    );
  }
} else {
  console.warn("Android package dir not found — skipping keep-alive service injection.");
}

const lines = [];
for (const name of permissions) {
  if (!xml.includes(`android:name="${name}"`)) {
    lines.push(`    <uses-permission android:name="${name}" />`);
  }
}

if (lines.length > 0) {
  // Insert just before the <application> element.
  xml = xml.replace(
    /(\n\s*)(<application)/,
    `\n${lines.join("\n")}$1$2`
  );
  console.log(`Added permissions:\n${lines.join("\n")}`);
} else {
  console.log("All required permissions already present.");
}

// Always write: the keep-alive service block above may have edited xml even
// when no permissions were missing (must not early-exit before this).
writeFileSync(manifestPath, xml, "utf8");
