/**
 * Injects the runtime permissions the translator needs into the generated
 * Android project. Run after `npx cap add android` (and safe to re-run).
 *
 * Capacitor's WebView grants the in-page getUserMedia() microphone request
 * automatically *as long as* RECORD_AUDIO is declared in the manifest — its
 * BridgeWebChromeClient maps RESOURCE_AUDIO_CAPTURE to the Android runtime
 * permission and prompts the user on first use.
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

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
];

const lines = [];
for (const name of permissions) {
  if (!xml.includes(`android:name="${name}"`)) {
    lines.push(`    <uses-permission android:name="${name}" />`);
  }
}

if (lines.length === 0) {
  console.log("All required permissions already present — nothing to do.");
  process.exit(0);
}

// Insert just before the <application> element.
xml = xml.replace(
  /(\n\s*)(<application)/,
  `\n${lines.join("\n")}$1$2`
);

writeFileSync(manifestPath, xml, "utf8");
console.log(`Added permissions:\n${lines.join("\n")}`);
