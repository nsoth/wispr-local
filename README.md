# Wispr Local

Local, privacy-first voice-to-text dictation for Windows. Hold a hotkey, speak, release — the
text is pasted wherever your cursor is. Powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
(CUDA, with CPU fallback), with optional AI formatting through OpenAI or Claude.

Built with Tauri 2 (Rust + React). Windows 10/11 only.

## Features

- **Hotkey dictation** with three modes: hold-to-talk (default), toggle, or hold-or-tap-for-hands-free.
  Default `Ctrl+Shift+Space`; any combination, or a key that never types (F13–F24, Pause, ScrollLock).
- **Cancel** a recording (default `Ctrl+Shift+Backspace`, the overlay's ×, or the tray) — nothing is pasted.
- **Local whisper.cpp transcription**, no internet needed; `large-v3-turbo` by default, model picker in Settings.
- **Russian / English** auto-detection constrained to those two languages (no Ukrainian/Polish drift), or pinned.
- **Floating pill overlay** on the monitor you are working on: waveform, RU/EN badge, timer, pin, cancel; it stays
  up through *Transcribing → Pasting → Pasted ✓* (or *No speech* / *Too short* / *Copied to clipboard*).
- **Streaming preview** in the main window while you speak (only while the window is visible).
- **Text pipeline**: filler removal (`эм`, `um`…), spoken line breaks (*"новая строка"*, *"new paragraph"*),
  a user dictionary for terms Whisper transliterates (*"три джей эс"* → `Three.js`), optional trailing space.
- **Paste safety**: the text goes only to the window that was focused when you released the key; if focus moved,
  the window is elevated or the session locked, the transcript lands in the clipboard instead. The previous
  clipboard content (text, HTML, files, images) is restored afterwards; transit writes are hidden from
  Clipboard History and cloud sync.
- **Crash insurance**: the audio of a recording is spooled to disk; after a crash it is transcribed into the
  history and the clipboard on the next start. A supervisor process restarts the app after a native CUDA abort
  and falls back to the CPU for one run.
- **Every recording is kept** (the last 30, up to 400 MB, as 16 kHz WAV in `dataecordings`). If a result
  looks wrong, *Tray → Re-transcribe last recording* runs the model again and copies the text.
- **A muted or wrong microphone cannot eat a dictation any more**: the pill names a fallback device, a
  hands-free recording without speech for 20 s says "No sound from the mic?" with a chime, and a capture
  without speech dynamics or a looping result ends as *No speech* with a toast explaining why, instead of
  pasting five minutes of "I'm going to be listening to the experience of the world."
- **History** of the last 100 dictations with time, target app and language; search, copy, paste again.
- **Usage card**: dictations, words, minutes and the no-result rate for today and the last 7 days
  (one JSON line per dictation in `stats.jsonl`).
- **Chimes** on the current Windows default output device, separate start/stop volume, custom sound files.
- **Tray**: start/stop (hands-free), cancel, language, AI formatting on/off, pause hotkey, restart on GPU,
  settings, log folder.
- **AI formatting** (optional) via OpenAI or Claude: punctuation, paragraphs, lists. Keys are DPAPI-encrypted
  and never shown again in the UI; the dictation is sent wrapped so it cannot be mistaken for instructions.
- **Diagnostics**: one log line per dictation with every stage's duration, a GPU watchdog that notices a
  power-capped card, build identity in the log, startup banner for a broken settings file (never overwritten).

## Requirements

- Windows 10/11
- [Rust](https://rustup.rs/) 1.77+
- [Node.js](https://nodejs.org/) 20+
- [CMake](https://cmake.org/) 3.5+
- [LLVM/Clang](https://releases.llvm.org/) (bindgen for whisper.cpp)
- [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/) with "Desktop development with C++"
- NVIDIA GPU + CUDA toolkit for the default (CUDA) build; see *CPU build* below otherwise

## Setup

```bash
git clone https://github.com/nsoth/wispr-local.git
cd wispr-local
npm install
```

Put the environment in place (paths for this machine live in `scripts/build-env.ps1`; adjust them once):

```powershell
. .\scripts\build-env.ps1
```

Download a multilingual GGML model, e.g. `ggml-large-v3-turbo.bin` from
[huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main), into

```
%APPDATA%\wispr-local\WisprLocal\data\models\
```

English-only `.en.bin` files are skipped on purpose (the app also handles Russian). Several models can
coexist; pick one in *Settings → Model* (no restart needed).

## Build and run

Development (hot reload):

```powershell
.\run-dev.ps1
```

Release executable (what the author runs daily; there is no installer):

```powershell
. .\scripts\build-env.ps1
npm run build
npx tauri build --no-bundle
# → src-tauri\target\release\wispr-local.exe
```

After changing anything under `src/`, delete `%LOCALAPPDATA%\com.wispr-local.app\EBWebView` while the
app is stopped: WebView2 caches the bundled UI and would keep serving the old one.

Tests (release profile, shares the whisper.cpp build with the app):

```powershell
npm test
```

### CPU build

CUDA is a default Cargo feature. Without an NVIDIA toolkit:

```powershell
cargo test --manifest-path src-tauri/Cargo.toml --no-default-features --lib
npx tauri build --no-bundle -- --no-default-features
```

The app detects the backend at load time and shows it in the footer (`Model ready · CUDA · large-v3-turbo`).

## Using it

1. Focus the app you want to type into.
2. Hold the hotkey and speak. The pill at the bottom of that monitor shows the waveform and the timer.
3. Release. The pill shows *Transcribing…* and then *Pasted ✓*.

Hands-free: click the pin in the pill (or tap the hotkey in *hybrid* mode, or start from the tray / the big
microphone button); press the hotkey again (or click the stop square) to finish. The limit is 30 minutes.

Spoken commands (between pauses): *новая строка* / *new line*, *новый абзац* / *new paragraph*.

## Settings worth knowing

- **General**: start with Windows, overlay on/off, hotkey mode, cancel key, language (Auto / Russian / English),
  microphone (system default follows Windows; a renamed USB mic is still recognized).
- **Sounds**: separate start/stop volume, custom files (peak-normalized), *Test* reports the output device.
- **Text**: spoken line breaks, what follows a paste (space / line break / nothing), restore clipboard, history
  retention (nothing / 20 / 100 / 500), the dictionary (*heard as → write as*, whole word, match case).
- **Model**: switch and reload the Whisper model.
- **AI formatting**: provider, key (stored encrypted; *Remove* forgets it), model (with suggestions), prompt
  (*Reset to default*), *Test* sends one small bilingual request and reports the latency or the provider's error.
  The dictionary's spellings are passed to the model as a glossary.

All of this lives in `%APPDATA%\wispr-local\WisprLocal\data\settings.json`. Editing it by hand is fine
while the app is closed; a file that cannot be parsed is moved aside as `settings.json.corrupt-<time>` and
reported in the window — it is never silently replaced.

## Logs and troubleshooting

- Log: `%APPDATA%\wispr-local\WisprLocal\data\wispr.log` (rotates into `.1`–`.3`); *Tray → Open log folder*.
  Each dictation leaves one line like
  `utt#12 audio=8.4s lock=1ms detect=180ms transcribe=620ms (rtf 0.10) format=0ms paste=410ms backend=CUDA lang=ru …`.
- **"Transcription is running very slowly"**: the GPU is probably power-capped (laptop unplugged → sleep →
  replugged). Check `nvidia-smi -q -d POWER`, replug or reboot.
- **Running on the CPU** after a crash: the supervisor retries CUDA automatically on the next run, or use
  *Tray → Restart on GPU*.
- **Toasts** are sent under the app's own identity; if Windows shows none, check *Settings → Notifications →
  Wispr Local*.
- Environment variables: `WISPR_NO_SUPERVISOR=1` runs the app without the watchdog; `WISPR_FORCE_CPU=1` forces
  the CPU backend; `RUST_LOG=debug` adds per-tick preview and hotkey lines to the log.
- Data folder (settings, history, keys, models, log): *Settings → About → Open data folder*.
- Recordings: *Settings → About → Open recordings* (`dataecordings\<time>-<outcome>.wav`); the newest one
  can be re-run from the tray. `recovered` files are what the crash recovery found at start.

## Project layout

```
src-tauri/src/
  lib.rs            wiring: state, plugins, hotkey handler, listeners
  pipeline.rs       start / preview / stop-transcribe-paste flows, model loader, overlay state
  hotkey.rs         hold / toggle / hybrid state machine
  overlay.rs        pill window placement and OS-level clipping
  supervisor.rs     crash watchdog + restart policy + log rotation
  settings.rs       settings.json (load never writes), secrets.rs (DPAPI keys), state.rs (history)
  text.rs           fillers, spoken commands, paste suffix; transcription/replacements.rs (dictionary)
  audio/            capture (cpal), device matching, sample buffer, crash spool
  transcription/    whisper engine, language gate, hallucination filters
  system/           chimes, text injection, focus/paste decisions, tray, toasts
src/                React UI (App.tsx main window, Overlay.tsx pill, ipc.ts event contract)
```

## License

MIT
