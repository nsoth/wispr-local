# Audit implementation plan (2026-10-02)

Spec: `../../../../wispr-local-audit-2026-10-02.md` (the full audit report, finding ids like
`audio-01`, `gap4-01` refer to it). Owner's instruction: take everything into work, rebuild,
redeploy, update docs and memory.

## Global constraints

- Windows-only Tauri 2 app used daily by the owner. Every task ends with a green Rust suite
  and, when `src/` changed, a green `npm run build` (tsc + vite).
- Test command (release profile shares the whisper.cpp build with the app):
  `powershell -NoProfile -ExecutionPolicy Bypass -Command ". ./scripts/build-env.ps1; cargo test --release --manifest-path src-tauri/Cargo.toml --lib"`
- Frontend check: `npm run build`.
- The running release exe must never be overwritten while a recording is active (check the
  tail of `%APPDATA%\wispr-local\WisprLocal\data\wispr.log`). Deploy = `npx tauri build
  --no-bundle` (with build-env), stop both `wispr-local` processes, delete
  `%LOCALAPPDATA%\com.wispr-local.app\EBWebView`, start the exe detached.
- Keep the intentional decisions listed in the audit's context block (opaque region-clipped
  overlay, WebView2 visibility cycle, overlay placement on the focused monitor, supervisor
  two-process model, filler policy, ru/en gate).
- TDD for every Rust behavior that can be unit-tested (pure functions, parsing, settings,
  text, supervisor policy). Tauri/Win32 integration is verified by build + runtime probes.
- Commit after each task with a descriptive message ending in the Claude attribution line.

## Task order

1. Commit pending async-loader work (done: 00f872a).
2. Build environment + test baseline.
3. Restructure `lib.rs` into modules, structured status protocol.
4. Sounds: per-play output stream, bounded wait, separate start/stop volume, fade-out.
5. Settings and secrets hardening, API-key lifecycle, startup diagnostics.
6. State managed before windows, ModelState, model picker/reload.
7. Text pipeline: fillers, hallucination patterns, voice commands, dictionary, spacing.
8. Microphone handling: name normalization, fallback, toast dedupe, structured mic error.
9. Timing logs, GPU watchdog, log noise.
10. Streaming preview gating and abort.
11. Pipeline feedback: overlay phases, tray tri-state, busy signal, pipeline guard, language badge.
12. Paste safety: foreground guard, elevation check, modifier wait, clipboard formats/flags.
13. Notifications: own AUMID, direct WinRT toasts, activation.
14. Supervisor: one-shot CPU fallback, restart-on-GPU, build identity, log rotation, soft hotkey failure.
15. Hotkey modes (hold/toggle/hybrid), cancel, extended keys, capture suspend, pause.
16. Tray menu: dynamic items, language submenu, AI toggle, tools.
17. History v2 and main-window layout.
18. UX visual pass: tokens, contrast, color-scheme, notices, settings extras, accessibility.
19. Engine quality: beam search final, persistent state, flash attention, normalization.
20. Lifecycle extras: single-instance plugin, history write under lock, lock/resume detection, exit hook, recording spool.
21. Security extras: error bodies, prompt wrapping, CSP, autostart, retention, deps.
22. Build hygiene: cuda feature, compile_error, gitattributes, icon script, eslint, CI, README.
23. Product extras: AI formatting improvements, usage stats, show window when model missing.
24. Deploy, push, tag, memory and README update, final review.

---

### Task 1: Commit pending work (build-01 part 1)

Done before this plan: commit `00f872a` holds the async model loader. Tag `v0.1.0` is applied
in Task 24 after the deploy build.

### Task 2: Build environment and test baseline

Files: `scripts/build-env.ps1` (new), `package.json` (`test` script uses release profile).

Steps:
1. `scripts/build-env.ps1` sets LIBCLANG_PATH, CMAKE, CUDA_PATH, PATH and enters the VS dev
   shell (`-SkipAutomaticLocation`).
2. Run the test command. Expected: `test result: ok. 27 passed`.
3. Change `package.json` `test` to `cargo test --release --manifest-path src-tauri/Cargo.toml --lib`.
4. Commit: "Add build-env helper and run tests in the release profile".

### Task 3: Restructure lib.rs and the status protocol (build-08, build-14, build-13 part)

Files: `src-tauri/src/lib.rs`, new `overlay.rs`, `pipeline.rs`, `text.rs`, `instance.rs`,
`events.rs`; `state.rs`; `commands.rs`; `src/App.tsx`, `src/Overlay.tsx`, new `src/ipc.ts`.

Design:
- Behavior-free move: overlay geometry/show/hide to `overlay.rs`; start/preview/stop flows,
  model loader and notify helpers to `pipeline.rs`; `remove_fillers` + tests to `text.rs`;
  single-instance mutex to `instance.rs`. `lib.rs` keeps `run()`, the hotkey closure and wiring.
- `events.rs`: `pub const STATUS_CHANGED: &str = "status-changed"` etc. `src/ipc.ts` mirrors
  event names and payload types.
- `AppStatus` gets `#[serde(tag = "state", content = "message", rename_all = "lowercase")]`.
  `pipeline::set_status(app, status)` is the only place that mutates `AppState.status` outside
  the atomic claim in stop flow, and it emits `status-changed` with the serialized enum.
  `get_status` returns the same payload. Frontend derives booleans from `status.state`.
- `pipeline::finish_without_result(app, reason, message)` replaces the four copy-pasted blocks.
- Tray emits the same `request-start-recording` / `request-stop-recording` events as the hotkey.

Tests (RED first): `AppStatus` serializes to `{"state":"error","message":"x"}` and
`{"state":"idle"}`; existing filler tests move with the function.

Verify: cargo tests green; `npm run build` green.
Commit: "Split lib.rs into overlay, pipeline, text and instance modules; structured status events".

### Task 4: Sounds (audio-01, audio-02, audio-03, audio-13, gap1-01..gap1-06)

Files: `src-tauri/src/system/sounds.rs`, `settings.rs`, `commands.rs`, `src/App.tsx`.

Design:
- `SoundConfig { start_sound, stop_sound, start_volume, stop_volume }`; `SoundKind { Start,
  Stop, Busy, Cancel }`.
- The sound thread never exits. Each `Play` opens `cpal::default_host().default_output_device()`
  → `OutputStream::try_from_device`, appends the source to a fresh `Sink`, waits with
  `while !sink.empty() && elapsed < expected + 500ms { sleep 10ms }`, then drops sink and
  stream. Device name logged on change ("Sound output: <name>"). Open failure logs and skips;
  never falls back to another device.
- Built-in chimes are synthesized into `SamplesBuffer<f32>` at 48 kHz with an 8 ms attack and
  25 ms release envelope (`synth_chime`). Start: 440 → 554 Hz, peaks 0.08/0.06. Stop: reverse.
  Busy: 2× 220 Hz 50 ms. Cancel: 330 Hz 90 ms.
- Custom files: decoded fully (cap 10 s), peak-normalized to 0.08, played via `SamplesBuffer`.
- Settings: `start_volume: Option<f32>`, `stop_volume: Option<f32>`; resolution
  `start = start_volume.unwrap_or(sound_volume * 0.6)`, `stop = stop_volume.unwrap_or(sound_volume)`.
  `set_sound_settings(start_sound, stop_sound, start_volume, stop_volume)` writes explicit values
  and keeps `sound_volume = stop_volume` for older builds.
- `test_sound(which, volume: Option<f32>) -> Result<String>` returns the device name via a
  reply channel (`recv_timeout(4 s)`).
- UI: two sliders (Start chime / Stop chime), Test buttons pass the live slider value, notice
  "Played on <device>". Saves only after user edits.

Tests (RED first): `synth_chime` first/last sample ≈ 0, peak ≤ configured, length correct;
`Settings` legacy `{"sound_volume":0.77}` resolves to start 0.462 / stop 0.77; explicit values
round-trip; `normalize_peak_to` scales a custom buffer.

Verify: cargo tests; `npm run build`; after deploy: Settings → Test follows the Windows default
output device.
Commit: "Play chimes on the current default output device with separate start/stop volume".

### Task 5: Settings and secrets hardening (gap4-01..07, security-01, security-11, ux-13, ux-23)

Files: `settings.rs`, `secrets.rs`, `formatting.rs`, `state.rs`, `commands.rs`, `lib.rs`,
`src/App.tsx`.

Design:
- `Settings::load_with_report(dir) -> SettingsLoad { settings, error: Option<String>, read_only: bool }`.
  Reads bytes, strips UTF-8 BOM, detects UTF-16 BOMs, `from_utf8`. Parse error → rename file
  to `settings.json.corrupt-<YYYYMMDD-HHMMSS>` and report; IO error → defaults, `read_only = true`
  (setters refuse with a clear message), file untouched. Unknown top-level keys are warned.
- `hotkey` gets `#[serde(default)]`; `AiProvider` and `LanguageMode` get `#[serde(other)] Unknown`
  normalized to None / Auto with a warning.
- `secrets.rs`: `ApiKeys { openai: String, claude: String }` stored DPAPI-encrypted as JSON in
  `api-keys.dat`; legacy `api-key.dat` migrates into both slots on first load (then removed);
  legacy plaintext `api_key` inside settings.json migrates the same way.
- `AiSettings` keeps `#[serde(skip)] openai_api_key / claude_api_key`; `fn api_key(&self)`.
- IPC: `get_ai_settings -> AiSettingsView { provider, openai_model, claude_model, prompt,
  openai_key_set, claude_key_set, key_error }`; `set_ai_settings(update: AiSettingsUpdate)` with
  `openai_api_key: Option<String>` / `claude_api_key: Option<String>` (None = unchanged, Some("")
  = remove). `Settings::apply_ai_update` is pure and returns whether keys changed.
- `state.rs`: `load_history_with_report`; corrupt history renamed `.corrupt-<ts>`; `push_history -> bool`;
  `save_history` skipped when nothing changed.
- `AppState.settings_load_error`, `history_load_error`; `get_startup_diagnostics` command; the
  frontend shows a persistent error banner (kind=error) with the serde message.
- Frontend: AI and sound auto-save only after a user edit (dirty refs). Key inputs show
  "Key stored" with Replace/Remove instead of the secret.

Tests (RED first): BOM file parses; UTF-16 file reports "not UTF-8"; trailing-comma file →
defaults + `.corrupt-*` exists with original bytes; `{"model_file":"x"}` without hotkey parses;
`"provider":"local"` → None with other fields kept; `"language":"russian"` → Auto; valid file is
byte-identical after load and no `.tmp` remains; `apply_ai_update` with no key fields keeps a
stored key; corrupt history → empty + renamed; `write_file_atomic` leaves no temp on success.

Commit: "Never overwrite settings with defaults; keep API keys out of the webview round-trip".

### Task 6: State before windows, ModelState, model picker (gap5-02, lifecycle-10, lifecycle-03, lifecycle-11, ux-06, gap3-06, gap5-08, engine-18, ux-07, product-09, build-07, security-09, product-16)

Files: `lib.rs`, `pipeline.rs`, `state.rs`, `commands.rs`, `transcription/models.rs` (delete),
`src/App.tsx`.

Design:
- Build `AppConfig`, settings, history, buffer/capture, engine, sound player before
  `tauri::Builder::default()`; register with `Builder::manage`. `setup()` keeps tray, overlay,
  hotkey, listeners and `spawn_model_loader`.
- `ModelState { Loading, Ready { backend, file, fallback: bool }, Missing, Failed(String) }` replaces
  `model_loaded`/`compute_backend`. `get_model_state` and `model-state-changed` share a payload
  `{ state, backend, file, fallback, error }`.
- Loader runs under `catch_unwind`; poison-tolerant locks; emits Failed on panic. Missing model
  shows and focuses the main window.
- `start_recording_flow`: Loading → record anyway and notice "Model is still loading"; Missing /
  Failed → specific toast.
- `reload_model(file: Option<String>)`, `get_model_files` (names + sizes), `set_model_file`.
  Settings > Model: select + Reload; footer shows file and backend; fallback notice.
- Frontend: initial batch retried with 300 ms / 1 s / 3 s backoff before the banner;
  `log_frontend_error` command.
- Delete `models.rs`.

Tests (RED first): `ModelState` payload serialization; candidate ordering puts the requested
file first and dedupes; `lock_or_recover` returns inner value for a poisoned mutex.

Commit: "Register state before windows are created; explicit model state with reload and picker".

### Task 7: Text pipeline (engine-03, engine-04, engine-05, engine-20, product-11, engine-14, product-03, system-08, product-04, engine-06 partial)

Files: `text.rs`, new `transcription/replacements.rs`, `settings.rs`, `commands.rs`, `pipeline.rs`,
`engine.rs`, `src/App.tsx`, `Cargo.toml` (`regex = "1"`).

Design:
- `remove_fillers` drops "мм"/"ммм"/"er"/"erm"; never removes a token right after a number;
  carries capitalization and terminal punctuation of a removed token.
- `clean_hallucinations` uses anchored patterns (segment starts with "субтитры сделал/создал",
  "редактор субтитров", "корректор а."; whole-segment equality for the stock phrases).
- `apply_voice_commands`: "новая строка"/"с новой строки"/"new line" → `\r\n`; "абзац"/"новый абзац"/
  "new paragraph" → `\r\n\r\n`; whole-phrase, case-insensitive, strips adjacent `, . ! ?`.
- `replacements.rs`: `ReplacementRule { from, to, whole_word, case_insensitive }` compiled once
  (`regex` with `(?iu)` and Unicode `\b`); `Vocabulary` in managed `RwLock`; default seed list
  (Three.js, WebGL, Tauri, React, Blender, glTF, GLB, retopology, mesh, API, GPU, CUDA).
- `TextPipeline::finalize(raw) -> String`: fillers → voice commands → replacements → trim.
  Applied to the final text and (fillers + replacements only) to the preview text.
- `paste_suffix: Space | Newline | None` (default Space) applied before injection when the text
  does not already end with whitespace; history stores the text without suffix.
- Settings UI: Dictionary table (add/remove rows), toggles for voice commands and paste suffix.

Tests (RED first): "толщина стенки 2 мм" unchanged; "went to the ER." unchanged; "Эм, привет,
э, как дела?" → "Привет, как дела?"; "Добавь субтитры к видео" survives; "Субтитры сделал
DimaTorzok" dropped; voice command cases; replacement whole-word and case rules; suffix helper.

Commit: "Text pipeline: safer fillers and hallucination filters, voice commands, user dictionary".

### Task 8: Microphone handling (gap2-02, audio-14, audio-04, audio-12 step 1, audio-15, ux-14)

Files: `audio/devices.rs`, `audio/capture.rs`, `pipeline.rs`, `commands.rs`, `state.rs`, `src/App.tsx`.

Design:
- `normalize_device_name` strips the Windows re-enumeration prefix `(N- ` → `(`; matching tries
  exact, then normalized; `used_fallback` only when neither matches.
- When the preferred device is found but fails to open, retry with the system default and report
  `used_fallback` with the error text.
- Fallback toast only when the fallback device changes (`last_fallback_device` in AppState);
  always `log::warn`.
- `AudioDeviceInfo.is_default`; UI marks "(default)" and mono/stereo; device list refreshes when
  Settings opens.
- Mic error becomes structured: `AppStatus::Error` payload `{ code: "mic", detail }`; UI shows
  "Microphone unavailable" with Choose microphone / Retry.

Tests (RED first): `normalize_device_name("Microphone (2- Wireless Mic Rx)") == "Microphone (Wireless Mic Rx)"`;
`match_device` prefers exact, then normalized; fallback dedupe helper.

Commit: "Match microphones by normalized name; fall back cleanly and stop the toast storm".

### Task 9: Timing logs, GPU watchdog, log noise (build-10, engine-17, product-15a, gap3-04, build-11)

Files: `engine.rs`, `pipeline.rs`, `state.rs`, `supervisor.rs`.

Design:
- `engine.transcribe*` returns `TranscriptionResult { text, language, detect_ms, full_ms, segments }`.
- One info line per utterance: `utt#N audio=12.3s lock=5ms transcribe=1480ms (rtf 0.12) format=640ms paste=410ms backend=CUDA lang=ru chars=214`.
- Preview baseline: EMA of tick duration per audio second after the first successful tick; a
  tick slower than `max(8 s, 6× baseline)` logs WARN, emits `operation-notice` "GPU is running
  very slowly: check charger / nvidia-smi power limit" and disables preview for the recording.
  Final slower than 5× realtime after 20 s emits "still transcribing" progress notice.
- Hotkey Pressed/Released and preview tick lines move to `debug`.
- Supervisor drain rotates the log inline (`wispr.log.1..3`, 5 MB each) and timestamps raw
  lines that do not start with `[`.

Tests (RED first): `Watchdog::is_slow` threshold math; `rotate_generations` renames in order.

Commit: "Log stage timings and warn when the GPU stalls; rotate logs in place".

### Task 10: Preview gating and abort (engine-01, ux-03b, product-05, engine-10, lifecycle-09)

Files: `pipeline.rs`, `engine.rs`, `state.rs`.

Design:
- Preview tick runs only when the main window is visible and not minimized (checked each tick);
  the loop keeps polling so opening the window mid-recording resumes previews.
- `preview_abort: Arc<AtomicBool>` set by the stop flow before locking the engine; the preview
  params use `set_abort_callback_safe`; preview text is not emitted once the status left Recording.
- Preview and final transcription run inside `tauri::async_runtime::spawn_blocking`.
- In CPU mode no preview is spawned.

Tests: abort flag helper; visibility gate is integration (manual).

Commit: "Gate the streaming preview on window visibility and make it abortable".

### Task 11: Pipeline feedback (lifecycle-02, ux-01, system-09, gap2-05, lifecycle-01 layer 1, engine-16, lifecycle-04, build-09, lifecycle-08, ux-18, ux-19, ux-05)

Files: `pipeline.rs`, `overlay.rs`, `state.rs`, `system/tray.rs`, `src/Overlay.tsx`,
`src/styles/overlay.css`, `tauri.conf.json`.

Design:
- Overlay stays visible until the pipeline ends. Rust emits `overlay-state` `{ phase, message,
  language }` with phases `recording | processing | result`; `finish_pipeline(app, outcome)` is
  the single exit: shows the result (Pasted ✓ / No speech / Too short / Failed / Copied to
  clipboard) for 1.2–2 s, then hides unless a new recording started.
- Tray: `AtomicU8 { Idle, Recording, Processing }`, amber dot and tooltip "Transcribing…".
- Busy press: short busy chime + notice "Still transcribing the previous recording".
- `PipelineGuard` (Drop) resets status to Idle, stops tray animation and hides the overlay if the
  stop task unwinds; status `Starting` while the capture opens.
- Overlay waveform: decay per incoming sample instead of per frame; timer m:ss; language badge
  from `language-detected`; border contrast raised; window title "Wispr Local Overlay".
- `too-short` no longer sends an OS toast (overlay + notice only).

Tests: `PipelineGuard` resets status when dropped armed; `format_timer`.

Commit: "Keep the overlay alive through transcription and paste; tray processing state; busy signal".

### Task 12: Paste safety (system-07, system-01, system-04, security-02, system-15, security-03, system-02, system-03, gap3-01, product-13)

Files: `system/text_injection.rs`, `pipeline.rs`, `settings.rs`, `commands.rs`, `src/App.tsx`.

Design:
- At stop, capture `GetForegroundWindow` + PID + title (`PasteTarget`). Before pasting: if the
  foreground PID differs or equals our own, or the target is elevated (`OpenProcess` denied), or
  the input desktop is not "Default": copy to clipboard (no restore), notify "Focus changed /
  elevated window: transcript copied to clipboard".
- Wait up to 500 ms for Shift/Alt/Win/Ctrl to be physically released before SendInput.
- Clipboard snapshot: file_list → html → text → image; restore in the same order; transit writes
  use `exclude_from_history().exclude_from_cloud()`; restore delayed 1 s on a background thread;
  `restore_clipboard: bool` (default true) setting ("keep transcript in clipboard").
- `inject_text` returns `InjectOutcome { pasted: bool, reason }` and logs "paste shortcut sent".

Tests (RED first): `modifiers_released` polling helper with injected state; clipboard snapshot
ordering with a fake backend trait; target-decision function (`decide_paste(target_now, target_then, own_pid, elevated)`).

Commit: "Paste only into the window that was focused at release; preserve more clipboard formats".

### Task 13: Notifications (gap2-01, gap2-03, gap2-04, gap2-06)

Files: new `system/notify.rs`, `lib.rs`, `Cargo.toml` (`tauri-winrt-notification = "0.7"`,
windows-sys `Win32_UI_Shell`), `autostart.rs` pattern reuse.

Design:
- `register_app_identity(data_dir)`: `HKCU\Software\Classes\AppUserModelId\com.wispr-local.app`
  with DisplayName and IconUri (icon extracted to data_dir once); `SetCurrentProcessExplicitAppUserModelID`
  in the child before windows.
- `notify_user` uses `Toast::new("com.wispr-local.app")` with `on_activated` showing the main
  window; failures logged at WARN and fall back to the plugin.

Tests: registry helper is integration; unit test for `aumid_registry_values()` string building.

Commit: "Send toasts under our own AppUserModelId".

### Task 14: Supervisor and build identity (lifecycle-05, engine-12, lifecycle-17 part, gap5-03, gap5-04, gap5-06, gap5-07, lifecycle-12, build-05, lifecycle-18, system-06)

Files: `supervisor.rs`, `lib.rs`, `build.rs`, `engine.rs`, `pipeline.rs`, `system/tray.rs`.

Design:
- `classify_exit(code, elapsed, fast_crashes, force_cpu) -> RestartDecision` (pure, tested).
  CPU fallback is one-shot: cleared once the child lived longer than FAST_CRASH; a second
  fast-fail within 10 min keeps CPU for the session. Exit code `0x5752` = "restart on GPU".
- CPU mode: small models first in the candidate list, `available_parallelism().min(16)` threads,
  no preview, footer text "CPU mode: GPU failed, restart to retry CUDA".
- `build.rs` exports `WISPR_GIT_HASH`, `WISPR_GIT_DIRTY`, `WISPR_BUILD_TIME`; logged at start and
  in the supervisor "app started" line with args and version.
- `RunEvent::Exit` writes "session ending" directly to the log file.
- Single-instance mutex waits up to 10 s for a tearing-down instance.
- Hotkey registration failure at start becomes a soft error (notice + status error) with 3 retries.

Tests (RED first): `classify_exit` table (clean, session-end, fast-fail → CPU once, give-up);
`candidate_order(force_cpu)`.

Commit: "Supervisor: one-shot CPU fallback, restart-on-GPU, build identity in logs".

### Task 15: Hotkey modes and cancel (product-01, ux-17, system-10, ux-02, product-02, system-11, product-12, ux-12, system-12)

Files: `lib.rs`, `pipeline.rs`, `state.rs`, `settings.rs`, `commands.rs`, `src/App.tsx`,
`src/Overlay.tsx`.

Design:
- `hotkey_mode: Hold | Toggle | Hybrid` (default Hold). `HotkeyMachine` (pure) maps
  `(mode, event, status, locked, elapsed_since_press)` → `Action { Start, Stop, Lock, Ignore }`.
  Hybrid: release within 350 ms of start locks the recording.
- `cancel_recording` command + overlay X button + optional cancel hotkey (default
  "Ctrl+Shift+Backspace", empty disables); cancel during processing marks the entry "not pasted".
- `parse_key_code` gains F13–F24, Numpad0–9/Add/Subtract, Pause, ScrollLock, CapsLock, Insert,
  Home, End, PageUp, PageDown; bare F13–F24/Pause/ScrollLock/CapsLock allowed without modifiers.
- Capture suspends the live shortcut (`is_capturing` flag ignores events) and translates plugin
  errors to "Already used by another app".
- Tray "Pause hotkey" check item (`hotkey_paused` in AppState).

Tests (RED first): `HotkeyMachine` table for the three modes; `parse_hotkey("F13")` ok,
`is_safe_bare_key`.

Commit: "Hold, toggle and hybrid hotkey modes; cancel a recording; more hotkey keys".

### Task 16: Tray menu (ux-11, system-13, lifecycle-17, ux-05, security-13 phase 1)

Files: `system/tray.rs`, `lib.rs`, `commands.rs`, `settings.rs`.

Design:
- `TrayMenu` in managed state with `set_enabled`/`set_text`: toggle Start/Stop item, status
  item, Language submenu (check items), "AI formatting" check (`ai.enabled`), "Pause hotkey",
  "Copy last transcript", "Open settings", "Open log folder", "Restart on GPU" (CPU only), Quit
  waits up to 2 s for Idle.
- Tray start is hands-free (`pinned = true`).

Tests: menu label helper `tray_labels(status)`.

Commit: "Dynamic tray menu with language, AI toggle and tools".

### Task 17: History v2 and layout (ux-04, product-10, ux-09, product-13)

Files: `state.rs`, `commands.rs`, `pipeline.rs`, `src/App.tsx`, `src/styles/global.css`, `tauri.conf.json`.

Design:
- `HistoryEntry { text, ts (unix ms), target, lang, duration_s, pasted }`; `HISTORY_LIMIT = 100`;
  legacy `Vec<String>` accepted via untagged enum. `history.json` stays (small).
- UI: relative time + target, click to expand, search box, per-row Copy / Paste again
  (`paste_history_item(idx)` activates the saved hwnd when it still exists, else copies), two-step
  inline Clear with `ask()`.
- Layout: `.app` fills the viewport, history list scrolls; default window 380×560, min 340×420.

Tests (RED first): legacy history parses; `push_history` dedup/cap; relative-time helper (TS).

Commit: "History with timestamps, targets and re-paste; scrollable main layout".

### Task 18: UX visual pass (ux-08, ux-20, ux-15, ux-16, ux-10, ux-21, ux-14 UI)

Files: `src/styles/global.css`, `index.html`, `src/App.tsx`, `src/styles/overlay.css`.

Design: color tokens on `:root` (`--text-1/2/3`, `--accent`, `--border`), WCAG AA for secondary
text, minimum 12–13 px, `color-scheme: dark`, thin scrollbars, `ask()` dialogs instead of
`window.confirm`, notice `{text, kind}` with timed info and sticky errors, Settings Advanced group
(Reset prompt, Open data folder, Open log, version/build), Escape closes Settings or hides the
window, Ctrl+, opens Settings, `lang` attribute per history item, mic circle becomes a hands-free
start button.

Verify: `npm run build`; screenshots via vite preview.
Commit: "Visual and accessibility pass for the main window".

### Task 19: Engine quality (engine-09, engine-11, engine-02, engine-07, engine-08, engine-19, audio-10, audio-06)

Files: `engine.rs`, `audio/buffer.rs`, `whisper_gotchas.md` (memory, outside repo).

Design: persistent `WhisperState` created at load (recreated on reload); final pass uses
`BeamSearch { beam_size: 5, patience: -1.0 }`, preview uses Greedy + `temperature_inc(0)`;
`flash_attn(true)` for the GPU attempt; comments corrected for `no_context`/no_speech granularity;
`normalize_peak` uses the 99.5th percentile with a 0.98 clamp, ignoring the first 200 ms;
`INITIAL_CAPACITY` 5 minutes; `assemble_segments` extracted and tested.

Tests (RED first): `assemble_segments` drops no-speech windows and stock phrases;
`normalize_peak` ignores one transient; buffer capacity.

Commit: "Beam search for the final pass, persistent whisper state, robust normalization".

### Task 20: Lifecycle extras (lifecycle-06, gap5-06, lifecycle-13, security-12, lifecycle-14, gap3-02, gap3-05, lifecycle-15)

Files: `lib.rs`, `instance.rs`, `state.rs`, `config.rs`, `commands.rs`, `pipeline.rs`,
new `audio/spool.rs`.

Design: `tauri-plugin-single-instance` shows the running window; history saved under the state
lock with unique temp names (`<file>.<pid>.<seq>.tmp`) and stale `*.tmp` cleaned at start;
device commands async; preview tick detects resume (wall-clock jump > 10 s) and lock
(`GetForegroundWindow() == NULL` twice) → stop capture, keep audio in history/clipboard without
auto-paste; `RunEvent::Exit` spools an in-flight recording; `pending.pcm` writer during recording,
recovered on next start (transcribed → history + clipboard + toast).

Tests (RED first): unique temp naming; spool writer/reader round trip; `detect_interruption`.

Commit: "Recover recordings across crashes, lock and sleep; single-instance activation".

### Task 21: Security extras (security-05, security-06, security-07, security-08, system-16, security-04, security-10, build-06, build-19)

Files: `formatting.rs`, `tauri.conf.json`, `capabilities/default.json`, `autostart.rs`,
`commands.rs`, `settings.rs`, `Cargo.toml`.

Design: provider error bodies capped at 4 KB with `error.message` extraction; transcript wrapped
in `<transcript>` with a fixed system suffix, OpenAI `max_completion_tokens`, output length lower
bound and alphabet check; CSP `default-src 'self'; connect-src ipc: http://ipc.localhost; img-src
'self' data:; style-src 'self' 'unsafe-inline'`, `withGlobalTauri: false`, capabilities trimmed;
autostart written only when different and skipped in debug, `StartupApproved` read; history
retention setting (off / 5 / 20 / 100); drop `hound`/`ringbuf`, `tokio` features trimmed,
`rust-version`, `directories = "6"`.

Tests (RED first): `summarize_error_body`; `validate_formatted_output` rejects < 40% and alphabet
flip; autostart value builder.

Commit: "Harden AI formatting, CSP and autostart; trim dependencies".

### Task 22: Build hygiene (build-02, build-15, build-16, build-18, build-20, build-03, build-04, build-17, build-21)

Files: `Cargo.toml`, `lib.rs`, `.gitattributes`, `.gitignore`, `scripts/generate_icon.py`,
`index.html`, `eslint.config.js`, `package.json`, `.github/workflows/ci.yml`, `README.md`.

Design: `cuda` as a default feature; `compile_error!` on non-Windows and stub removal;
`.gitattributes` + renormalize; `src-tauri/gen/schemas` ignored; icon script path fix; favicon
link removed; eslint flat config with react-hooks; `npm run check` = lint + fmt check + build +
test; CI workflow on windows-latest with `--no-default-features`; README rewritten for actual
behavior (no installer, logs, env vars, WebView cache, troubleshooting); module doc comments.

Commit: "Build hygiene: optional CUDA feature, lint gates, CI and an accurate README".

### Task 23: Product extras (product-07, product-15b)

Files: `formatting.rs`, `settings.rs`, `commands.rs`, new `stats.rs`, `src/App.tsx`.

Design: bilingual default prompt with glossary from the dictionary, 8 s timeout, `test_ai_connection`
command + Test button, model `<datalist>`; `stats.jsonl` one line per dictation, `get_stats(days)`,
"Today / 7 days" card (words, minutes, no-result rate).

Tests (RED first): stats aggregation over sample lines.

Commit: "AI formatting test button and usage stats".

### Task 24: Deploy, push, memory

Steps: `npx tauri build --no-bundle` with build-env; wait for an idle moment; stop both processes;
delete EBWebView cache; start the exe; probe: log shows build identity, model loaded, toast test,
Settings → Test sound logs "Sound output"; overlay probe via keybd_event hold; `git push origin
main`; `git tag -a v0.2.0`; `git push --tags`; update memory files and MEMORY.md; final review.
