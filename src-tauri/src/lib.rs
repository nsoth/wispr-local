pub mod audio;
pub mod autostart;
pub mod commands;
pub mod config;
pub mod formatting;
pub mod secrets;
pub mod settings;
pub mod state;
pub mod supervisor;
pub mod system;
pub mod transcription;

use std::sync::Mutex;
use tauri::{Emitter, Listener, Manager};

use audio::buffer::AudioBuffer;
use audio::capture::AudioCapture;
use config::AppConfig;
use settings::Settings;
use state::{AppState, AppStatus};
use system::sounds::SoundPlayer;
use system::tray::TrayAnimator;
use transcription::engine::WhisperEngine;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // The process the user (or autostart) launches is only a crash watchdog;
    // the actual app runs as its supervised child. whisper.cpp aborts the
    // whole process on CUDA errors, and the supervisor turns that into an
    // automatic restart instead of silently dead dictation (supervisor.rs).
    if !supervisor::should_run_app() {
        supervisor::run_supervisor();
    }

    // Only the supervised child owns the instance guard. A second launch
    // starts its own supervisor/child pair, sees this guard, then exits cleanly
    // instead of repeatedly crashing on the already-registered global hotkey.
    if !claim_single_instance() {
        eprintln!("Wispr Local is already running");
        return;
    }

    // When launched via autostart, detach from the parent console so no
    // terminal window lingers on the desktop. In release builds the binary
    // already uses windows_subsystem = "windows", so this is a no-op there;
    // it matters for debug builds started from the Run registry key.
    #[cfg(windows)]
    {
        if std::env::args().any(|a| a == "--hidden") {
            unsafe {
                windows_sys::Win32::System::Console::FreeConsole();
            }
        }
    }

    env_logger::init();
    // Route whisper.cpp/GGML native log output through the `log` crate →
    // env_logger → stderr → supervisor pipe → wispr.log. Without this the
    // CUDA error text printed right before a GGML abort was lost with the
    // invisible stderr of a windows-subsystem process.
    whisper_rs::install_logging_hooks();

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    use tauri_plugin_global_shortcut::ShortcutState;
                    log::info!("Hotkey event: {:?} state={:?}", shortcut, event.state);

                    let (recording, locked) = {
                        let state = app.state::<Mutex<AppState>>();
                        let s = state.lock().unwrap();
                        (s.status == AppStatus::Recording, s.recording_locked)
                    };

                    match event.state {
                        ShortcutState::Pressed => {
                            if recording && locked {
                                // Pinned recording: a fresh press stops it.
                                log::info!("Hotkey PRESSED - stopping pinned recording");
                                let _ = app.emit("hotkey-stop-recording", ());
                            } else if !recording {
                                log::info!("Hotkey PRESSED - starting recording");
                                let _ = app.emit("hotkey-start-recording", ());
                            }
                            // recording && !locked: key-repeat while holding — ignore.
                        }
                        ShortcutState::Released => {
                            if locked {
                                // Pinned via the overlay button: keep recording
                                // after the key is released.
                                log::info!("Hotkey RELEASED - recording pinned, ignoring");
                            } else {
                                log::info!("Hotkey RELEASED - stopping recording");
                                let _ = app.emit("hotkey-stop-recording", ());
                            }
                        }
                    }
                })
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            // Initialize configuration
            let config = AppConfig::new();
            config
                .ensure_dirs()
                .expect("Failed to create app directories");

            // Initialize audio pipeline
            let buffer = AudioBuffer::new();
            let capture = AudioCapture::new(buffer.clone());

            // Load settings (needed below for model selection)
            let user_settings = Settings::load(&config.data_dir);
            log::info!("Loaded hotkey setting: {}", user_settings.hotkey);

            // Initialize Whisper engine. Try the configured model first, then
            // fall back to older models so an incomplete download doesn't
            // leave the app without transcription.
            let mut engine = WhisperEngine::new();
            let mut initial_state = AppState {
                history: state::load_history(&config.data_dir),
                ..AppState::default()
            };

            let mut candidates = vec![user_settings.model_file.clone()];
            for fallback in [
                settings::default_model_file(),
                "ggml-medium.bin".to_string(),
            ] {
                if !candidates.contains(&fallback) {
                    candidates.push(fallback);
                }
            }
            for discovered in config.available_model_files() {
                if !candidates.contains(&discovered) {
                    candidates.push(discovered);
                }
            }

            for model_filename in &candidates {
                let model_path = config.model_path(model_filename);
                if !model_path.exists() {
                    log::warn!("Model not found at {:?}", model_path);
                    continue;
                }
                match engine.load_model(&model_path) {
                    Ok(_) => {
                        log::info!("Model loaded from {:?}", model_path);
                        initial_state.model_loaded = true;
                        break;
                    }
                    Err(e) => log::error!("Failed to load model {}: {}", model_filename, e),
                }
            }
            if !initial_state.model_loaded {
                log::error!(
                    "No usable Whisper model found in {:?}. Download one to enable transcription.",
                    config.models_dir
                );
            } else if std::env::var("WISPR_FORCE_CPU").is_ok() {
                notify_user(
                    app.handle(),
                    "Wispr Local recovered from a GPU failure and is using CPU transcription until restart.",
                );
            }

            // Sync autostart state with saved settings
            if user_settings.run_on_startup {
                let _ = autostart::set_autostart_registry(true);
                log::info!("Autostart enabled");
            }

            // Initialize sound player (persistent output stream) with settings
            let sound_player = SoundPlayer::new(
                user_settings.start_sound.clone(),
                user_settings.stop_sound.clone(),
                user_settings.sound_volume,
            );

            // Register state
            app.manage(Mutex::new(initial_state));
            app.manage(Mutex::new(capture));
            app.manage(buffer.clone());
            app.manage(Mutex::new(engine));
            app.manage(config);
            app.manage(sound_player);
            app.manage(Mutex::new(user_settings.clone()));

            // Setup system tray (also manages TrayAnimator state).
            system::tray::setup_tray(app.handle())?;

            // Position the overlay window just above the Windows taskbar and
            // ensure it respects the current show_overlay setting.
            place_overlay_window(app.handle());
            if let Some(overlay) = app.get_webview_window("overlay") {
                let _ = overlay.hide();
                sync_overlay_webview_visibility(&overlay, false);
                // WS_EX_NOACTIVATE: the pin button must be clickable without
                // stealing focus from the app the user is dictating into —
                // otherwise the eventual paste would land in the wrong window.
                let _ = overlay.set_focusable(false);
            }

            // Register global hotkey from settings
            {
                use tauri_plugin_global_shortcut::GlobalShortcutExt;
                let shortcut = commands::parse_hotkey(&user_settings.hotkey)
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                app.global_shortcut().register(shortcut)?;
                log::info!(
                    "Global hotkey registered: {} (hold to dictate)",
                    user_settings.hotkey
                );
            }

            // Make close button hide the window instead of destroying it
            if let Some(window) = app.get_webview_window("main") {
                let w = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = w.hide();
                    }
                });
            }

            // Handle start recording (from hotkey or tray)
            let app_handle = app.handle().clone();
            app.listen("hotkey-start-recording", move |_event| {
                start_recording_flow(&app_handle);
            });

            let app_handle = app.handle().clone();
            app.listen("tray-start-recording", move |_event| {
                start_recording_flow(&app_handle);
            });

            // Handle stop recording (from hotkey or tray)
            let app_handle = app.handle().clone();
            app.listen("hotkey-stop-recording", move |_event| {
                let app = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    stop_and_transcribe_flow(&app).await;
                });
            });

            let app_handle = app.handle().clone();
            app.listen("tray-stop-recording", move |_event| {
                let app = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    stop_and_transcribe_flow(&app).await;
                });
            });

            // A pinned recording cannot grow memory forever. The audio callback
            // emits this once at 30 minutes, then the normal finalization path
            // transcribes everything captured so far.
            let app_handle = app.handle().clone();
            app.listen("recording-limit-reached", move |_event| {
                let app = app_handle.clone();
                notify_user(
                    &app,
                    "Recording reached the 30-minute limit and is being processed.",
                );
                tauri::async_runtime::spawn(async move {
                    stop_and_transcribe_flow(&app).await;
                });
            });

            let app_handle = app.handle().clone();
            app.listen("audio-stream-error", move |_event| {
                let app = app_handle.clone();
                let message =
                    "The microphone disconnected or stopped responding; processing captured audio.";
                let _ = app.emit("operation-notice", message);
                notify_user(&app, message);
                tauri::async_runtime::spawn(async move {
                    stop_and_transcribe_flow(&app).await;
                });
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_status,
            commands::is_model_loaded,
            commands::get_compute_backend,
            commands::get_last_transcription,
            commands::toggle_recording_lock,
            commands::get_history,
            commands::clear_history,
            commands::copy_text,
            commands::get_models_dir,
            commands::open_models_dir,
            commands::get_hotkey,
            commands::set_hotkey,
            commands::get_sound_settings,
            commands::set_sound_settings,
            commands::test_sound,
            commands::get_ai_settings,
            commands::set_ai_settings,
            commands::get_autostart,
            commands::set_autostart,
            commands::get_show_overlay,
            commands::set_show_overlay,
            commands::get_language,
            commands::set_language,
            commands::get_input_devices,
            commands::get_input_device,
            commands::set_input_device,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(windows)]
struct InstanceMutex(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
unsafe impl Send for InstanceMutex {}
#[cfg(windows)]
unsafe impl Sync for InstanceMutex {}

#[cfg(windows)]
impl Drop for InstanceMutex {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

#[cfg(windows)]
fn claim_single_instance() -> bool {
    use std::os::windows::ffi::OsStrExt;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{
        GetLastError, ERROR_ALREADY_EXISTS, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::System::Threading::CreateMutexW;

    static INSTANCE_MUTEX: OnceLock<InstanceMutex> = OnceLock::new();
    let name: Vec<u16> = std::ffi::OsStr::new("Local\\WisprLocalAppInstance")
        .encode_wide()
        .chain(Some(0))
        .collect();
    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    if handle.is_null() || handle == INVALID_HANDLE_VALUE {
        // Failure to create the guard should not make dictation unavailable.
        return true;
    }
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(handle);
        }
        return false;
    }
    INSTANCE_MUTEX.set(InstanceMutex(handle)).is_ok()
}

#[cfg(not(windows))]
fn claim_single_instance() -> bool {
    true
}

/// Clip the overlay window to a rounded pill (corner radius = height/2) via
/// SetWindowRgn, so nothing outside the pill exists at the compositor level.
/// WebView2 transparency proved unreliable here (square backdrop behind the
/// rounded CSS pill), so instead the window is opaque and simply has no pixels
/// outside the rounded region. `w`/`h` are physical pixels — the region must
/// be recomputed whenever the window lands on a monitor with a different DPI.
#[cfg(windows)]
fn apply_pill_region(overlay: &tauri::WebviewWindow, w: i32, h: i32) {
    let Ok(hwnd) = overlay.hwnd() else {
        log::warn!("Could not get overlay hwnd for pill region");
        return;
    };
    unsafe {
        use windows_sys::Win32::Graphics::Gdi::{CreateRoundRectRgn, DeleteObject, SetWindowRgn};
        // Ellipse w/h = window height -> fully rounded ends, matching the
        // CSS border-radius: 999px pill. The region takes ownership of rgn.
        let rgn = CreateRoundRectRgn(0, 0, w + 1, h + 1, h, h);
        if rgn.is_null() {
            log::warn!("Could not create overlay pill region");
        } else if SetWindowRgn(hwnd.0 as _, rgn, 1) == 0 {
            // Windows takes ownership only on success.
            let _ = DeleteObject(rgn as _);
            log::warn!("Could not apply overlay pill region");
        }
    }
}

#[cfg(not(windows))]
fn apply_pill_region(_overlay: &tauri::WebviewWindow, _w: i32, _h: i32) {}

/// Physical-pixel rect of the window that currently has keyboard focus — the
/// app the user is dictating into.
#[cfg(windows)]
fn foreground_window_rect() -> Option<windows_sys::Win32::Foundation::RECT> {
    use windows_sys::Win32::Foundation::RECT;
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowRect};
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return None;
        }
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        if GetWindowRect(hwnd, &mut rect) == 0 {
            return None;
        }
        Some(rect)
    }
}

/// The monitor hosting the focused window, falling back to the primary.
/// Dictation pastes into the focused app, so that monitor is where the user
/// is looking; pinning the overlay to the primary monitor made it invisible
/// whenever the user worked on another display.
fn target_monitor(overlay: &tauri::WebviewWindow) -> Option<tauri::Monitor> {
    #[cfg(windows)]
    if let Some(r) = foreground_window_rect() {
        let cx = r.left + (r.right - r.left) / 2;
        let cy = r.top + (r.bottom - r.top) / 2;
        if let Ok(monitors) = overlay.available_monitors() {
            let hit = monitors.into_iter().find(|m| {
                let p = m.position();
                let s = m.size();
                cx >= p.x && cx < p.x + s.width as i32 && cy >= p.y && cy < p.y + s.height as i32
            });
            if hit.is_some() {
                return hit;
            }
        }
    }
    overlay.primary_monitor().ok().flatten()
}

/// Position the overlay bottom-center on the monitor the user is working on
/// and clip it to the pill shape at that monitor's DPI.
fn place_overlay_window(app: &tauri::AppHandle) {
    let Some(overlay) = app.get_webview_window("overlay") else {
        return;
    };

    // Logical size mirrors the overlay entry in tauri.conf.json.
    const LOGICAL_W: f64 = 312.0;
    const LOGICAL_H: f64 = 52.0;
    // Room for the default Windows taskbar (48 logical px) plus a small gap
    // so the overlay doesn't feel glued to it.
    const BOTTOM_MARGIN: f64 = 64.0;

    let Some(monitor) = target_monitor(&overlay) else {
        log::warn!("Overlay positioning skipped: no monitor found");
        return;
    };

    let scale = monitor.scale_factor();
    let mon_pos = monitor.position();
    let mon_size = monitor.size();

    // Physical size the window will have ON the target monitor. outer_size()
    // can't be used here: it reports the size at the window's current DPI,
    // which is stale while moving between monitors with different scales.
    let win_w = (LOGICAL_W * scale).round() as i32;
    let win_h = (LOGICAL_H * scale).round() as i32;
    let margin = (BOTTOM_MARGIN * scale).round() as i32;

    let x = mon_pos.x + (mon_size.width as i32 - win_w) / 2;
    let y = mon_pos.y + mon_size.height as i32 - win_h - margin;

    if let Err(e) = overlay.set_position(tauri::PhysicalPosition { x, y }) {
        log::warn!("Failed to position overlay: {}", e);
    }
    apply_pill_region(&overlay, win_w, win_h);
}

/// Sync the WebView2 controller's own visibility flag with the overlay
/// window. Since the late-July 2026 updates (WebView2 150.x runtime /
/// KB5101711), ShowWindow alone no longer resumes composition that was
/// suspended when this WS_EX_NOACTIVATE window was hidden: Win32 reports the
/// window visible with correct rect and region, but not a single pixel (not
/// even backgroundColor) reaches the screen. Dropping the controller to
/// hidden and back forces WebView2 to resume drawing; keeping it hidden
/// while the window is hidden also stops pointless background compositing.
#[cfg(windows)]
fn sync_overlay_webview_visibility(overlay: &tauri::WebviewWindow, visible: bool) {
    let result = overlay.with_webview(move |webview| unsafe {
        let controller = webview.controller();
        let _ = controller.SetIsVisible(false);
        if visible {
            if let Err(e) = controller.SetIsVisible(true) {
                log::error!("WebView2 SetIsVisible(true) failed: {e}");
            }
        }
    });
    if let Err(e) = result {
        log::error!("Overlay webview visibility sync failed: {e}");
    }
}

#[cfg(not(windows))]
fn sync_overlay_webview_visibility(_overlay: &tauri::WebviewWindow, _visible: bool) {}

fn show_overlay_if_enabled(app: &tauri::AppHandle) {
    let show = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = settings.lock().unwrap();
        guard.show_overlay
    };
    if !show {
        return;
    }
    let Some(overlay) = app.get_webview_window("overlay") else {
        log::error!("Overlay window is gone — recording indicator unavailable");
        return;
    };
    place_overlay_window(app);
    if let Err(e) = overlay.show() {
        log::error!("Failed to show overlay: {}", e);
    }
    sync_overlay_webview_visibility(&overlay, true);
    // Crossing to a monitor with a different scale factor resizes the window
    // shortly after set_position; re-clip once the size has settled so the
    // pill isn't left with a stale region.
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        let recording = {
            let state = app.state::<Mutex<AppState>>();
            let s = state.lock().unwrap();
            s.status == AppStatus::Recording
        };
        if !recording {
            return;
        }
        if let Some(overlay) = app.get_webview_window("overlay") {
            if let Ok(size) = overlay.outer_size() {
                apply_pill_region(&overlay, size.width as i32, size.height as i32);
            }
        }
    });
}

fn hide_overlay(app: &tauri::AppHandle) {
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.hide();
        sync_overlay_webview_visibility(&overlay, false);
    }
}

fn start_recording_flow(app: &tauri::AppHandle) {
    log::info!("start_recording_flow called");
    let state = app.state::<Mutex<AppState>>();
    let capture = app.state::<Mutex<AudioCapture>>();
    let buffer = app.state::<AudioBuffer>();

    let can_start = {
        let Ok(mut s) = state.lock() else {
            log::error!("App state lock poisoned while starting recording");
            return;
        };
        if !matches!(s.status, AppStatus::Idle | AppStatus::Error(_)) {
            log::info!("Ignoring recording request while app is busy");
            return;
        }
        if !s.model_loaded {
            false
        } else {
            s.status = AppStatus::Recording;
            s.recording_locked = false;
            true
        }
    };

    if !can_start {
        notify_user(
            app,
            "Whisper model is not loaded. Open Wispr Local for setup help.",
        );
        let _ = app.emit("operation-notice", "Whisper model is not loaded");
        return;
    }

    {
        buffer.clear();
    }

    let preferred_device = {
        let settings = app.state::<Mutex<Settings>>();
        settings
            .lock()
            .map(|s| s.input_device.clone())
            .unwrap_or_default()
    };
    let start_result = capture
        .lock()
        .map_err(|e| e.to_string())
        .and_then(|mut cap| {
            cap.start(
                Some(app.clone()),
                (!preferred_device.is_empty()).then_some(preferred_device.as_str()),
            )
        });
    match start_result {
        Ok(start) => {
            if let Ok(mut s) = state.lock() {
                s.device_sample_rate = start.sample_rate;
            }
            log::info!(
                "Recording started at {} Hz with input device {}",
                start.sample_rate,
                start.device_name
            );
            if start.used_fallback {
                let message = format!(
                    "Selected microphone is unavailable; using {}.",
                    start.device_name
                );
                let _ = app.emit("operation-notice", &message);
                notify_user(app, &message);
            }
        }
        Err(e) => {
            log::error!("Failed to start recording: {}", e);
            if let Ok(mut s) = state.lock() {
                s.status = AppStatus::Error(e.clone());
            }
            let message = format!("Microphone error: {e}");
            let _ = app.emit("status-changed", &message);
            let _ = app.emit("operation-notice", &message);
            notify_user(app, &message);
            app.state::<TrayAnimator>().stop();
            hide_overlay(app);
            return;
        }
    }

    let _ = app.emit("status-changed", "Recording");
    let _ = app.emit("lock-changed", false);
    app.state::<SoundPlayer>().play_start();

    // Kick off tray animation and reveal overlay (if user hasn't disabled it).
    app.state::<TrayAnimator>().start();
    show_overlay_if_enabled(app);

    // Spawn streaming preview: transcribe every ~2s while recording
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        streaming_preview_loop(app_clone).await;
    });
}

async fn streaming_preview_loop(app: tauri::AppHandle) {
    use std::time::Duration;

    // Max audio to transcribe in preview mode (10s at 16kHz) — keeps preview fast
    const MAX_PREVIEW_SAMPLES: usize = 16000 * 10;

    // Wait 1.5s before first preview (need enough audio)
    for _ in 0..15 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let state = app.state::<Mutex<AppState>>();
        let still_recording = state.lock().unwrap().status == AppStatus::Recording;
        if !still_recording {
            return;
        }
    }

    // Language detected on the first preview cycle is reused for the rest of
    // this recording — see WhisperEngine::transcribe_cached for the trade-off.
    let mut lang_cache: Option<&'static str> = None;

    loop {
        let buffer = app.state::<AudioBuffer>();
        let samples = buffer.snapshot_tail(MAX_PREVIEW_SAMPLES);

        if samples.len() >= 16000 {
            // Check if still recording right before locking the engine
            {
                let state = app.state::<Mutex<AppState>>();
                if state.lock().unwrap().status != AppStatus::Recording {
                    return;
                }
            }

            // Try non-blocking lock — skip if final transcription holds it
            let engine = app.state::<Mutex<WhisperEngine>>();
            let lock_result = engine.try_lock();
            if let Ok(eng) = lock_result {
                let duration = samples.len() as f32 / 16000.0;
                log::info!("Streaming preview: transcribing {:.1}s", duration);
                let language = {
                    let settings = app.state::<Mutex<Settings>>();
                    let guard = settings.lock().unwrap();
                    guard.language
                };
                match eng.transcribe_cached(&samples, language, &mut lang_cache) {
                    Ok(text) if !text.is_empty() => {
                        log::info!("Streaming preview ready ({} chars)", text.chars().count());
                        let _ = app.emit("streaming-preview", &text);
                    }
                    _ => {}
                }
            } else {
                log::info!("Streaming preview: engine locked, skipping");
            }
        }

        // Wait 2s before next preview, checking every 100ms if still recording
        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let state = app.state::<Mutex<AppState>>();
            let still_recording = state.lock().unwrap().status == AppStatus::Recording;
            if !still_recording {
                return;
            }
        }
    }
}

/// Remove filler interjections from transcription (Russian + English).
/// Only pure interjections that carry no meaning in any context — semantic
/// words ("ну", "значит", "like", "so", "well") stay, because stripping them
/// blindly corrupts real sentences ("I like this" → "I this"). Contextual
/// filler cleanup is the AI formatting step's job.
fn remove_fillers(text: &str) -> String {
    const FILLERS: &[&str] = &[
        // Russian
        "э", "ээ", "эээ", "эм", "ээм", "эмм", "ам", "хм", "мм", "ммм", // English
        "um", "umm", "uh", "uhh", "hmm", "er", "erm", "ah", "mhm",
    ];

    let cleaned: Vec<&str> = text
        .split_whitespace()
        .filter(|w| {
            let lower = w.to_lowercase();
            let stripped =
                lower.trim_matches(|c: char| matches!(c, ',' | '.' | '!' | '?' | '…' | '-' | '—'));
            !FILLERS.contains(&stripped)
        })
        .collect();

    cleaned.join(" ").trim().to_string()
}

/// Tell the user why nothing was pasted instead of failing silently.
/// Emits an event for the UI and shows a system toast (the main window is
/// usually hidden in the tray during dictation).
fn notify_no_result(app: &tauri::AppHandle, reason: &str, message: &str) {
    log::warn!("No transcription result: {}", reason);
    let _ = app.emit("transcription-empty", reason);

    notify_user(app, message);
}

fn notify_user(app: &tauri::AppHandle, message: &str) {
    use tauri_plugin_notification::NotificationExt;
    let _ = app
        .notification()
        .builder()
        .title("Wispr Local")
        .body(message)
        .show();
}

async fn stop_and_transcribe_flow(app: &tauri::AppHandle) {
    log::info!("stop_and_transcribe_flow called");
    let state = app.state::<Mutex<AppState>>();
    let capture = app.state::<Mutex<AudioCapture>>();
    let buffer = app.state::<AudioBuffer>();
    let engine = app.state::<Mutex<WhisperEngine>>();

    // Claim the stop atomically. Hotkey release, tray, pin button, and the
    // recording-limit event can arrive together; only one may finalize audio.
    {
        let mut s = state.lock().unwrap();
        if s.status != AppStatus::Recording {
            return;
        }
        s.recording_locked = false;
        s.status = AppStatus::Transcribing;
    }
    let _ = app.emit("lock-changed", false);
    let _ = app.emit("status-changed", "Transcribing");

    // Stop capture
    {
        capture.lock().unwrap().stop();
    }
    app.state::<SoundPlayer>().play_stop();

    // End the recording indicator as soon as the mic is released; transcription
    // runs afterwards and doesn't need the red pulse.
    app.state::<TrayAnimator>().stop();
    hide_overlay(app);

    let samples = buffer.take_samples();
    // Under ~0.5s is an accidental hotkey tap — not enough audio for even one
    // word, and short buffers are prime hallucination bait for Whisper.
    const MIN_SAMPLES: usize = 8000; // 0.5s at 16kHz
    if samples.len() < MIN_SAMPLES {
        state.lock().unwrap().status = AppStatus::Idle;
        let _ = app.emit("status-changed", "Idle");
        notify_no_result(app, "too-short", "Recording too short — nothing captured");
        return;
    }

    log::info!(
        "Transcribing {:.1}s of audio",
        samples.len() as f32 / 16000.0
    );

    let language = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = settings.lock().unwrap();
        guard.language
    };
    let text = {
        let eng = engine.lock().unwrap();
        match eng.transcribe(&samples, language) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Transcription failed: {}", e);
                state.lock().unwrap().status = AppStatus::Idle;
                let _ = app.emit("status-changed", "Idle");
                notify_no_result(app, "error", "Transcription failed — check logs");
                return;
            }
        }
    };

    if text.is_empty() {
        state.lock().unwrap().status = AppStatus::Idle;
        let _ = app.emit("status-changed", "Idle");
        notify_no_result(app, "no-speech", "No speech detected — try again");
        return;
    }

    let text = remove_fillers(&text);
    log::info!("Transcription cleaned ({} chars)", text.chars().count());

    if text.is_empty() {
        state.lock().unwrap().status = AppStatus::Idle;
        let _ = app.emit("status-changed", "Idle");
        notify_no_result(app, "no-speech", "No speech detected — try again");
        return;
    }

    // AI formatting step
    let ai_settings = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = settings.lock().unwrap();
        guard.ai.clone()
    };

    let text = if ai_settings.provider != formatting::AiProvider::None {
        {
            state.lock().unwrap().status = AppStatus::Formatting;
        }
        let _ = app.emit("status-changed", "Formatting");
        match formatting::format_text(&text, &ai_settings).await {
            Ok(formatted) => formatted,
            Err(e) => {
                log::error!("AI formatting failed; using raw text: {e}");
                let message = "AI formatting failed; pasted the raw transcript instead.";
                let _ = app.emit("operation-notice", message);
                notify_user(app, message);
                text
            }
        }
    } else {
        text
    };

    {
        state.lock().unwrap().status = AppStatus::Injecting;
    }
    let _ = app.emit("status-changed", "Injecting");

    match system::text_injection::inject_text(&text) {
        Ok(_) => log::info!("Text injected successfully"),
        Err(e) => {
            log::error!("Text injection failed: {}", e);
            let copied = commands::copy_text(text.clone()).is_ok();
            let message = if copied {
                "Automatic paste failed; the transcript was copied to your clipboard."
            } else {
                "Automatic paste failed; open Wispr Local to copy the transcript from history."
            };
            let _ = app.emit("operation-notice", message);
            notify_user(app, message);
        }
    }

    let history = {
        let mut s = state.lock().unwrap();
        s.last_transcription = text.clone();
        s.push_history(&text);
        s.status = AppStatus::Idle;
        s.history.clone()
    };
    if let Err(e) = state::save_history(&app.state::<AppConfig>().data_dir, &history) {
        log::warn!("Failed to save transcription history: {e}");
        let _ = app.emit("operation-notice", "History could not be saved to disk");
    }
    let _ = app.emit("status-changed", "Idle");
    let _ = app.emit("history-changed", &history);
    let _ = app.emit("transcription-complete", text);
}

#[cfg(test)]
mod filler_tests {
    use super::remove_fillers;

    #[test]
    fn removes_interjections() {
        assert_eq!(
            remove_fillers("Эм, привет, э, как дела?"),
            "привет, как дела?"
        );
        assert_eq!(
            remove_fillers("Um, hello there, uh, okay"),
            "hello there, okay"
        );
    }

    #[test]
    fn keeps_semantic_words() {
        assert_eq!(
            remove_fillers("I like this approach"),
            "I like this approach"
        );
        assert_eq!(
            remove_fillers("Ну, это значит, что всё хорошо"),
            "Ну, это значит, что всё хорошо"
        );
        assert_eq!(
            remove_fillers("So, well, basically it works"),
            "So, well, basically it works"
        );
    }

    #[test]
    fn handles_empty_and_filler_only() {
        assert_eq!(remove_fillers("эм... ээ"), "");
        assert_eq!(remove_fillers(""), "");
    }
}
