pub mod audio;
pub mod autostart;
pub mod commands;
pub mod config;
pub mod formatting;
pub mod settings;
pub mod state;
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
            config.ensure_dirs().expect("Failed to create app directories");

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
            let mut initial_state = AppState::default();
            initial_state.history = state::load_history(&config.data_dir);

            let mut candidates = vec![user_settings.model_file.clone()];
            for fallback in [settings::default_model_file(), "ggml-medium.bin".to_string()] {
                if !candidates.contains(&fallback) {
                    candidates.push(fallback);
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
            position_overlay_window(app.handle());
            if let Some(overlay) = app.get_webview_window("overlay") {
                let _ = overlay.hide();
                // WS_EX_NOACTIVATE: the pin button must be clickable without
                // stealing focus from the app the user is dictating into —
                // otherwise the eventual paste would land in the wrong window.
                let _ = overlay.set_focusable(false);
                // Clip the window itself to a pill shape at the OS level.
                // WebView2 transparency proved unreliable here (square backdrop
                // behind the rounded CSS pill), so instead the window is opaque
                // and simply has no pixels outside the rounded region.
                apply_pill_region(&overlay);
            }

            // Register global hotkey from settings
            {
                use tauri_plugin_global_shortcut::GlobalShortcutExt;
                let shortcut = commands::parse_hotkey(&user_settings.hotkey)
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
                app.global_shortcut().register(shortcut)?;
                log::info!("Global hotkey registered: {} (hold to dictate)", user_settings.hotkey);
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
                let app = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    start_recording_flow(&app);
                });
            });

            let app_handle = app.handle().clone();
            app.listen("tray-start-recording", move |_event| {
                let app = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    start_recording_flow(&app);
                });
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

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_recording,
            commands::stop_recording_and_transcribe,
            commands::get_status,
            commands::is_model_loaded,
            commands::get_last_transcription,
            commands::toggle_recording_lock,
            commands::get_history,
            commands::copy_text,
            commands::get_models_dir,
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Clip the overlay window to a rounded pill (corner radius = height/2) via
/// SetWindowRgn, so nothing outside the pill exists at the compositor level.
/// The region uses physical pixels, so this works at any DPI scale.
#[cfg(windows)]
fn apply_pill_region(overlay: &tauri::WebviewWindow) {
    let (Ok(hwnd), Ok(size)) = (overlay.hwnd(), overlay.outer_size()) else {
        log::warn!("Could not get overlay hwnd/size for pill region");
        return;
    };
    unsafe {
        use windows_sys::Win32::Graphics::Gdi::{CreateRoundRectRgn, SetWindowRgn};
        let w = size.width as i32;
        let h = size.height as i32;
        // Ellipse w/h = window height -> fully rounded ends, matching the
        // CSS border-radius: 999px pill. The region takes ownership of rgn.
        let rgn = CreateRoundRectRgn(0, 0, w + 1, h + 1, h, h);
        SetWindowRgn(hwnd.0 as _, rgn, 1);
    }
}

#[cfg(not(windows))]
fn apply_pill_region(_overlay: &tauri::WebviewWindow) {}

fn position_overlay_window(app: &tauri::AppHandle) {
    let Some(overlay) = app.get_webview_window("overlay") else {
        return;
    };
    let Ok(Some(monitor)) = overlay.primary_monitor() else {
        return;
    };

    let monitor_size = monitor.size();
    let monitor_pos = monitor.position();

    let win_size = overlay.outer_size().unwrap_or(tauri::PhysicalSize {
        width: 312,
        height: 52,
    });

    // Center horizontally, park above the taskbar. 64px leaves room for the
    // default Windows taskbar (48px) plus a small gap so the overlay doesn't
    // feel glued to it.
    const BOTTOM_MARGIN: i32 = 64;
    let x = monitor_pos.x + (monitor_size.width as i32 - win_size.width as i32) / 2;
    let y = monitor_pos.y + monitor_size.height as i32 - win_size.height as i32 - BOTTOM_MARGIN;

    let _ = overlay.set_position(tauri::PhysicalPosition { x, y });
}

fn show_overlay_if_enabled(app: &tauri::AppHandle) {
    let show = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = settings.lock().unwrap();
        guard.show_overlay
    };
    if !show {
        return;
    }
    if let Some(overlay) = app.get_webview_window("overlay") {
        position_overlay_window(app);
        let _ = overlay.show();
    }
}

fn hide_overlay(app: &tauri::AppHandle) {
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.hide();
    }
}

fn start_recording_flow(app: &tauri::AppHandle) {
    log::info!("start_recording_flow called");
    let state = app.state::<Mutex<AppState>>();
    let capture = app.state::<Mutex<AudioCapture>>();
    let buffer = app.state::<AudioBuffer>();

    {
        let mut s = state.lock().unwrap();
        if s.status == AppStatus::Recording {
            return;
        }
        buffer.clear();
        s.status = AppStatus::Recording;
        s.recording_locked = false;
    }

    let _ = app.emit("status-changed", "Recording");
    let _ = app.emit("lock-changed", false);
    app.state::<SoundPlayer>().play_start();

    // Kick off tray animation and reveal overlay (if user hasn't disabled it).
    app.state::<TrayAnimator>().start();
    show_overlay_if_enabled(app);

    let mut cap = capture.lock().unwrap();
    match cap.start(Some(app.clone())) {
        Ok(rate) => log::info!("Recording started at {} Hz", rate),
        Err(e) => {
            log::error!("Failed to start recording: {}", e);
            state.lock().unwrap().status = AppStatus::Error(e);
            let _ = app.emit("status-changed", "Error");
            app.state::<TrayAnimator>().stop();
            hide_overlay(app);
            return;
        }
    }

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

    loop {
        let buffer = app.state::<AudioBuffer>();
        let full_samples = buffer.snapshot();

        if full_samples.len() >= 16000 {
            // Only transcribe the last 10s for speed; show full context on final
            let samples = if full_samples.len() > MAX_PREVIEW_SAMPLES {
                &full_samples[full_samples.len() - MAX_PREVIEW_SAMPLES..]
            } else {
                &full_samples
            };

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
                match eng.transcribe(samples) {
                    Ok(text) if !text.is_empty() => {
                        log::info!("Preview: {}", text);
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
        "э", "ээ", "эээ", "эм", "ээм", "эмм", "ам", "хм", "мм", "ммм",
        // English
        "um", "umm", "uh", "uhh", "hmm", "er", "erm", "ah", "mhm",
    ];

    let cleaned: Vec<&str> = text
        .split_whitespace()
        .filter(|w| {
            let lower = w.to_lowercase();
            let stripped = lower
                .trim_matches(|c: char| matches!(c, ',' | '.' | '!' | '?' | '…' | '-' | '—'));
            !FILLERS.contains(&stripped)
        })
        .collect();

    cleaned.join(" ").trim().to_string()
}

#[cfg(test)]
mod filler_tests {
    use super::remove_fillers;

    #[test]
    fn removes_interjections() {
        assert_eq!(remove_fillers("Эм, привет, э, как дела?"), "привет, как дела?");
        assert_eq!(remove_fillers("Um, hello there, uh, okay"), "hello there, okay");
    }

    #[test]
    fn keeps_semantic_words() {
        assert_eq!(remove_fillers("I like this approach"), "I like this approach");
        assert_eq!(remove_fillers("Ну, это значит, что всё хорошо"), "Ну, это значит, что всё хорошо");
        assert_eq!(remove_fillers("So, well, basically it works"), "So, well, basically it works");
    }

    #[test]
    fn handles_empty_and_filler_only() {
        assert_eq!(remove_fillers("эм... ээ"), "");
        assert_eq!(remove_fillers(""), "");
    }
}

/// Tell the user why nothing was pasted instead of failing silently.
/// Emits an event for the UI and shows a system toast (the main window is
/// usually hidden in the tray during dictation).
fn notify_no_result(app: &tauri::AppHandle, reason: &str, message: &str) {
    log::warn!("No transcription result: {}", reason);
    let _ = app.emit("transcription-empty", reason);

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

    // Only stop if we're actually recording
    {
        let mut s = state.lock().unwrap();
        if s.status != AppStatus::Recording {
            return;
        }
        s.recording_locked = false;
    }
    let _ = app.emit("lock-changed", false);

    // Stop capture
    {
        capture.lock().unwrap().stop();
    }
    app.state::<SoundPlayer>().play_stop();

    // End the recording indicator as soon as the mic is released; transcription
    // runs afterwards and doesn't need the red pulse.
    app.state::<TrayAnimator>().stop();
    hide_overlay(app);

    {
        state.lock().unwrap().status = AppStatus::Transcribing;
    }
    let _ = app.emit("status-changed", "Transcribing");

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

    let text = {
        let eng = engine.lock().unwrap();
        match eng.transcribe(&samples) {
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
    log::info!("Transcription (cleaned): {}", text);

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
        formatting::format_text(&text, &ai_settings).await
    } else {
        text
    };

    {
        state.lock().unwrap().status = AppStatus::Injecting;
    }
    let _ = app.emit("status-changed", "Injecting");

    match system::text_injection::inject_text(&text) {
        Ok(_) => log::info!("Text injected successfully"),
        Err(e) => log::error!("Text injection failed: {}", e),
    }

    let history = {
        let mut s = state.lock().unwrap();
        s.last_transcription = text.clone();
        s.push_history(&text);
        s.status = AppStatus::Idle;
        s.history.clone()
    };
    state::save_history(&app.state::<AppConfig>().data_dir, &history);
    let _ = app.emit("status-changed", "Idle");
    let _ = app.emit("history-changed", &history);
    let _ = app.emit("transcription-complete", text);
}
