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
                    match event.state {
                        ShortcutState::Pressed => {
                            log::info!("Hotkey PRESSED - starting recording");
                            let _ = app.emit("hotkey-start-recording", ());
                        }
                        ShortcutState::Released => {
                            log::info!("Hotkey RELEASED - stopping recording");
                            let _ = app.emit("hotkey-stop-recording", ());
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

            // Initialize Whisper engine and try loading model
            let mut engine = WhisperEngine::new();
            let model_filename = "ggml-medium.bin";
            let model_path = config.model_path(model_filename);

            let mut initial_state = AppState::default();

            if model_path.exists() {
                match engine.load_model(&model_path) {
                    Ok(_) => {
                        log::info!("Model loaded from {:?}", model_path);
                        initial_state.model_loaded = true;
                    }
                    Err(e) => log::error!("Failed to load model: {}", e),
                }
            } else {
                log::warn!(
                    "Model not found at {:?}. Download it to enable transcription.",
                    model_path
                );
            }

            // Load settings
            let user_settings = Settings::load(&config.data_dir);
            log::info!("Loaded hotkey setting: {}", user_settings.hotkey);

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
        width: 280,
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
    }

    let _ = app.emit("status-changed", "Recording");
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

/// Remove common filler words from transcription (Russian + English)
fn remove_fillers(text: &str) -> String {
    // Regex-free approach: split by words, filter fillers, rejoin
    let fillers_ru = [
        "ну", "эм", "э", "ээ", "эээ", "ам", "хм", "ммм", "мм",
        "типа", "короче", "как бы", "это самое", "в общем", "так сказать",
        "слушай", "значит", "ну вот",
    ];
    let fillers_en = [
        "um", "uh", "uh", "uhh", "umm", "hmm", "er", "ah", "like",
        "you know", "i mean", "so", "well", "basically",
    ];

    let mut result = text.to_string();

    // Remove multi-word fillers first (longer patterns first)
    for filler in fillers_ru.iter().chain(fillers_en.iter()) {
        if filler.contains(' ') {
            // Case-insensitive removal of multi-word fillers
            let lower = result.to_lowercase();
            let filler_lower = filler.to_lowercase();
            while let Some(pos) = lower.find(&filler_lower) {
                // Remove filler and any trailing comma/space
                let end = pos + filler.len();
                let end = if result[end..].starts_with(", ") {
                    end + 2
                } else if result[end..].starts_with(' ') {
                    end + 1
                } else {
                    end
                };
                result = format!("{}{}", &result[..pos], &result[end..]);
                break; // re-check from start since indices changed
            }
        }
    }

    // Remove single-word fillers
    let words: Vec<&str> = result.split_whitespace().collect();
    let cleaned: Vec<&str> = words
        .into_iter()
        .filter(|w| {
            let lower = w.to_lowercase();
            let stripped = lower.trim_matches(|c: char| c == ',' || c == '.' || c == '!' || c == '?');
            !fillers_ru.contains(&stripped)
                && !fillers_en.contains(&stripped)
        })
        .collect();

    let result = cleaned.join(" ");
    // Clean up double spaces and trim
    result.trim().to_string()
}

async fn stop_and_transcribe_flow(app: &tauri::AppHandle) {
    log::info!("stop_and_transcribe_flow called");
    let state = app.state::<Mutex<AppState>>();
    let capture = app.state::<Mutex<AudioCapture>>();
    let buffer = app.state::<AudioBuffer>();
    let engine = app.state::<Mutex<WhisperEngine>>();

    // Only stop if we're actually recording
    {
        let s = state.lock().unwrap();
        if s.status != AppStatus::Recording {
            return;
        }
    }

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
    if samples.is_empty() {
        state.lock().unwrap().status = AppStatus::Idle;
        let _ = app.emit("status-changed", "Idle");
        log::warn!("No audio recorded");
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
                return;
            }
        }
    };

    if text.is_empty() {
        log::warn!("No speech detected");
        state.lock().unwrap().status = AppStatus::Idle;
        let _ = app.emit("status-changed", "Idle");
        return;
    }

    let text = remove_fillers(&text);
    log::info!("Transcription (cleaned): {}", text);

    if text.is_empty() {
        log::warn!("No speech after filler removal");
        state.lock().unwrap().status = AppStatus::Idle;
        let _ = app.emit("status-changed", "Idle");
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

    {
        let mut s = state.lock().unwrap();
        s.last_transcription = text.clone();
        s.status = AppStatus::Idle;
    }
    let _ = app.emit("status-changed", "Idle");
    let _ = app.emit("transcription-complete", text);
}
