//! Wispr Local — local hold-to-dictate voice-to-text for Windows.
//!
//! Process model: the exe the user launches is a crash supervisor
//! ([`supervisor`]); the real app runs as its child. Inside the child, this
//! module wires everything together: managed state, the global hotkey, the
//! tray, the overlay window and the event listeners that drive the
//! [`pipeline`]. The modules are:
//!
//! - [`audio`]      microphone capture, device selection, sample buffer
//! - [`transcription`] whisper.cpp engine and language handling
//! - [`pipeline`]   start/stop flows, streaming preview, model loader
//! - [`overlay`]    the floating recording pill (Win32 geometry)
//! - [`system`]     chimes, text injection, tray
//! - [`settings`] / [`secrets`] / [`config`] persisted configuration
//! - [`commands`]   the Tauri IPC surface used by the webviews
//! - [`events`]     event names shared with the frontend

pub mod audio;
pub mod autostart;
pub mod commands;
pub mod config;
pub mod events;
pub mod formatting;
pub mod instance;
pub mod overlay;
pub mod pipeline;
pub mod secrets;
pub mod settings;
pub mod state;
pub mod supervisor;
pub mod system;
pub mod text;
pub mod transcription;

use std::sync::Mutex;
use tauri::{Emitter, Listener, Manager};

use audio::buffer::AudioBuffer;
use audio::capture::AudioCapture;
use config::AppConfig;
use pipeline::{notify_user, spawn_model_loader, start_recording_flow, stop_and_transcribe_flow};
use settings::Settings;
use state::{AppState, AppStatus};
use system::sounds::SoundPlayer;
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
    if !instance::claim_single_instance() {
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
                                let _ = app.emit(events::REQUEST_STOP_RECORDING, ());
                            } else if !recording {
                                log::info!("Hotkey PRESSED - starting recording");
                                let _ = app.emit(events::REQUEST_START_RECORDING, ());
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
                                let _ = app.emit(events::REQUEST_STOP_RECORDING, ());
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

            // Initialize an empty Whisper engine; the model is loaded on a
            // background thread after state is registered (see below). Loading
            // synchronously here — especially the slower CPU fallback after a
            // GPU crash — let the frontend's initial queries race an as-yet
            // unmanaged state and stick on a false "Model not loaded" banner.
            let engine = WhisperEngine::new();
            let initial_state = AppState {
                history: state::load_history(&config.data_dir),
                ..AppState::default()
            };

            // Try the configured model first, then fall back to older models so
            // an incomplete download doesn't leave the app without transcription.
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
            // Resolve to full paths now, before `config` is moved into state.
            let candidate_paths: Vec<std::path::PathBuf> = candidates
                .iter()
                .map(|name| config.model_path(name))
                .collect();
            let models_dir = config.models_dir.clone();

            // Sync autostart state with saved settings
            if user_settings.run_on_startup {
                let _ = autostart::set_autostart_registry(true);
                log::info!("Autostart enabled");
            }

            // Initialize the sound thread (opens the output device per chime).
            let sound_player = SoundPlayer::new(user_settings.sound_config());

            // Register state
            app.manage(Mutex::new(initial_state));
            app.manage(Mutex::new(capture));
            app.manage(buffer.clone());
            app.manage(Mutex::new(engine));
            app.manage(config);
            app.manage(sound_player);
            app.manage(Mutex::new(user_settings.clone()));

            // Load the model off-thread now that state is managed. Commands
            // (is_model_loaded, get_compute_backend) read the mirrored fields in
            // AppState, so the window never races an unmanaged state; the load
            // result is pushed to the frontend via `model-state-changed`.
            spawn_model_loader(app.handle().clone(), candidate_paths, models_dir);

            // Setup system tray (also manages TrayAnimator state).
            system::tray::setup_tray(app.handle())?;

            // Position the overlay window just above the Windows taskbar and
            // ensure it respects the current show_overlay setting.
            overlay::place_overlay_window(app.handle());
            if let Some(overlay_window) = app.get_webview_window("overlay") {
                let _ = overlay_window.hide();
                overlay::sync_overlay_webview_visibility(&overlay_window, false);
                // WS_EX_NOACTIVATE: the pin button must be clickable without
                // stealing focus from the app the user is dictating into —
                // otherwise the eventual paste would land in the wrong window.
                let _ = overlay_window.set_focusable(false);
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

            // Start requests come from the hotkey, the tray and (later) the UI.
            let app_handle = app.handle().clone();
            app.listen(events::REQUEST_START_RECORDING, move |_event| {
                start_recording_flow(&app_handle);
            });

            // Stop requests: hotkey release, tray, overlay pin button.
            let app_handle = app.handle().clone();
            app.listen(events::REQUEST_STOP_RECORDING, move |_event| {
                let app = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    stop_and_transcribe_flow(&app).await;
                });
            });

            // A pinned recording cannot grow memory forever. The audio callback
            // emits this once at 30 minutes, then the normal finalization path
            // transcribes everything captured so far.
            let app_handle = app.handle().clone();
            app.listen(events::RECORDING_LIMIT_REACHED, move |_event| {
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
            app.listen(events::AUDIO_STREAM_ERROR, move |_event| {
                let app = app_handle.clone();
                let message =
                    "The microphone disconnected or stopped responding; processing captured audio.";
                let _ = app.emit(events::OPERATION_NOTICE, message);
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
