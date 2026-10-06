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
//!
//! Everything that does not need an `AppHandle` is built *before* the Tauri
//! builder and registered with `Builder::manage`, so it exists before the
//! windows are created. Registering state inside `setup()` let the main
//! window's first IPC calls race an as-yet unmanaged state ("state not
//! managed" → "Some settings could not be loaded" banner on ~1 in 4 starts).

#[cfg(not(windows))]
compile_error!(
    "wispr-local is Windows-only: it relies on Win32 window regions, WASAPI, DPAPI and toast APIs"
);

pub mod audio;
pub mod autostart;
pub mod commands;
pub mod config;
pub mod events;
pub mod formatting;
pub mod hotkey;
pub mod instance;
pub mod overlay;
pub mod pipeline;
pub mod secrets;
pub mod settings;
pub mod state;
pub mod stats;
pub mod supervisor;
pub mod system;
pub mod text;
pub mod transcription;
pub mod watchdog;

use std::sync::Mutex;
use tauri::{Emitter, Listener, Manager};

use audio::buffer::AudioBuffer;
use audio::capture::AudioCapture;
use config::AppConfig;
use pipeline::{
    cancel_recording, notify_user, spawn_model_loader, start_recording_flow,
    stop_and_transcribe_flow,
};
use settings::Settings;
use state::{AppState, AppStatus, ModelState, StartupDiagnostics};
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
    log::info!("wispr-local {} starting", supervisor::build_identity());
    // Route whisper.cpp/GGML native log output through the `log` crate →
    // env_logger → stderr → supervisor pipe → wispr.log. Without this the
    // CUDA error text printed right before a GGML abort was lost with the
    // invisible stderr of a windows-subsystem process.
    whisper_rs::install_logging_hooks();

    // ---- State that needs no AppHandle: built before any window exists ----
    let config = AppConfig::new();
    config
        .ensure_dirs()
        .expect("Failed to create app directories");
    // Toasts and the taskbar need our own identity; without it every toast
    // is attributed to "Windows PowerShell".
    if let Err(e) = system::notify::register_app_identity(&config.data_dir) {
        log::warn!("Notification identity not registered: {e}");
    }
    // Leftovers of an interrupted atomic write from a previous run.
    config::remove_stale_temp_files(&config.data_dir);

    // A broken settings file is quarantined and reported, never overwritten
    // with defaults.
    let settings_load = Settings::load_with_report(&config.data_dir);
    let user_settings = settings_load.settings.clone();
    log::info!("Loaded hotkey setting: {}", user_settings.hotkey);
    let (history, history_error) = state::load_history_with_report(&config.data_dir);
    let diagnostics = StartupDiagnostics {
        settings_error: settings_load.error.clone(),
        settings_read_only: settings_load.read_only,
        unknown_settings_keys: settings_load.unknown_keys.clone(),
        settings_adjustments: settings_load.adjustments.clone(),
        history_error,
        api_key_error: settings_load.api_key_error.clone(),
    };

    let buffer = AudioBuffer::new();
    let capture = AudioCapture::new(buffer.clone());
    // Empty engine; the model loads on a background thread from setup().
    let engine = WhisperEngine::new();
    // The sound thread opens the output device per chime.
    let sound_player = SoundPlayer::new(user_settings.sound_config());

    // Sync autostart state with saved settings
    if user_settings.run_on_startup {
        let _ = autostart::set_autostart_registry(true);
        log::info!("Autostart enabled");
    }

    let initial_state = AppState {
        history,
        diagnostics: diagnostics.clone(),
        model: ModelState::Loading,
        ..AppState::default()
    };
    let hotkey_string = user_settings.hotkey.clone();
    let cancel_hotkey_string = user_settings.cancel_hotkey.clone();
    let requested_model = user_settings.model_file.clone();

    tauri::Builder::default()
        .manage(Mutex::new(initial_state))
        .manage(Mutex::new(capture))
        .manage(buffer)
        .manage(Mutex::new(engine))
        .manage(config)
        .manage(sound_player)
        .manage(Mutex::new(user_settings))
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    use tauri_plugin_global_shortcut::ShortcutState;
                    log::debug!("Hotkey event: {:?} state={:?}", shortcut, event.state);

                    let (mode, cancel_shortcut) = {
                        let settings = app.state::<Mutex<Settings>>();
                        let s = state::lock_or_recover(&settings);
                        (s.hotkey_mode, commands::parse_hotkey(&s.cancel_hotkey).ok())
                    };
                    if cancel_shortcut.as_ref() == Some(shortcut) {
                        if event.state == ShortcutState::Pressed {
                            let _ = app.emit(events::REQUEST_CANCEL_RECORDING, ());
                        }
                        return;
                    }

                    let hotkey_event = match event.state {
                        ShortcutState::Pressed => hotkey::HotkeyEvent::Pressed,
                        ShortcutState::Released => hotkey::HotkeyEvent::Released,
                    };
                    let action = {
                        let state = app.state::<Mutex<AppState>>();
                        let mut s = state::lock_or_recover(&state);
                        if s.hotkey_paused || s.capturing_hotkey {
                            return;
                        }
                        let ctx = hotkey::HotkeyContext {
                            mode,
                            recording: s.status == AppStatus::Recording,
                            locked: s.recording_locked,
                            held_ms: s
                                .recording_started_at
                                .map(|t| t.elapsed().as_millis())
                                .unwrap_or(0),
                            key_down: s.key_down,
                        };
                        let (action, key_down) = hotkey::decide(hotkey_event, ctx);
                        s.key_down = key_down;
                        if action == hotkey::HotkeyAction::Lock {
                            s.recording_locked = true;
                        }
                        action
                    };

                    match action {
                        hotkey::HotkeyAction::Start => {
                            let _ = app.emit(events::REQUEST_START_RECORDING, ());
                        }
                        hotkey::HotkeyAction::Stop => {
                            let _ = app.emit(events::REQUEST_STOP_RECORDING, ());
                        }
                        hotkey::HotkeyAction::Lock => {
                            log::info!("Tap: recording continues hands-free");
                            let _ = app.emit(events::LOCK_CHANGED, true);
                        }
                        hotkey::HotkeyAction::Ignore => {}
                    }
                })
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .setup(move |app| {
            // Load the model off-thread. The state is already managed, so the
            // window can query it at any time; the result arrives through the
            // model-state-changed event.
            spawn_model_loader(app.handle().clone(), requested_model);

            // The window is hidden at this point, so a quarantined settings
            // file also gets a toast; the banner appears once the window opens.
            if let Some(problem) = diagnostics.settings_error.as_deref() {
                notify_user(app.handle(), &format!("Settings problem: {problem}"));
            }

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

            // Register the global hotkey. A conflict with another app used to
            // panic setup → exit 101 → three supervisor restarts → "keeps
            // crashing" dialog; now it is a soft error with retries, and the
            // tray, settings and hotkey editor keep working.
            register_hotkey_with_retries(app.handle().clone(), hotkey_string);
            commands::register_cancel_hotkey(app.handle(), &cancel_hotkey_string);

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

            // Start requests come from the hotkey (hold) and the tray / window
            // (hands-free).
            let app_handle = app.handle().clone();
            app.listen(events::REQUEST_START_RECORDING, move |_event| {
                start_recording_flow(&app_handle, false);
            });
            let app_handle = app.handle().clone();
            app.listen(events::REQUEST_START_HANDS_FREE, move |_event| {
                start_recording_flow(&app_handle, true);
            });

            // Cancel: cancel hotkey, overlay X button, tray.
            let app_handle = app.handle().clone();
            app.listen(events::REQUEST_CANCEL_RECORDING, move |_event| {
                cancel_recording(&app_handle);
            });

            // Stop requests: hotkey release, tray, overlay pin button.
            let app_handle = app.handle().clone();
            app.listen(events::REQUEST_RETRANSCRIBE, move |_event| {
                let app = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    crate::pipeline::retranscribe_latest(&app).await;
                });
            });

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
            commands::get_model_state,
            commands::get_model_files,
            commands::set_model_file,
            commands::reload_model,
            commands::log_frontend_error,
            commands::get_app_info,
            commands::restart_on_gpu,
            commands::get_last_transcription,
            commands::toggle_recording_lock,
            commands::get_history,
            commands::paste_history_item,
            commands::clear_history,
            commands::copy_text,
            commands::get_models_dir,
            commands::open_path,
            commands::get_default_prompt,
            commands::start_hands_free,
            commands::stop_recording,
            commands::get_hotkey,
            commands::set_hotkey,
            commands::get_hotkey_mode,
            commands::set_hotkey_mode,
            commands::get_cancel_hotkey,
            commands::set_cancel_hotkey,
            commands::begin_hotkey_capture,
            commands::end_hotkey_capture,
            commands::set_hotkey_paused,
            commands::cancel_recording,
            commands::get_sound_settings,
            commands::set_sound_settings,
            commands::test_sound,
            commands::get_ai_settings,
            commands::set_ai_settings,
            commands::get_startup_diagnostics,
            commands::get_text_settings,
            commands::set_text_settings,
            commands::get_autostart,
            commands::set_autostart,
            commands::get_show_overlay,
            commands::set_show_overlay,
            commands::get_language,
            commands::set_language,
            commands::get_input_devices,
            commands::get_input_device,
            commands::set_input_device,
            commands::probe_input_device,
            commands::test_ai_connection,
            commands::get_stats,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| match event {
            tauri::RunEvent::ExitRequested { code, .. } => {
                log_exit_line(app, &format!("exit requested (code {code:?})"));
            }
            tauri::RunEvent::Exit => {
                log_exit_line(app, "exiting (quit or session end)");
            }
            _ => {}
        });
}

/// Write an exit marker straight into wispr.log: at logoff the pipe to the
/// supervisor may already be gone, so `log::info!` alone leaves no trace.
fn log_exit_line(app: &tauri::AppHandle, message: &str) {
    log::info!("{message}");
    use std::io::Write;
    let path = app.state::<AppConfig>().data_dir.join("wispr.log");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let ts = humantime::format_rfc3339_seconds(std::time::SystemTime::now());
        let _ = writeln!(f, "[{ts} child] {message}");
    }
}

/// Register the hotkey now; on failure retry a few times in the background
/// (another app may still be starting at logon), then surface the problem
/// instead of crashing.
fn register_hotkey_with_retries(app: tauri::AppHandle, hotkey: String) {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;
    let shortcut = match commands::parse_hotkey(&hotkey) {
        Ok(s) => s,
        Err(e) => {
            log::error!("Saved hotkey '{hotkey}' is invalid ({e}); use Settings to set a new one");
            pipeline::set_status(
                &app,
                AppStatus::error("hotkey", format!("Saved hotkey '{hotkey}' is invalid: {e}")),
            );
            return;
        }
    };
    match app.global_shortcut().register(shortcut) {
        Ok(()) => {
            log::info!("Global hotkey registered: {hotkey} (hold to dictate)");
            return;
        }
        Err(e) => log::warn!("Hotkey {hotkey} could not be registered ({e}); retrying"),
    }
    std::thread::spawn(move || {
        for attempt in 1..=3 {
            std::thread::sleep(std::time::Duration::from_secs(2));
            match app.global_shortcut().register(shortcut) {
                Ok(()) => {
                    log::info!("Global hotkey registered on retry {attempt}: {hotkey}");
                    return;
                }
                Err(e) => log::warn!("Hotkey retry {attempt} failed: {e}"),
            }
        }
        let message =
            format!("Hotkey {hotkey} is used by another app; choose a different one in Settings");
        log::error!("{message}");
        pipeline::set_status(&app, AppStatus::error("hotkey", message.clone()));
        let _ = app.emit(events::OPERATION_NOTICE, &message);
        pipeline::notify_user_long(&app, &message);
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.show();
            let _ = window.set_focus();
        }
    });
}
