use std::sync::Mutex;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

use crate::config::AppConfig;
use crate::settings::Settings;
use crate::state::{AppState, AppStatus, ModelState};
use crate::system::sounds::SoundPlayer;

/// Current pipeline status, in the same shape as the `status-changed` event.
#[tauri::command]
pub fn get_status(state: State<'_, Mutex<AppState>>) -> Result<AppStatus, String> {
    let app_state = state.lock().map_err(|e| e.to_string())?;
    Ok(app_state.status.clone())
}

/// Model state mirrored in AppState (never blocks on the engine mutex, which
/// is held for the whole of a load or a transcription).
#[tauri::command]
pub fn get_model_state(state: State<'_, Mutex<AppState>>) -> Result<ModelState, String> {
    Ok(crate::state::lock_or_recover(&state).model.clone())
}

#[derive(serde::Serialize)]
pub struct ModelFileInfo {
    pub name: String,
    pub size_bytes: u64,
    /// The file named in settings.json.
    pub configured: bool,
    /// The file currently loaded in the engine.
    pub loaded: bool,
}

/// Multilingual whisper.cpp models present in the models directory.
#[tauri::command(async)]
pub fn get_model_files(
    config: State<'_, AppConfig>,
    settings: State<'_, Mutex<Settings>>,
    state: State<'_, Mutex<AppState>>,
) -> Result<Vec<ModelFileInfo>, String> {
    let configured = settings
        .lock()
        .map_err(|e| e.to_string())?
        .model_file
        .clone();
    let loaded = match &crate::state::lock_or_recover(&state).model {
        ModelState::Ready { file, .. } => file.clone(),
        _ => String::new(),
    };
    Ok(config
        .available_model_files()
        .into_iter()
        .map(|name| {
            let size_bytes = std::fs::metadata(config.model_path(&name))
                .map(|m| m.len())
                .unwrap_or(0);
            ModelFileInfo {
                configured: name == configured,
                loaded: name == loaded,
                name,
                size_bytes,
            }
        })
        .collect())
}

/// Choose another model file and load it right away.
#[tauri::command]
pub fn set_model_file(
    app: AppHandle,
    name: String,
    config: State<'_, AppConfig>,
    settings: State<'_, Mutex<Settings>>,
    state: State<'_, Mutex<AppState>>,
) -> Result<(), String> {
    if !config.available_model_files().iter().any(|f| *f == name) {
        return Err(format!("{name} is not in the models folder"));
    }
    ensure_model_reload_allowed(&state)?;
    {
        let mut s = settings.lock().map_err(|e| e.to_string())?;
        let previous = s.model_file.clone();
        s.model_file = name.clone();
        if let Err(e) = s.save(&config.data_dir) {
            s.model_file = previous;
            return Err(e);
        }
    }
    log::info!("Model file changed to {name}; reloading");
    crate::pipeline::spawn_model_loader(app, name);
    Ok(())
}

/// Reload the configured model (after adding a file, or to retry a failure).
#[tauri::command]
pub fn reload_model(
    app: AppHandle,
    settings: State<'_, Mutex<Settings>>,
    state: State<'_, Mutex<AppState>>,
) -> Result<(), String> {
    ensure_model_reload_allowed(&state)?;
    let requested = settings
        .lock()
        .map_err(|e| e.to_string())?
        .model_file
        .clone();
    log::info!("Model reload requested");
    crate::pipeline::spawn_model_loader(app, requested);
    Ok(())
}

fn ensure_model_reload_allowed(state: &Mutex<AppState>) -> Result<(), String> {
    let s = crate::state::lock_or_recover(state);
    if !matches!(s.status, AppStatus::Idle | AppStatus::Error { .. }) {
        return Err("Finish the current dictation before reloading the model".to_string());
    }
    if s.model == ModelState::Loading {
        return Err("The model is already loading".to_string());
    }
    Ok(())
}

/// Version and build identity for the footer/About line.
#[tauri::command]
pub fn get_app_info() -> String {
    crate::supervisor::build_identity()
}

/// Ask the supervisor to respawn the app on CUDA (after a CPU fallback).
#[tauri::command]
pub fn restart_on_gpu(app: AppHandle) {
    log::info!("Restart on GPU requested");
    app.exit(crate::supervisor::RESTART_ON_GPU_CODE);
}

/// Let the webview put its own failures into wispr.log, next to the backend's.
#[tauri::command]
pub fn log_frontend_error(command: String, message: String) {
    log::warn!("Frontend: {command}: {message}");
}

#[tauri::command]
pub fn get_last_transcription(state: State<'_, Mutex<AppState>>) -> Result<String, String> {
    let app_state = state.lock().map_err(|e| e.to_string())?;
    Ok(app_state.last_transcription.clone())
}

/// Pin (lock) the active recording so the hotkey can be released, or stop a
/// pinned recording. Called from the overlay's pin/stop button.
/// Returns the new lock state.
#[tauri::command]
pub fn toggle_recording_lock(
    app: AppHandle,
    state: State<'_, Mutex<AppState>>,
) -> Result<bool, String> {
    use tauri::Emitter;

    let action = {
        let mut s = state.lock().map_err(|e| e.to_string())?;
        if s.status != AppStatus::Recording {
            return Ok(false);
        }
        if s.recording_locked {
            // Second click: stop the pinned recording.
            "stop"
        } else {
            s.recording_locked = true;
            "lock"
        }
    };

    match action {
        "lock" => {
            log::info!("Recording pinned via overlay button");
            let _ = app.emit(crate::events::LOCK_CHANGED, true);
            Ok(true)
        }
        _ => {
            log::info!("Pinned recording stopped via overlay button");
            let _ = app.emit(crate::events::REQUEST_STOP_RECORDING, ());
            Ok(false)
        }
    }
}

#[tauri::command]
pub fn get_history(
    state: State<'_, Mutex<AppState>>,
) -> Result<Vec<crate::state::HistoryEntry>, String> {
    Ok(crate::state::lock_or_recover(&state).history.clone())
}

/// Paste a history entry again: bring its original window back to the
/// front when it still exists, otherwise leave the text in the clipboard.
/// Returns "pasted" or "copied".
#[tauri::command]
pub fn paste_history_item(
    index: usize,
    state: State<'_, Mutex<AppState>>,
) -> Result<String, String> {
    let (entry, restore_clipboard) = {
        let s = crate::state::lock_or_recover(&state);
        let entry = s
            .history
            .get(index)
            .cloned()
            .ok_or_else(|| "That history entry no longer exists".to_string())?;
        (entry, true)
    };
    if entry.hwnd != 0 && activate_window(entry.hwnd) {
        std::thread::sleep(std::time::Duration::from_millis(150));
        let now = crate::system::focus::foreground_target();
        if now.as_ref().map(|t| t.hwnd) == Some(entry.hwnd) {
            crate::system::text_injection::inject_text(&entry.text, restore_clipboard)?;
            log::info!("History entry {index} pasted again into {}", entry.target);
            return Ok("pasted".to_string());
        }
    }
    crate::system::text_injection::copy_only(&entry.text)?;
    Ok("copied".to_string())
}

#[cfg(windows)]
fn activate_window(hwnd: isize) -> bool {
    use windows_sys::Win32::UI::WindowsAndMessaging::{IsWindow, SetForegroundWindow};
    unsafe {
        let handle = hwnd as *mut core::ffi::c_void;
        IsWindow(handle) != 0 && SetForegroundWindow(handle) != 0
    }
}

#[cfg(not(windows))]
fn activate_window(_hwnd: isize) -> bool {
    false
}

#[tauri::command]
pub fn clear_history(
    app: AppHandle,
    state: State<'_, Mutex<AppState>>,
    config: State<'_, AppConfig>,
) -> Result<(), String> {
    use tauri::Emitter;

    {
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        let previous = std::mem::take(&mut app_state.history);
        if let Err(e) = crate::state::save_history(&config.data_dir, &app_state.history) {
            app_state.history = previous;
            return Err(e);
        }
    }
    let history: Vec<crate::state::HistoryEntry> = Vec::new();
    app.emit(crate::events::HISTORY_CHANGED, &history)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Copy arbitrary text to the system clipboard (used by the history list).
#[tauri::command]
pub fn copy_text(text: String) -> Result<(), String> {
    crate::system::text_injection::copy_only(&text)
}

#[tauri::command]
pub fn get_models_dir(config: State<'_, crate::config::AppConfig>) -> Result<String, String> {
    Ok(config.models_dir.to_string_lossy().to_string())
}

/// Open one of the app's locations in Explorer: "models", "data" or "log"
/// (the log is selected inside its folder).
#[tauri::command]
pub fn open_path(kind: String, config: State<'_, AppConfig>) -> Result<(), String> {
    let mut command = std::process::Command::new("explorer.exe");
    match kind.as_str() {
        "models" => {
            command.arg(&config.models_dir);
        }
        "data" => {
            command.arg(&config.data_dir);
        }
        "log" => {
            command.arg(format!(
                "/select,{}",
                config.data_dir.join("wispr.log").display()
            ));
        }
        other => return Err(format!("Unknown location: {other}")),
    }
    command
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Failed to open {kind}: {e}"))
}

/// The built-in AI formatting prompt (for "Reset to default").
#[tauri::command]
pub fn get_default_prompt() -> String {
    crate::formatting::default_prompt()
}

/// Main-window mic button: start a hands-free recording.
#[tauri::command]
pub fn start_hands_free(app: AppHandle) {
    use tauri::Emitter;
    let _ = app.emit(crate::events::REQUEST_START_HANDS_FREE, ());
}

/// Main-window mic button while recording: stop and paste.
#[tauri::command]
pub fn stop_recording(app: AppHandle) {
    use tauri::Emitter;
    let _ = app.emit(crate::events::REQUEST_STOP_RECORDING, ());
}

#[tauri::command]
pub fn get_hotkey(settings: State<'_, Mutex<Settings>>) -> Result<String, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(s.hotkey.clone())
}

#[tauri::command]
pub fn set_hotkey(
    app: AppHandle,
    hotkey: String,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<String, String> {
    // Parse the new hotkey string
    let new_shortcut = parse_hotkey(&hotkey)?;
    if !has_modifier(&hotkey) && !is_safe_bare_key(&hotkey) {
        return Err(
            "Use at least one modifier (Ctrl, Shift, Alt, Win) or a key that never types: \
             F13-F24, Pause, ScrollLock, CapsLock"
                .to_string(),
        );
    }

    // Get the old hotkey to unregister
    let old_hotkey = {
        let s = settings.lock().map_err(|e| e.to_string())?;
        s.hotkey.clone()
    };
    if hotkey.eq_ignore_ascii_case(&old_hotkey) {
        return Ok(old_hotkey);
    }
    let old_shortcut = parse_hotkey(&old_hotkey)?;

    // Register first so a conflict never removes the working shortcut. If a
    // later step fails, roll back to the previous registration and setting.
    let gs = app.global_shortcut();
    gs.register(new_shortcut)
        .map_err(|e| friendly_register_error(&e.to_string()))?;
    if let Err(e) = gs.unregister(old_shortcut) {
        let _ = gs.unregister(new_shortcut);
        return Err(format!("Failed to replace old hotkey: {e}"));
    }

    // Save to settings
    let save_result = {
        let mut s = settings.lock().map_err(|e| e.to_string())?;
        s.hotkey = hotkey.clone();
        s.save(&config.data_dir)
    };
    if let Err(e) = save_result {
        let _ = gs.unregister(new_shortcut);
        let _ = gs.register(old_shortcut);
        if let Ok(mut s) = settings.lock() {
            s.hotkey = old_hotkey;
        }
        return Err(e);
    }

    log::info!("Hotkey changed to: {}", hotkey);
    Ok(hotkey)
}

/// Keys that never produce text, so they may be a hotkey on their own.
pub fn is_safe_bare_key(hotkey: &str) -> bool {
    let key = hotkey.trim().to_ascii_lowercase();
    if key == "pause" || key == "scrolllock" || key == "capslock" {
        return true;
    }
    key.strip_prefix('f')
        .and_then(|n| n.parse::<u32>().ok())
        .is_some_and(|n| (13..=24).contains(&n))
}

/// The plugin's error for a taken combination is a bare "already registered"
/// enum name; say what to do instead.
fn friendly_register_error(error: &str) -> String {
    let lower = error.to_ascii_lowercase();
    if lower.contains("already") {
        "That combination is already used by another app; pick a different one".to_string()
    } else {
        format!("Could not register the hotkey: {error}")
    }
}

/// Register the cancel shortcut. Failure is logged, never fatal.
pub fn register_cancel_hotkey(app: &AppHandle, hotkey: &str) {
    if hotkey.trim().is_empty() {
        return;
    }
    match parse_hotkey(hotkey) {
        Ok(shortcut) => match app.global_shortcut().register(shortcut) {
            Ok(()) => log::info!("Cancel hotkey registered: {hotkey}"),
            Err(e) => log::warn!("Cancel hotkey {hotkey} not registered: {e}"),
        },
        Err(e) => log::warn!("Cancel hotkey '{hotkey}' is invalid: {e}"),
    }
}

#[tauri::command]
pub fn get_hotkey_mode(
    settings: State<'_, Mutex<Settings>>,
) -> Result<crate::hotkey::HotkeyMode, String> {
    Ok(settings.lock().map_err(|e| e.to_string())?.hotkey_mode)
}

#[tauri::command]
pub fn set_hotkey_mode(
    mode: crate::hotkey::HotkeyMode,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<(), String> {
    let mut s = settings.lock().map_err(|e| e.to_string())?;
    let previous = s.hotkey_mode;
    s.hotkey_mode = mode;
    if let Err(e) = s.save(&config.data_dir) {
        s.hotkey_mode = previous;
        return Err(e);
    }
    log::info!("Hotkey mode set to {mode:?}");
    Ok(())
}

#[tauri::command]
pub fn get_cancel_hotkey(settings: State<'_, Mutex<Settings>>) -> Result<String, String> {
    Ok(settings
        .lock()
        .map_err(|e| e.to_string())?
        .cancel_hotkey
        .clone())
}

/// Change the cancel shortcut (empty string disables it).
#[tauri::command]
pub fn set_cancel_hotkey(
    app: AppHandle,
    hotkey: String,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<String, String> {
    let hotkey = hotkey.trim().to_string();
    let old = settings
        .lock()
        .map_err(|e| e.to_string())?
        .cancel_hotkey
        .clone();
    if hotkey.eq_ignore_ascii_case(&old) {
        return Ok(old);
    }
    let gs = app.global_shortcut();
    if !hotkey.is_empty() {
        let shortcut = parse_hotkey(&hotkey)?;
        if !has_modifier(&hotkey) && !is_safe_bare_key(&hotkey) {
            return Err("Use at least one modifier for the cancel key".to_string());
        }
        gs.register(shortcut)
            .map_err(|e| friendly_register_error(&e.to_string()))?;
    }
    if let Ok(old_shortcut) = parse_hotkey(&old) {
        let _ = gs.unregister(old_shortcut);
    }
    let mut s = settings.lock().map_err(|e| e.to_string())?;
    s.cancel_hotkey = hotkey.clone();
    if let Err(e) = s.save(&config.data_dir) {
        s.cancel_hotkey = old.clone();
        if let Ok(new_shortcut) = parse_hotkey(&hotkey) {
            let _ = gs.unregister(new_shortcut);
        }
        if let Ok(old_shortcut) = parse_hotkey(&old) {
            let _ = gs.register(old_shortcut);
        }
        return Err(e);
    }
    log::info!("Cancel hotkey changed to: {hotkey}");
    Ok(hotkey)
}

/// While the Settings page records a new combination the live shortcut must
/// not start a recording when the user presses it.
#[tauri::command]
pub fn begin_hotkey_capture(state: State<'_, Mutex<AppState>>) {
    crate::state::lock_or_recover(&state).capturing_hotkey = true;
}

#[tauri::command]
pub fn end_hotkey_capture(state: State<'_, Mutex<AppState>>) {
    crate::state::lock_or_recover(&state).capturing_hotkey = false;
}

/// Tray → "Pause hotkey".
#[tauri::command]
pub fn set_hotkey_paused(app: AppHandle, paused: bool, state: State<'_, Mutex<AppState>>) {
    crate::state::lock_or_recover(&state).hotkey_paused = paused;
    log::info!("Hotkey {}", if paused { "paused" } else { "resumed" });
    crate::system::tray::refresh(&app);
}

/// Overlay X button / tray: discard the recording or skip the paste.
#[tauri::command]
pub fn cancel_recording(app: AppHandle) {
    crate::pipeline::cancel_recording(&app);
}

fn has_modifier(hotkey: &str) -> bool {
    hotkey.split('+').any(|part| {
        matches!(
            part.trim().to_ascii_lowercase().as_str(),
            "ctrl" | "control" | "shift" | "alt" | "super" | "win" | "meta" | "cmd"
        )
    })
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SoundSettings {
    pub start_sound: String,
    pub stop_sound: String,
    pub start_volume: f32,
    pub stop_volume: f32,
}

#[tauri::command]
pub fn get_sound_settings(settings: State<'_, Mutex<Settings>>) -> Result<SoundSettings, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(SoundSettings {
        start_sound: s.start_sound.clone(),
        stop_sound: s.stop_sound.clone(),
        start_volume: s.start_volume(),
        stop_volume: s.stop_volume(),
    })
}

#[tauri::command]
pub fn set_sound_settings(
    start_sound: String,
    stop_sound: String,
    start_volume: f32,
    stop_volume: f32,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
    player: State<'_, SoundPlayer>,
) -> Result<(), String> {
    // Persist first; don't leave runtime behavior different from the UI if the
    // settings file is temporarily unavailable.
    let runtime_config = {
        let mut s = settings.lock().map_err(|e| e.to_string())?;
        let previous = s.clone();
        s.start_sound = start_sound;
        s.stop_sound = stop_sound;
        s.set_volumes(start_volume, stop_volume);
        if let Err(e) = s.save(&config.data_dir) {
            *s = previous;
            return Err(e);
        }
        s.sound_config()
    };
    player.update_config(runtime_config);

    Ok(())
}

/// Play a chime now and report which output device it went to. `volume`
/// overrides the saved level so the Settings slider can be auditioned before
/// its debounced save lands.
#[tauri::command]
pub fn test_sound(
    which: String,
    volume: Option<f32>,
    player: State<'_, SoundPlayer>,
) -> Result<String, String> {
    let kind = crate::system::sounds::SoundKind::parse(&which)
        .ok_or_else(|| "Unknown sound: use 'start', 'stop', 'busy' or 'cancel'".to_string())?;
    player.play_and_wait(kind, volume, std::time::Duration::from_secs(6))
}

/// What the Settings page shows for the AI section: no secret ever crosses
/// into the webview, only whether a key is stored.
#[derive(serde::Serialize)]
pub struct AiSettingsView {
    pub provider: crate::formatting::AiProvider,
    pub enabled: bool,
    pub openai_model: String,
    pub claude_model: String,
    pub prompt: String,
    pub openai_key_set: bool,
    pub claude_key_set: bool,
    pub key_error: Option<String>,
}

fn ai_settings_view(s: &Settings, state: &AppState) -> AiSettingsView {
    AiSettingsView {
        provider: s.ai.provider.clone(),
        enabled: s.ai.enabled,
        openai_model: s.ai.openai_model.clone(),
        claude_model: s.ai.claude_model.clone(),
        prompt: s.ai.prompt.clone(),
        openai_key_set: !s.ai.keys.openai.is_empty(),
        claude_key_set: !s.ai.keys.claude.is_empty(),
        key_error: state.diagnostics.api_key_error.clone(),
    }
}

#[tauri::command]
pub fn get_ai_settings(
    settings: State<'_, Mutex<Settings>>,
    state: State<'_, Mutex<AppState>>,
) -> Result<AiSettingsView, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    let app_state = state.lock().map_err(|e| e.to_string())?;
    Ok(ai_settings_view(&s, &app_state))
}

/// Apply an edit from the Settings page. Keys are written to the encrypted
/// store only when the update carries them; settings.json only when a visible
/// field changed. Returns the refreshed view.
#[tauri::command]
pub fn set_ai_settings(
    app: AppHandle,
    update: crate::settings::AiSettingsUpdate,
    settings: State<'_, Mutex<Settings>>,
    state: State<'_, Mutex<AppState>>,
    config: State<'_, AppConfig>,
) -> Result<AiSettingsView, String> {
    let mut s = settings.lock().map_err(|e| e.to_string())?;
    let previous = s.clone();
    let change = s.apply_ai_update(&update);
    if change.keys_changed {
        if let Err(e) = crate::secrets::save_api_keys(&config.data_dir, &s.ai.keys) {
            *s = previous;
            return Err(e);
        }
    }
    if change.settings_changed {
        if let Err(e) = s.save(&config.data_dir) {
            if change.keys_changed {
                if let Err(e2) = crate::secrets::save_api_keys(&config.data_dir, &previous.ai.keys)
                {
                    log::warn!("Could not roll back the API keys after a failed save: {e2}");
                }
            }
            *s = previous;
            return Err(e);
        }
        log::info!(
            "AI settings updated: provider={:?} enabled={}",
            s.ai.provider,
            s.ai.enabled
        );
        crate::system::tray::refresh(&app);
    }
    if change.keys_changed {
        log::info!(
            "API keys updated (openai: {}, claude: {})",
            if s.ai.keys.openai.is_empty() {
                "none"
            } else {
                "set"
            },
            if s.ai.keys.claude.is_empty() {
                "none"
            } else {
                "set"
            }
        );
    }
    let app_state = state.lock().map_err(|e| e.to_string())?;
    Ok(ai_settings_view(&s, &app_state))
}

/// The text post-processing section of Settings.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TextSettings {
    pub voice_commands: bool,
    pub paste_suffix: crate::text::PasteSuffix,
    pub replacements: Vec<crate::transcription::replacements::ReplacementRule>,
    #[serde(default = "default_true")]
    pub restore_clipboard: bool,
}

fn default_true() -> bool {
    true
}

#[tauri::command]
pub fn get_text_settings(settings: State<'_, Mutex<Settings>>) -> Result<TextSettings, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(TextSettings {
        voice_commands: s.voice_commands,
        paste_suffix: s.paste_suffix,
        replacements: s.replacements.clone(),
        restore_clipboard: s.restore_clipboard,
    })
}

#[tauri::command]
pub fn set_text_settings(
    update: TextSettings,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<(), String> {
    let mut s = settings.lock().map_err(|e| e.to_string())?;
    let previous = s.clone();
    s.voice_commands = update.voice_commands;
    s.paste_suffix = update.paste_suffix;
    s.restore_clipboard = update.restore_clipboard;
    s.replacements = update
        .replacements
        .into_iter()
        .filter(|r| !r.from.trim().is_empty())
        .collect();
    if let Err(e) = s.save(&config.data_dir) {
        *s = previous;
        return Err(e);
    }
    log::info!(
        "Text settings updated ({} dictionary rules, voice commands {})",
        s.replacements.len(),
        if s.voice_commands { "on" } else { "off" }
    );
    Ok(())
}

/// Tray toggle for AI formatting; keeps provider and keys, flips the switch.
pub fn apply_ai_enabled(app: &AppHandle, enabled: bool) -> Result<(), String> {
    let settings = app.state::<Mutex<Settings>>();
    let config = app.state::<AppConfig>();
    {
        let mut s = crate::state::lock_or_recover(&settings);
        let previous = s.ai.enabled;
        s.ai.enabled = enabled;
        if let Err(e) = s.save(&config.data_dir) {
            s.ai.enabled = previous;
            return Err(e);
        }
    }
    log::info!(
        "AI formatting {}",
        if enabled { "enabled" } else { "disabled" }
    );
    use tauri::Emitter;
    let _ = app.emit(crate::events::AI_SETTINGS_CHANGED, ());
    crate::system::tray::refresh(app);
    Ok(())
}

/// Problems found while loading state files at startup (settings.json moved
/// aside, unreadable history, undecryptable keys).
#[tauri::command]
pub fn get_startup_diagnostics(
    state: State<'_, Mutex<AppState>>,
) -> Result<crate::state::StartupDiagnostics, String> {
    let app_state = state.lock().map_err(|e| e.to_string())?;
    Ok(app_state.diagnostics.clone())
}

#[tauri::command]
pub fn get_show_overlay(settings: State<'_, Mutex<Settings>>) -> Result<bool, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(s.show_overlay)
}

#[tauri::command]
pub fn set_show_overlay(
    app: AppHandle,
    show: bool,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<(), String> {
    {
        let mut s = settings.lock().map_err(|e| e.to_string())?;
        let previous = s.show_overlay;
        s.show_overlay = show;
        if let Err(e) = s.save(&config.data_dir) {
            s.show_overlay = previous;
            return Err(e);
        }
    }

    // If turning off, hide overlay immediately. Do not force-show on toggle
    // on — it appears naturally when recording starts.
    if !show {
        if let Some(w) = app.get_webview_window("overlay") {
            let _ = w.hide();
        }
    }

    Ok(())
}

#[tauri::command]
pub fn get_language(
    settings: State<'_, Mutex<Settings>>,
) -> Result<crate::transcription::engine::LanguageMode, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(s.language)
}

#[tauri::command]
pub fn set_language(
    app: AppHandle,
    language: crate::transcription::engine::LanguageMode,
) -> Result<(), String> {
    apply_language(&app, language)
}

/// Change the language mode from the Settings page or the tray; both stay
/// in sync through the `language-mode-changed` event and the tray refresh.
pub fn apply_language(
    app: &AppHandle,
    language: crate::transcription::engine::LanguageMode,
) -> Result<(), String> {
    let settings = app.state::<Mutex<Settings>>();
    let config = app.state::<AppConfig>();
    {
        let mut s = crate::state::lock_or_recover(&settings);
        if s.language == language {
            return Ok(());
        }
        let previous = s.language;
        s.language = language;
        if let Err(e) = s.save(&config.data_dir) {
            s.language = previous;
            return Err(e);
        }
    }
    log::info!("Language mode updated: {:?}", language);
    use tauri::Emitter;
    let _ = app.emit(crate::events::LANGUAGE_MODE_CHANGED, language);
    crate::system::tray::refresh(app);
    Ok(())
}

#[tauri::command(async)]
pub fn get_input_devices() -> Result<Vec<crate::audio::devices::AudioDeviceInfo>, String> {
    Ok(crate::audio::devices::list_input_devices())
}

#[tauri::command]
pub fn get_input_device(settings: State<'_, Mutex<Settings>>) -> Result<String, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(s.input_device.clone())
}

#[tauri::command(async)]
pub fn set_input_device(
    input_device: String,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<(), String> {
    if !input_device.is_empty()
        && !crate::audio::devices::list_input_devices()
            .iter()
            .any(|device| device.name == input_device)
    {
        return Err("That microphone is no longer available".to_string());
    }

    let mut s = settings.lock().map_err(|e| e.to_string())?;
    let previous = s.input_device.clone();
    s.input_device = input_device;
    if let Err(e) = s.save(&config.data_dir) {
        s.input_device = previous;
        return Err(e);
    }
    Ok(())
}

/// Check that the configured microphone (or the system default) can be
/// resolved right now; clears a standing microphone error. Returns the name
/// of the device that would be used.
#[tauri::command(async)]
pub fn probe_input_device(
    app: AppHandle,
    settings: State<'_, Mutex<Settings>>,
    state: State<'_, Mutex<AppState>>,
) -> Result<String, String> {
    let preferred = settings
        .lock()
        .map_err(|e| e.to_string())?
        .input_device
        .clone();
    let selected = crate::audio::devices::select_input_device(
        (!preferred.is_empty()).then_some(preferred.as_str()),
    )?;
    let cleared = {
        let mut s = crate::state::lock_or_recover(&state);
        if matches!(&s.status, AppStatus::Error { code, .. } if code == "mic") {
            s.status = AppStatus::Idle;
            true
        } else {
            false
        }
    };
    if cleared {
        crate::pipeline::emit_status(&app, &AppStatus::Idle);
    }
    Ok(if selected.used_fallback {
        format!("{} (fallback)", selected.name)
    } else {
        selected.name
    })
}

#[tauri::command]
pub fn get_autostart(settings: State<'_, Mutex<Settings>>) -> Result<bool, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(s.run_on_startup)
}

#[tauri::command]
pub fn set_autostart(
    enabled: bool,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<(), String> {
    crate::autostart::set_autostart_registry(enabled)?;
    let mut s = settings.lock().map_err(|e| e.to_string())?;
    let previous = s.run_on_startup;
    s.run_on_startup = enabled;
    if let Err(e) = s.save(&config.data_dir) {
        s.run_on_startup = previous;
        let _ = crate::autostart::set_autostart_registry(previous);
        return Err(e);
    }
    log::info!("Autostart set to: {}", enabled);
    Ok(())
}

/// Parse a hotkey string like "Ctrl+Shift+Space" into a tauri Shortcut.
pub fn parse_hotkey(hotkey: &str) -> Result<Shortcut, String> {
    let parts: Vec<&str> = hotkey.split('+').map(|s| s.trim()).collect();
    if parts.is_empty() {
        return Err("Empty hotkey".to_string());
    }

    let mut modifiers = Modifiers::empty();
    let mut key_code: Option<Code> = None;

    for part in &parts {
        match part.to_lowercase().as_str() {
            "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
            "shift" => modifiers |= Modifiers::SHIFT,
            "alt" => modifiers |= Modifiers::ALT,
            "super" | "win" | "meta" | "cmd" => modifiers |= Modifiers::SUPER,
            key => {
                if key_code.is_some() {
                    return Err(format!("Multiple keys in hotkey: {}", hotkey));
                }
                key_code = Some(parse_key_code(key)?);
            }
        }
    }

    let code = key_code.ok_or_else(|| format!("No key specified in hotkey: {}", hotkey))?;
    let mods = if modifiers.is_empty() {
        None
    } else {
        Some(modifiers)
    };

    Ok(Shortcut::new(mods, code))
}

fn parse_key_code(key: &str) -> Result<Code, String> {
    match key.to_lowercase().as_str() {
        "space" => Ok(Code::Space),
        "enter" | "return" => Ok(Code::Enter),
        "tab" => Ok(Code::Tab),
        "escape" | "esc" => Ok(Code::Escape),
        "backspace" => Ok(Code::Backspace),
        "delete" | "del" => Ok(Code::Delete),
        "insert" => Ok(Code::Insert),
        "home" => Ok(Code::Home),
        "end" => Ok(Code::End),
        "pageup" => Ok(Code::PageUp),
        "pagedown" => Ok(Code::PageDown),
        "up" => Ok(Code::ArrowUp),
        "down" => Ok(Code::ArrowDown),
        "left" => Ok(Code::ArrowLeft),
        "right" => Ok(Code::ArrowRight),
        "f1" => Ok(Code::F1),
        "f2" => Ok(Code::F2),
        "f3" => Ok(Code::F3),
        "f4" => Ok(Code::F4),
        "f5" => Ok(Code::F5),
        "f6" => Ok(Code::F6),
        "f7" => Ok(Code::F7),
        "f8" => Ok(Code::F8),
        "f9" => Ok(Code::F9),
        "f10" => Ok(Code::F10),
        "f11" => Ok(Code::F11),
        "f12" => Ok(Code::F12),
        "f13" => Ok(Code::F13),
        "f14" => Ok(Code::F14),
        "f15" => Ok(Code::F15),
        "f16" => Ok(Code::F16),
        "f17" => Ok(Code::F17),
        "f18" => Ok(Code::F18),
        "f19" => Ok(Code::F19),
        "f20" => Ok(Code::F20),
        "f21" => Ok(Code::F21),
        "f22" => Ok(Code::F22),
        "f23" => Ok(Code::F23),
        "f24" => Ok(Code::F24),
        "pause" => Ok(Code::Pause),
        "scrolllock" => Ok(Code::ScrollLock),
        "capslock" => Ok(Code::CapsLock),
        "numlock" => Ok(Code::NumLock),
        "printscreen" => Ok(Code::PrintScreen),
        "numpad0" => Ok(Code::Numpad0),
        "numpad1" => Ok(Code::Numpad1),
        "numpad2" => Ok(Code::Numpad2),
        "numpad3" => Ok(Code::Numpad3),
        "numpad4" => Ok(Code::Numpad4),
        "numpad5" => Ok(Code::Numpad5),
        "numpad6" => Ok(Code::Numpad6),
        "numpad7" => Ok(Code::Numpad7),
        "numpad8" => Ok(Code::Numpad8),
        "numpad9" => Ok(Code::Numpad9),
        "numpadadd" => Ok(Code::NumpadAdd),
        "numpadsubtract" => Ok(Code::NumpadSubtract),
        "numpadmultiply" => Ok(Code::NumpadMultiply),
        "numpaddivide" => Ok(Code::NumpadDivide),
        "numpaddecimal" => Ok(Code::NumpadDecimal),
        "numpadenter" => Ok(Code::NumpadEnter),
        "`" | "backquote" => Ok(Code::Backquote),
        "-" | "minus" => Ok(Code::Minus),
        "=" | "equal" => Ok(Code::Equal),
        "[" | "bracketleft" => Ok(Code::BracketLeft),
        "]" | "bracketright" => Ok(Code::BracketRight),
        "\\" | "backslash" => Ok(Code::Backslash),
        ";" | "semicolon" => Ok(Code::Semicolon),
        "'" | "quote" => Ok(Code::Quote),
        "," | "comma" => Ok(Code::Comma),
        "." | "period" => Ok(Code::Period),
        "/" | "slash" => Ok(Code::Slash),
        "0" => Ok(Code::Digit0),
        "1" => Ok(Code::Digit1),
        "2" => Ok(Code::Digit2),
        "3" => Ok(Code::Digit3),
        "4" => Ok(Code::Digit4),
        "5" => Ok(Code::Digit5),
        "6" => Ok(Code::Digit6),
        "7" => Ok(Code::Digit7),
        "8" => Ok(Code::Digit8),
        "9" => Ok(Code::Digit9),
        "a" => Ok(Code::KeyA),
        "b" => Ok(Code::KeyB),
        "c" => Ok(Code::KeyC),
        "d" => Ok(Code::KeyD),
        "e" => Ok(Code::KeyE),
        "f" => Ok(Code::KeyF),
        "g" => Ok(Code::KeyG),
        "h" => Ok(Code::KeyH),
        "i" => Ok(Code::KeyI),
        "j" => Ok(Code::KeyJ),
        "k" => Ok(Code::KeyK),
        "l" => Ok(Code::KeyL),
        "m" => Ok(Code::KeyM),
        "n" => Ok(Code::KeyN),
        "o" => Ok(Code::KeyO),
        "p" => Ok(Code::KeyP),
        "q" => Ok(Code::KeyQ),
        "r" => Ok(Code::KeyR),
        "s" => Ok(Code::KeyS),
        "t" => Ok(Code::KeyT),
        "u" => Ok(Code::KeyU),
        "v" => Ok(Code::KeyV),
        "w" => Ok(Code::KeyW),
        "x" => Ok(Code::KeyX),
        "y" => Ok(Code::KeyY),
        "z" => Ok(Code::KeyZ),
        other => Err(format!("Unknown key: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::{has_modifier, parse_hotkey};

    #[test]
    fn hotkey_parser_accepts_supported_combo() {
        assert!(parse_hotkey("Ctrl+Shift+Space").is_ok());
        assert!(parse_hotkey("Win+Alt+F12").is_ok());
    }

    #[test]
    fn hotkey_parser_rejects_unknown_and_multiple_keys() {
        assert!(parse_hotkey("Ctrl+NotAKey").is_err());
        assert!(parse_hotkey("Ctrl+A+B").is_err());
    }

    #[test]
    fn modifier_guard_protects_normal_typing() {
        assert!(has_modifier("Ctrl+A"));
        assert!(!has_modifier("A"));
    }

    #[test]
    fn extended_keys_parse_and_safe_bare_keys_are_allowed() {
        use super::{is_safe_bare_key, parse_hotkey as parse};
        use tauri_plugin_global_shortcut::{Code, Shortcut};
        assert_eq!(parse("F13").unwrap(), Shortcut::new(None, Code::F13));
        assert!(parse("Ctrl+NumpadAdd").is_ok());
        assert!(parse("Pause").is_ok());
        assert!(parse("ScrollLock").is_ok());
        assert!(is_safe_bare_key("F13"));
        assert!(is_safe_bare_key("f24"));
        assert!(is_safe_bare_key("Pause"));
        assert!(is_safe_bare_key("CapsLock"));
        assert!(!is_safe_bare_key("F12"), "F12 types in some apps");
        assert!(!is_safe_bare_key("Space"));
        assert!(!is_safe_bare_key("Ctrl+F13"));
    }

    #[test]
    fn taken_combination_error_is_actionable() {
        let msg = super::friendly_register_error("AlreadyRegistered(Shortcut(..))");
        assert!(msg.contains("already used"));
        assert!(super::friendly_register_error("boom").contains("boom"));
    }
}
