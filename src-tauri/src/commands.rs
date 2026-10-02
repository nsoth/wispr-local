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
#[tauri::command]
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
    if !matches!(s.status, AppStatus::Idle | AppStatus::Error(_)) {
        return Err("Finish the current dictation before reloading the model".to_string());
    }
    if s.model == ModelState::Loading {
        return Err("The model is already loading".to_string());
    }
    Ok(())
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
pub fn get_history(state: State<'_, Mutex<AppState>>) -> Result<Vec<String>, String> {
    let app_state = state.lock().map_err(|e| e.to_string())?;
    Ok(app_state.history.clone())
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
    let history: Vec<String> = Vec::new();
    app.emit(crate::events::HISTORY_CHANGED, &history)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Copy arbitrary text to the system clipboard (used by the history list).
#[tauri::command]
pub fn copy_text(text: String) -> Result<(), String> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|e| format!("Failed to open clipboard: {}", e))?;
    clipboard
        .set_text(&text)
        .map_err(|e| format!("Failed to set clipboard text: {}", e))?;
    Ok(())
}

#[tauri::command]
pub fn get_models_dir(config: State<'_, crate::config::AppConfig>) -> Result<String, String> {
    Ok(config.models_dir.to_string_lossy().to_string())
}

#[tauri::command]
pub fn open_models_dir(config: State<'_, AppConfig>) -> Result<(), String> {
    #[cfg(windows)]
    let mut command = std::process::Command::new("explorer.exe");
    #[cfg(target_os = "macos")]
    let mut command = std::process::Command::new("open");
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = std::process::Command::new("xdg-open");

    command
        .arg(&config.models_dir)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Failed to open models folder: {e}"))
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
    if !has_modifier(&hotkey) {
        return Err("Use at least one modifier: Ctrl, Shift, Alt, or Win".to_string());
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
        .map_err(|e| format!("Failed to register new hotkey: {}", e))?;
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
        log::info!("AI settings updated: provider={:?}", s.ai.provider);
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
}

#[tauri::command]
pub fn get_text_settings(settings: State<'_, Mutex<Settings>>) -> Result<TextSettings, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(TextSettings {
        voice_commands: s.voice_commands,
        paste_suffix: s.paste_suffix,
        replacements: s.replacements.clone(),
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
    language: crate::transcription::engine::LanguageMode,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<(), String> {
    let mut s = settings.lock().map_err(|e| e.to_string())?;
    log::info!("Language mode updated: {:?}", language);
    let previous = s.language;
    s.language = language;
    if let Err(e) = s.save(&config.data_dir) {
        s.language = previous;
        return Err(e);
    }
    Ok(())
}

#[tauri::command]
pub fn get_input_devices() -> Result<Vec<crate::audio::devices::AudioDeviceInfo>, String> {
    Ok(crate::audio::devices::list_input_devices())
}

#[tauri::command]
pub fn get_input_device(settings: State<'_, Mutex<Settings>>) -> Result<String, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(s.input_device.clone())
}

#[tauri::command]
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
}
