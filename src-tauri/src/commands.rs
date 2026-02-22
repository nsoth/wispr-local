use std::sync::Mutex;
use tauri::{AppHandle, State};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

use crate::audio::buffer::AudioBuffer;
use crate::audio::capture::AudioCapture;
use crate::config::AppConfig;
use crate::settings::Settings;
use crate::state::{AppState, AppStatus};
use crate::system::sounds::SoundPlayer;
use crate::system::text_injection;
use crate::transcription::engine::WhisperEngine;

#[tauri::command]
pub async fn start_recording(
    state: State<'_, Mutex<AppState>>,
    capture: State<'_, Mutex<AudioCapture>>,
    buffer: State<'_, AudioBuffer>,
) -> Result<String, String> {
    {
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        if app_state.status == AppStatus::Recording {
            return Err("Already recording".to_string());
        }
        buffer.clear();
        app_state.status = AppStatus::Recording;
    }

    let mut cap = capture.lock().map_err(|e| e.to_string())?;
    let sample_rate = cap.start()?;

    {
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        app_state.device_sample_rate = sample_rate;
    }

    Ok(format!("Recording at {} Hz", sample_rate))
}

#[tauri::command]
pub async fn stop_recording_and_transcribe(
    state: State<'_, Mutex<AppState>>,
    capture: State<'_, Mutex<AudioCapture>>,
    buffer: State<'_, AudioBuffer>,
    engine: State<'_, Mutex<WhisperEngine>>,
) -> Result<String, String> {
    // Stop recording
    {
        let mut cap = capture.lock().map_err(|e| e.to_string())?;
        cap.stop();
    }

    {
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        app_state.status = AppStatus::Transcribing;
    }

    let samples = buffer.take_samples();
    if samples.is_empty() {
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        app_state.status = AppStatus::Idle;
        return Err("No audio recorded".to_string());
    }

    log::info!(
        "Transcribing {} samples ({:.1}s of audio)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    // Transcribe
    let text = {
        let eng = engine.lock().map_err(|e| e.to_string())?;
        eng.transcribe(&samples)?
    };

    if text.is_empty() {
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        app_state.status = AppStatus::Idle;
        return Err("No speech detected".to_string());
    }

    log::info!("Transcription: {}", text);

    // Inject text
    {
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        app_state.status = AppStatus::Injecting;
    }

    text_injection::inject_text(&text)?;

    // Done
    {
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        app_state.last_transcription = text.clone();
        app_state.status = AppStatus::Idle;
    }

    Ok(text)
}

#[tauri::command]
pub fn get_status(state: State<'_, Mutex<AppState>>) -> Result<String, String> {
    let app_state = state.lock().map_err(|e| e.to_string())?;
    let status = match &app_state.status {
        AppStatus::Idle => "Idle".to_string(),
        AppStatus::Recording => "Recording".to_string(),
        AppStatus::Transcribing => "Transcribing".to_string(),
        AppStatus::Formatting => "Formatting".to_string(),
        AppStatus::Injecting => "Injecting".to_string(),
        AppStatus::Error(e) => format!("Error: {}", e),
    };
    Ok(status)
}

#[tauri::command]
pub fn is_model_loaded(engine: State<'_, Mutex<WhisperEngine>>) -> Result<bool, String> {
    let eng = engine.lock().map_err(|e| e.to_string())?;
    Ok(eng.is_loaded())
}

#[tauri::command]
pub fn get_last_transcription(state: State<'_, Mutex<AppState>>) -> Result<String, String> {
    let app_state = state.lock().map_err(|e| e.to_string())?;
    Ok(app_state.last_transcription.clone())
}

#[tauri::command]
pub fn get_models_dir(config: State<'_, crate::config::AppConfig>) -> Result<String, String> {
    Ok(config.models_dir.to_string_lossy().to_string())
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

    // Get the old hotkey to unregister
    let old_hotkey = {
        let s = settings.lock().map_err(|e| e.to_string())?;
        s.hotkey.clone()
    };
    let old_shortcut = parse_hotkey(&old_hotkey)?;

    // Unregister old, register new
    let gs = app.global_shortcut();
    gs.unregister(old_shortcut)
        .map_err(|e| format!("Failed to unregister old hotkey: {}", e))?;
    gs.register(new_shortcut)
        .map_err(|e| format!("Failed to register new hotkey: {}", e))?;

    // Save to settings
    {
        let mut s = settings.lock().map_err(|e| e.to_string())?;
        s.hotkey = hotkey.clone();
        s.save(&config.data_dir)?;
    }

    log::info!("Hotkey changed to: {}", hotkey);
    Ok(hotkey)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SoundSettings {
    pub start_sound: String,
    pub stop_sound: String,
    pub sound_volume: f32,
}

#[tauri::command]
pub fn get_sound_settings(settings: State<'_, Mutex<Settings>>) -> Result<SoundSettings, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(SoundSettings {
        start_sound: s.start_sound.clone(),
        stop_sound: s.stop_sound.clone(),
        sound_volume: s.sound_volume,
    })
}

#[tauri::command]
pub fn set_sound_settings(
    start_sound: String,
    stop_sound: String,
    sound_volume: f32,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
    player: State<'_, SoundPlayer>,
) -> Result<(), String> {
    let volume = sound_volume.clamp(0.0, 1.0);

    // Update sound player at runtime
    player.update_config(start_sound.clone(), stop_sound.clone(), volume);

    // Save to settings
    {
        let mut s = settings.lock().map_err(|e| e.to_string())?;
        s.start_sound = start_sound;
        s.stop_sound = stop_sound;
        s.sound_volume = volume;
        s.save(&config.data_dir)?;
    }

    Ok(())
}

#[tauri::command]
pub fn test_sound(which: String, player: State<'_, SoundPlayer>) -> Result<(), String> {
    match which.as_str() {
        "start" => player.play_start(),
        "stop" => player.play_stop(),
        _ => return Err("Unknown sound: use 'start' or 'stop'".to_string()),
    }
    Ok(())
}

#[tauri::command]
pub fn get_ai_settings(settings: State<'_, Mutex<Settings>>) -> Result<crate::formatting::AiSettings, String> {
    let s = settings.lock().map_err(|e| e.to_string())?;
    Ok(s.ai.clone())
}

#[tauri::command]
pub fn set_ai_settings(
    ai: crate::formatting::AiSettings,
    settings: State<'_, Mutex<Settings>>,
    config: State<'_, AppConfig>,
) -> Result<(), String> {
    let mut s = settings.lock().map_err(|e| e.to_string())?;
    log::info!("AI settings updated: provider={:?}", ai.provider);
    s.ai = ai;
    s.save(&config.data_dir)?;
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
