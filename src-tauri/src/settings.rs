//! Persisted user settings (`settings.json` in the data directory).
//!
//! Loading never writes the file back: a file that cannot be decoded or
//! parsed is moved aside as `settings.json.corrupt-<timestamp>` and reported
//! through [`SettingsLoad`], and an unreadable file leaves the in-memory
//! defaults marked read-only so no setter can overwrite bytes we never saw.
//! The one exception is the migration of a legacy plaintext `api_key`, which
//! rewrites the file only after the key was stored encrypted.

use crate::formatting::{AiProvider, AiSettings};
use crate::hotkey::HotkeyMode;
use crate::secrets::{self, ApiKeys};
use crate::text::{PasteSuffix, TextPipeline};
use crate::transcription::engine::LanguageMode;
use crate::transcription::replacements::{self, ReplacementRule, Vocabulary};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_hotkey")]
    pub hotkey: String,
    /// Hold (push-to-talk), Toggle (press to start/stop) or Hybrid (hold,
    /// or tap to go hands-free).
    #[serde(default)]
    pub hotkey_mode: HotkeyMode,
    /// Discards the active recording. Empty disables the shortcut.
    #[serde(default = "default_cancel_hotkey")]
    pub cancel_hotkey: String,
    #[serde(default)]
    pub start_sound: String,
    #[serde(default)]
    pub stop_sound: String,
    /// Legacy master volume. Still written (as the stop volume) so an older
    /// build reading this file keeps a sensible level.
    #[serde(default = "default_volume")]
    pub sound_volume: f32,
    /// Explicit chime volumes. `None` resolves from `sound_volume`: the start
    /// chime at [`START_VOLUME_FACTOR`] of it, the stop chime at it exactly.
    #[serde(default)]
    pub start_volume: Option<f32>,
    #[serde(default)]
    pub stop_volume: Option<f32>,
    #[serde(default)]
    pub ai: AiSettings,
    #[serde(default)]
    pub run_on_startup: bool,
    #[serde(default = "default_show_overlay")]
    pub show_overlay: bool,
    /// Whisper model filename inside the models dir. The loader falls back to
    /// ggml-medium.bin when this file is missing.
    #[serde(default = "default_model_file")]
    pub model_file: String,
    /// Transcription language. `Auto` detects Russian vs English per utterance;
    /// `Russian`/`English` pin the decoder. Defaults to Auto.
    #[serde(default)]
    pub language: LanguageMode,
    /// Empty uses the current Windows default. Otherwise this is the CPAL
    /// device name selected by the user.
    #[serde(default)]
    pub input_device: String,
    /// Turn "новая строка" / "new paragraph" (spoken between pauses) into
    /// line breaks.
    #[serde(default = "default_true")]
    pub voice_commands: bool,
    /// What follows a pasted transcript so the next dictation does not glue
    /// onto it.
    #[serde(default)]
    pub paste_suffix: PasteSuffix,
    /// User dictionary applied after filler removal.
    #[serde(default = "replacements::default_rules")]
    pub replacements: Vec<ReplacementRule>,
    /// Put the previous clipboard content back after pasting (off = keep the
    /// transcript in the clipboard).
    #[serde(default = "default_true")]
    pub restore_clipboard: bool,
    /// How many dictations to keep in history.json (0 = none).
    #[serde(default = "default_history_limit")]
    pub history_limit: usize,
    /// Set when settings.json existed but could not be read at startup. Every
    /// save is refused until a restart so a transient IO error can never turn
    /// into "defaults written over the user's file".
    #[serde(skip)]
    pub read_only: bool,
}

/// Top-level keys this build understands; anything else is reported (a typo
/// like `"model"` for `model_file` is otherwise silently ignored).
const KNOWN_KEYS: &[&str] = &[
    "hotkey",
    "hotkey_mode",
    "cancel_hotkey",
    "start_sound",
    "stop_sound",
    "sound_volume",
    "start_volume",
    "stop_volume",
    "ai",
    "run_on_startup",
    "show_overlay",
    "model_file",
    "language",
    "input_device",
    "voice_commands",
    "paste_suffix",
    "replacements",
    "restore_clipboard",
    "history_limit",
];

fn default_history_limit() -> usize {
    crate::state::HISTORY_LIMIT
}

fn default_true() -> bool {
    true
}

fn default_hotkey() -> String {
    "Ctrl+Shift+Space".to_string()
}

fn default_cancel_hotkey() -> String {
    "Ctrl+Shift+Backspace".to_string()
}

fn default_volume() -> f32 {
    0.5
}

/// Default start-chime level relative to the stop chime: the start cue is a
/// private "I'm listening" signal, the stop cue confirms a paste.
pub const START_VOLUME_FACTOR: f32 = 0.6;

fn default_show_overlay() -> bool {
    true
}

pub fn default_model_file() -> String {
    "ggml-large-v3-turbo.bin".to_string()
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: default_hotkey(),
            hotkey_mode: HotkeyMode::default(),
            cancel_hotkey: default_cancel_hotkey(),
            start_sound: String::new(),
            stop_sound: String::new(),
            sound_volume: default_volume(),
            start_volume: None,
            stop_volume: None,
            ai: AiSettings::default(),
            run_on_startup: false,
            show_overlay: true,
            model_file: default_model_file(),
            language: LanguageMode::default(),
            input_device: String::new(),
            voice_commands: true,
            paste_suffix: PasteSuffix::default(),
            replacements: replacements::default_rules(),
            restore_clipboard: true,
            history_limit: crate::state::HISTORY_LIMIT,
            read_only: false,
        }
    }
}

/// Result of [`Settings::load_with_report`].
#[derive(Debug)]
pub struct SettingsLoad {
    pub settings: Settings,
    /// Human-readable problem with settings.json, if any.
    pub error: Option<String>,
    pub read_only: bool,
    pub unknown_keys: Vec<String>,
    /// Values that could not be used as written and were replaced by their
    /// defaults (e.g. a hotkey that does not parse). Shown at startup.
    pub adjustments: Vec<String>,
    /// Problem decrypting the stored API keys, if any.
    pub api_key_error: Option<String>,
}

/// What the Settings page sends when the AI section changes. Key fields are
/// `None` when untouched, `Some("")` to remove a key, `Some(key)` to replace.
#[derive(Debug, Clone, Deserialize)]
pub struct AiSettingsUpdate {
    pub provider: AiProvider,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub openai_model: String,
    pub claude_model: String,
    pub prompt: String,
    #[serde(default)]
    pub openai_api_key: Option<String>,
    #[serde(default)]
    pub claude_api_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiChange {
    pub keys_changed: bool,
    pub settings_changed: bool,
}

impl Settings {
    /// Effective start-chime volume in 0..=1.
    pub fn start_volume(&self) -> f32 {
        self.start_volume
            .unwrap_or(self.sound_volume * START_VOLUME_FACTOR)
            .clamp(0.0, 1.0)
    }

    /// Effective stop-chime volume in 0..=1.
    pub fn stop_volume(&self) -> f32 {
        self.stop_volume
            .unwrap_or(self.sound_volume)
            .clamp(0.0, 1.0)
    }

    /// Store explicit chime volumes; keeps the legacy master in sync.
    pub fn set_volumes(&mut self, start: f32, stop: f32) {
        let start = start.clamp(0.0, 1.0);
        let stop = stop.clamp(0.0, 1.0);
        self.start_volume = Some(start);
        self.stop_volume = Some(stop);
        self.sound_volume = stop;
    }

    /// Post-processing steps for one utterance, compiled from the current
    /// dictionary (a few dozen small regexes; cheap per dictation).
    pub fn text_pipeline(&self) -> TextPipeline {
        TextPipeline {
            voice_commands: self.voice_commands,
            vocabulary: Vocabulary::compile(&self.replacements),
        }
    }

    pub fn sound_config(&self) -> crate::system::sounds::SoundConfig {
        crate::system::sounds::SoundConfig {
            start_sound: self.start_sound.clone(),
            stop_sound: self.stop_sound.clone(),
            start_volume: self.start_volume(),
            stop_volume: self.stop_volume(),
        }
    }

    /// Apply an AI-settings update from the UI. Pure: nothing is written.
    pub fn apply_ai_update(&mut self, update: &AiSettingsUpdate) -> AiChange {
        let mut keys_changed = false;
        let mut settings_changed = false;

        let prompt = if update.prompt.trim().is_empty() {
            crate::formatting::default_prompt()
        } else {
            update.prompt.clone()
        };
        if self.ai.provider != update.provider
            || self.ai.enabled != update.enabled
            || self.ai.openai_model != update.openai_model
            || self.ai.claude_model != update.claude_model
            || self.ai.prompt != prompt
        {
            settings_changed = true;
        }
        self.ai.provider = update.provider.clone();
        self.ai.enabled = update.enabled;
        self.ai.openai_model = update.openai_model.clone();
        self.ai.claude_model = update.claude_model.clone();
        self.ai.prompt = prompt;

        if let Some(key) = &update.openai_api_key {
            let key = key.trim().to_string();
            if key != self.ai.keys.openai {
                self.ai.keys.openai = key;
                keys_changed = true;
            }
        }
        if let Some(key) = &update.claude_api_key {
            let key = key.trim().to_string();
            if key != self.ai.keys.claude {
                self.ai.keys.claude = key;
                keys_changed = true;
            }
        }
        AiChange {
            keys_changed,
            settings_changed,
        }
    }

    pub fn file_path(data_dir: &Path) -> PathBuf {
        data_dir.join("settings.json")
    }

    pub fn load(data_dir: &Path) -> Self {
        Self::load_with_report(data_dir).settings
    }

    pub fn load_with_report(data_dir: &Path) -> SettingsLoad {
        let path = Self::file_path(data_dir);
        let mut load = SettingsLoad {
            settings: Self::default(),
            error: None,
            read_only: false,
            unknown_keys: Vec::new(),
            adjustments: Vec::new(),
            api_key_error: None,
        };

        let bytes = match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                load.error = Some(format!(
                    "settings.json could not be read ({e}); using defaults for this session and \
                     refusing to save until Wispr Local is restarted"
                ));
                load.read_only = true;
                load.settings.read_only = true;
                log::error!("{}", load.error.as_deref().unwrap_or_default());
                return load;
            }
        };

        if let Some(bytes) = bytes {
            match decode_utf8(&bytes).and_then(|text| parse_settings(&text)) {
                Ok((settings, unknown_keys)) => {
                    if !unknown_keys.is_empty() {
                        log::warn!(
                            "settings.json has keys this build does not know: {}",
                            unknown_keys.join(", ")
                        );
                    }
                    load.settings = settings;
                    load.unknown_keys = unknown_keys;
                }
                Err(problem) => {
                    let moved = crate::config::quarantine_file(&path);
                    load.error = Some(match &moved {
                        Some(p) => format!(
                            "{problem}. The file was moved to {} and default settings are in use.",
                            p.file_name().unwrap_or_default().to_string_lossy()
                        ),
                        None => format!(
                            "{problem}. The file could not be moved aside; default settings are in \
                             use and will not be saved."
                        ),
                    });
                    if moved.is_none() {
                        load.read_only = true;
                        load.settings.read_only = true;
                    }
                    log::error!("{}", load.error.as_deref().unwrap_or_default());
                }
            }
        }

        load.adjustments = normalize(&mut load.settings);
        for adjustment in &load.adjustments {
            log::warn!("{adjustment}");
        }

        match secrets::load_api_keys(data_dir) {
            Ok(keys) => load.settings.ai.keys = keys,
            Err(e) => {
                log::warn!("Could not load the encrypted API keys: {e}");
                load.api_key_error = Some(e);
            }
        }
        migrate_legacy_plaintext_key(&mut load, data_dir);
        load
    }

    pub fn save(&self, data_dir: &Path) -> Result<(), String> {
        if self.read_only {
            return Err(
                "settings.json could not be read at startup; restart Wispr Local before changing \
                 settings"
                    .to_string(),
            );
        }
        let path = Self::file_path(data_dir);
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        crate::config::write_file_atomic(&path, json.as_bytes())
    }
}

/// Reject UTF-16, strip a UTF-8 BOM, validate UTF-8.
fn decode_utf8(bytes: &[u8]) -> Result<String, String> {
    if bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]) {
        return Err(
            "settings.json is saved as UTF-16; save it as UTF-8 without BOM (PowerShell's Out-File \
             defaults to UTF-16)"
                .to_string(),
        );
    }
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    String::from_utf8(bytes.to_vec()).map_err(|e| format!("settings.json is not valid UTF-8 ({e})"))
}

fn parse_settings(text: &str) -> Result<(Settings, Vec<String>), String> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| format!("settings.json could not be parsed ({e})"))?;
    let unknown_keys = value
        .as_object()
        .map(|map| {
            map.keys()
                .filter(|k| !KNOWN_KEYS.contains(&k.as_str()))
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let settings: Settings = serde_json::from_value(value)
        .map_err(|e| format!("settings.json could not be parsed ({e})"))?;
    Ok((settings, unknown_keys))
}

/// Clamp and default values from older or hand-edited files. Returns the
/// user-facing description of every value that had to be replaced.
fn normalize(settings: &mut Settings) -> Vec<String> {
    let mut adjustments = Vec::new();
    if let Err(e) = crate::hotkey::parse_hotkey(&settings.hotkey) {
        adjustments.push(format!(
            "The hotkey {:?} in settings.json is not valid ({e}); using {} until you set a new one.",
            settings.hotkey,
            default_hotkey()
        ));
        settings.hotkey = default_hotkey();
    }
    if !settings.cancel_hotkey.trim().is_empty() {
        if let Err(e) = crate::hotkey::parse_hotkey(&settings.cancel_hotkey) {
            adjustments.push(format!(
                "The cancel hotkey {:?} in settings.json is not valid ({e}); using {}.",
                settings.cancel_hotkey,
                default_cancel_hotkey()
            ));
            settings.cancel_hotkey = default_cancel_hotkey();
        }
    }
    settings.sound_volume = settings.sound_volume.clamp(0.0, 1.0);
    settings.history_limit = settings.history_limit.min(crate::state::HISTORY_MAX);
    if settings.model_file.trim().is_empty() {
        settings.model_file = default_model_file();
    }
    if settings.ai.provider == AiProvider::Unknown {
        log::warn!("settings.json names an AI provider this build does not know; using None");
        settings.ai.provider = AiProvider::None;
    }
    if settings.language == LanguageMode::Unknown {
        log::warn!("settings.json names a language mode this build does not know; using Auto");
        settings.language = LanguageMode::Auto;
    }
    if settings.ai.prompt.trim().is_empty() {
        settings.ai.prompt = crate::formatting::default_prompt();
    }
    adjustments
}

/// A pre-DPAPI settings file stored the key in plaintext. Move it into the
/// encrypted store, then rewrite settings.json without it — the only write
/// `load` ever performs, and only after the encrypted copy exists.
fn migrate_legacy_plaintext_key(load: &mut SettingsLoad, data_dir: &Path) {
    let legacy = std::mem::take(&mut load.settings.ai.legacy_api_key);
    if legacy.is_empty() {
        return;
    }
    if !load.settings.ai.keys.is_empty() {
        log::info!("Ignoring the plaintext api_key in settings.json: encrypted keys already exist");
    } else {
        load.settings.ai.keys = ApiKeys {
            openai: legacy.clone(),
            claude: legacy.clone(),
        };
        if let Err(e) = secrets::save_api_keys(data_dir, &load.settings.ai.keys) {
            log::warn!("Could not migrate the plaintext API key to encrypted storage: {e}");
            load.api_key_error = Some(e);
            return;
        }
    }
    if load.read_only {
        return;
    }
    match load.settings.save(data_dir) {
        Ok(()) => log::info!("Removed the plaintext API key from settings.json"),
        Err(e) => log::warn!("API key was encrypted but settings.json could not be rewritten: {e}"),
    }
}

#[cfg(test)]
mod volume_tests {
    use super::Settings;

    #[test]
    fn legacy_single_volume_resolves_to_quieter_start() {
        let s: Settings =
            serde_json::from_str(r#"{"hotkey":"Ctrl+Shift+Space","sound_volume":0.77}"#).unwrap();
        assert!(
            (s.start_volume() - 0.462).abs() < 1e-3,
            "start = 0.6 x master"
        );
        assert!((s.stop_volume() - 0.77).abs() < 1e-6, "stop = master");
    }

    #[test]
    fn explicit_volumes_round_trip_and_keep_master_for_older_builds() {
        let mut s = Settings::default();
        s.set_volumes(0.3, 0.9);
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert!((back.start_volume() - 0.3).abs() < 1e-6);
        assert!((back.stop_volume() - 0.9).abs() < 1e-6);
        assert!((back.sound_volume - 0.9).abs() < 1e-6);
    }

    #[test]
    fn volumes_are_clamped() {
        let mut s = Settings::default();
        s.set_volumes(-1.0, 7.0);
        assert_eq!(s.start_volume(), 0.0);
        assert_eq!(s.stop_volume(), 1.0);
    }
}

#[cfg(test)]
mod load_tests {
    use super::{AiSettingsUpdate, Settings};
    use crate::config::test_dir;
    use crate::formatting::AiProvider;
    use crate::transcription::engine::LanguageMode;

    fn write(dir: &std::path::Path, bytes: &[u8]) -> std::path::PathBuf {
        let path = Settings::file_path(dir);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn listing(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn valid_file_is_untouched_by_load() {
        let dir = test_dir("valid");
        let original = br#"{
  "hotkey": "Ctrl+Alt+Space",
  "language": "ru",
  "input_device": "Microphone (Wireless Mic Rx)"
}"#;
        let path = write(&dir, original);
        let load = Settings::load_with_report(&dir);
        assert!(load.error.is_none(), "{:?}", load.error);
        assert!(!load.read_only);
        assert_eq!(load.settings.hotkey, "Ctrl+Alt+Space");
        assert_eq!(load.settings.language, LanguageMode::Russian);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            original,
            "load must not rewrite the file"
        );
        assert_eq!(
            listing(&dir),
            vec!["settings.json"],
            "no temp or backup files"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_file_gives_defaults_without_error() {
        let dir = test_dir("missing");
        let load = Settings::load_with_report(&dir);
        assert!(load.error.is_none());
        assert!(!load.read_only);
        assert_eq!(load.settings.hotkey, "Ctrl+Shift+Space");
        assert!(listing(&dir).is_empty(), "load must not create files");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_hotkey_strings_fall_back_to_defaults_and_are_reported() {
        let dir = test_dir("bad-hotkey");
        write(
            &dir,
            br#"{"hotkey":"Ctrl+NotAKey","cancel_hotkey":"Nope+Nope"}"#,
        );
        let load = Settings::load_with_report(&dir);
        assert!(load.error.is_none(), "{:?}", load.error);
        assert_eq!(load.settings.hotkey, "Ctrl+Shift+Space");
        assert_eq!(load.settings.cancel_hotkey, "Ctrl+Shift+Backspace");
        assert_eq!(load.adjustments.len(), 2, "{:?}", load.adjustments);
        assert!(load.adjustments[0].contains("Ctrl+NotAKey"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bom_prefixed_file_parses() {
        let dir = test_dir("bom");
        write(&dir, b"\xEF\xBB\xBF{\"hotkey\":\"Ctrl+Shift+F9\"}");
        let load = Settings::load_with_report(&dir);
        assert!(load.error.is_none(), "{:?}", load.error);
        assert_eq!(load.settings.hotkey, "Ctrl+Shift+F9");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn utf16_file_is_quarantined_and_reported() {
        let dir = test_dir("utf16");
        let bytes = b"\xFF\xFE{\x00}\x00".to_vec();
        write(&dir, &bytes);
        let load = Settings::load_with_report(&dir);
        let error = load.error.expect("error reported");
        assert!(error.contains("UTF-16"), "{error}");
        assert!(!load.read_only, "defaults stay editable after a quarantine");
        let names = listing(&dir);
        assert_eq!(names.len(), 1, "{names:?}");
        assert!(names[0].starts_with("settings.json.corrupt-"), "{names:?}");
        assert_eq!(std::fs::read(dir.join(&names[0])).unwrap(), bytes);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn syntax_error_is_quarantined_with_original_bytes() {
        let dir = test_dir("syntax");
        let bytes = b"{\"hotkey\": \"Ctrl+Shift+Space\",}".to_vec();
        write(&dir, &bytes);
        let load = Settings::load_with_report(&dir);
        let error = load.error.expect("error reported");
        assert!(
            error.contains("line 1"),
            "serde position is surfaced: {error}"
        );
        let names = listing(&dir);
        assert_eq!(names.len(), 1, "{names:?}");
        assert!(names[0].starts_with("settings.json.corrupt-"));
        assert_eq!(std::fs::read(dir.join(&names[0])).unwrap(), bytes);
        assert_eq!(load.settings.hotkey, "Ctrl+Shift+Space");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_hotkey_uses_default() {
        let dir = test_dir("nohotkey");
        write(&dir, br#"{"model_file":"ggml-small.bin"}"#);
        let load = Settings::load_with_report(&dir);
        assert!(load.error.is_none(), "{:?}", load.error);
        assert_eq!(load.settings.hotkey, "Ctrl+Shift+Space");
        assert_eq!(load.settings.model_file, "ggml-small.bin");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unknown_enum_values_fall_back_and_keep_other_fields() {
        let dir = test_dir("enums");
        write(
            &dir,
            br#"{"hotkey":"Ctrl+Shift+F8","ai":{"provider":"local","openai_model":"gpt-x"},"language":"russian"}"#,
        );
        let load = Settings::load_with_report(&dir);
        assert!(load.error.is_none(), "{:?}", load.error);
        assert_eq!(load.settings.ai.provider, AiProvider::None);
        assert_eq!(load.settings.ai.openai_model, "gpt-x");
        assert_eq!(load.settings.language, LanguageMode::Auto);
        assert_eq!(load.settings.hotkey, "Ctrl+Shift+F8");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unknown_top_level_keys_are_reported() {
        let dir = test_dir("unknown");
        write(
            &dir,
            br#"{"hotkey":"Ctrl+Shift+Space","model":"ggml-small.bin"}"#,
        );
        let load = Settings::load_with_report(&dir);
        assert_eq!(load.unknown_keys, vec!["model".to_string()]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn apply_ai_update_without_key_fields_keeps_stored_keys() {
        let mut s = Settings::default();
        s.ai.keys.openai = "sk-openai".into();
        s.ai.keys.claude = "sk-ant".into();
        let change = s.apply_ai_update(&AiSettingsUpdate {
            provider: AiProvider::OpenAi,
            enabled: true,
            openai_model: "gpt-4o-mini".into(),
            claude_model: "claude-haiku-4-5".into(),
            prompt: "Format it".into(),
            openai_api_key: None,
            claude_api_key: None,
        });
        assert!(!change.keys_changed);
        assert!(change.settings_changed);
        assert_eq!(s.ai.keys.openai, "sk-openai");
        assert_eq!(s.ai.keys.claude, "sk-ant");
        assert_eq!(s.ai.api_key(), "sk-openai");
    }

    #[test]
    fn apply_ai_update_with_empty_key_removes_only_that_key() {
        let mut s = Settings::default();
        s.ai.keys.openai = "sk-openai".into();
        s.ai.keys.claude = "sk-ant".into();
        let change = s.apply_ai_update(&AiSettingsUpdate {
            provider: AiProvider::Claude,
            enabled: true,
            openai_model: s.ai.openai_model.clone(),
            claude_model: s.ai.claude_model.clone(),
            prompt: s.ai.prompt.clone(),
            openai_api_key: Some(String::new()),
            claude_api_key: Some("  sk-new  ".into()),
        });
        assert!(change.keys_changed);
        assert_eq!(s.ai.keys.openai, "");
        assert_eq!(s.ai.keys.claude, "sk-new", "keys are trimmed");
        assert_eq!(s.ai.api_key(), "sk-new");
    }

    #[test]
    fn blank_prompt_falls_back_to_default() {
        let mut s = Settings::default();
        s.apply_ai_update(&AiSettingsUpdate {
            provider: AiProvider::None,
            enabled: true,
            openai_model: s.ai.openai_model.clone(),
            claude_model: s.ai.claude_model.clone(),
            prompt: "   ".into(),
            openai_api_key: None,
            claude_api_key: None,
        });
        assert_eq!(s.ai.prompt, crate::formatting::default_prompt());
    }

    #[test]
    fn read_only_settings_refuse_to_save() {
        let dir = test_dir("readonly");
        let mut s = Settings::default();
        s.read_only = true;
        assert!(s.save(&dir).is_err());
        assert!(listing(&dir).is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
