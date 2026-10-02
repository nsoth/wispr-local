use crate::formatting::AiSettings;
use crate::transcription::engine::LanguageMode;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey: String,
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
            hotkey: "Ctrl+Shift+Space".to_string(),
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
        }
    }
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

    pub fn sound_config(&self) -> crate::system::sounds::SoundConfig {
        crate::system::sounds::SoundConfig {
            start_sound: self.start_sound.clone(),
            stop_sound: self.stop_sound.clone(),
            start_volume: self.start_volume(),
            stop_volume: self.stop_volume(),
        }
    }

    pub fn file_path(data_dir: &Path) -> PathBuf {
        data_dir.join("settings.json")
    }

    pub fn load(data_dir: &Path) -> Self {
        let path = Self::file_path(data_dir);
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str::<Settings>(&contents) {
                    Ok(mut settings) => {
                        // Normalize values from older or manually edited files.
                        settings.sound_volume = settings.sound_volume.clamp(0.0, 1.0);
                        if settings.model_file.trim().is_empty() {
                            settings.model_file = default_model_file();
                        }
                        let legacy_key = settings.ai.api_key.clone();
                        match crate::secrets::load_api_key(data_dir) {
                            Ok(Some(key)) => settings.ai.api_key = key,
                            Ok(None) if !legacy_key.is_empty() => {
                                match crate::secrets::save_api_key(data_dir, &legacy_key) {
                                    Ok(()) => {
                                        // `api_key` is skipped during serialization, removing
                                        // the legacy plaintext value after successful migration.
                                        if let Err(e) = settings.save(data_dir) {
                                            log::warn!(
                                                "API key was encrypted but plaintext settings migration failed: {e}"
                                            );
                                        }
                                    }
                                    Err(e) => log::warn!(
                                        "Could not migrate API key to encrypted storage: {e}"
                                    ),
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                log::warn!("Could not load encrypted API key: {e}");
                                settings.ai.api_key = legacy_key;
                            }
                        }
                        return settings;
                    }
                    Err(e) => log::warn!("Failed to parse settings: {}, using defaults", e),
                },
                Err(e) => log::warn!("Failed to read settings: {}, using defaults", e),
            }
        }
        Self::default()
    }

    pub fn save(&self, data_dir: &Path) -> Result<(), String> {
        if !self.ai.api_key.is_empty() {
            let securely_stored = crate::secrets::load_api_key(data_dir)
                .ok()
                .flatten()
                .is_some_and(|key| key == self.ai.api_key);
            if !securely_stored {
                // Do not rewrite a legacy settings file without first securing
                // its key; otherwise a temporary DPAPI failure could erase the
                // only recoverable copy on the next unrelated settings change.
                crate::secrets::save_api_key(data_dir, &self.ai.api_key)?;
            }
        }
        let path = Self::file_path(data_dir);
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        crate::config::write_file_atomic(&path, json.as_bytes())
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
