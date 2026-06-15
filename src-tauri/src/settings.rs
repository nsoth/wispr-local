use crate::formatting::AiSettings;
use crate::transcription::engine::LanguageMode;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey: String,
    #[serde(default)]
    pub start_sound: String,
    #[serde(default)]
    pub stop_sound: String,
    #[serde(default = "default_volume")]
    pub sound_volume: f32,
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
}

fn default_volume() -> f32 {
    0.5
}

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
            ai: AiSettings::default(),
            run_on_startup: false,
            show_overlay: true,
            model_file: default_model_file(),
            language: LanguageMode::default(),
        }
    }
}

impl Settings {
    pub fn file_path(data_dir: &PathBuf) -> PathBuf {
        data_dir.join("settings.json")
    }

    pub fn load(data_dir: &PathBuf) -> Self {
        let path = Self::file_path(data_dir);
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str(&contents) {
                    Ok(settings) => return settings,
                    Err(e) => log::warn!("Failed to parse settings: {}, using defaults", e),
                },
                Err(e) => log::warn!("Failed to read settings: {}, using defaults", e),
            }
        }
        Self::default()
    }

    pub fn save(&self, data_dir: &PathBuf) -> Result<(), String> {
        let path = Self::file_path(data_dir);
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, json).map_err(|e| e.to_string())?;
        Ok(())
    }
}
