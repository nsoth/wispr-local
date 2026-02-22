use crate::formatting::AiSettings;
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
}

fn default_volume() -> f32 {
    0.5
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: "Ctrl+Shift+Space".to_string(),
            start_sound: String::new(),
            stop_sound: String::new(),
            sound_volume: default_volume(),
            ai: AiSettings::default(),
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
