use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Keep this many recent transcriptions for the in-app history.
pub const HISTORY_LIMIT: usize = 5;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum AppStatus {
    #[default]
    Idle,
    Recording,
    Transcribing,
    Formatting,
    Injecting,
    Error(String),
}

pub struct AppState {
    pub status: AppStatus,
    pub model_loaded: bool,
    pub last_transcription: String,
    pub device_sample_rate: u32,
    /// True while a hands-free (pinned) recording is active — the hotkey
    /// release is ignored and the next press/pin-click stops the recording.
    pub recording_locked: bool,
    /// Most recent transcriptions, newest first, capped at HISTORY_LIMIT.
    pub history: Vec<String>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            status: AppStatus::Idle,
            model_loaded: false,
            last_transcription: String::new(),
            device_sample_rate: 48000,
            recording_locked: false,
            history: Vec::new(),
        }
    }
}

impl AppState {
    /// Prepend a transcription to the history, keeping it deduplicated
    /// against the most recent entry and capped at HISTORY_LIMIT.
    pub fn push_history(&mut self, text: &str) {
        if self.history.first().map(|s| s.as_str()) == Some(text) {
            return;
        }
        self.history.insert(0, text.to_string());
        self.history.truncate(HISTORY_LIMIT);
    }
}

fn history_path(data_dir: &Path) -> PathBuf {
    data_dir.join("history.json")
}

pub fn load_history(data_dir: &Path) -> Vec<String> {
    let path = history_path(data_dir);
    if let Ok(contents) = std::fs::read_to_string(&path) {
        if let Ok(mut history) = serde_json::from_str::<Vec<String>>(&contents) {
            history.retain(|item| !item.trim().is_empty());
            history.truncate(HISTORY_LIMIT);
            return history;
        }
    }
    Vec::new()
}

pub fn save_history(data_dir: &Path, history: &[String]) -> Result<(), String> {
    let path = history_path(data_dir);
    let json = serde_json::to_string_pretty(history).map_err(|e| e.to_string())?;
    crate::config::write_file_atomic(&path, json.as_bytes())
}
