use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Keep this many recent transcriptions for the in-app history.
pub const HISTORY_LIMIT: usize = 5;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AppStatus {
    Idle,
    Recording,
    Transcribing,
    Formatting,
    Injecting,
    Error(String),
}

impl Default for AppStatus {
    fn default() -> Self {
        AppStatus::Idle
    }
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

fn history_path(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("history.json")
}

pub fn load_history(data_dir: &PathBuf) -> Vec<String> {
    let path = history_path(data_dir);
    if let Ok(contents) = std::fs::read_to_string(&path) {
        if let Ok(history) = serde_json::from_str::<Vec<String>>(&contents) {
            return history;
        }
    }
    Vec::new()
}

pub fn save_history(data_dir: &PathBuf, history: &[String]) {
    let path = history_path(data_dir);
    match serde_json::to_string_pretty(history) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                log::warn!("Failed to save history: {}", e);
            }
        }
        Err(e) => log::warn!("Failed to serialize history: {}", e),
    }
}
