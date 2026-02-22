use serde::{Deserialize, Serialize};

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
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            status: AppStatus::Idle,
            model_loaded: false,
            last_transcription: String::new(),
            device_sample_rate: 48000,
        }
    }
}
