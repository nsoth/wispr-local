//! In-memory application state shared between the hotkey handler, the
//! pipeline and the IPC commands, plus the on-disk transcription history.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Keep this many recent transcriptions for the in-app history.
pub const HISTORY_LIMIT: usize = 5;

/// Pipeline state as seen by the UI. Serializes as `{"state": "idle"}` or
/// `{"state": "error", "message": "..."}` so both webviews branch on `state`
/// instead of parsing free-form strings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "message", rename_all = "lowercase")]
pub enum AppStatus {
    #[default]
    Idle,
    Recording,
    Transcribing,
    Formatting,
    Injecting,
    Error(String),
}

/// Problems found while loading state files at startup. Shown in the main
/// window so a quarantined settings file is never a silent surprise.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StartupDiagnostics {
    pub settings_error: Option<String>,
    /// True when settings.json could not be read (as opposed to parsed): the
    /// in-memory defaults must not be written back over a file we never saw.
    pub settings_read_only: bool,
    pub unknown_settings_keys: Vec<String>,
    pub history_error: Option<String>,
    pub api_key_error: Option<String>,
}

pub struct AppState {
    pub status: AppStatus,
    pub model_loaded: bool,
    /// "CUDA" or "CPU" once the model finishes loading; empty until then.
    /// Mirrored here from the engine so UI queries never block on the engine
    /// mutex (held for the full duration of a transcription).
    pub compute_backend: String,
    pub last_transcription: String,
    pub device_sample_rate: u32,
    /// True while a hands-free (pinned) recording is active — the hotkey
    /// release is ignored and the next press/pin-click stops the recording.
    pub recording_locked: bool,
    /// Most recent transcriptions, newest first, capped at HISTORY_LIMIT.
    pub history: Vec<String>,
    pub diagnostics: StartupDiagnostics,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            status: AppStatus::Idle,
            model_loaded: false,
            compute_backend: String::new(),
            last_transcription: String::new(),
            device_sample_rate: 48000,
            recording_locked: false,
            history: Vec::new(),
            diagnostics: StartupDiagnostics::default(),
        }
    }
}

impl AppState {
    /// Prepend a transcription to the history, keeping it deduplicated
    /// against the most recent entry and capped at HISTORY_LIMIT. Returns
    /// whether the history changed (so callers can skip a disk write).
    pub fn push_history(&mut self, text: &str) -> bool {
        if self.history.first().map(|s| s.as_str()) == Some(text) {
            return false;
        }
        self.history.insert(0, text.to_string());
        self.history.truncate(HISTORY_LIMIT);
        true
    }
}

fn history_path(data_dir: &Path) -> PathBuf {
    data_dir.join("history.json")
}

/// Load the history; a corrupt file is moved aside (never overwritten) and the
/// problem is returned for the startup diagnostics banner.
pub fn load_history_with_report(data_dir: &Path) -> (Vec<String>, Option<String>) {
    let path = history_path(data_dir);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (Vec::new(), None),
        Err(e) => {
            let message = format!("history.json could not be read: {e}");
            log::warn!("{message}");
            return (Vec::new(), Some(message));
        }
    };
    match serde_json::from_slice::<Vec<String>>(&bytes) {
        Ok(mut history) => {
            history.retain(|item| !item.trim().is_empty());
            history.truncate(HISTORY_LIMIT);
            (history, None)
        }
        Err(e) => {
            let moved = crate::config::quarantine_file(&path);
            let message = match moved {
                Some(p) => format!(
                    "history.json was unreadable ({e}); it was moved to {}",
                    p.file_name().unwrap_or_default().to_string_lossy()
                ),
                None => format!("history.json is unreadable ({e})"),
            };
            log::warn!("{message}");
            (Vec::new(), Some(message))
        }
    }
}

pub fn load_history(data_dir: &Path) -> Vec<String> {
    load_history_with_report(data_dir).0
}

pub fn save_history(data_dir: &Path, history: &[String]) -> Result<(), String> {
    let path = history_path(data_dir);
    let json = serde_json::to_string_pretty(history).map_err(|e| e.to_string())?;
    crate::config::write_file_atomic(&path, json.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::{load_history_with_report, AppState, AppStatus};
    use crate::config::test_dir;
    use serde_json::json;

    #[test]
    fn status_serializes_as_tagged_state_object() {
        assert_eq!(
            serde_json::to_value(AppStatus::Idle).unwrap(),
            json!({ "state": "idle" })
        );
        assert_eq!(
            serde_json::to_value(AppStatus::Transcribing).unwrap(),
            json!({ "state": "transcribing" })
        );
        assert_eq!(
            serde_json::to_value(AppStatus::Error("mic gone".to_string())).unwrap(),
            json!({ "state": "error", "message": "mic gone" })
        );
    }

    #[test]
    fn corrupt_history_is_quarantined_and_reported() {
        let dir = test_dir("history");
        std::fs::write(dir.join("history.json"), b"[\"one\",").unwrap();
        let (history, error) = load_history_with_report(&dir);
        assert!(history.is_empty());
        assert!(error.is_some());
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names.len(), 1);
        assert!(names[0].starts_with("history.json.corrupt-"), "{names:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_history_is_not_an_error() {
        let dir = test_dir("nohistory");
        let (history, error) = load_history_with_report(&dir);
        assert!(history.is_empty());
        assert!(error.is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn push_history_reports_whether_anything_changed() {
        let mut s = AppState::default();
        assert!(s.push_history("first"));
        assert!(!s.push_history("first"), "duplicate of the newest entry");
        assert!(s.push_history("second"));
        assert_eq!(s.history, vec!["second".to_string(), "first".to_string()]);
    }
}
