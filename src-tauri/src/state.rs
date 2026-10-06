//! In-memory application state shared between the hotkey handler, the
//! pipeline and the IPC commands, plus the on-disk transcription history.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard};

/// Default number of recent transcriptions kept in the in-app history.
pub const HISTORY_LIMIT: usize = 100;
/// Hard cap regardless of settings.
pub const HISTORY_MAX: usize = 500;

fn default_true() -> bool {
    true
}

/// One dictation in the history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub text: String,
    /// Unix time in milliseconds (0 for entries migrated from the old format).
    #[serde(default)]
    pub ts: u64,
    /// Executable or title of the window the text was pasted into.
    #[serde(default)]
    pub target: String,
    /// "ru" / "en".
    #[serde(default)]
    pub lang: String,
    #[serde(default)]
    pub duration_s: f32,
    /// False when the paste was skipped (focus moved, cancelled) and the text
    /// only lives here.
    #[serde(default = "default_true")]
    pub pasted: bool,
    /// Window handle of the paste target in this session (not persisted).
    #[serde(skip)]
    pub hwnd: isize,
}

impl HistoryEntry {
    pub fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

/// history.json before 2026-10 held a bare list of strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum HistoryFile {
    Entries(Vec<HistoryEntry>),
    Legacy(Vec<String>),
}

/// Pipeline state as seen by the UI. Serializes as `{"state": "idle"}` or
/// `{"state": "error", "code": "mic", "message": "..."}` so both webviews
/// branch on `state` (and `code`) instead of parsing free-form strings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum AppStatus {
    #[default]
    Idle,
    Recording,
    Transcribing,
    Formatting,
    Injecting,
    Error {
        /// Machine-readable class: `mic` for a microphone that failed to open.
        code: String,
        message: String,
    },
}

impl AppStatus {
    pub fn error(code: &str, message: impl Into<String>) -> Self {
        AppStatus::Error {
            code: code.to_string(),
            message: message.into(),
        }
    }
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
    /// Settings values replaced by defaults because they could not be used.
    pub settings_adjustments: Vec<String>,
    pub history_error: Option<String>,
    pub api_key_error: Option<String>,
}

/// Where the Whisper model stands. Mirrored here from the engine so UI
/// queries never block on the engine mutex (held for the full duration of a
/// transcription and of a load). Serialized for `get_model_state` and the
/// `model-state-changed` event.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum ModelState {
    Loading,
    Ready {
        /// "CUDA" or "CPU".
        backend: String,
        /// File name inside the models directory.
        file: String,
        /// True when the configured model was unavailable and another one
        /// was loaded instead.
        fallback: bool,
    },
    Missing,
    Failed {
        error: String,
    },
}

impl ModelState {
    pub fn is_ready(&self) -> bool {
        matches!(self, ModelState::Ready { .. })
    }

    pub fn backend(&self) -> &str {
        match self {
            ModelState::Ready { backend, .. } => backend,
            _ => "",
        }
    }
}

/// Lock a mutex even if a previous holder panicked: the protected state is
/// plain data that is always left consistent, and a poisoned lock must not
/// turn one panic into a permanently dead dictation pipeline.
pub fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Lock order, everywhere: `Settings` before `AppState` (and either before the
/// engine). A path that takes them the other way round can deadlock against
/// the stop flow and the AI/language commands.
pub struct AppState {
    pub status: AppStatus,
    pub model: ModelState,
    pub last_transcription: String,
    pub device_sample_rate: u32,
    /// True while a hands-free (pinned) recording is active — the hotkey
    /// release is ignored and the next press/pin-click stops the recording.
    pub recording_locked: bool,
    /// Most recent transcriptions, newest first, capped at HISTORY_LIMIT.
    pub history: Vec<HistoryEntry>,
    pub diagnostics: StartupDiagnostics,
    /// The fallback microphone that was last announced, so the toast fires
    /// once per device instead of on every recording.
    pub last_fallback_device: Option<String>,
    /// Set by the stop flow so an in-flight preview tick aborts and releases
    /// the engine to the final transcription; cleared when a recording starts.
    pub preview_abort: Arc<AtomicBool>,
    /// "ru" / "en" for the current utterance once known (pinned or detected).
    pub detected_language: String,
    /// When the active recording started (for tap-vs-hold detection).
    pub recording_started_at: Option<std::time::Instant>,
    /// Whether the hotkey is believed to be physically down.
    pub key_down: bool,
    /// Tray → "Pause hotkey": events are ignored while set.
    pub hotkey_paused: bool,
    /// The Settings page is recording a new combination: ignore the live one.
    pub capturing_hotkey: bool,
    /// Set by a cancel request while transcribing/formatting: skip the paste.
    pub cancel_requested: Arc<AtomicBool>,
    /// Set when the recording was interrupted by sleep or a locked session:
    /// the text goes to history and clipboard, never auto-pasted.
    pub suppress_paste: Arc<AtomicBool>,
    /// Writer thread spooling the active recording to disk.
    pub spool: Option<crate::audio::spool::SpoolWriter>,
    /// Set by the tray's "Stop and paste": the menu itself had focus, so the
    /// stop flow pastes into whichever foreign window is focused when the
    /// text is ready instead of insisting on the origin window.
    pub tray_stop_pending: bool,
    /// Wall-clock start of the active recording; compared with the captured
    /// audio length at stop to notice a sleep (no samples while suspended).
    pub recording_started_wall: Option<std::time::SystemTime>,
    /// Input device of the active (or last) recording, as Windows names it.
    pub recording_device: String,
    /// Whether that device was a fallback for an unavailable preferred one.
    pub recording_device_fallback: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            status: AppStatus::Idle,
            model: ModelState::Loading,
            last_transcription: String::new(),
            device_sample_rate: 48000,
            recording_locked: false,
            history: Vec::new(),
            diagnostics: StartupDiagnostics::default(),
            last_fallback_device: None,
            preview_abort: Arc::new(AtomicBool::new(false)),
            detected_language: String::new(),
            recording_started_at: None,
            key_down: false,
            hotkey_paused: false,
            capturing_hotkey: false,
            cancel_requested: Arc::new(AtomicBool::new(false)),
            suppress_paste: Arc::new(AtomicBool::new(false)),
            spool: None,
            tray_stop_pending: false,
            recording_started_wall: None,
            recording_device: String::new(),
            recording_device_fallback: false,
        }
    }
}

impl AppState {
    /// Prepend a transcription to the history, keeping it deduplicated
    /// against the most recent entry and capped at HISTORY_LIMIT. Returns
    /// whether the history changed (so callers can skip a disk write).
    pub fn push_history(&mut self, entry: HistoryEntry, limit: usize) -> bool {
        let limit = limit.min(HISTORY_MAX);
        if limit == 0 {
            let had_entries = !self.history.is_empty();
            self.history.clear();
            return had_entries;
        }
        if self.history.first().map(|e| e.text.as_str()) == Some(entry.text.as_str()) {
            return false;
        }
        self.history.insert(0, entry);
        self.history.truncate(limit);
        true
    }
}

fn history_path(data_dir: &Path) -> PathBuf {
    data_dir.join("history.json")
}

/// Load the history; a corrupt file is moved aside (never overwritten) and the
/// problem is returned for the startup diagnostics banner.
pub fn load_history_with_report(data_dir: &Path) -> (Vec<HistoryEntry>, Option<String>) {
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
    match serde_json::from_slice::<HistoryFile>(&bytes) {
        Ok(file) => {
            let mut history = match file {
                HistoryFile::Entries(entries) => entries,
                HistoryFile::Legacy(texts) => texts
                    .into_iter()
                    .map(|text| HistoryEntry {
                        text,
                        ts: 0,
                        target: String::new(),
                        lang: String::new(),
                        duration_s: 0.0,
                        pasted: true,
                        hwnd: 0,
                    })
                    .collect(),
            };
            history.retain(|item| !item.text.trim().is_empty());
            history.truncate(HISTORY_MAX);
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

pub fn load_history(data_dir: &Path) -> Vec<HistoryEntry> {
    load_history_with_report(data_dir).0
}

pub fn save_history(data_dir: &Path, history: &[HistoryEntry]) -> Result<(), String> {
    let path = history_path(data_dir);
    let json = serde_json::to_string_pretty(history).map_err(|e| e.to_string())?;
    crate::config::write_file_atomic(&path, json.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::{
        load_history_with_report, lock_or_recover, AppState, AppStatus, HistoryEntry, ModelState,
    };

    fn entry(text: &str) -> HistoryEntry {
        HistoryEntry {
            text: text.into(),
            ts: 1,
            target: "Telegram.exe".into(),
            lang: "ru".into(),
            duration_s: 2.5,
            pasted: true,
            hwnd: 0,
        }
    }

    #[test]
    fn legacy_string_history_is_migrated() {
        let dir = test_dir("legacy-history");
        std::fs::write(dir.join("history.json"), br#"["one", "two", "  "]"#).unwrap();
        let (history, error) = load_history_with_report(&dir);
        assert!(error.is_none());
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].text, "one");
        assert_eq!(history[0].ts, 0);
        assert!(history[0].pasted);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn entries_round_trip_through_the_file() {
        let dir = test_dir("entry-history");
        let entries = vec![entry("first"), entry("second")];
        super::save_history(&dir, &entries).unwrap();
        let (loaded, error) = load_history_with_report(&dir);
        assert!(error.is_none());
        assert_eq!(loaded, entries);
        std::fs::remove_dir_all(dir).unwrap();
    }
    use crate::config::test_dir;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    #[test]
    fn model_state_serializes_with_state_tag() {
        assert_eq!(
            serde_json::to_value(ModelState::Loading).unwrap(),
            json!({ "state": "loading" })
        );
        assert_eq!(
            serde_json::to_value(ModelState::Ready {
                backend: "CUDA".into(),
                file: "ggml-large-v3-turbo.bin".into(),
                fallback: false,
            })
            .unwrap(),
            json!({ "state": "ready", "backend": "CUDA", "file": "ggml-large-v3-turbo.bin", "fallback": false })
        );
        assert_eq!(
            serde_json::to_value(ModelState::Failed {
                error: "boom".into()
            })
            .unwrap(),
            json!({ "state": "failed", "error": "boom" })
        );
    }

    #[test]
    fn lock_or_recover_survives_a_poisoned_mutex() {
        let shared = Arc::new(Mutex::new(5));
        let poisoner = Arc::clone(&shared);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert!(shared.lock().is_err(), "mutex is poisoned");
        assert_eq!(*lock_or_recover(&shared), 5);
    }

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
            serde_json::to_value(AppStatus::error("mic", "mic gone")).unwrap(),
            json!({ "state": "error", "code": "mic", "message": "mic gone" })
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
        assert!(s.push_history(entry("first"), 100));
        assert!(
            !s.push_history(entry("first"), 100),
            "duplicate of the newest entry"
        );
        assert!(s.push_history(entry("second"), 100));
        let texts: Vec<&str> = s.history.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(texts, vec!["second", "first"]);
    }

    #[test]
    fn history_is_capped() {
        let mut s = AppState::default();
        for i in 0..(super::HISTORY_LIMIT + 10) {
            s.push_history(entry(&format!("item {i}")), super::HISTORY_LIMIT);
        }
        assert_eq!(s.history.len(), super::HISTORY_LIMIT);
        assert_eq!(
            s.history[0].text,
            format!("item {}", super::HISTORY_LIMIT + 9)
        );
        assert!(s.push_history(entry("gone"), 0), "limit 0 clears");
        assert!(s.history.is_empty());
        assert!(!s.push_history(entry("still gone"), 0));
    }
}
