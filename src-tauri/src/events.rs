//! Names of the events exchanged between the Rust backend and the two
//! webviews (main window and overlay). Mirrored in `src/ipc.ts`; keep both
//! lists in sync when adding an event.

/// Payload: serialized [`crate::state::AppStatus`] (`{"state": "...", "message"?: "..."}`).
pub const STATUS_CHANGED: &str = "status-changed";
/// Payload: `bool` — whether the active recording is pinned (hands-free).
pub const LOCK_CHANGED: &str = "lock-changed";
/// Payload: `f32` RMS level scaled for the overlay waveform, ~20 Hz while recording.
pub const AUDIO_LEVEL: &str = "audio-level";
/// Payload: `String` — partial transcript of the last ~10 s while recording.
pub const STREAMING_PREVIEW: &str = "streaming-preview";
/// Payload: reason string (`too-short`, `no-speech`, `error`).
pub const TRANSCRIPTION_EMPTY: &str = "transcription-empty";
/// Payload: final text that was pasted (or copied) for this utterance.
pub const TRANSCRIPTION_COMPLETE: &str = "transcription-complete";
/// Payload: `String` — short human-readable notice for the main window banner.
pub const OPERATION_NOTICE: &str = "operation-notice";
/// Payload: `Vec<String>` — the whole history, newest first.
pub const HISTORY_CHANGED: &str = "history-changed";
/// Payload: serialized [`crate::state::ModelState`].
pub const MODEL_STATE_CHANGED: &str = "model-state-changed";
/// Payload: [`crate::pipeline::OverlayState`] — what the pill shows.
pub const OVERLAY_STATE: &str = "overlay-state";
/// Payload: `{ language: "ru" | "en", source: "auto" | "pinned" }`.
pub const LANGUAGE_DETECTED: &str = "language-detected";

/// Internal (Rust → Rust) requests raised by the hotkey, tray and overlay.
pub const REQUEST_START_RECORDING: &str = "request-start-recording";
pub const REQUEST_STOP_RECORDING: &str = "request-stop-recording";
/// Discard the active recording (or skip the paste of the one in progress).
pub const REQUEST_CANCEL_RECORDING: &str = "request-cancel-recording";
/// Emitted once by the capture callback when the 30-minute cap is reached.
pub const RECORDING_LIMIT_REACHED: &str = "recording-limit-reached";
/// Emitted once by the capture error callback (device disconnected etc.).
pub const AUDIO_STREAM_ERROR: &str = "audio-stream-error";
