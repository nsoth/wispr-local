//! The dictation pipeline: start capture → streaming preview → stop, transcribe,
//! post-process, optional AI formatting, paste → history. Also owns the
//! background model loader and the user-notification helpers.
//!
//! Status changes go through [`set_status`] so the UI always receives the
//! same serialized [`AppStatus`] payload that `get_status` returns. The
//! overlay pill follows [`OverlayState`] events: it stays visible from the
//! hotkey press until the outcome (pasted / no speech / failed) has been
//! shown, instead of vanishing the moment the key is released.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{Emitter, Manager};

use crate::audio::buffer::AudioBuffer;
use crate::audio::capture::AudioCapture;
use crate::audio::devices;
use crate::audio::spool::{self, SpoolWriter};
use crate::config::AppConfig;
use crate::events;
use crate::overlay::{hide_overlay, show_overlay_if_enabled};
use crate::settings::{self, Settings};
use crate::state::{self, lock_or_recover, AppState, AppStatus, HistoryEntry, ModelState};
use crate::stats;
use crate::system::focus::{self, PasteDecision};
use crate::system::sounds::{SoundKind, SoundPlayer};
use crate::system::tray::{TrayAnimator, TrayPhase};
use crate::text::apply_paste_suffix;
use crate::transcription::engine::{
    has_speech_dynamics, speech_stats, LanguageMode, SpeechStats, TranscribeOptions, WhisperEngine,
};
use crate::watchdog::{Verdict, Watchdog};
use crate::{formatting, system};

/// Counts finished recordings for the per-utterance log line.
static UTTERANCES: AtomicU64 = AtomicU64::new(0);

/// Record the new status in [`AppState`] and tell every webview about it.
pub fn set_status(app: &tauri::AppHandle, status: AppStatus) {
    {
        let state = app.state::<Mutex<AppState>>();
        lock_or_recover(&state).status = status.clone();
    }
    emit_status(app, &status);
}

/// Broadcast a status that was already stored (e.g. under an existing lock).
pub fn emit_status(app: &tauri::AppHandle, status: &AppStatus) {
    let _ = app.emit(events::STATUS_CHANGED, status);
    system::tray::refresh(app);
}

/// What the overlay pill shows. `phase` drives the layout (waveform vs text),
/// `tone` the colour of a result, `language` the RU/EN badge.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OverlayState {
    /// "recording" | "processing" | "result"
    pub phase: &'static str,
    pub message: String,
    /// "" | "ok" | "warn" | "error"
    pub tone: &'static str,
    /// "ru" | "en" | "" (not known yet)
    pub language: String,
}

pub fn emit_overlay_state(
    app: &tauri::AppHandle,
    phase: &'static str,
    message: &str,
    tone: &'static str,
) {
    let language = {
        let state = app.state::<Mutex<AppState>>();
        let s = lock_or_recover(&state);
        s.detected_language.clone()
    };
    let _ = app.emit(
        events::OVERLAY_STATE,
        OverlayState {
            phase,
            message: message.to_string(),
            tone,
            language,
        },
    );
}

/// Remember the language used for the current utterance and tell both
/// webviews (badge in the overlay, suffix in the main window status).
pub fn announce_language(app: &tauri::AppHandle, language: &str, source: &'static str) {
    {
        let state = app.state::<Mutex<AppState>>();
        lock_or_recover(&state).detected_language = language.to_string();
    }
    let _ = app.emit(
        events::LANGUAGE_DETECTED,
        serde_json::json!({ "language": language, "source": source }),
    );
}

/// Runs a closure when dropped unless disarmed: the stop flow arms one right
/// after claiming the pipeline so a panic anywhere in the flow still returns
/// the app to Idle instead of leaving a zombie "Transcribing" state.
pub struct ArmedGuard<F: FnOnce()> {
    on_drop: Option<F>,
}

impl<F: FnOnce()> ArmedGuard<F> {
    pub fn new(on_drop: F) -> Self {
        Self {
            on_drop: Some(on_drop),
        }
    }

    pub fn disarm(&mut self) {
        self.on_drop = None;
    }
}

impl<F: FnOnce()> Drop for ArmedGuard<F> {
    fn drop(&mut self) {
        if let Some(f) = self.on_drop.take() {
            f();
        }
    }
}

/// `pinned`: keep recording after the hotkey is released (tray / window
/// start buttons have no key to hold).
pub fn start_recording_flow(app: &tauri::AppHandle, pinned: bool) {
    log::debug!("start_recording_flow called (pinned={pinned})");
    let state = app.state::<Mutex<AppState>>();
    let capture = app.state::<Mutex<AudioCapture>>();
    let buffer = app.state::<AudioBuffer>();

    let model = {
        let mut s = lock_or_recover(&state);
        if !matches!(s.status, AppStatus::Idle | AppStatus::Error { .. }) {
            if s.status != AppStatus::Recording {
                // The previous recording is still being transcribed or pasted.
                // Say so instead of silently eating the press (the user holds
                // the key and talks into nothing otherwise).
                log::info!("Recording request while busy ({:?}): signalling", s.status);
                drop(s);
                app.state::<SoundPlayer>().play(SoundKind::Busy);
                let _ = app.emit(
                    events::OPERATION_NOTICE,
                    "Still processing the previous recording; try again in a moment",
                );
            }
            return;
        }
        match &s.model {
            ModelState::Ready { .. } | ModelState::Loading => {
                s.status = AppStatus::Recording;
                s.recording_locked = pinned;
                s.detected_language.clear();
                s.recording_started_at = Some(Instant::now());
                s.recording_started_wall = Some(std::time::SystemTime::now());
                s.tray_stop_pending = false;
                s.preview_abort.store(false, Ordering::Relaxed);
                s.cancel_requested.store(false, Ordering::Relaxed);
                s.suppress_paste.store(false, Ordering::Relaxed);
            }
            ModelState::Missing | ModelState::Failed { .. } => {}
        }
        s.model.clone()
    };

    match &model {
        ModelState::Ready { .. } => {}
        ModelState::Loading => {
            // Capture does not need the model; the final transcription waits
            // for the engine lock that the loader holds.
            log::info!("Recording while the model is still loading");
            let _ = app.emit(
                events::OPERATION_NOTICE,
                "Model is still loading; this recording is transcribed when it is ready",
            );
        }
        ModelState::Missing => {
            log::warn!("Recording refused: no Whisper model found");
            notify_user(app, "No Whisper model found. Open Wispr Local to add one.");
            let _ = app.emit(events::OPERATION_NOTICE, "No Whisper model found");
            return;
        }
        ModelState::Failed { error } => {
            log::warn!("Recording refused: model failed to load ({error})");
            notify_user(
                app,
                "The Whisper model failed to load. Open Wispr Local to retry.",
            );
            let _ = app.emit(
                events::OPERATION_NOTICE,
                format!("Model failed to load: {error}"),
            );
            return;
        }
    }

    buffer.clear();

    let (preferred_device, language_mode) = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        (guard.input_device.clone(), guard.language)
    };
    // A pinned language is known before any audio arrives.
    match language_mode {
        LanguageMode::Russian => announce_language(app, "ru", "pinned"),
        LanguageMode::English => announce_language(app, "en", "pinned"),
        _ => {}
    }

    let start_result = {
        let mut cap = lock_or_recover(&capture);
        cap.start(
            Some(app.clone()),
            (!preferred_device.is_empty()).then_some(preferred_device.as_str()),
        )
    };
    match start_result {
        Ok(start) => {
            {
                let mut s = lock_or_recover(&state);
                s.device_sample_rate = start.sample_rate;
                s.recording_device = start.device_name.clone();
                s.recording_device_fallback = start.used_fallback;
            }
            log::info!(
                "Recording started at {} Hz, {} ch, input device {}",
                start.sample_rate,
                start.channels,
                start.device_name
            );
            if start.used_fallback {
                log::warn!(
                    "Preferred microphone unavailable ({}); using {}",
                    start.fallback_reason.as_deref().unwrap_or("not found"),
                    start.device_name
                );
                // Announce once per fallback device, not on every recording.
                let announce = {
                    let mut s = lock_or_recover(&state);
                    let yes = devices::should_announce_fallback(
                        s.last_fallback_device.as_deref(),
                        &start.device_name,
                    );
                    if yes {
                        s.last_fallback_device = Some(start.device_name.clone());
                    }
                    yes
                };
                if announce {
                    let message = format!(
                        "Selected microphone is unavailable; using {} until it is back.",
                        start.device_name
                    );
                    let _ = app.emit(events::OPERATION_NOTICE, &message);
                    notify_user(app, &message);
                }
            } else {
                lock_or_recover(&state).last_fallback_device = None;
            }
        }
        Err(e) => {
            log::error!("Failed to start recording: {}", e);
            set_status(app, AppStatus::error("mic", e.clone()));
            let message = format!("Microphone error: {e}");
            let _ = app.emit(events::OPERATION_NOTICE, &message);
            notify_user(app, &message);
            app.state::<TrayAnimator>().set_phase(TrayPhase::Idle);
            hide_overlay(app);
            return;
        }
    }

    // A release that arrived while the device was opening has already claimed
    // the stop; do not start the indicators for a recording that is over.
    if lock_or_recover(&state).status != AppStatus::Recording {
        log::info!("Recording was stopped while the microphone was opening");
        return;
    }

    emit_status(app, &AppStatus::Recording);
    let _ = app.emit(events::LOCK_CHANGED, pinned);
    app.state::<SoundPlayer>().play_start();

    // Crash insurance: spool the audio to disk while recording.
    let spool_path = spool::spool_path(&app.state::<AppConfig>().data_dir);
    match SpoolWriter::start(spool_path, app.state::<AudioBuffer>().inner().clone()) {
        Ok(writer) => lock_or_recover(&state).spool = Some(writer),
        Err(e) => log::warn!("{e}; recording continues without crash insurance"),
    }
    let guard_app = app.clone();
    tauri::async_runtime::spawn(async move {
        recording_guard_loop(guard_app).await;
    });

    // Kick off tray animation and reveal overlay (if user hasn't disabled it).
    app.state::<TrayAnimator>().set_phase(TrayPhase::Recording);
    show_overlay_if_enabled(app);
    // A fallback microphone is named in the pill for the whole recording:
    // five minutes into the wrong device is what this prevents.
    let fallback_label = {
        let s = lock_or_recover(&state);
        if s.recording_device_fallback {
            mic_label(&s.recording_device)
        } else {
            String::new()
        }
    };
    emit_overlay_state(
        app,
        "recording",
        &fallback_label,
        if fallback_label.is_empty() {
            ""
        } else {
            "warn"
        },
    );

    // Spawn streaming preview: transcribe every ~2s while recording
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        streaming_preview_loop(app_clone).await;
    });
}

/// While recording, watch for a locked session or a sleep/resume: in both
/// cases the paste target is gone, so the recording is finished without an
/// automatic paste (text goes to history and clipboard).
async fn recording_guard_loop(app: tauri::AppHandle) {
    let mut last_tick = std::time::SystemTime::now();
    let mut no_foreground_ticks = 0u32;
    let mut ticks = 0u32;
    let mut mic_warned = false;
    loop {
        tokio::time::sleep(Duration::from_millis(1000)).await;
        ticks += 1;
        if ticks % 5 == 0 && is_recording(&app) {
            warn_if_mic_is_dead(&app, &mut mic_warned);
        }
        let now = std::time::SystemTime::now();
        let jumped = now
            .duration_since(last_tick)
            .map(|d| d > Duration::from_secs(10))
            .unwrap_or(true);
        last_tick = now;
        if jumped {
            // Flag first: the resume may already have stopped the recording
            // (audio stream error) and the stop flow may be transcribing.
            let state = app.state::<Mutex<AppState>>();
            lock_or_recover(&state)
                .suppress_paste
                .store(true, Ordering::Relaxed);
        }
        if !is_recording(&app) {
            return;
        }
        if crate::system::focus::foreground_target().is_none() {
            no_foreground_ticks += 1;
        } else {
            no_foreground_ticks = 0;
        }
        let reason = if jumped {
            Some("the computer slept")
        } else if no_foreground_ticks >= 2 {
            Some("the session was locked")
        } else {
            None
        };
        if let Some(reason) = reason {
            log::warn!("Recording interrupted: {reason}; finishing without auto-paste");
            {
                let state = app.state::<Mutex<AppState>>();
                lock_or_recover(&state)
                    .suppress_paste
                    .store(true, Ordering::Relaxed);
            }
            let _ = app.emit(
                events::OPERATION_NOTICE,
                format!("Recording stopped because {reason}; the text is in the clipboard."),
            );
            let _ = app.emit(events::REQUEST_STOP_RECORDING, ());
            return;
        }
    }
}

/// Every five seconds of a recording: if the last twenty seconds carry no
/// speech dynamics, say so in the pill, the window and a toast (once), and
/// clear the warning when speech shows up again.
fn warn_if_mic_is_dead(app: &tauri::AppHandle, warned: &mut bool) {
    let state = app.state::<Mutex<AppState>>();
    let (elapsed, device, fallback) = {
        let s = lock_or_recover(&state);
        (
            s.recording_started_at
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0),
            s.recording_device.clone(),
            s.recording_device_fallback,
        )
    };
    let tail = app.state::<AudioBuffer>().snapshot_tail(16_000 * 20);
    let stats = speech_stats(&tail);
    if mic_seems_dead(stats.as_ref(), elapsed) {
        if !*warned {
            *warned = true;
            log::warn!(
                "No speech dynamics in the last 20 s of the recording ({device}); warning the user"
            );
            emit_overlay_state(app, "recording", "No sound from the mic?", "warn");
            let message = if fallback {
                format!(
                    "No speech is reaching {device} (the selected microphone is unavailable). \
                     Check the microphone; the recording continues."
                )
            } else {
                format!("No speech is reaching {device}. Is it muted? The recording continues.")
            };
            let _ = app.emit(events::OPERATION_NOTICE, &message);
            notify_user(app, &message);
            app.state::<SoundPlayer>().play(SoundKind::Busy);
        }
    } else if *warned && stats.as_ref().is_some_and(has_speech_dynamics) {
        *warned = false;
        let label = if fallback {
            mic_label(&device)
        } else {
            String::new()
        };
        emit_overlay_state(
            app,
            "recording",
            &label,
            if label.is_empty() { "" } else { "warn" },
        );
    }
}

fn is_recording(app: &tauri::AppHandle) -> bool {
    let state = app.state::<Mutex<AppState>>();
    let s = lock_or_recover(&state);
    s.status == AppStatus::Recording
}

/// The only consumer of the streaming preview is the main window; when it is
/// hidden in the tray (the normal case) every preview tick would be a full
/// GPU encoder pass for nobody.
fn main_window_visible(app: &tauri::AppHandle) -> bool {
    app.get_webview_window("main")
        .map(|w| w.is_visible().unwrap_or(false) && !w.is_minimized().unwrap_or(false))
        .unwrap_or(false)
}

async fn streaming_preview_loop(app: tauri::AppHandle) {
    // Max audio to transcribe in preview mode (10s at 16kHz) — keeps preview fast
    const MAX_PREVIEW_SAMPLES: usize = 16000 * 10;

    // Wait 1.5s before first preview (need enough audio)
    for _ in 0..15 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if !is_recording(&app) {
            return;
        }
    }

    let (abort, on_cpu) = {
        let state = app.state::<Mutex<AppState>>();
        let s = lock_or_recover(&state);
        (s.preview_abort.clone(), s.model.backend() == "CPU")
    };
    if on_cpu {
        // On the CPU a preview tick takes longer than the two-second cadence
        // and would only delay the final transcription.
        log::info!("Streaming preview disabled on the CPU backend");
        return;
    }

    // Language detected on the first preview cycle is reused for the rest of
    // this recording — see WhisperEngine::transcribe_cached for the trade-off.
    let mut lang_cache: Option<&'static str> = None;
    let mut watchdog = Watchdog::new();
    let mut paused_logged = false;

    loop {
        if !is_recording(&app) {
            return;
        }
        if !main_window_visible(&app) {
            if !paused_logged {
                log::debug!("Streaming preview paused: main window hidden");
                paused_logged = true;
            }
        } else {
            paused_logged = false;
            let samples = app
                .state::<AudioBuffer>()
                .snapshot_tail(MAX_PREVIEW_SAMPLES);
            if samples.len() >= 16000 {
                let audio_s = samples.len() as f64 / 16000.0;
                let (language, pipeline) = {
                    let settings = app.state::<Mutex<Settings>>();
                    let guard = lock_or_recover(&settings);
                    (guard.language, guard.text_pipeline())
                };
                let blocking_app = app.clone();
                let abort_flag = abort.clone();
                let cache_in = lang_cache;
                let started = Instant::now();
                // Whisper blocks for hundreds of milliseconds; keep it off the
                // async runtime's worker threads.
                let outcome = tauri::async_runtime::spawn_blocking(move || {
                    let engine = blocking_app.state::<Mutex<WhisperEngine>>();
                    // Non-blocking: skip the tick if the final pass holds the engine.
                    let Ok(mut eng) = engine.try_lock() else {
                        return (cache_in, None);
                    };
                    let mut cache = cache_in;
                    let result = eng.transcribe_cached(
                        &samples,
                        language,
                        &mut cache,
                        TranscribeOptions::preview(Some(abort_flag)),
                    );
                    (cache, Some(result))
                })
                .await;

                match outcome {
                    Ok((cache, Some(Ok(result)))) => {
                        if lang_cache.is_none() && cache.is_some() && is_recording(&app) {
                            announce_language(&app, result.language, "auto");
                        }
                        lang_cache = cache;
                        let elapsed_ms = started.elapsed().as_millis() as f64;
                        log::debug!(
                            "Streaming preview: {audio_s:.1}s of audio in {elapsed_ms:.0} ms ({} chars)",
                            result.text.chars().count()
                        );
                        if is_recording(&app) && !result.text.is_empty() {
                            let text = pipeline.preview(&result.text);
                            let _ = app.emit(events::STREAMING_PREVIEW, &text);
                        }
                        if let Verdict::Slow {
                            elapsed_ms,
                            threshold_ms,
                        } = watchdog.observe(elapsed_ms, audio_s)
                        {
                            log::warn!(
                                "GPU watchdog: a preview of {audio_s:.1}s took {elapsed_ms:.0} ms \
                                 (threshold {threshold_ms:.0} ms); preview disabled for this recording"
                            );
                            let message =
                                "Transcription is running very slowly: check the charger \
                                           and the GPU power limit (nvidia-smi).";
                            let _ = app.emit(events::OPERATION_NOTICE, message);
                            notify_user(&app, message);
                            return;
                        }
                    }
                    Ok((cache, Some(Err(e)))) => {
                        lang_cache = cache;
                        log::debug!("Streaming preview skipped: {e}");
                    }
                    Ok((_, None)) => log::debug!("Streaming preview: engine locked, skipping"),
                    Err(e) => log::warn!("Streaming preview task failed: {e}"),
                }
            }
        }

        // Wait 2s before next preview, checking every 100ms if still recording
        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if !is_recording(&app) {
                return;
            }
        }
    }
}

/// Discard the active recording, or — while the previous one is still being
/// transcribed/formatted — skip its paste. Triggered by the cancel hotkey,
/// the overlay's X button and the tray.
pub fn cancel_recording(app: &tauri::AppHandle) {
    let state = app.state::<Mutex<AppState>>();
    let (was_recording, writer) = {
        let mut s = lock_or_recover(&state);
        match s.status {
            AppStatus::Recording => {
                s.status = AppStatus::Idle;
                s.recording_locked = false;
                s.recording_started_at = None;
                s.recording_started_wall = None;
                s.preview_abort.store(true, Ordering::Relaxed);
                (true, s.spool.take())
            }
            AppStatus::Transcribing | AppStatus::Formatting => {
                s.cancel_requested.store(true, Ordering::Relaxed);
                (false, None)
            }
            _ => return,
        }
    };
    if was_recording {
        log::info!("Recording cancelled");
        // Stop the spool outside the state lock (it joins a thread) and drop
        // the file: a cancelled dictation must not come back as "recovered".
        if let Some(writer) = writer {
            writer.stop();
        }
        lock_or_recover(&app.state::<Mutex<AudioCapture>>()).stop();
        app.state::<AudioBuffer>().clear();
        spool::remove_spool(&app.state::<AppConfig>().data_dir);
        app.state::<SoundPlayer>().play(SoundKind::Cancel);
        emit_status(app, &AppStatus::Idle);
        let _ = app.emit(events::LOCK_CHANGED, false);
        app.state::<TrayAnimator>().set_phase(TrayPhase::Idle);
        emit_overlay_state(app, "result", "Cancelled", "warn");
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1000)).await;
            if !is_recording(&app) {
                hide_overlay(&app);
            }
        });
    } else {
        log::info!("Cancel requested while processing: the result will not be pasted");
        emit_overlay_state(app, "processing", "Cancelling", "warn");
    }
}

/// How a stopped recording ended; drives the overlay's result flash, the
/// toast policy and the main-window notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Pasted,
    /// Paste failed; the text is in the clipboard (or only in history).
    CopiedToClipboard {
        in_clipboard: bool,
    },
    TooShort,
    NoSpeech,
    Failed,
    /// The user cancelled while processing; the text went to history only.
    Cancelled,
}

impl Outcome {
    /// Short name used in the usage stats file.
    pub fn label(&self) -> &'static str {
        match self {
            Outcome::Pasted => "pasted",
            Outcome::CopiedToClipboard { .. } => "copied",
            Outcome::TooShort => "too-short",
            Outcome::NoSpeech => "no-speech",
            Outcome::Failed => "failed",
            Outcome::Cancelled => "cancelled",
        }
    }

    fn overlay(self) -> (&'static str, &'static str, u64) {
        match self {
            Outcome::Cancelled => ("Cancelled", "warn", 1200),
            Outcome::Pasted => ("Pasted", "ok", 1200),
            Outcome::CopiedToClipboard { in_clipboard: true } => {
                ("Copied to clipboard", "warn", 2200)
            }
            Outcome::CopiedToClipboard {
                in_clipboard: false,
            } => ("Paste failed, see history", "error", 2500),
            Outcome::TooShort => ("Too short", "warn", 1400),
            Outcome::NoSpeech => ("No speech", "warn", 1800),
            Outcome::Failed => ("Transcription failed", "error", 2500),
        }
    }

    /// `transcription-empty` reason for the main window, if any.
    fn empty_reason(self) -> Option<&'static str> {
        match self {
            Outcome::TooShort => Some("too-short"),
            Outcome::NoSpeech => Some("no-speech"),
            Outcome::Failed => Some("error"),
            _ => None,
        }
    }

    /// OS toast text when the overlay is disabled (otherwise the pill tells).
    /// An accidental tap ("too short") never deserves a toast.
    fn toast(self) -> Option<&'static str> {
        match self {
            Outcome::NoSpeech => Some("No speech detected — try again"),
            Outcome::Failed => Some("Transcription failed — check wispr.log"),
            _ => None,
        }
    }
}

/// Append one line to the usage stats; a failure is logged, never surfaced.
fn record_stat(app: &tauri::AppHandle, outcome: &Outcome, audio_s: f64, words: u32, lang: &str) {
    let data_dir = app.state::<AppConfig>().data_dir.clone();
    let record = stats::StatRecord {
        ts: HistoryEntry::now_ms(),
        audio_s: audio_s as f32,
        words,
        outcome: outcome.label().to_string(),
        lang: lang.to_string(),
    };
    if let Err(e) = stats::append(&data_dir, &record) {
        log::warn!("Could not record usage stats: {e}");
    }
}

/// Hands-free recordings into a muted or dead microphone used to run for
/// minutes without a hint. After twenty seconds of audio without speech
/// dynamics the overlay says so.
fn mic_seems_dead(stats: Option<&SpeechStats>, elapsed_s: f64) -> bool {
    elapsed_s >= 20.0 && stats.is_some_and(|s| !has_speech_dynamics(s))
}

/// "Microphone (NVIDIA Broadcast)" → "NVIDIA Broadcast" for the pill.
fn mic_label(device: &str) -> String {
    let d = device.trim();
    d.strip_prefix("Microphone (")
        .or_else(|| d.strip_prefix("Microphone Array ("))
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or(d)
        .to_string()
}

/// How many finished recordings stay on disk for a re-transcription.
const KEEP_RECORDINGS: usize = 30;
/// ...and how much space they may take (a minute is 1.9 MB).
const KEEP_RECORDING_BYTES: u64 = 400 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpoolAction {
    /// Leave `pending.pcm` for the crash recovery on the next start.
    Keep,
    /// The user discarded it or it was never a recording.
    Remove,
    /// Keep the audio in the recordings folder so the text can be redone.
    Archive,
}

/// What happens to the spool of a finished recording.
fn spool_action(outcome: &Outcome) -> SpoolAction {
    match outcome {
        Outcome::Failed => SpoolAction::Keep,
        Outcome::TooShort | Outcome::Cancelled => SpoolAction::Remove,
        Outcome::Pasted | Outcome::CopiedToClipboard { .. } | Outcome::NoSpeech => {
            SpoolAction::Archive
        }
    }
}

/// A recording whose wall-clock duration exceeds its captured audio by more
/// than ten seconds spanned a sleep (no samples arrive while suspended) or a
/// stalled microphone; either way the paste target is no longer trustworthy.
fn sleep_gap_detected(wall_s: f64, audio_s: f64) -> bool {
    wall_s - audio_s > 10.0
}

/// The single exit of the stop pipeline: back to Idle, result shown in the
/// pill for a moment, tray back to idle, overlay hidden afterwards unless a
/// new recording has started in the meantime.
fn finish_pipeline(app: &tauri::AppHandle, outcome: Outcome) {
    let data_dir = app.state::<AppConfig>().data_dir.clone();
    match spool_action(&outcome) {
        SpoolAction::Keep => {}
        SpoolAction::Remove => spool::remove_spool(&data_dir),
        SpoolAction::Archive => {
            if let Some(path) = spool::archive_spool(&data_dir, outcome.label()) {
                log::info!("Recording kept at {}", path.display());
            }
            spool::prune_recordings(&data_dir, KEEP_RECORDINGS, KEEP_RECORDING_BYTES);
        }
    }
    let (message, tone, linger_ms) = outcome.overlay();
    if let Some(reason) = outcome.empty_reason() {
        log::warn!("No transcription result: {reason}");
        let _ = app.emit(events::TRANSCRIPTION_EMPTY, reason);
    }
    set_status(app, AppStatus::Idle);
    app.state::<TrayAnimator>().set_phase(TrayPhase::Idle);

    let overlay_enabled = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        guard.show_overlay
    };
    if overlay_enabled {
        emit_overlay_state(app, "result", message, tone);
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(Duration::from_millis(linger_ms)).await;
            if !is_recording(&app) {
                hide_overlay(&app);
            }
        });
    } else if let Some(text) = outcome.toast() {
        notify_user(app, text);
    }
}

/// Show a toast under our own identity; fall back to the notification
/// plugin (attributed to PowerShell) if WinRT refuses, and log either failure
/// instead of dropping the message silently.
pub fn notify_user(app: &tauri::AppHandle, message: &str) {
    notify_user_with(app, message, false);
}

/// Like [`notify_user`], but the toast stays for ~25 s (session-level news).
pub fn notify_user_long(app: &tauri::AppHandle, message: &str) {
    notify_user_with(app, message, true);
}

fn notify_user_with(app: &tauri::AppHandle, message: &str, long: bool) {
    match system::notify::show_toast(app.clone(), message, long) {
        Ok(()) => return,
        Err(e) => log::warn!("Toast via WinRT failed ({e}); using the notification plugin"),
    }
    use tauri_plugin_notification::NotificationExt;
    if let Err(e) = app
        .notification()
        .builder()
        .title("Wispr Local")
        .body(message)
        .show()
    {
        log::warn!("Toast via plugin failed: {e}");
    }
}

/// Record the model state and tell the main window.
pub fn set_model_state(app: &tauri::AppHandle, model: ModelState) {
    {
        let state = app.state::<Mutex<AppState>>();
        lock_or_recover(&state).model = model.clone();
    }
    let _ = app.emit(events::MODEL_STATE_CHANGED, &model);
    system::tray::refresh(app);
}

/// Model files to try, in order: the configured one, the shipped defaults,
/// then anything else discovered in the models directory. Deduplicated.
/// With `prefer_small` (CPU backend) the small/base models go first: the
/// 1.5 GB turbo model takes minutes per utterance on the CPU.
pub fn model_candidates(requested: &str, discovered: &[String], prefer_small: bool) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    let mut push = |name: &str| {
        if !name.trim().is_empty() && !candidates.iter().any(|c| c == name) {
            candidates.push(name.to_string());
        }
    };
    if prefer_small {
        let mut small: Vec<&String> = discovered
            .iter()
            .filter(|n| n.contains("small") || n.contains("base") || n.contains("tiny"))
            .collect();
        // Quantized first, then by name (tiny < base < small sorts the wrong
        // way; a stable preference list is simpler than ranking by size).
        small.sort_by_key(|n| (!n.contains("q5"), (*n).clone()));
        for name in small {
            push(name);
        }
    }
    push(requested);
    push(&settings::default_model_file());
    push("ggml-medium.bin");
    for name in discovered {
        push(name);
    }
    candidates
}

/// Load the Whisper model on a background thread: marks the state Loading,
/// tries each candidate in order, mirrors the result into AppState and emits
/// `model-state-changed`. A panic inside whisper-rs becomes `Failed` instead
/// of a poisoned engine. Must be called only after the engine and AppState
/// mutexes are managed.
pub fn spawn_model_loader(app: tauri::AppHandle, requested: String) {
    set_model_state(&app, ModelState::Loading);
    std::thread::Builder::new()
        .name("wispr-model-loader".into())
        .spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                load_first_available(&app, &requested)
            }));
            let model = match outcome {
                Ok(model) => model,
                Err(_) => {
                    log::error!("Model loader panicked; see the lines above");
                    ModelState::Failed {
                        error: "the model loader crashed (see wispr.log)".to_string(),
                    }
                }
            };
            set_model_state(&app, model.clone());

            match &model {
                ModelState::Ready {
                    backend,
                    file,
                    fallback,
                } => {
                    recover_spooled_recording(&app);
                    if *fallback {
                        let message = format!(
                            "Configured model {requested} is not available; using {file} instead."
                        );
                        log::warn!("{message}");
                        let _ = app.emit(events::OPERATION_NOTICE, &message);
                        notify_user(&app, &message);
                    }
                    if backend != "CUDA" {
                        let message = if std::env::var("WISPR_FORCE_CPU").is_ok() {
                            "Wispr Local recovered from a GPU failure and is using CPU \
                             transcription until restart."
                        } else {
                            "CUDA is unavailable; transcription runs on the CPU and will be slow."
                        };
                        notify_user_long(&app, message);
                    }
                }
                ModelState::Missing => {
                    let models_dir = app.state::<AppConfig>().models_dir.clone();
                    log::error!(
                        "No Whisper model found in {}. Download one to enable transcription.",
                        models_dir.display()
                    );
                    // First-run dead end otherwise: the window is hidden in
                    // the tray and the hotkey only shows a toast.
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                }
                ModelState::Failed { error } => {
                    notify_user_long(&app, &format!("Whisper model failed to load: {error}"));
                }
                ModelState::Loading => {}
            }
        })
        .expect("spawn model loader thread");
}

/// A spool file left by a crashed run holds a dictation the user never got:
/// transcribe it, keep it in history and the clipboard, and say so.
fn recover_spooled_recording(app: &tauri::AppHandle) {
    let data_dir = app.state::<AppConfig>().data_dir.clone();
    let Some(path) = spool::pending_recovery(&data_dir) else {
        return;
    };
    let samples = match spool::read_spool(&path) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("Recording spool unreadable: {e}");
            spool::remove_spool(&data_dir);
            return;
        }
    };
    let audio_s = samples.len() as f64 / 16000.0;
    log::info!("Recovering {audio_s:.1}s of audio spooled by a previous run");
    let (language, pipeline) = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        (guard.language, guard.text_pipeline())
    };
    let result = {
        let engine = app.state::<Mutex<WhisperEngine>>();
        let mut eng = lock_or_recover(&engine);
        eng.transcribe(&samples, language, TranscribeOptions::final_pass())
    };
    match spool::archive_spool(&data_dir, "recovered") {
        Some(path) => log::info!("Recovered audio kept at {}", path.display()),
        None => spool::remove_spool(&data_dir),
    }
    let text = match result {
        Ok(r) => pipeline.finalize(&r.text),
        Err(e) => {
            log::warn!("Recovered audio could not be transcribed: {e}");
            return;
        }
    };
    if text.is_empty() {
        log::info!("Recovered audio contained no speech");
        return;
    }
    // Lock order: Settings before AppState, like every other path.
    let limit = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        guard.history_limit
    };
    let history = {
        let state = app.state::<Mutex<AppState>>();
        let mut s = lock_or_recover(&state);
        s.last_transcription = text.clone();
        s.push_history(
            HistoryEntry {
                text: text.clone(),
                ts: HistoryEntry::now_ms(),
                target: String::new(),
                lang: String::new(),
                duration_s: audio_s as f32,
                pasted: false,
                hwnd: 0,
            },
            limit,
        );
        let _ = state::save_history(&data_dir, &s.history);
        s.history.clone()
    };
    let _ = app.emit(events::HISTORY_CHANGED, &history);
    let _ = system::text_injection::copy_only(&text);
    notify_user_long(
        app,
        "Recovered the last dictation after a crash: it is in the clipboard and the history.",
    );
}

/// Tray → "Re-transcribe last recording": run the final pass again on the
/// newest kept recording; the text goes to the clipboard and the history.
pub async fn retranscribe_latest(app: &tauri::AppHandle) {
    let data_dir = app.state::<AppConfig>().data_dir.clone();
    let Some(path) = spool::list_recordings(&data_dir).into_iter().next() else {
        let message = "No kept recordings yet.";
        let _ = app.emit(events::OPERATION_NOTICE, message);
        notify_user(app, message);
        return;
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    {
        let state = app.state::<Mutex<AppState>>();
        let mut s = lock_or_recover(&state);
        if !matches!(s.status, AppStatus::Idle | AppStatus::Error { .. }) {
            let message = "Busy with a recording; try again in a moment.";
            drop(s);
            let _ = app.emit(events::OPERATION_NOTICE, message);
            return;
        }
        s.status = AppStatus::Transcribing;
    }
    emit_status(app, &AppStatus::Transcribing);
    app.state::<TrayAnimator>().set_phase(TrayPhase::Processing);
    show_overlay_if_enabled(app);
    emit_overlay_state(app, "processing", "Re-transcribing", "");
    log::info!("Re-transcribing {name}");

    let (language, pipeline, history_limit) = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        (guard.language, guard.text_pipeline(), guard.history_limit)
    };
    let blocking_app = app.clone();
    let file = path.clone();
    let outcome = tauri::async_runtime::spawn_blocking(move || {
        let samples = spool::read_recording(&file)?;
        let engine = blocking_app.state::<Mutex<WhisperEngine>>();
        let mut eng = lock_or_recover(&engine);
        eng.transcribe(&samples, language, TranscribeOptions::final_pass())
            .map(|r| (r, samples.len()))
    })
    .await;
    let (result, samples) = match outcome {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => {
            log::error!("Re-transcription of {name} failed: {e}");
            let message = format!("Re-transcription failed: {e}");
            let _ = app.emit(events::OPERATION_NOTICE, &message);
            notify_user(app, &message);
            finish_pipeline(app, Outcome::Failed);
            return;
        }
        Err(e) => {
            log::error!("Re-transcription task failed: {e}");
            finish_pipeline(app, Outcome::Failed);
            return;
        }
    };
    let audio_s = samples as f64 / 16000.0;
    let text = pipeline.finalize(&result.text);
    if text.is_empty() {
        let message = format!(
            "Still nothing recognizable in {name} ({}).",
            result.empty_reason.unwrap_or("no speech")
        );
        log::warn!("{message}");
        let _ = app.emit(events::OPERATION_NOTICE, &message);
        notify_user(app, &message);
        finish_pipeline(app, Outcome::NoSpeech);
        return;
    }
    let copied = system::text_injection::copy_only(&text).is_ok();
    let history = {
        let state = app.state::<Mutex<AppState>>();
        let mut s = lock_or_recover(&state);
        s.last_transcription = text.clone();
        s.push_history(
            HistoryEntry {
                text: text.clone(),
                ts: HistoryEntry::now_ms(),
                target: format!("re-transcribed {name}"),
                lang: result.language.to_string(),
                duration_s: audio_s as f32,
                pasted: false,
                hwnd: 0,
            },
            history_limit,
        );
        let _ = state::save_history(&data_dir, &s.history);
        s.history.clone()
    };
    let _ = app.emit(events::HISTORY_CHANGED, &history);
    let message = if copied {
        format!(
            "Re-transcribed {name}: {} characters copied to the clipboard.",
            text.chars().count()
        )
    } else {
        format!("Re-transcribed {name}: the text is in the history.")
    };
    log::info!("{message}");
    let _ = app.emit(events::OPERATION_NOTICE, &message);
    notify_user(app, &message);
    finish_pipeline(
        app,
        Outcome::CopiedToClipboard {
            in_clipboard: copied,
        },
    );
}

fn load_first_available(app: &tauri::AppHandle, requested: &str) -> ModelState {
    let config = app.state::<AppConfig>();
    let discovered = config.available_model_files();
    let prefer_small = std::env::var("WISPR_FORCE_CPU").is_ok();
    if prefer_small {
        log::info!("CPU backend forced by the supervisor: preferring small models");
    }
    let candidates = model_candidates(requested, &discovered, prefer_small);
    let engine = app.state::<Mutex<WhisperEngine>>();

    let mut any_present = false;
    let mut last_error = String::new();
    for name in &candidates {
        let path = config.model_path(name);
        if !path.exists() {
            log::debug!("Model not found at {}", path.display());
            continue;
        }
        any_present = true;
        let started = Instant::now();
        let result = {
            let mut eng = lock_or_recover(&engine);
            eng.unload();
            eng.load_model(&path)
        };
        match result {
            Ok(()) => {
                let backend = lock_or_recover(&engine).compute_backend().to_string();
                log::info!(
                    "Model loaded from {} in {} ms ({backend})",
                    path.display(),
                    started.elapsed().as_millis()
                );
                return ModelState::Ready {
                    backend,
                    file: name.clone(),
                    fallback: name != requested,
                };
            }
            Err(e) => {
                log::error!("Failed to load model {}: {e}", path.display());
                last_error = e;
            }
        }
    }
    if any_present {
        ModelState::Failed { error: last_error }
    } else {
        ModelState::Missing
    }
}

pub async fn stop_and_transcribe_flow(app: &tauri::AppHandle) {
    log::debug!("stop_and_transcribe_flow called");
    let state = app.state::<Mutex<AppState>>();
    let capture = app.state::<Mutex<AudioCapture>>();
    let buffer = app.state::<AudioBuffer>();

    // Claim the stop atomically. Hotkey release, tray, pin button, and the
    // recording-limit event can arrive together; only one may finalize audio.
    let (preview_abort, origin_unknown, started_wall) = {
        let mut s = lock_or_recover(&state);
        if s.status != AppStatus::Recording {
            return;
        }
        s.recording_locked = false;
        s.status = AppStatus::Transcribing;
        (
            s.preview_abort.clone(),
            std::mem::take(&mut s.tray_stop_pending),
            s.recording_started_wall.take(),
        )
    };
    // The window the user was in when the key went up is where the text
    // belongs; anything focused later (settings, another app) must not get it.
    let origin = focus::foreground_target();
    // Whatever happens below (including a panic), the app returns to Idle.
    let guard_app = app.clone();
    let mut guard = ArmedGuard::new(move || {
        log::error!("Stop pipeline aborted unexpectedly; resetting to Idle");
        finish_pipeline(&guard_app, Outcome::Failed);
    });

    // An in-flight preview tick holds the engine; make it bail out so the
    // final pass does not queue behind it.
    preview_abort.store(true, Ordering::Relaxed);
    let _ = app.emit(events::LOCK_CHANGED, false);
    emit_status(app, &AppStatus::Transcribing);
    app.state::<TrayAnimator>().set_phase(TrayPhase::Processing);
    emit_overlay_state(app, "processing", "Transcribing", "");

    // Stop capture
    lock_or_recover(&capture).stop();
    app.state::<SoundPlayer>().play_stop();
    // Flush the crash spool before the buffer is drained.
    if let Some(writer) = lock_or_recover(&state).spool.take() {
        writer.stop();
    }

    let samples = buffer.take_samples();
    // Under ~0.5s is an accidental hotkey tap — not enough audio for even one
    // word, and short buffers are prime hallucination bait for Whisper.
    const MIN_SAMPLES: usize = 8000; // 0.5s at 16kHz
    if samples.len() < MIN_SAMPLES {
        guard.disarm();
        record_stat(
            app,
            &Outcome::TooShort,
            samples.len() as f64 / 16000.0,
            0,
            "",
        );
        finish_pipeline(app, Outcome::TooShort);
        return;
    }

    let audio_s = samples.len() as f64 / 16000.0;
    let utterance = UTTERANCES.fetch_add(1, Ordering::Relaxed) + 1;
    log::info!("utt#{utterance}: transcribing {audio_s:.1}s of audio");
    let wall_s = started_wall
        .and_then(|t| t.elapsed().ok())
        .map(|d| d.as_secs_f64())
        .unwrap_or(audio_s);
    if sleep_gap_detected(wall_s, audio_s) {
        log::warn!(
            "utt#{utterance}: {wall_s:.0}s of wall time for {audio_s:.1}s of audio; the machine \
             slept or the microphone stalled, so the text will not be auto-pasted"
        );
        lock_or_recover(&state)
            .suppress_paste
            .store(true, Ordering::Relaxed);
    }

    let (language, pipeline, paste_suffix, restore_clipboard, history_limit, backend) = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        let backend = lock_or_recover(&state).model.backend().to_string();
        (
            guard.language,
            guard.text_pipeline(),
            guard.paste_suffix,
            guard.restore_clipboard,
            guard.history_limit,
            backend,
        )
    };

    // Whisper blocks for seconds; run it off the async worker threads. The
    // engine lock waits for a preview tick (now aborting) or a model load.
    let lock_started = Instant::now();
    let blocking_app = app.clone();
    let outcome = tauri::async_runtime::spawn_blocking(move || {
        let engine = blocking_app.state::<Mutex<WhisperEngine>>();
        let mut eng = lock_or_recover(&engine);
        let lock_ms = lock_started.elapsed().as_millis();
        let result = eng.transcribe(&samples, language, TranscribeOptions::final_pass());
        (lock_ms, result)
    })
    .await;
    let (lock_ms, result) = match outcome {
        Ok(pair) => pair,
        Err(e) => {
            log::error!("Transcription task failed: {e}");
            guard.disarm();
            record_stat(app, &Outcome::Failed, audio_s, 0, "");
            finish_pipeline(app, Outcome::Failed);
            return;
        }
    };
    let result = match result {
        Ok(r) => r,
        Err(e) => {
            log::error!("Transcription failed: {}", e);
            guard.disarm();
            record_stat(app, &Outcome::Failed, audio_s, 0, "");
            finish_pipeline(app, Outcome::Failed);
            return;
        }
    };
    announce_language(app, result.language, "auto");
    let transcribe_ms = result.detect_ms + result.full_ms;
    let rtf = transcribe_ms as f64 / 1000.0 / audio_s;
    if rtf > 1.0 && backend == "CUDA" {
        log::warn!(
            "utt#{utterance}: transcription slower than realtime ({transcribe_ms} ms for {audio_s:.1}s); \
             the GPU may be power-capped"
        );
        let _ = app.emit(
            events::OPERATION_NOTICE,
            format!(
                "Transcription took {:.0} s for {audio_s:.0} s of audio: check the charger and the \
                 GPU power limit (nvidia-smi).",
                transcribe_ms as f64 / 1000.0
            ),
        );
    }

    if let Some(s) = result.stats.as_ref() {
        log::info!(
            "utt#{utterance}: audio floor={:.4} loud={:.4} active={:.2}",
            s.floor,
            s.loud,
            s.active
        );
    }
    if result.text.is_empty() {
        guard.disarm();
        if let Some(reason) = result.empty_reason.filter(|_| audio_s >= 4.0) {
            let (device, fallback) = {
                let s = lock_or_recover(&state);
                (s.recording_device.clone(), s.recording_device_fallback)
            };
            let what = match reason {
                "hallucination-loop" => "the model produced only noise text",
                "no-dynamics" => "the microphone delivered a steady signal without speech",
                _ => "the recording is silent",
            };
            let hint = if fallback {
                format!("the selected microphone is unavailable and {device} was used instead")
            } else {
                format!("check that {device} is not muted")
            };
            let message = format!(
                "Nothing recognizable in {}:{:02} of audio: {what}; {hint}. The audio is kept \
                 (tray: Re-transcribe last recording).",
                (audio_s as u64) / 60,
                (audio_s as u64) % 60
            );
            log::warn!("utt#{utterance}: {message}");
            let _ = app.emit(events::OPERATION_NOTICE, &message);
            notify_user_long(app, &message);
        }
        record_stat(
            app,
            &Outcome::NoSpeech,
            audio_s,
            0,
            &result.language.to_string(),
        );
        finish_pipeline(app, Outcome::NoSpeech);
        return;
    }

    let text = pipeline.finalize(&result.text);
    log::info!("Transcription cleaned ({} chars)", text.chars().count());

    if text.is_empty() {
        guard.disarm();
        record_stat(
            app,
            &Outcome::NoSpeech,
            audio_s,
            0,
            &result.language.to_string(),
        );
        finish_pipeline(app, Outcome::NoSpeech);
        return;
    }

    // AI formatting step
    let (ai_settings, glossary) = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        (
            guard.ai.clone(),
            formatting::glossary_from_rules(&guard.replacements),
        )
    };

    let cancel_requested = lock_or_recover(&state).cancel_requested.clone();
    if cancel_requested.load(Ordering::Relaxed) {
        log::info!("utt#{utterance}: cancelled before formatting; kept in history only");
        lock_or_recover(&state).push_history(
            HistoryEntry {
                text: text.clone(),
                ts: HistoryEntry::now_ms(),
                target: String::new(),
                lang: result.language.to_string(),
                duration_s: audio_s as f32,
                pasted: false,
                hwnd: 0,
            },
            history_limit,
        );
        guard.disarm();
        record_stat(
            app,
            &Outcome::Cancelled,
            audio_s,
            stats::count_words(&text),
            &result.language.to_string(),
        );
        finish_pipeline(app, Outcome::Cancelled);
        let history = lock_or_recover(&state).history.clone();
        let _ = app.emit(events::HISTORY_CHANGED, &history);
        return;
    }

    let format_started = Instant::now();
    let text = if ai_settings.is_active() {
        set_status(app, AppStatus::Formatting);
        emit_overlay_state(app, "processing", "Formatting", "");
        match formatting::format_text(&text, &ai_settings, &glossary).await {
            Ok(formatted) => formatted,
            Err(e) => {
                log::error!("AI formatting failed; using raw text: {e}");
                let message = "AI formatting failed; pasted the raw transcript instead.";
                let _ = app.emit(events::OPERATION_NOTICE, message);
                notify_user(app, message);
                text
            }
        }
    } else {
        text
    };
    let format_ms = if ai_settings.is_active() {
        format_started.elapsed().as_millis()
    } else {
        0
    };

    if cancel_requested.load(Ordering::Relaxed) {
        log::info!("utt#{utterance}: cancelled before pasting; kept in history only");
        lock_or_recover(&state).push_history(
            HistoryEntry {
                text: text.clone(),
                ts: HistoryEntry::now_ms(),
                target: String::new(),
                lang: result.language.to_string(),
                duration_s: audio_s as f32,
                pasted: false,
                hwnd: 0,
            },
            history_limit,
        );
        guard.disarm();
        record_stat(
            app,
            &Outcome::Cancelled,
            audio_s,
            stats::count_words(&text),
            &result.language.to_string(),
        );
        finish_pipeline(app, Outcome::Cancelled);
        let history = lock_or_recover(&state).history.clone();
        let _ = app.emit(events::HISTORY_CHANGED, &history);
        return;
    }

    set_status(app, AppStatus::Injecting);
    emit_overlay_state(app, "processing", "Pasting", "");

    let paste_started = Instant::now();
    let to_paste = apply_paste_suffix(&text, paste_suffix);
    let now = focus::foreground_target();
    let elevated = now
        .as_ref()
        .map(|t| focus::is_elevated_pid(t.pid))
        .unwrap_or(false);
    let target_name = now
        .as_ref()
        .map(|t| t.describe())
        .unwrap_or_else(|| "none".to_string());
    let suppressed = lock_or_recover(&state)
        .suppress_paste
        .load(Ordering::Relaxed);
    let decision = if suppressed {
        PasteDecision::CopyOnly("the recording was interrupted by sleep or lock".to_string())
    } else {
        focus::decide_paste(
            origin.as_ref(),
            now.as_ref(),
            std::process::id(),
            elevated,
            origin_unknown,
        )
    };
    let outcome = match decision {
        PasteDecision::Paste => {
            match system::text_injection::inject_text(&to_paste, restore_clipboard) {
                Ok(_) => {
                    log::debug!("Paste shortcut sent to {target_name}");
                    Outcome::Pasted
                }
                Err(e) => {
                    log::error!("Text injection failed: {}", e);
                    let copied = system::text_injection::copy_only(&text).is_ok();
                    let message = if copied {
                        "Automatic paste failed; the transcript was copied to your clipboard."
                    } else {
                        "Automatic paste failed; open Wispr Local to copy the transcript from history."
                    };
                    let _ = app.emit(events::OPERATION_NOTICE, message);
                    notify_user(app, message);
                    Outcome::CopiedToClipboard {
                        in_clipboard: copied,
                    }
                }
            }
        }
        PasteDecision::CopyOnly(reason) => {
            log::warn!("Not pasting: {reason}; transcript left in the clipboard");
            let copied = system::text_injection::copy_only(&text).is_ok();
            let message = if copied {
                format!("Not pasted ({reason}); the transcript is in your clipboard.")
            } else {
                format!("Not pasted ({reason}); copy it from the history.")
            };
            let _ = app.emit(events::OPERATION_NOTICE, &message);
            notify_user(app, &message);
            Outcome::CopiedToClipboard {
                in_clipboard: copied,
            }
        }
    };
    let paste_ms = paste_started.elapsed().as_millis();

    // History is saved under the state lock so "Clear history" from the UI
    // can never interleave with this write.
    let data_dir = app.state::<AppConfig>().data_dir.clone();
    let (history, save_error) = {
        let mut s = lock_or_recover(&state);
        s.last_transcription = text.clone();
        let changed = s.push_history(
            HistoryEntry {
                text: text.clone(),
                ts: HistoryEntry::now_ms(),
                target: now.as_ref().map(|t| t.describe()).unwrap_or_default(),
                lang: result.language.to_string(),
                duration_s: audio_s as f32,
                pasted: outcome == Outcome::Pasted,
                hwnd: now.as_ref().map(|t| t.hwnd).unwrap_or(0),
            },
            history_limit,
        );
        let error = if changed {
            state::save_history(&data_dir, &s.history).err()
        } else {
            None
        };
        (s.history.clone(), error)
    };
    if let Some(e) = save_error {
        log::warn!("Failed to save transcription history: {e}");
        let _ = app.emit(
            events::OPERATION_NOTICE,
            "History could not be saved to disk",
        );
    }
    log::info!(
        "utt#{utterance} audio={audio_s:.1}s lock={lock_ms}ms detect={}ms transcribe={}ms (rtf {rtf:.2}) \
         format={format_ms}ms paste={paste_ms}ms backend={backend} lang={} chars={} segments={} dropped={} target={target_name} outcome={outcome:?}",
        result.detect_ms,
        result.full_ms,
        result.language,
        text.chars().count(),
        result.segments,
        result.dropped
    );
    guard.disarm();
    record_stat(
        app,
        &outcome,
        audio_s,
        stats::count_words(&text),
        &result.language.to_string(),
    );
    finish_pipeline(app, outcome);
    let _ = app.emit(events::HISTORY_CHANGED, &history);
    let _ = app.emit(events::TRANSCRIPTION_COMPLETE, text);
}

#[cfg(test)]
mod tests {
    use super::{model_candidates, ArmedGuard, Outcome};
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn a_recording_without_dynamics_is_flagged_after_twenty_seconds() {
        use super::mic_seems_dead;
        use crate::transcription::engine::SpeechStats;
        let dead = SpeechStats {
            floor: 0.006,
            loud: 0.007,
            active: 0.0,
        };
        let live = SpeechStats {
            floor: 0.006,
            loud: 0.05,
            active: 0.3,
        };
        assert!(!mic_seems_dead(Some(&dead), 10.0), "too early to judge");
        assert!(mic_seems_dead(Some(&dead), 20.0));
        assert!(!mic_seems_dead(Some(&live), 60.0));
        assert!(!mic_seems_dead(None, 60.0), "no statistics, no warning");
    }

    #[test]
    fn spool_policy_keeps_failed_removes_discarded_and_archives_the_rest() {
        use super::{spool_action, SpoolAction};
        assert_eq!(spool_action(&Outcome::Failed), SpoolAction::Keep);
        assert_eq!(spool_action(&Outcome::TooShort), SpoolAction::Remove);
        assert_eq!(spool_action(&Outcome::Cancelled), SpoolAction::Remove);
        for outcome in [
            Outcome::Pasted,
            Outcome::CopiedToClipboard { in_clipboard: true },
            Outcome::NoSpeech,
        ] {
            assert_eq!(spool_action(&outcome), SpoolAction::Archive, "{outcome:?}");
        }
    }

    #[test]
    fn mic_label_strips_the_windows_wrapper() {
        use super::mic_label;
        assert_eq!(
            mic_label("Microphone (NVIDIA Broadcast)"),
            "NVIDIA Broadcast"
        );
        assert_eq!(
            mic_label("Microphone (2- HyperX SoloCast)"),
            "2- HyperX SoloCast"
        );
        assert_eq!(mic_label("Headset (Soundcore)"), "Headset (Soundcore)");
    }

    #[test]
    fn a_wall_clock_gap_beyond_the_audio_means_the_machine_slept() {
        use super::sleep_gap_detected;
        assert!(!sleep_gap_detected(12.0, 10.0), "normal jitter");
        assert!(sleep_gap_detected(600.0, 10.0), "ten minutes of sleep");
        assert!(
            !sleep_gap_detected(5.0, 10.0),
            "clock went backwards is not a gap"
        );
    }

    #[test]
    fn outcome_labels_match_the_stats_vocabulary() {
        use crate::stats::is_no_result;
        assert!(is_no_result(Outcome::TooShort.label()));
        assert!(is_no_result(Outcome::NoSpeech.label()));
        assert!(is_no_result(Outcome::Failed.label()));
        assert!(!is_no_result(Outcome::Pasted.label()));
        assert!(!is_no_result(Outcome::Cancelled.label()));
        assert!(!is_no_result(
            Outcome::CopiedToClipboard { in_clipboard: true }.label()
        ));
    }

    #[test]
    fn candidates_start_with_the_requested_file_and_dedupe() {
        let discovered = vec![
            "ggml-base.bin".to_string(),
            "ggml-large-v3-turbo.bin".to_string(),
            "ggml-small-q5_1.bin".to_string(),
        ];
        assert_eq!(
            model_candidates("ggml-small-q5_1.bin", &discovered, false),
            vec![
                "ggml-small-q5_1.bin",
                "ggml-large-v3-turbo.bin",
                "ggml-medium.bin",
                "ggml-base.bin",
            ]
        );
    }

    #[test]
    fn empty_request_falls_back_to_defaults() {
        assert_eq!(
            model_candidates("  ", &[], false),
            vec!["ggml-large-v3-turbo.bin", "ggml-medium.bin"]
        );
    }

    #[test]
    fn cpu_backend_prefers_small_models_first() {
        let discovered = vec![
            "ggml-base.bin".to_string(),
            "ggml-large-v3-turbo.bin".to_string(),
            "ggml-small-q5_1.bin".to_string(),
            "ggml-small.bin".to_string(),
        ];
        assert_eq!(
            model_candidates("ggml-large-v3-turbo.bin", &discovered, true),
            vec![
                "ggml-small-q5_1.bin",
                "ggml-base.bin",
                "ggml-small.bin",
                "ggml-large-v3-turbo.bin",
                "ggml-medium.bin",
            ]
        );
    }

    #[test]
    fn armed_guard_fires_on_drop_unless_disarmed() {
        let fired = Rc::new(Cell::new(0));
        {
            let f = Rc::clone(&fired);
            let _guard = ArmedGuard::new(move || f.set(f.get() + 1));
        }
        assert_eq!(fired.get(), 1, "armed guard fires");
        {
            let f = Rc::clone(&fired);
            let mut guard = ArmedGuard::new(move || f.set(f.get() + 1));
            guard.disarm();
        }
        assert_eq!(fired.get(), 1, "disarmed guard stays quiet");
    }

    #[test]
    fn accidental_taps_never_toast() {
        assert!(Outcome::TooShort.toast().is_none());
        assert!(Outcome::Cancelled.toast().is_none());
        assert!(Outcome::Cancelled.empty_reason().is_none());
        assert!(Outcome::Pasted.toast().is_none());
        assert!(Outcome::NoSpeech.toast().is_some());
        assert_eq!(Outcome::TooShort.empty_reason(), Some("too-short"));
    }
}
