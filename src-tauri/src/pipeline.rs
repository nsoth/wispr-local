//! The dictation pipeline: start capture → streaming preview → stop, transcribe,
//! post-process, optional AI formatting, paste → history. Also owns the
//! background model loader and the user-notification helpers.
//!
//! Status changes go through [`set_status`] so the UI always receives the
//! same serialized [`AppStatus`] payload that `get_status` returns.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use tauri::{Emitter, Manager};

use crate::audio::buffer::AudioBuffer;
use crate::audio::capture::AudioCapture;
use crate::audio::devices;
use crate::config::AppConfig;
use crate::events;
use crate::overlay::{hide_overlay, show_overlay_if_enabled};
use crate::settings::{self, Settings};
use crate::state::{self, lock_or_recover, AppState, AppStatus, ModelState};
use crate::system::sounds::SoundPlayer;
use crate::system::tray::TrayAnimator;
use crate::text::apply_paste_suffix;
use crate::transcription::engine::{TranscribeOptions, WhisperEngine};
use crate::watchdog::{Verdict, Watchdog};
use crate::{commands, formatting, system};

/// Counts finished recordings for the per-utterance log line.
static UTTERANCES: AtomicU64 = AtomicU64::new(0);

/// Record the new status in [`AppState`] and tell every webview about it.
pub fn set_status(app: &tauri::AppHandle, status: AppStatus) {
    {
        let state = app.state::<Mutex<AppState>>();
        if let Ok(mut s) = state.lock() {
            s.status = status.clone();
        };
    }
    emit_status(app, &status);
}

/// Broadcast a status that was already stored (e.g. under an existing lock).
pub fn emit_status(app: &tauri::AppHandle, status: &AppStatus) {
    let _ = app.emit(events::STATUS_CHANGED, status);
}

pub fn start_recording_flow(app: &tauri::AppHandle) {
    log::debug!("start_recording_flow called");
    let state = app.state::<Mutex<AppState>>();
    let capture = app.state::<Mutex<AudioCapture>>();
    let buffer = app.state::<AudioBuffer>();

    let model = {
        let mut s = lock_or_recover(&state);
        if !matches!(s.status, AppStatus::Idle | AppStatus::Error { .. }) {
            log::info!("Ignoring recording request while app is busy");
            return;
        }
        match &s.model {
            ModelState::Ready { .. } | ModelState::Loading => {
                s.status = AppStatus::Recording;
                s.recording_locked = false;
                s.preview_abort.store(false, Ordering::Relaxed);
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

    {
        buffer.clear();
    }

    let preferred_device = {
        let settings = app.state::<Mutex<Settings>>();
        settings
            .lock()
            .map(|s| s.input_device.clone())
            .unwrap_or_default()
    };
    let start_result = capture
        .lock()
        .map_err(|e| e.to_string())
        .and_then(|mut cap| {
            cap.start(
                Some(app.clone()),
                (!preferred_device.is_empty()).then_some(preferred_device.as_str()),
            )
        });
    match start_result {
        Ok(start) => {
            lock_or_recover(&state).device_sample_rate = start.sample_rate;
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
            app.state::<TrayAnimator>().stop();
            hide_overlay(app);
            return;
        }
    }

    emit_status(app, &AppStatus::Recording);
    let _ = app.emit(events::LOCK_CHANGED, false);
    app.state::<SoundPlayer>().play_start();

    // Kick off tray animation and reveal overlay (if user hasn't disabled it).
    app.state::<TrayAnimator>().start();
    show_overlay_if_enabled(app);

    // Spawn streaming preview: transcribe every ~2s while recording
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        streaming_preview_loop(app_clone).await;
    });
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
    use std::time::Duration;

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
                    let Ok(eng) = engine.try_lock() else {
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

/// Tell the user why nothing was pasted instead of failing silently.
/// Emits an event for the UI and shows a system toast (the main window is
/// usually hidden in the tray during dictation).
fn notify_no_result(app: &tauri::AppHandle, reason: &str, message: &str) {
    log::warn!("No transcription result: {}", reason);
    let _ = app.emit(events::TRANSCRIPTION_EMPTY, reason);

    notify_user(app, message);
}

/// Return the pipeline to Idle and explain the missing result. Used by every
/// early exit of [`stop_and_transcribe_flow`].
fn finish_without_result(app: &tauri::AppHandle, reason: &str, message: &str) {
    set_status(app, AppStatus::Idle);
    notify_no_result(app, reason, message);
}

pub fn notify_user(app: &tauri::AppHandle, message: &str) {
    use tauri_plugin_notification::NotificationExt;
    let _ = app
        .notification()
        .builder()
        .title("Wispr Local")
        .body(message)
        .show();
}

/// Record the model state and tell the main window.
pub fn set_model_state(app: &tauri::AppHandle, model: ModelState) {
    {
        let state = app.state::<Mutex<AppState>>();
        lock_or_recover(&state).model = model.clone();
    }
    let _ = app.emit(events::MODEL_STATE_CHANGED, &model);
}

/// Model files to try, in order: the configured one, the shipped defaults,
/// then anything else discovered in the models directory. Deduplicated.
pub fn model_candidates(requested: &str, discovered: &[String]) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    let mut push = |name: &str| {
        if !name.trim().is_empty() && !candidates.iter().any(|c| c == name) {
            candidates.push(name.to_string());
        }
    };
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
                            "Wispr Local recovered from a GPU failure and is using CPU                              transcription until restart."
                        } else {
                            "CUDA is unavailable; transcription runs on the CPU and will be slow."
                        };
                        notify_user(&app, message);
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
                    notify_user(&app, &format!("Whisper model failed to load: {error}"));
                }
                ModelState::Loading => {}
            }
        })
        .expect("spawn model loader thread");
}

fn load_first_available(app: &tauri::AppHandle, requested: &str) -> ModelState {
    let config = app.state::<AppConfig>();
    let discovered = config.available_model_files();
    let candidates = model_candidates(requested, &discovered);
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
        let started = std::time::Instant::now();
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
    let preview_abort = {
        let mut s = lock_or_recover(&state);
        if s.status != AppStatus::Recording {
            return;
        }
        s.recording_locked = false;
        s.status = AppStatus::Transcribing;
        s.preview_abort.clone()
    };
    // An in-flight preview tick holds the engine; make it bail out so the
    // final pass does not queue behind it.
    preview_abort.store(true, Ordering::Relaxed);
    let _ = app.emit(events::LOCK_CHANGED, false);
    emit_status(app, &AppStatus::Transcribing);

    // Stop capture
    lock_or_recover(&capture).stop();
    app.state::<SoundPlayer>().play_stop();

    // End the recording indicator as soon as the mic is released; transcription
    // runs afterwards and doesn't need the red pulse.
    app.state::<TrayAnimator>().stop();
    hide_overlay(app);

    let samples = buffer.take_samples();
    // Under ~0.5s is an accidental hotkey tap — not enough audio for even one
    // word, and short buffers are prime hallucination bait for Whisper.
    const MIN_SAMPLES: usize = 8000; // 0.5s at 16kHz
    if samples.len() < MIN_SAMPLES {
        finish_without_result(app, "too-short", "Recording too short — nothing captured");
        return;
    }

    let audio_s = samples.len() as f64 / 16000.0;
    let utterance = UTTERANCES.fetch_add(1, Ordering::Relaxed) + 1;
    log::info!("utt#{utterance}: transcribing {audio_s:.1}s of audio");

    let (language, pipeline, paste_suffix, backend) = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        let backend = lock_or_recover(&state).model.backend().to_string();
        (
            guard.language,
            guard.text_pipeline(),
            guard.paste_suffix,
            backend,
        )
    };

    // Whisper blocks for seconds; run it off the async worker threads. The
    // engine lock waits for a preview tick (now aborting) or a model load.
    let lock_started = Instant::now();
    let blocking_app = app.clone();
    let outcome = tauri::async_runtime::spawn_blocking(move || {
        let engine = blocking_app.state::<Mutex<WhisperEngine>>();
        let eng = lock_or_recover(&engine);
        let lock_ms = lock_started.elapsed().as_millis();
        let result = eng.transcribe(&samples, language, TranscribeOptions::final_pass());
        (lock_ms, result)
    })
    .await;
    let (lock_ms, result) = match outcome {
        Ok(pair) => pair,
        Err(e) => {
            log::error!("Transcription task failed: {e}");
            finish_without_result(app, "error", "Transcription failed — check logs");
            return;
        }
    };
    let result = match result {
        Ok(r) => r,
        Err(e) => {
            log::error!("Transcription failed: {}", e);
            finish_without_result(app, "error", "Transcription failed — check logs");
            return;
        }
    };
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

    if result.text.is_empty() {
        finish_without_result(app, "no-speech", "No speech detected — try again");
        return;
    }

    let text = pipeline.finalize(&result.text);
    log::info!("Transcription cleaned ({} chars)", text.chars().count());

    if text.is_empty() {
        finish_without_result(app, "no-speech", "No speech detected — try again");
        return;
    }

    // AI formatting step
    let ai_settings = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = lock_or_recover(&settings);
        guard.ai.clone()
    };

    let format_started = Instant::now();
    let text = if ai_settings.provider != formatting::AiProvider::None {
        set_status(app, AppStatus::Formatting);
        match formatting::format_text(&text, &ai_settings).await {
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
    let format_ms = if ai_settings.provider != formatting::AiProvider::None {
        format_started.elapsed().as_millis()
    } else {
        0
    };

    set_status(app, AppStatus::Injecting);

    let paste_started = Instant::now();
    let to_paste = apply_paste_suffix(&text, paste_suffix);
    let pasted = match system::text_injection::inject_text(&to_paste) {
        Ok(_) => {
            log::debug!("Paste shortcut sent");
            true
        }
        Err(e) => {
            log::error!("Text injection failed: {}", e);
            let copied = commands::copy_text(text.clone()).is_ok();
            let message = if copied {
                "Automatic paste failed; the transcript was copied to your clipboard."
            } else {
                "Automatic paste failed; open Wispr Local to copy the transcript from history."
            };
            let _ = app.emit(events::OPERATION_NOTICE, message);
            notify_user(app, message);
            false
        }
    };
    let paste_ms = paste_started.elapsed().as_millis();

    let (history, history_changed) = {
        let mut s = lock_or_recover(&state);
        s.last_transcription = text.clone();
        let changed = s.push_history(&text);
        s.status = AppStatus::Idle;
        (s.history.clone(), changed)
    };
    if history_changed {
        if let Err(e) = state::save_history(&app.state::<AppConfig>().data_dir, &history) {
            log::warn!("Failed to save transcription history: {e}");
            let _ = app.emit(
                events::OPERATION_NOTICE,
                "History could not be saved to disk",
            );
        }
    }
    log::info!(
        "utt#{utterance} audio={audio_s:.1}s lock={lock_ms}ms detect={}ms transcribe={}ms (rtf {rtf:.2}) \
         format={format_ms}ms paste={paste_ms}ms backend={backend} lang={} chars={} segments={} dropped={} pasted={pasted}",
        result.detect_ms,
        result.full_ms,
        result.language,
        text.chars().count(),
        result.segments,
        result.dropped
    );
    emit_status(app, &AppStatus::Idle);
    let _ = app.emit(events::HISTORY_CHANGED, &history);
    let _ = app.emit(events::TRANSCRIPTION_COMPLETE, text);
}

#[cfg(test)]
mod tests {
    use super::model_candidates;

    #[test]
    fn candidates_start_with_the_requested_file_and_dedupe() {
        let discovered = vec![
            "ggml-base.bin".to_string(),
            "ggml-large-v3-turbo.bin".to_string(),
            "ggml-small-q5_1.bin".to_string(),
        ];
        assert_eq!(
            model_candidates("ggml-small-q5_1.bin", &discovered),
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
            model_candidates("  ", &[]),
            vec!["ggml-large-v3-turbo.bin", "ggml-medium.bin"]
        );
    }
}
