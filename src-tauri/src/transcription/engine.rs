//! whisper.cpp wrapper: model loading (CUDA with CPU fallback), constrained
//! Russian/English language detection, decoding parameters for the preview
//! and final passes, and hallucination filtering of the decoded segments.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Which language Whisper decodes as.
///
/// `Auto` runs a cheap detection pass constrained to Russian vs English — every
/// other language is ignored, so a Russian speaker's audio can never leak into
/// Ukrainian/Belarusian/Polish. `Russian` and `English` pin the decoder
/// deterministically and skip the detection pass entirely.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum LanguageMode {
    #[serde(rename = "auto")]
    #[default]
    Auto,
    #[serde(rename = "ru")]
    Russian,
    #[serde(rename = "en")]
    English,
    /// Any value this build does not know; normalized to `Auto` after loading.
    #[serde(other, skip_serializing)]
    Unknown,
}

/// Preview ticks trade a little accuracy for speed; the final pass does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassMode {
    Preview,
    Final,
}

pub struct TranscribeOptions {
    pub mode: PassMode,
    /// Checked between decoder steps; `true` makes whisper.cpp stop early.
    pub abort: Option<Arc<AtomicBool>>,
}

impl TranscribeOptions {
    pub fn preview(abort: Option<Arc<AtomicBool>>) -> Self {
        Self {
            mode: PassMode::Preview,
            abort,
        }
    }

    pub fn final_pass() -> Self {
        Self {
            mode: PassMode::Final,
            abort: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    /// "ru" or "en" — detected or pinned.
    pub language: &'static str,
    /// Auto-detection pass duration (0 when pinned or cached).
    pub detect_ms: u128,
    /// `whisper_full` duration.
    pub full_ms: u128,
    pub segments: usize,
    /// Segments removed as silence or hallucination.
    pub dropped: usize,
}

pub struct WhisperEngine {
    context: Option<WhisperContext>,
    using_gpu: bool,
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperEngine {
    pub fn new() -> Self {
        Self {
            context: None,
            using_gpu: false,
        }
    }

    /// Load the Whisper model from disk. Expensive (~200-1100ms).
    /// Call once at startup and keep warm.
    pub fn load_model(&mut self, model_path: &Path) -> Result<(), String> {
        log::info!("Loading Whisper model from {:?}...", model_path);
        let path = model_path.to_str().ok_or("Invalid model path")?;
        let force_cpu = std::env::var("WISPR_FORCE_CPU").is_ok();
        let mut params = WhisperContextParameters::default();
        params.use_gpu(!force_cpu);
        let (ctx, using_gpu) = match WhisperContext::new_with_params(path, params) {
            Ok(ctx) => (ctx, !force_cpu),
            Err(gpu_error) if !force_cpu => {
                log::warn!(
                    "GPU model initialization failed; retrying on CPU: {}",
                    gpu_error
                );
                let mut cpu_params = WhisperContextParameters::default();
                cpu_params.use_gpu(false);
                let ctx = WhisperContext::new_with_params(path, cpu_params)
                    .map_err(|e| format!("GPU load failed ({gpu_error}); CPU load failed ({e})"))?;
                (ctx, false)
            }
            Err(e) => return Err(format!("Failed to load Whisper model: {e}")),
        };

        self.context = Some(ctx);
        self.using_gpu = using_gpu;
        log::info!(
            "Whisper model loaded successfully ({})",
            if using_gpu { "CUDA" } else { "CPU" }
        );
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.context.is_some()
    }

    /// Drop the current model (frees its VRAM) before loading another one.
    pub fn unload(&mut self) {
        self.context = None;
        self.using_gpu = false;
    }

    pub fn compute_backend(&self) -> &'static str {
        if self.using_gpu {
            "CUDA"
        } else {
            "CPU"
        }
    }

    /// Number of decoder threads: all logical cores on the CPU path (the
    /// hard-coded 8 left half of a 20-thread laptop idle), fewer on CUDA where
    /// the threads only feed the GPU.
    fn decode_threads(&self) -> i32 {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        if self.using_gpu {
            cores.min(8) as i32
        } else {
            cores.min(16) as i32
        }
    }

    /// Transcribe audio samples (must be 16kHz, mono, f32) with a fresh
    /// language decision.
    pub fn transcribe(
        &self,
        audio: &[f32],
        language: LanguageMode,
        options: TranscribeOptions,
    ) -> Result<TranscriptionResult, String> {
        self.transcribe_cached(audio, language, &mut None, options)
    }

    /// Like [`Self::transcribe`], but with a caller-held cache for the Auto
    /// language decision. The streaming preview passes the same cache on every
    /// ~2s cycle, so the detection pass — a full GPU encoder run — happens once
    /// per recording instead of once per cycle. Half the CUDA launches means
    /// half the exposure to whisper.cpp's abort-on-CUDA-error (see
    /// supervisor.rs), and faster previews. The trade-off: the preview's
    /// language sticks for the rest of the recording; the final transcription
    /// uses a fresh cache, so the pasted result always re-detects on the full
    /// utterance (more precisely, on its first 30 s: whisper's detector only
    /// looks at one window).
    pub fn transcribe_cached(
        &self,
        audio: &[f32],
        language: LanguageMode,
        lang_cache: &mut Option<&'static str>,
        options: TranscribeOptions,
    ) -> Result<TranscriptionResult, String> {
        let ctx = self.context.as_ref().ok_or("Whisper model not loaded")?;

        // Peak-normalize so quiet mics still register, without the clipping
        // distortion a fixed capture-time gain caused on loud speech. The gain
        // cap keeps near-silent recordings from being blown up into noise.
        let audio = normalize_peak(audio);

        let mut state = ctx
            .create_state()
            .map_err(|e| format!("Failed to create Whisper state: {}", e))?;

        // Decide the decode language before building params.
        // History: we used to hardcode `set_language(Some("ru"))`, which ran
        // pure-English speech through the Russian decoder head and "translated"
        // it to Russian. A bare `set_language(None)` auto-detect (the even
        // earlier approach) leaked into Ukrainian/Belarusian/Polish ~5-10% of
        // the time for Russian speakers, which is why Auto detection is now
        // clamped to just ru vs en (see detect_ru_or_en).
        let mut detect_ms = 0;
        let lang = match language {
            LanguageMode::Russian => "ru",
            LanguageMode::English => "en",
            LanguageMode::Auto | LanguageMode::Unknown => match *lang_cache {
                Some(cached) => cached,
                None => {
                    let started = Instant::now();
                    let detected = detect_ru_or_en(&mut state, &audio)?;
                    detect_ms = started.elapsed().as_millis();
                    *lang_cache = Some(detected);
                    detected
                }
            },
        };

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // The medium+ models handle English code-switching (technical terms,
        // mixed phrases) inside a Russian utterance fine when the language is
        // pinned, so a ru-detected utterance keeps embedded English terms.
        params.set_language(Some(lang));
        params.set_suppress_blank(true);
        // Ban non-speech "meta" tokens ([музыка], [текст на русском], ♪ …) at
        // decode time. These are subtitle artifacts the model hallucinates on
        // silent/noisy stretches and they were leaking out as the final output.
        params.set_suppress_nst(true);
        // whisper.cpp's default; with a fresh state per call this only matters
        // for recordings longer than one 30 s window, where it stops a
        // hallucination in a quiet first window from seeding the next ones.
        params.set_no_context(true);
        params.set_n_threads(self.decode_threads());
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_single_segment(false);
        if options.mode == PassMode::Preview {
            // Preview text is replaced every two seconds: skip the temperature
            // fallback re-decodes and use a shorter encoder context (the
            // preview never exceeds 10 s of audio; 768 covers 15 s).
            params.set_temperature_inc(0.0);
            params.set_audio_ctx(768);
        }
        if let Some(abort) = options.abort.clone() {
            params.set_abort_callback_safe(move || abort.load(Ordering::Relaxed));
        }

        let started = Instant::now();
        state
            .full(params, &audio)
            .map_err(|e| format!("Whisper transcription failed: {}", e))?;
        let full_ms = started.elapsed().as_millis();

        let num_segments = state.full_n_segments();
        let decoded = (0..num_segments).filter_map(|i| {
            state
                .get_segment(i)
                .map(|segment| (segment.to_string(), segment.no_speech_probability()))
        });
        let (text, dropped) = assemble_segments(decoded);

        Ok(TranscriptionResult {
            text,
            language: lang,
            detect_ms,
            full_ms,
            segments: num_segments as usize,
            dropped,
        })
    }
}

/// Join decoded segments into the transcript, dropping windows whisper
/// classified as silence and segments that are pure hallucination. Returns
/// the text and the number of dropped segments.
fn assemble_segments(segments: impl IntoIterator<Item = (String, f32)>) -> (String, usize) {
    let mut parts: Vec<String> = Vec::new();
    let mut dropped = 0;
    for (seg_text, no_speech) in segments {
        // Window classified as near-certain silence — whatever the decoder
        // produced there is noise, not dictation. (The probability is computed
        // once per 30 s window and copied to every segment inside it.)
        if no_speech > 0.85 {
            log::info!(
                "Dropping silent segment (no_speech_prob={:.2}, {} chars)",
                no_speech,
                seg_text.trim().chars().count()
            );
            dropped += 1;
            continue;
        }

        // Safety net behind suppress_nst: strip bracketed meta-text
        // and known subtitle-credit hallucinations.
        match clean_hallucinations(&seg_text) {
            Some(clean) => parts.push(clean),
            None => {
                log::info!(
                    "Dropping hallucinated segment ({} chars)",
                    seg_text.trim().chars().count()
                );
                dropped += 1;
            }
        }
    }
    (parts.join(" ").trim().to_string(), dropped)
}

/// Scale audio so its peak sits at ~0.95, with the gain capped at 8x so pure
/// noise-floor recordings aren't amplified into garbage.
fn normalize_peak(audio: &[f32]) -> Vec<f32> {
    let peak = audio.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak <= 0.0 {
        return audio.to_vec();
    }
    let gain = (0.95 / peak).min(8.0);
    audio.iter().map(|&s| s * gain).collect()
}

/// English must beat Russian by a comfortable margin in the two-way ru/en race
/// before we switch the decoder off Russian — English's share must be at least
/// this fraction of `P(ru) + P(en)`.
const EN_THRESHOLD: f32 = 0.60;

/// Switching to English additionally requires Russian to be essentially
/// ABSENT: p_ru below this absolute probability. Field data (wispr.log
/// 2026-07-22): Russian dictation dense with English tech terms ("artist
/// mesh", "retop", "queue") accumulated enough English mass over a long
/// utterance to win the share race (share_en 0.61–0.80) while p_ru stayed at
/// 0.19–0.38 — and whole utterances got pasted as English translations.
/// Genuine English speech scores p_ru ≈ 0.02–0.03, far below this bar, so it
/// still switches; any audible Russian keeps the Russian decoder head.
const RU_PRESENCE_MAX: f32 = 0.15;

/// Pick "ru" or "en" from their detection probabilities, biased toward Russian.
/// Only these two probabilities matter; every other language is ignored by the
/// caller, so this is a strict two-way decision.
fn pick_language(p_ru: f32, p_en: f32) -> &'static str {
    let total = p_ru + p_en;
    if total <= 0.0 {
        // Degenerate detection — fall back to the historical default.
        return "ru";
    }
    if p_ru < RU_PRESENCE_MAX && p_en / total >= EN_THRESHOLD {
        "en"
    } else {
        "ru"
    }
}

/// Run Whisper's language detector but consider ONLY Russian and English —
/// every other language's probability is discarded, so acoustically-close
/// Slavic languages (Ukrainian, Belarusian, Polish) can never win for a Russian
/// speaker. Costs one extra encoder pass over the first 30 s window (Auto
/// mode only).
fn detect_ru_or_en(state: &mut WhisperState, audio: &[f32]) -> Result<&'static str, String> {
    state
        .pcm_to_mel(audio, 8)
        .map_err(|e| format!("Language detection (mel) failed: {}", e))?;
    let (_, probs) = state
        .lang_detect(0, 8)
        .map_err(|e| format!("Language detection failed: {}", e))?;

    let prob_of = |code: &str| -> f32 {
        whisper_rs::get_lang_id(code)
            .and_then(|id| probs.get(id as usize).copied())
            .unwrap_or(0.0)
    };
    let (p_ru, p_en) = (prob_of("ru"), prob_of("en"));
    let lang = pick_language(p_ru, p_en);
    log::info!(
        "Auto language: {} (p_ru={:.3}, p_en={:.3})",
        lang,
        p_ru,
        p_en
    );
    Ok(lang)
}

/// Stock phrases Whisper hallucinates on silence — subtitle credits and
/// sign-offs from its training data. Matched as anchored patterns so a real
/// short sentence containing "субтитры" survives (see tests).
const HALLUCINATION_SEGMENTS: &[&str] = &[
    "продолжение следует",
    "спасибо за просмотр",
    "thanks for watching",
    "thank you for watching",
    "субтитры",
    "редактор субтитров",
    "редактор субтитров а. семкин",
    "корректор а. егорова",
];

/// Segment prefixes that only ever come from subtitle credits.
const HALLUCINATION_PREFIXES: &[&str] = &[
    "субтитры сделал",
    "субтитры создал",
    "субтитры подготовил",
    "субтитры добавил",
    "редактор субтитров",
    "корректор а.",
    "субтитры по",
    "subtitles by",
];

/// Tokens that never appear in real dictation.
const HALLUCINATION_TOKENS: &[&str] = &["dimatorzok"];

/// Strip bracketed meta-annotations ("[текст на русском]", "(музыка)", "♪…")
/// from a segment and reject segments that are pure hallucination.
/// Returns None when nothing real remains.
fn clean_hallucinations(text: &str) -> Option<String> {
    // Remove [...] / (...) / {...} chunks. An unclosed opening bracket
    // swallows the rest of the segment — Whisper often truncates these.
    let mut depth = 0usize;
    let mut clean = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth = depth.saturating_sub(1),
            '♪' | '♫' => {}
            _ if depth == 0 => clean.push(ch),
            _ => {}
        }
    }

    // Collapse whitespace left behind by removed chunks.
    let clean = clean.split_whitespace().collect::<Vec<_>>().join(" ");

    // Drop if nothing but punctuation remains.
    if !clean.chars().any(|c| c.is_alphanumeric()) {
        return None;
    }

    let lower = clean.to_lowercase();
    let core = lower.trim_matches(|c: char| !c.is_alphanumeric());
    if HALLUCINATION_SEGMENTS.contains(&core) {
        return None;
    }
    if HALLUCINATION_PREFIXES
        .iter()
        .any(|prefix| core.starts_with(prefix))
    {
        return None;
    }
    if HALLUCINATION_TOKENS
        .iter()
        .any(|token| lower.contains(token))
    {
        return None;
    }

    Some(clean)
}

#[cfg(test)]
mod tests {
    use super::{assemble_segments, clean_hallucinations, normalize_peak};

    #[test]
    fn passes_normal_text_through() {
        assert_eq!(
            clean_hallucinations(" Привет, это тестовая диктовка. "),
            Some("Привет, это тестовая диктовка.".to_string())
        );
    }

    #[test]
    fn drops_bracket_only_segments() {
        assert_eq!(clean_hallucinations("[текст на русском]"), None);
        assert_eq!(clean_hallucinations(" [text in English] "), None);
        assert_eq!(clean_hallucinations("(музыка)"), None);
        assert_eq!(clean_hallucinations("♪♪♪"), None);
    }

    #[test]
    fn drops_unclosed_bracket_segments() {
        assert_eq!(clean_hallucinations("[текст на русском"), None);
    }

    #[test]
    fn strips_inline_brackets_keeps_speech() {
        assert_eq!(
            clean_hallucinations("Привет [музыка] мир"),
            Some("Привет мир".to_string())
        );
    }

    #[test]
    fn drops_subtitle_credits() {
        assert_eq!(clean_hallucinations("Субтитры сделал DimaTorzok"), None);
        assert_eq!(clean_hallucinations("Продолжение следует..."), None);
        assert_eq!(clean_hallucinations("Спасибо за просмотр!"), None);
        assert_eq!(clean_hallucinations("Редактор субтитров А.Семкин"), None);
        assert_eq!(clean_hallucinations("Корректор А.Егорова"), None);
        assert_eq!(clean_hallucinations("Субтитры"), None);
    }

    #[test]
    fn keeps_short_real_sentences_that_mention_stock_words() {
        assert_eq!(
            clean_hallucinations("Добавь субтитры к видео"),
            Some("Добавь субтитры к видео".to_string())
        );
        assert_eq!(
            clean_hallucinations("Нужен корректор цвета"),
            Some("Нужен корректор цвета".to_string())
        );
        assert_eq!(
            clean_hallucinations("Спасибо за просмотр макета, продолжаем"),
            Some("Спасибо за просмотр макета, продолжаем".to_string())
        );
    }

    #[test]
    fn keeps_long_sentences_mentioning_stock_words() {
        let text = "Я хочу добавить в приложение редактор субтитров с поддержкой нескольких дорожек и экспортом";
        assert_eq!(clean_hallucinations(text), Some(text.to_string()));
    }

    #[test]
    fn drops_punctuation_only() {
        assert_eq!(clean_hallucinations("..."), None);
        assert_eq!(clean_hallucinations(""), None);
    }

    #[test]
    fn assemble_drops_silent_windows_and_hallucinations() {
        let (text, dropped) = assemble_segments(vec![
            (" Привет ".to_string(), 0.1),
            (" [музыка] ".to_string(), 0.2),
            (" шум ".to_string(), 0.95),
            (" мир. ".to_string(), 0.1),
        ]);
        assert_eq!(text, "Привет мир.");
        assert_eq!(dropped, 2);
    }

    #[test]
    fn normalize_peak_caps_the_gain() {
        let quiet = vec![0.0, 0.01, -0.01];
        let out = normalize_peak(&quiet);
        assert!((out[1] - 0.08).abs() < 1e-6, "8x cap: {out:?}");
        let silent = normalize_peak(&[0.0, 0.0]);
        assert_eq!(silent, vec![0.0, 0.0]);
        let loud = normalize_peak(&[0.5, -1.0]);
        assert!((loud[1].abs() - 0.95).abs() < 1e-6);
    }

    use super::pick_language;

    #[test]
    fn pick_language_pure_english() {
        assert_eq!(pick_language(0.02, 0.97), "en");
    }

    #[test]
    fn pick_language_pure_russian() {
        assert_eq!(pick_language(0.96, 0.03), "ru");
    }

    #[test]
    fn pick_language_near_tie_stays_russian() {
        // Russian with embedded English terms: English edges ahead but not by
        // the required margin, so it stays Russian and code-switching survives.
        assert_eq!(pick_language(0.45, 0.50), "ru");
    }

    #[test]
    fn pick_language_russian_presence_blocks_english() {
        // English share is high, but Russian is clearly audible (p_ru well
        // above RU_PRESENCE_MAX) — term-heavy Russian, not English speech.
        assert_eq!(pick_language(0.20, 0.80), "ru");
    }

    #[test]
    fn pick_language_field_data_regressions_stay_russian() {
        // Real mis-switches from wispr.log 2026-07-22: Russian dictation about
        // "artist mesh"/"retop"/"queue" pasted as English translations. All
        // must stay Russian under the presence gate.
        assert_eq!(pick_language(0.285, 0.643), "ru");
        assert_eq!(pick_language(0.363, 0.601), "ru");
        assert_eq!(pick_language(0.379, 0.590), "ru");
        assert_eq!(pick_language(0.188, 0.769), "ru");
    }

    #[test]
    fn pick_language_genuine_english_still_switches() {
        // Genuine English utterances score p_ru far below the presence bar
        // (~0.02-0.03 observed), so the switch to English survives the gate.
        assert_eq!(pick_language(0.03, 0.90), "en");
        assert_eq!(pick_language(0.10, 0.55), "en");
    }

    #[test]
    fn pick_language_zero_defaults_russian() {
        assert_eq!(pick_language(0.0, 0.0), "ru");
    }

    #[test]
    fn pick_language_boundaries() {
        // p_ru exactly at RU_PRESENCE_MAX is not "below": stays Russian.
        assert_eq!(pick_language(0.15, 0.85), "ru");
        // Clearly above the share threshold with Russian absent: English.
        assert_eq!(pick_language(0.14, 0.22), "en");
    }
}
