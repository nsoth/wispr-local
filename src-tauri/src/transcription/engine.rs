use serde::{Deserialize, Serialize};
use std::path::Path;
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

    pub fn compute_backend(&self) -> &'static str {
        if self.using_gpu {
            "CUDA"
        } else {
            "CPU"
        }
    }

    /// Transcribe audio samples (must be 16kHz, mono, f32).
    pub fn transcribe(&self, audio: &[f32], language: LanguageMode) -> Result<String, String> {
        self.transcribe_cached(audio, language, &mut None)
    }

    /// Like [`Self::transcribe`], but with a caller-held cache for the Auto
    /// language decision. The streaming preview passes the same cache on every
    /// ~2s cycle, so the detection pass — a full GPU encoder run — happens once
    /// per recording instead of once per cycle. Half the CUDA launches means
    /// half the exposure to whisper.cpp's abort-on-CUDA-error (see
    /// supervisor.rs), and faster previews. The trade-off: the preview's
    /// language sticks for the rest of the recording; the final transcription
    /// uses a fresh cache, so the pasted result always re-detects on the full
    /// utterance.
    pub fn transcribe_cached(
        &self,
        audio: &[f32],
        language: LanguageMode,
        lang_cache: &mut Option<&'static str>,
    ) -> Result<String, String> {
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
        let lang = match language {
            LanguageMode::Russian => "ru",
            LanguageMode::English => "en",
            LanguageMode::Auto => match *lang_cache {
                Some(cached) => cached,
                None => {
                    let detected = detect_ru_or_en(&mut state, &audio)?;
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
        // Don't condition each 30s window on the previous window's text — a
        // hallucination in the first (quiet) window otherwise cascades and
        // takes over the whole transcription.
        params.set_no_context(true);
        params.set_n_threads(8);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_single_segment(false);

        state
            .full(params, &audio)
            .map_err(|e| format!("Whisper transcription failed: {}", e))?;

        let num_segments = state.full_n_segments();

        let mut parts: Vec<String> = Vec::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                let seg_text = segment.to_string();
                let no_speech = segment.no_speech_probability();

                // Window classified as near-certain silence — whatever the
                // decoder produced there is noise, not dictation.
                if no_speech > 0.85 {
                    log::info!(
                        "Dropping silent segment (no_speech_prob={:.2}, {} chars)",
                        no_speech,
                        seg_text.trim().chars().count()
                    );
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
                    }
                }
            }
        }

        Ok(parts.join(" ").trim().to_string())
    }
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
/// speaker. Costs one extra encoder pass over the audio (Auto mode only).
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
/// sign-offs from its training data. A short segment that is nothing but one
/// of these is never real dictation.
const HALLUCINATION_PHRASES: &[&str] = &[
    "субтитры",
    "субтитров",
    "редактор субтитров",
    "корректор",
    "продолжение следует",
    "спасибо за просмотр",
    "thanks for watching",
    "thank you for watching",
    "dimatorzok",
];

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

    // Drop short segments that are just a stock hallucination phrase.
    let lower = clean.to_lowercase();
    let word_count = lower.split_whitespace().count();
    if word_count <= 6 && HALLUCINATION_PHRASES.iter().any(|p| lower.contains(p)) {
        return None;
    }

    Some(clean)
}

#[cfg(test)]
mod tests {
    use super::clean_hallucinations;

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
}
