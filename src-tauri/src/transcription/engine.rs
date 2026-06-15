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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LanguageMode {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "ru")]
    Russian,
    #[serde(rename = "en")]
    English,
}

impl Default for LanguageMode {
    fn default() -> Self {
        LanguageMode::Auto
    }
}

pub struct WhisperEngine {
    context: Option<WhisperContext>,
}

impl WhisperEngine {
    pub fn new() -> Self {
        Self { context: None }
    }

    /// Load the Whisper model from disk. Expensive (~200-1100ms).
    /// Call once at startup and keep warm.
    pub fn load_model(&mut self, model_path: &Path) -> Result<(), String> {
        log::info!("Loading Whisper model from {:?}...", model_path);
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().ok_or("Invalid model path")?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("Failed to load Whisper model: {}", e))?;

        self.context = Some(ctx);
        log::info!("Whisper model loaded successfully");
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.context.is_some()
    }

    /// Transcribe audio samples (must be 16kHz, mono, f32).
    pub fn transcribe(&self, audio: &[f32], language: LanguageMode) -> Result<String, String> {
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
            LanguageMode::Auto => detect_ru_or_en(&mut state, &audio)?,
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
                        "Dropping segment (no_speech_prob={:.2}): {}",
                        no_speech,
                        seg_text.trim()
                    );
                    continue;
                }

                // Safety net behind suppress_nst: strip bracketed meta-text
                // and known subtitle-credit hallucinations.
                match clean_hallucinations(&seg_text) {
                    Some(clean) => parts.push(clean),
                    None => {
                        log::info!("Dropping hallucinated segment: {}", seg_text.trim());
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
/// this fraction of `P(ru) + P(en)`. A near-tie (Russian speech with embedded
/// English terms) therefore stays Russian, so code-switching isn't mangled into
/// English. Pure English scores ~0.97 here and clears the bar easily.
const EN_THRESHOLD: f32 = 0.60;

/// Pick "ru" or "en" from their detection probabilities, biased toward Russian.
/// Only these two probabilities matter; every other language is ignored by the
/// caller, so this is a strict two-way decision.
fn pick_language(p_ru: f32, p_en: f32) -> &'static str {
    let total = p_ru + p_en;
    if total <= 0.0 {
        // Degenerate detection — fall back to the historical default.
        return "ru";
    }
    if p_en / total >= EN_THRESHOLD {
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
    if word_count <= 6
        && HALLUCINATION_PHRASES
            .iter()
            .any(|p| lower.contains(p))
    {
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
    fn pick_language_clear_english_majority_switches() {
        // English well past the 0.60 share threshold.
        assert_eq!(pick_language(0.20, 0.80), "en");
    }

    #[test]
    fn pick_language_zero_defaults_russian() {
        assert_eq!(pick_language(0.0, 0.0), "ru");
    }
}
