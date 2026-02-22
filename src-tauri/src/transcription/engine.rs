use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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
    pub fn transcribe(&self, audio: &[f32]) -> Result<String, String> {
        let ctx = self.context.as_ref().ok_or("Whisper model not loaded")?;

        let mut state = ctx
            .create_state()
            .map_err(|e| format!("Failed to create Whisper state: {}", e))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(None); // auto-detect language
        // Bias model toward Russian and English only (suppresses Polish/Czech/etc.)
        params.set_initial_prompt("Текст на русском или английском языке. Text in Russian or English.");
        params.set_n_threads(8);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_single_segment(false);

        state
            .full(params, audio)
            .map_err(|e| format!("Whisper transcription failed: {}", e))?;

        let num_segments = state.full_n_segments();

        let mut text = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                let seg_text = segment.to_string();
                text.push_str(seg_text.trim());
                if i < num_segments - 1 {
                    text.push(' ');
                }
            }
        }

        Ok(text.trim().to_string())
    }
}
