//! Post-processing of raw Whisper output before it is pasted.

/// Remove filler interjections from transcription (Russian + English).
/// Only pure interjections that carry no meaning in any context — semantic
/// words ("ну", "значит", "like", "so", "well") stay, because stripping them
/// blindly corrupts real sentences ("I like this" → "I this"). Contextual
/// filler cleanup is the AI formatting step's job.
pub fn remove_fillers(text: &str) -> String {
    const FILLERS: &[&str] = &[
        // Russian
        "э", "ээ", "эээ", "эм", "ээм", "эмм", "ам", "хм", "мм", "ммм", // English
        "um", "umm", "uh", "uhh", "hmm", "er", "erm", "ah", "mhm",
    ];

    let cleaned: Vec<&str> = text
        .split_whitespace()
        .filter(|w| {
            let lower = w.to_lowercase();
            let stripped =
                lower.trim_matches(|c: char| matches!(c, ',' | '.' | '!' | '?' | '…' | '-' | '—'));
            !FILLERS.contains(&stripped)
        })
        .collect();

    cleaned.join(" ").trim().to_string()
}

#[cfg(test)]
mod filler_tests {
    use super::remove_fillers;

    #[test]
    fn removes_interjections() {
        assert_eq!(
            remove_fillers("Эм, привет, э, как дела?"),
            "привет, как дела?"
        );
        assert_eq!(
            remove_fillers("Um, hello there, uh, okay"),
            "hello there, okay"
        );
    }

    #[test]
    fn keeps_semantic_words() {
        assert_eq!(
            remove_fillers("I like this approach"),
            "I like this approach"
        );
        assert_eq!(
            remove_fillers("Ну, это значит, что всё хорошо"),
            "Ну, это значит, что всё хорошо"
        );
        assert_eq!(
            remove_fillers("So, well, basically it works"),
            "So, well, basically it works"
        );
    }

    #[test]
    fn handles_empty_and_filler_only() {
        assert_eq!(remove_fillers("эм... ээ"), "");
        assert_eq!(remove_fillers(""), "");
    }
}
