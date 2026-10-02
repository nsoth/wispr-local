//! Post-processing of raw Whisper output before it is pasted:
//! filler removal → spoken formatting commands → user dictionary → suffix.

use serde::{Deserialize, Serialize};

use crate::transcription::replacements::Vocabulary;

/// What to append after a pasted transcript so consecutive dictations do not
/// run together ("sentence.Next sentence").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PasteSuffix {
    #[default]
    Space,
    Newline,
    None,
}

/// Punctuation that may cling to a spoken word on either side.
const EDGE_PUNCT: &[char] = &[',', '.', '!', '?', '…', '-', '—', ';', ':', '"', '«', '»'];

fn is_terminal(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

fn ends_with_punct(word: &str) -> bool {
    word.chars().last().is_some_and(|c| EDGE_PUNCT.contains(&c))
}

fn capitalize_first(word: &str) -> String {
    let mut chars = word.chars();
    let mut out = String::with_capacity(word.len());
    for c in chars.by_ref() {
        if c.is_alphabetic() {
            out.extend(c.to_uppercase());
            break;
        }
        out.push(c);
    }
    out.extend(chars);
    out
}

/// Move the sentence-ending punctuation of a removed word onto the word
/// before it ("важно, эм." → "важно.").
fn attach_terminal_punct(previous: &mut String, punct: char) {
    while previous.ends_with([',', ';', ':']) {
        previous.pop();
    }
    if !previous.chars().last().is_some_and(is_terminal) {
        previous.push(punct);
    }
}

/// Remove filler interjections from transcription (Russian + English).
/// Only pure interjections that carry no meaning in any context — semantic
/// words ("ну", "значит", "like", "so", "well") stay, because stripping them
/// blindly corrupts real sentences ("I like this" → "I this"). Contextual
/// filler cleanup is the AI formatting step's job. Capitalization and
/// sentence-ending punctuation of a removed word are carried over so the
/// sentence still reads correctly.
pub fn remove_fillers(text: &str) -> String {
    // "мм" (millimetres), "er" (emergency room) and "erm" were removed from
    // this list: they are real words far more often than fillers.
    const FILLERS: &[&str] = &[
        // Russian
        "э", "ээ", "эээ", "эм", "ээм", "эмм", "ам", "хм", // English
        "um", "umm", "uh", "uhh", "hmm", "ah", "mhm",
    ];

    let mut out: Vec<String> = Vec::new();
    let mut capitalize_next = false;
    for word in text.split_whitespace() {
        let core = word.trim_matches(|c: char| EDGE_PUNCT.contains(&c));
        let lower = core.to_lowercase();
        let is_filler = !lower.is_empty() && FILLERS.contains(&lower.as_str());
        if is_filler {
            // A filler that opened the text or was capitalized started a
            // sentence; the next real word takes over that role.
            if out.is_empty() || core.chars().next().is_some_and(char::is_uppercase) {
                capitalize_next = true;
            }
            let terminal = word
                .chars()
                .rev()
                .take_while(|c| EDGE_PUNCT.contains(c))
                .find(|c| is_terminal(*c));
            if let Some(punct) = terminal {
                if let Some(previous) = out.last_mut() {
                    attach_terminal_punct(previous, punct);
                }
                capitalize_next = true;
            }
            continue;
        }
        let mut w = word.to_string();
        if capitalize_next {
            w = capitalize_first(&w);
            capitalize_next = false;
        }
        out.push(w);
    }
    out.join(" ")
}

/// Spoken formatting commands. Each is only recognized as a command when it
/// stands alone between punctuation marks (or at the text edges), so "это
/// новая строка в таблице" keeps its words while "Привет. Новая строка. Как
/// дела." breaks the line.
const VOICE_COMMANDS: &[(&[&str], &str)] = &[
    (&["новая", "строка"], "\r\n"),
    (&["с", "новой", "строки"], "\r\n"),
    (&["перенос", "строки"], "\r\n"),
    (&["new", "line"], "\r\n"),
    (&["новый", "абзац"], "\r\n\r\n"),
    (&["new", "paragraph"], "\r\n\r\n"),
];

fn match_voice_command(
    tokens: &[&str],
    cores: &[String],
    i: usize,
) -> Option<(usize, &'static str)> {
    for (phrase, brk) in VOICE_COMMANDS {
        let n = phrase.len();
        if i + n > tokens.len() {
            continue;
        }
        if !(0..n).all(|k| cores[i + k] == phrase[k]) {
            continue;
        }
        let previous_ok = i == 0 || ends_with_punct(tokens[i - 1]);
        let last_ok = i + n == tokens.len() || ends_with_punct(tokens[i + n - 1]);
        if previous_ok && last_ok {
            return Some((n, brk));
        }
    }
    None
}

/// Turn spoken formatting commands into line breaks (CRLF, which every
/// Windows text control accepts). Breaks at the very start or end are dropped.
pub fn apply_voice_commands(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let cores: Vec<String> = tokens
        .iter()
        .map(|w| {
            w.trim_matches(|c: char| EDGE_PUNCT.contains(&c))
                .to_lowercase()
        })
        .collect();

    let mut out = String::new();
    let mut pending_break: Option<&str> = None;
    let mut capitalize_next = false;
    let mut i = 0;
    while i < tokens.len() {
        if let Some((len, brk)) = match_voice_command(&tokens, &cores, i) {
            pending_break = Some(match pending_break {
                Some(existing) if existing.len() >= brk.len() => existing,
                _ => brk,
            });
            capitalize_next = true;
            i += len;
            continue;
        }
        let mut w = tokens[i].to_string();
        if capitalize_next {
            w = capitalize_first(&w);
            capitalize_next = false;
        }
        if !out.is_empty() {
            match pending_break.take() {
                Some(brk) => out.push_str(brk),
                None => out.push(' '),
            }
        }
        out.push_str(&w);
        i += 1;
    }
    out
}

/// Append the configured suffix unless the text already ends with whitespace.
pub fn apply_paste_suffix(text: &str, suffix: PasteSuffix) -> String {
    if text.is_empty() || text.ends_with(char::is_whitespace) {
        return text.to_string();
    }
    match suffix {
        PasteSuffix::Space => format!("{text} "),
        PasteSuffix::Newline => format!("{text}\r\n"),
        PasteSuffix::None => text.to_string(),
    }
}

/// The ordered post-processing steps for one utterance.
pub struct TextPipeline {
    pub voice_commands: bool,
    pub vocabulary: Vocabulary,
}

impl TextPipeline {
    /// Final text: fillers → voice commands → dictionary → trim.
    pub fn finalize(&self, raw: &str) -> String {
        let text = remove_fillers(raw);
        let text = if self.voice_commands {
            apply_voice_commands(&text)
        } else {
            text
        };
        self.vocabulary.apply(&text).trim().to_string()
    }

    /// Preview text while recording: fillers → dictionary (no line breaks,
    /// the preview is a single running line).
    pub fn preview(&self, raw: &str) -> String {
        self.vocabulary
            .apply(&remove_fillers(raw))
            .trim()
            .to_string()
    }
}

#[cfg(test)]
mod filler_tests {
    use super::remove_fillers;

    #[test]
    fn removes_interjections_and_carries_capitalization() {
        assert_eq!(
            remove_fillers("Эм, привет, э, как дела?"),
            "Привет, как дела?"
        );
        assert_eq!(
            remove_fillers("Um, hello there, uh, okay"),
            "Hello there, okay"
        );
    }

    #[test]
    fn carries_terminal_punctuation_to_the_previous_word() {
        assert_eq!(remove_fillers("Привет э. Как дела"), "Привет. Как дела");
        assert_eq!(remove_fillers("Это важно, эм."), "Это важно.");
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
    fn keeps_units_and_abbreviations_that_look_like_fillers() {
        assert_eq!(remove_fillers("толщина стенки 2 мм"), "толщина стенки 2 мм");
        assert_eq!(remove_fillers("went to the ER."), "went to the ER.");
    }

    #[test]
    fn handles_empty_and_filler_only() {
        assert_eq!(remove_fillers("эм... ээ"), "");
        assert_eq!(remove_fillers(""), "");
    }
}

#[cfg(test)]
mod voice_command_tests {
    use super::apply_voice_commands;

    #[test]
    fn new_line_between_sentences() {
        assert_eq!(
            apply_voice_commands("Привет. Новая строка. Как дела."),
            "Привет.\r\nКак дела."
        );
        assert_eq!(
            apply_voice_commands("Привет, новая строка, как дела"),
            "Привет,\r\nКак дела"
        );
    }

    #[test]
    fn new_paragraph_in_english() {
        assert_eq!(
            apply_voice_commands("First line. New paragraph. second line"),
            "First line.\r\n\r\nSecond line"
        );
    }

    #[test]
    fn command_at_the_edges() {
        assert_eq!(apply_voice_commands("Новая строка. Привет"), "Привет");
        assert_eq!(apply_voice_commands("Привет. Новая строка."), "Привет.");
    }

    #[test]
    fn phrase_inside_a_sentence_is_not_a_command() {
        assert_eq!(
            apply_voice_commands("Это новая строка в таблице"),
            "Это новая строка в таблице"
        );
        assert_eq!(
            apply_voice_commands("add a new line to the file"),
            "add a new line to the file"
        );
    }
}

#[cfg(test)]
mod suffix_tests {
    use super::{apply_paste_suffix, PasteSuffix};

    #[test]
    fn appends_the_configured_suffix() {
        assert_eq!(
            apply_paste_suffix("Привет.", PasteSuffix::Space),
            "Привет. "
        );
        assert_eq!(
            apply_paste_suffix("Привет.", PasteSuffix::Newline),
            "Привет.\r\n"
        );
        assert_eq!(apply_paste_suffix("Привет.", PasteSuffix::None), "Привет.");
    }

    #[test]
    fn never_doubles_whitespace_or_touches_empty_text() {
        assert_eq!(
            apply_paste_suffix("Привет. ", PasteSuffix::Space),
            "Привет. "
        );
        assert_eq!(
            apply_paste_suffix("line\r\n", PasteSuffix::Newline),
            "line\r\n"
        );
        assert_eq!(apply_paste_suffix("", PasteSuffix::Space), "");
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::TextPipeline;
    use crate::transcription::replacements::{ReplacementRule, Vocabulary};

    fn pipeline() -> TextPipeline {
        TextPipeline {
            voice_commands: true,
            vocabulary: Vocabulary::compile(&[ReplacementRule {
                from: "три джей эс".into(),
                to: "Three.js".into(),
                whole_word: true,
                case_insensitive: true,
            }]),
        }
    }

    #[test]
    fn finalize_runs_every_step_in_order() {
        assert_eq!(
            pipeline().finalize("Эм, сцена на три джей эс. Новая строка. Готово."),
            "Сцена на Three.js.\r\nГотово."
        );
    }

    #[test]
    fn preview_keeps_a_single_line() {
        assert_eq!(
            pipeline().preview("эм сцена на Три Джей Эс новая строка готово"),
            "Сцена на Three.js новая строка готово"
        );
    }
}
