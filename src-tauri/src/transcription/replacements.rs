//! User dictionary: ordered find/replace rules applied to the transcript so
//! product names and terms the model transliterates ("три джей эс") come out
//! the way the user writes them ("Three.js").

use regex::{NoExpand, Regex};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplacementRule {
    pub from: String,
    pub to: String,
    #[serde(default = "default_true")]
    pub whole_word: bool,
    #[serde(default = "default_true")]
    pub case_insensitive: bool,
}

fn default_true() -> bool {
    true
}

/// Compiled rules, applied in order.
pub struct Vocabulary {
    compiled: Vec<(Regex, String)>,
}

impl Vocabulary {
    /// Compile the rules; a rule that cannot be compiled is skipped with a log
    /// line rather than disabling the whole dictionary.
    pub fn compile(rules: &[ReplacementRule]) -> Vocabulary {
        let mut compiled = Vec::with_capacity(rules.len());
        for rule in rules {
            let from = rule.from.trim();
            if from.is_empty() {
                continue;
            }
            let mut pattern = String::new();
            if rule.case_insensitive {
                pattern.push_str("(?i)");
            }
            // `\b` only works next to a word character; a term such as "C++"
            // gets a boundary on its letter side only.
            let starts_word = from.chars().next().is_some_and(is_word_char);
            let ends_word = from.chars().last().is_some_and(is_word_char);
            if rule.whole_word && starts_word {
                pattern.push_str(r"\b");
            }
            let words: Vec<String> = from.split_whitespace().map(regex::escape).collect();
            pattern.push_str(&words.join(r"\s+"));
            if rule.whole_word && ends_word {
                pattern.push_str(r"\b");
            }
            match Regex::new(&pattern) {
                Ok(re) => compiled.push((re, rule.to.clone())),
                Err(e) => log::warn!("Dictionary rule '{}' skipped: {e}", rule.from),
            }
        }
        Vocabulary { compiled }
    }

    pub fn is_empty(&self) -> bool {
        self.compiled.is_empty()
    }

    pub fn apply(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (re, to) in &self.compiled {
            out = re.replace_all(&out, NoExpand(to)).into_owned();
        }
        out
    }
}

/// Starter dictionary for a web/3D developer dictating Russian with English
/// terms. Only product names that are always written in Latin letters.
pub fn default_rules() -> Vec<ReplacementRule> {
    const SEED: &[(&str, &str)] = &[
        ("три джей эс", "Three.js"),
        ("три джейэс", "Three.js"),
        ("трижейэс", "Three.js"),
        ("three js", "Three.js"),
        ("веб джи эль", "WebGL"),
        ("вебджиэль", "WebGL"),
        ("вебгл", "WebGL"),
        ("web gl", "WebGL"),
        ("таури", "Tauri"),
        ("реакт", "React"),
        ("блендер", "Blender"),
        ("юнити", "Unity"),
        ("анрил", "Unreal"),
        ("гитхаб", "GitHub"),
        ("фигма", "Figma"),
        ("вс код", "VS Code"),
        ("ви эс код", "VS Code"),
        ("эр три эф", "R3F"),
        ("глтф", "glTF"),
        ("джи эл ти эф", "glTF"),
    ];
    SEED.iter()
        .map(|(from, to)| ReplacementRule {
            from: (*from).to_string(),
            to: (*to).to_string(),
            whole_word: true,
            case_insensitive: true,
        })
        .collect()
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::{default_rules, ReplacementRule, Vocabulary};

    fn rule(from: &str, to: &str, whole_word: bool, case_insensitive: bool) -> ReplacementRule {
        ReplacementRule {
            from: from.into(),
            to: to.into(),
            whole_word,
            case_insensitive,
        }
    }

    #[test]
    fn whole_word_rules_do_not_touch_longer_words() {
        let v = Vocabulary::compile(&[rule("меш", "mesh", true, true)]);
        assert_eq!(v.apply("этот меш и мешать"), "этот mesh и мешать");
        assert_eq!(v.apply("Меш."), "mesh.");
    }

    #[test]
    fn substring_rules_match_inside_words() {
        let v = Vocabulary::compile(&[rule("меш", "mesh", false, true)]);
        assert_eq!(v.apply("мешать"), "meshать");
    }

    #[test]
    fn case_sensitivity_is_per_rule() {
        let v = Vocabulary::compile(&[rule("api", "API", true, false)]);
        assert_eq!(v.apply("api and Api"), "API and Api");
    }

    #[test]
    fn multi_word_phrases_tolerate_extra_spaces_and_punctuation_edges() {
        let v = Vocabulary::compile(&[rule("три джей эс", "Three.js", true, true)]);
        assert_eq!(
            v.apply("сцена на Три  Джей Эс, готово"),
            "сцена на Three.js, готово"
        );
    }

    #[test]
    fn non_word_edges_still_match_whole_words() {
        let v = Vocabulary::compile(&[rule("си плюс плюс", "C++", true, true)]);
        assert_eq!(v.apply("пишу на си плюс плюс сейчас"), "пишу на C++ сейчас");
        let back = Vocabulary::compile(&[rule("C++", "си плюс плюс", true, true)]);
        assert_eq!(back.apply("на C++ пишу"), "на си плюс плюс пишу");
    }

    #[test]
    fn replacement_text_is_literal() {
        let v = Vocabulary::compile(&[rule("цена", "$100", true, true)]);
        assert_eq!(v.apply("цена вопроса"), "$100 вопроса");
    }

    #[test]
    fn rules_apply_in_order_and_empty_from_is_skipped() {
        let v = Vocabulary::compile(&[
            rule("a", "b", true, true),
            rule("b", "c", true, true),
            rule("", "x", true, true),
        ]);
        assert_eq!(v.apply("a"), "c");
    }

    #[test]
    fn default_rules_cover_the_owner_stack() {
        let v = Vocabulary::compile(&default_rules());
        assert_eq!(
            v.apply("сцена на три джей эс и вебджиэль"),
            "сцена на Three.js и WebGL"
        );
        assert_eq!(v.apply("приложение на таури"), "приложение на Tauri");
        assert!(!v.is_empty());
    }
}
