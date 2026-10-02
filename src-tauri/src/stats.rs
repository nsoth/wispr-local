//! Usage statistics: one JSON line per dictation appended to `stats.jsonl`
//! in the data directory, read back as "today / 7 days" totals for the main
//! window. Nothing else depends on the file; a missing or damaged line is
//! simply skipped.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

pub const STATS_FILE: &str = "stats.jsonl";

/// One finished dictation attempt, whatever its outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatRecord {
    /// Unix time in milliseconds.
    pub ts: u64,
    pub audio_s: f32,
    pub words: u32,
    /// `pasted`, `copied`, `too-short`, `no-speech`, `failed` or `cancelled`.
    pub outcome: String,
    pub lang: String,
}

/// Totals over a time window.
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct StatsSummary {
    pub dictations: u32,
    pub words: u32,
    pub audio_s: f32,
    /// Attempts that produced nothing (too short, no speech, failed).
    pub no_result: u32,
}

pub fn count_words(text: &str) -> u32 {
    text.split_whitespace().count() as u32
}

/// Outcomes that gave the user no text at all.
pub fn is_no_result(outcome: &str) -> bool {
    matches!(outcome, "too-short" | "no-speech" | "failed")
}

/// Sum every record with `ts >= since_ms`; unparsable lines are ignored.
pub fn aggregate(lines: &str, since_ms: u64) -> StatsSummary {
    let mut summary = StatsSummary::default();
    for line in lines.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<StatRecord>(line) else {
            continue;
        };
        if record.ts < since_ms {
            continue;
        }
        summary.dictations += 1;
        summary.words += record.words;
        summary.audio_s += record.audio_s;
        if is_no_result(&record.outcome) {
            summary.no_result += 1;
        }
    }
    summary
}

pub fn append(data_dir: &Path, record: &StatRecord) -> Result<(), String> {
    let path = data_dir.join(STATS_FILE);
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut line = serde_json::to_string(record).map_err(|e| e.to_string())?;
    line.push('\n');
    file.write_all(line.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Totals since `since_ms` from the stats file (empty when there is none).
pub fn summarize(data_dir: &Path, since_ms: u64) -> StatsSummary {
    match std::fs::read_to_string(data_dir.join(STATS_FILE)) {
        Ok(contents) => aggregate(&contents, since_ms),
        Err(_) => StatsSummary::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::{aggregate, append, count_words, summarize, StatRecord, StatsSummary};

    fn line(ts: u64, audio_s: f32, words: u32, outcome: &str) -> String {
        format!(
            r#"{{"ts":{ts},"audio_s":{audio_s},"words":{words},"outcome":"{outcome}","lang":"ru"}}"#
        )
    }

    #[test]
    fn aggregate_counts_only_records_since_the_cutoff() {
        let lines = [
            line(100, 4.0, 10, "pasted"),
            line(200, 6.0, 15, "no-speech"),
            line(300, 2.0, 5, "copied"),
        ]
        .join("\n");
        assert_eq!(
            aggregate(&lines, 200),
            StatsSummary {
                dictations: 2,
                words: 20,
                audio_s: 8.0,
                no_result: 1,
            }
        );
        assert_eq!(aggregate(&lines, 301), StatsSummary::default());
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let lines = format!(
            "{}\nnot json at all\n{{\"ts\":5}}\n\n{}\n",
            line(10, 1.0, 1, "failed"),
            line(20, 1.5, 3, "too-short")
        );
        let summary = aggregate(&lines, 0);
        assert_eq!(summary.dictations, 2);
        assert_eq!(summary.no_result, 2);
        assert_eq!(summary.words, 4);
    }

    #[test]
    fn count_words_splits_on_any_whitespace() {
        assert_eq!(count_words("Привет,  мир\nThree.js"), 3);
        assert_eq!(count_words("   "), 0);
    }

    #[test]
    fn append_then_summarize_round_trips() {
        let dir = crate::config::test_dir("stats");
        assert_eq!(summarize(&dir, 0), StatsSummary::default(), "no file yet");
        for (ts, outcome) in [(1_000, "pasted"), (2_000, "too-short")] {
            append(
                &dir,
                &StatRecord {
                    ts,
                    audio_s: 3.0,
                    words: 7,
                    outcome: outcome.to_string(),
                    lang: "en".to_string(),
                },
            )
            .expect("append");
        }
        let summary = summarize(&dir, 0);
        assert_eq!(summary.dictations, 2);
        assert_eq!(summary.words, 14);
        assert_eq!(summary.no_result, 1);
        assert_eq!(summarize(&dir, 1_500).dictations, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
