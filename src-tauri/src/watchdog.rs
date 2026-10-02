//! Detects a GPU that has become pathologically slow (the RTX 4060 pinned at
//! 10 W after unplug → sleep → replug, 2026-09-02) from the duration of the
//! streaming-preview ticks, so the user is told to check the charger instead
//! of staring at a silent app for 20 minutes.

/// A tick slower than this is suspicious regardless of history (a healthy
/// CUDA preview of ≤10 s of audio finishes in well under a second).
pub const MIN_SLOW_MS: f64 = 8_000.0;
/// ...or slower than this multiple of the established pace.
pub const SLOW_FACTOR: f64 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    Normal,
    Slow { elapsed_ms: f64, threshold_ms: f64 },
}

/// Learns the normal "milliseconds per audio second" of the machine from
/// healthy ticks and flags ticks that fall far outside it.
#[derive(Debug, Default, Clone)]
pub struct Watchdog {
    baseline_ms_per_s: Option<f64>,
}

impl Watchdog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, elapsed_ms: f64, audio_s: f64) -> Verdict {
        let audio_s = audio_s.max(0.1);
        let expected_ms = self.baseline_ms_per_s.map(|b| b * audio_s).unwrap_or(0.0);
        let threshold_ms = MIN_SLOW_MS.max(SLOW_FACTOR * expected_ms);
        if elapsed_ms > threshold_ms {
            return Verdict::Slow {
                elapsed_ms,
                threshold_ms,
            };
        }
        let rate = elapsed_ms / audio_s;
        self.baseline_ms_per_s = Some(match self.baseline_ms_per_s {
            Some(b) => 0.7 * b + 0.3 * rate,
            None => rate,
        });
        Verdict::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::{Verdict, Watchdog};

    #[test]
    fn healthy_ticks_are_normal_and_build_a_baseline() {
        let mut w = Watchdog::new();
        assert_eq!(w.observe(300.0, 3.0), Verdict::Normal);
        assert_eq!(w.observe(900.0, 9.0), Verdict::Normal);
        assert!((w.baseline_ms_per_s.unwrap() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn a_very_slow_first_tick_is_flagged_even_without_history() {
        let mut w = Watchdog::new();
        assert!(matches!(w.observe(35_100.0, 1.5), Verdict::Slow { .. }));
        assert!(
            w.baseline_ms_per_s.is_none(),
            "a slow tick never becomes the baseline"
        );
    }

    #[test]
    fn a_regression_against_the_learned_pace_is_flagged() {
        let mut w = Watchdog::new();
        for _ in 0..3 {
            assert_eq!(w.observe(1_000.0, 10.0), Verdict::Normal); // 100 ms / s
        }
        // 10 s of audio normally takes 1 s; 9 s is 9x slower → flagged.
        assert!(matches!(w.observe(9_000.0, 10.0), Verdict::Slow { .. }));
        // 4 s (4x) is within the factor but under the floor? 4000 < 8000 → normal.
        assert_eq!(w.observe(4_000.0, 10.0), Verdict::Normal);
    }

    #[test]
    fn a_consistently_slow_cpu_is_not_flagged_once_learned() {
        let mut w = Watchdog::new();
        // CPU mode: 2 s per audio second, every tick ≤ 8 s floor on 3 s audio.
        assert_eq!(w.observe(6_000.0, 3.0), Verdict::Normal);
        assert_eq!(w.observe(7_000.0, 3.5), Verdict::Normal);
        // 10 s of audio at the learned 2 s/s pace = 20 s expected; 25 s is fine.
        assert_eq!(w.observe(25_000.0, 10.0), Verdict::Normal);
    }
}
