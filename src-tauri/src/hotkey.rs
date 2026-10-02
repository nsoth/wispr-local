//! Hotkey semantics: what a press or release means in each mode.
//!
//! - `Hold`: press starts, release stops (push-to-talk). A recording pinned
//!   via the overlay ignores the release and stops on the next press.
//! - `Toggle`: press starts or stops; release does nothing.
//! - `Hybrid`: hold works like `Hold`, but a tap shorter than [`TAP_MAX_MS`]
//!   pins the recording (hands-free) instead of stopping it; the next press
//!   stops.
//!
//! The global-shortcut plugin occasionally delivers duplicate Pressed events
//! mid-hold (Windows update regression), so every rule tolerates a Pressed
//! while the key is already considered down.

use serde::{Deserialize, Serialize};

/// Releases quicker than this are taps, not holds.
pub const TAP_MAX_MS: u128 = 350;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HotkeyMode {
    #[default]
    Hold,
    Toggle,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyEvent {
    Pressed,
    Released,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyAction {
    Start,
    Stop,
    /// Keep recording hands-free (tap in Hybrid mode).
    Lock,
    Ignore,
}

#[derive(Debug, Clone, Copy)]
pub struct HotkeyContext {
    pub mode: HotkeyMode,
    pub recording: bool,
    pub locked: bool,
    /// Milliseconds since the recording started (0 when not recording).
    pub held_ms: u128,
    /// The key was already down according to the previous events.
    pub key_down: bool,
}

/// Returns the action and the new `key_down` flag.
pub fn decide(event: HotkeyEvent, ctx: HotkeyContext) -> (HotkeyAction, bool) {
    match (ctx.mode, event) {
        (HotkeyMode::Toggle, HotkeyEvent::Pressed) => {
            if ctx.key_down {
                // Duplicate Pressed from the plugin while the key is held.
                (HotkeyAction::Ignore, true)
            } else if ctx.recording {
                (HotkeyAction::Stop, true)
            } else {
                (HotkeyAction::Start, true)
            }
        }
        (HotkeyMode::Toggle, HotkeyEvent::Released) => (HotkeyAction::Ignore, false),

        (HotkeyMode::Hold | HotkeyMode::Hybrid, HotkeyEvent::Pressed) => {
            if ctx.recording && ctx.locked {
                // Pinned recording: a fresh press stops it.
                (HotkeyAction::Stop, true)
            } else if !ctx.recording {
                (HotkeyAction::Start, true)
            } else {
                // Key repeat / duplicate while holding.
                (HotkeyAction::Ignore, true)
            }
        }
        (HotkeyMode::Hold | HotkeyMode::Hybrid, HotkeyEvent::Released) => {
            if !ctx.recording || ctx.locked {
                (HotkeyAction::Ignore, false)
            } else if ctx.mode == HotkeyMode::Hybrid && ctx.held_ms < TAP_MAX_MS {
                (HotkeyAction::Lock, false)
            } else {
                (HotkeyAction::Stop, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{decide, HotkeyAction, HotkeyContext, HotkeyEvent, HotkeyMode, TAP_MAX_MS};

    fn ctx(
        mode: HotkeyMode,
        recording: bool,
        locked: bool,
        held_ms: u128,
        key_down: bool,
    ) -> HotkeyContext {
        HotkeyContext {
            mode,
            recording,
            locked,
            held_ms,
            key_down,
        }
    }

    #[test]
    fn hold_press_starts_and_release_stops() {
        let (a, down) = decide(
            HotkeyEvent::Pressed,
            ctx(HotkeyMode::Hold, false, false, 0, false),
        );
        assert_eq!((a, down), (HotkeyAction::Start, true));
        let (a, down) = decide(
            HotkeyEvent::Released,
            ctx(HotkeyMode::Hold, true, false, 2000, true),
        );
        assert_eq!((a, down), (HotkeyAction::Stop, false));
    }

    #[test]
    fn hold_ignores_duplicate_presses_while_recording() {
        let (a, _) = decide(
            HotkeyEvent::Pressed,
            ctx(HotkeyMode::Hold, true, false, 1000, true),
        );
        assert_eq!(a, HotkeyAction::Ignore);
    }

    #[test]
    fn hold_pinned_recording_ignores_release_and_stops_on_next_press() {
        let (a, _) = decide(
            HotkeyEvent::Released,
            ctx(HotkeyMode::Hold, true, true, 5000, true),
        );
        assert_eq!(a, HotkeyAction::Ignore);
        let (a, _) = decide(
            HotkeyEvent::Pressed,
            ctx(HotkeyMode::Hold, true, true, 9000, false),
        );
        assert_eq!(a, HotkeyAction::Stop);
    }

    #[test]
    fn toggle_press_starts_then_stops_and_release_is_inert() {
        let (a, down) = decide(
            HotkeyEvent::Pressed,
            ctx(HotkeyMode::Toggle, false, false, 0, false),
        );
        assert_eq!((a, down), (HotkeyAction::Start, true));
        let (a, down) = decide(
            HotkeyEvent::Released,
            ctx(HotkeyMode::Toggle, true, false, 100, true),
        );
        assert_eq!((a, down), (HotkeyAction::Ignore, false));
        let (a, down) = decide(
            HotkeyEvent::Pressed,
            ctx(HotkeyMode::Toggle, true, false, 4000, false),
        );
        assert_eq!((a, down), (HotkeyAction::Stop, true));
    }

    #[test]
    fn toggle_ignores_a_duplicate_press_while_the_key_is_down() {
        // The plugin's mid-hold duplicate Pressed must not stop the recording.
        let (a, down) = decide(
            HotkeyEvent::Pressed,
            ctx(HotkeyMode::Toggle, true, false, 500, true),
        );
        assert_eq!((a, down), (HotkeyAction::Ignore, true));
    }

    #[test]
    fn hybrid_tap_locks_and_hold_stops() {
        let (a, _) = decide(
            HotkeyEvent::Released,
            ctx(HotkeyMode::Hybrid, true, false, TAP_MAX_MS - 50, true),
        );
        assert_eq!(a, HotkeyAction::Lock);
        let (a, _) = decide(
            HotkeyEvent::Released,
            ctx(HotkeyMode::Hybrid, true, false, TAP_MAX_MS + 50, true),
        );
        assert_eq!(a, HotkeyAction::Stop);
        // Once locked by a tap, the next press stops.
        let (a, _) = decide(
            HotkeyEvent::Pressed,
            ctx(HotkeyMode::Hybrid, true, true, 6000, false),
        );
        assert_eq!(a, HotkeyAction::Stop);
        let (a, _) = decide(
            HotkeyEvent::Released,
            ctx(HotkeyMode::Hybrid, true, true, 6100, true),
        );
        assert_eq!(a, HotkeyAction::Ignore);
    }

    #[test]
    fn releases_without_a_recording_are_ignored_in_every_mode() {
        for mode in [HotkeyMode::Hold, HotkeyMode::Toggle, HotkeyMode::Hybrid] {
            let (a, down) = decide(HotkeyEvent::Released, ctx(mode, false, false, 0, true));
            assert_eq!((a, down), (HotkeyAction::Ignore, false), "{mode:?}");
        }
    }
}
