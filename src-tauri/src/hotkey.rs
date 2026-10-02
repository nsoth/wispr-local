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
use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut};

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

/// Parse a hotkey string like "Ctrl+Shift+Space" into a tauri Shortcut.
pub fn parse_hotkey(hotkey: &str) -> Result<Shortcut, String> {
    let parts: Vec<&str> = hotkey.split('+').map(|s| s.trim()).collect();
    if parts.is_empty() {
        return Err("Empty hotkey".to_string());
    }

    let mut modifiers = Modifiers::empty();
    let mut key_code: Option<Code> = None;

    for part in &parts {
        match part.to_lowercase().as_str() {
            "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
            "shift" => modifiers |= Modifiers::SHIFT,
            "alt" => modifiers |= Modifiers::ALT,
            "super" | "win" | "meta" | "cmd" => modifiers |= Modifiers::SUPER,
            key => {
                if key_code.is_some() {
                    return Err(format!("Multiple keys in hotkey: {}", hotkey));
                }
                key_code = Some(parse_key_code(key)?);
            }
        }
    }

    let code = key_code.ok_or_else(|| format!("No key specified in hotkey: {}", hotkey))?;
    let mods = if modifiers.is_empty() {
        None
    } else {
        Some(modifiers)
    };

    Ok(Shortcut::new(mods, code))
}

pub fn parse_key_code(key: &str) -> Result<Code, String> {
    match key.to_lowercase().as_str() {
        "space" => Ok(Code::Space),
        "enter" | "return" => Ok(Code::Enter),
        "tab" => Ok(Code::Tab),
        "escape" | "esc" => Ok(Code::Escape),
        "backspace" => Ok(Code::Backspace),
        "delete" | "del" => Ok(Code::Delete),
        "insert" => Ok(Code::Insert),
        "home" => Ok(Code::Home),
        "end" => Ok(Code::End),
        "pageup" => Ok(Code::PageUp),
        "pagedown" => Ok(Code::PageDown),
        "up" => Ok(Code::ArrowUp),
        "down" => Ok(Code::ArrowDown),
        "left" => Ok(Code::ArrowLeft),
        "right" => Ok(Code::ArrowRight),
        "f1" => Ok(Code::F1),
        "f2" => Ok(Code::F2),
        "f3" => Ok(Code::F3),
        "f4" => Ok(Code::F4),
        "f5" => Ok(Code::F5),
        "f6" => Ok(Code::F6),
        "f7" => Ok(Code::F7),
        "f8" => Ok(Code::F8),
        "f9" => Ok(Code::F9),
        "f10" => Ok(Code::F10),
        "f11" => Ok(Code::F11),
        "f12" => Ok(Code::F12),
        "f13" => Ok(Code::F13),
        "f14" => Ok(Code::F14),
        "f15" => Ok(Code::F15),
        "f16" => Ok(Code::F16),
        "f17" => Ok(Code::F17),
        "f18" => Ok(Code::F18),
        "f19" => Ok(Code::F19),
        "f20" => Ok(Code::F20),
        "f21" => Ok(Code::F21),
        "f22" => Ok(Code::F22),
        "f23" => Ok(Code::F23),
        "f24" => Ok(Code::F24),
        "pause" => Ok(Code::Pause),
        "scrolllock" => Ok(Code::ScrollLock),
        "capslock" => Ok(Code::CapsLock),
        "numlock" => Ok(Code::NumLock),
        "printscreen" => Ok(Code::PrintScreen),
        "numpad0" => Ok(Code::Numpad0),
        "numpad1" => Ok(Code::Numpad1),
        "numpad2" => Ok(Code::Numpad2),
        "numpad3" => Ok(Code::Numpad3),
        "numpad4" => Ok(Code::Numpad4),
        "numpad5" => Ok(Code::Numpad5),
        "numpad6" => Ok(Code::Numpad6),
        "numpad7" => Ok(Code::Numpad7),
        "numpad8" => Ok(Code::Numpad8),
        "numpad9" => Ok(Code::Numpad9),
        "numpadadd" => Ok(Code::NumpadAdd),
        "numpadsubtract" => Ok(Code::NumpadSubtract),
        "numpadmultiply" => Ok(Code::NumpadMultiply),
        "numpaddivide" => Ok(Code::NumpadDivide),
        "numpaddecimal" => Ok(Code::NumpadDecimal),
        "numpadenter" => Ok(Code::NumpadEnter),
        "`" | "backquote" => Ok(Code::Backquote),
        "-" | "minus" => Ok(Code::Minus),
        "=" | "equal" => Ok(Code::Equal),
        "[" | "bracketleft" => Ok(Code::BracketLeft),
        "]" | "bracketright" => Ok(Code::BracketRight),
        "\\" | "backslash" => Ok(Code::Backslash),
        ";" | "semicolon" => Ok(Code::Semicolon),
        "'" | "quote" => Ok(Code::Quote),
        "," | "comma" => Ok(Code::Comma),
        "." | "period" => Ok(Code::Period),
        "/" | "slash" => Ok(Code::Slash),
        "0" => Ok(Code::Digit0),
        "1" => Ok(Code::Digit1),
        "2" => Ok(Code::Digit2),
        "3" => Ok(Code::Digit3),
        "4" => Ok(Code::Digit4),
        "5" => Ok(Code::Digit5),
        "6" => Ok(Code::Digit6),
        "7" => Ok(Code::Digit7),
        "8" => Ok(Code::Digit8),
        "9" => Ok(Code::Digit9),
        "a" => Ok(Code::KeyA),
        "b" => Ok(Code::KeyB),
        "c" => Ok(Code::KeyC),
        "d" => Ok(Code::KeyD),
        "e" => Ok(Code::KeyE),
        "f" => Ok(Code::KeyF),
        "g" => Ok(Code::KeyG),
        "h" => Ok(Code::KeyH),
        "i" => Ok(Code::KeyI),
        "j" => Ok(Code::KeyJ),
        "k" => Ok(Code::KeyK),
        "l" => Ok(Code::KeyL),
        "m" => Ok(Code::KeyM),
        "n" => Ok(Code::KeyN),
        "o" => Ok(Code::KeyO),
        "p" => Ok(Code::KeyP),
        "q" => Ok(Code::KeyQ),
        "r" => Ok(Code::KeyR),
        "s" => Ok(Code::KeyS),
        "t" => Ok(Code::KeyT),
        "u" => Ok(Code::KeyU),
        "v" => Ok(Code::KeyV),
        "w" => Ok(Code::KeyW),
        "x" => Ok(Code::KeyX),
        "y" => Ok(Code::KeyY),
        "z" => Ok(Code::KeyZ),
        other => Err(format!("Unknown key: {}", other)),
    }
}

/// Keys that never produce text, so they may be a hotkey on their own.
pub fn is_safe_bare_key(hotkey: &str) -> bool {
    let key = hotkey.trim().to_ascii_lowercase();
    if key == "pause" || key == "scrolllock" || key == "capslock" {
        return true;
    }
    key.strip_prefix('f')
        .and_then(|n| n.parse::<u32>().ok())
        .is_some_and(|n| (13..=24).contains(&n))
}

/// True when the string names at least one modifier key.
pub fn has_modifier(hotkey: &str) -> bool {
    hotkey.split('+').any(|part| {
        matches!(
            part.trim().to_ascii_lowercase().as_str(),
            "ctrl" | "control" | "shift" | "alt" | "super" | "win" | "meta" | "cmd"
        )
    })
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
