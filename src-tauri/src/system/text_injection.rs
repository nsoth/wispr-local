//! Paste the transcript into the focused application through the clipboard:
//! snapshot the clipboard, put the text in it, send Ctrl+V, restore the
//! snapshot a moment later on a background thread.

use arboard::{Clipboard, ImageData, SetExtWindows};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use std::borrow::Cow;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use super::focus;

/// How long slower applications (JetBrains while indexing, RDP sessions) get
/// to read the clipboard before the original content comes back.
const RESTORE_DELAY: Duration = Duration::from_millis(800);
/// How long to wait for the user to lift the hotkey's modifiers before the
/// synthesized Ctrl+V would turn into Ctrl+Shift+V.
const MODIFIER_WAIT: Duration = Duration::from_millis(500);

/// The clipboard content worth putting back. Richer formats come first so a
/// copied file selection or formatted text survives a dictation.
enum Snapshot {
    Files(Vec<PathBuf>),
    Html {
        html: String,
        alt: Option<String>,
    },
    Text(String),
    Image {
        width: usize,
        height: usize,
        bytes: Vec<u8>,
    },
    Empty,
}

fn snapshot(clipboard: &mut Clipboard) -> Snapshot {
    if let Ok(files) = clipboard.get().file_list() {
        if !files.is_empty() {
            return Snapshot::Files(files);
        }
    }
    if let Ok(html) = clipboard.get().html() {
        if !html.is_empty() {
            let alt = clipboard.get().text().ok();
            return Snapshot::Html { html, alt };
        }
    }
    if let Ok(text) = clipboard.get().text() {
        return Snapshot::Text(text);
    }
    if let Ok(image) = clipboard.get().image() {
        return Snapshot::Image {
            width: image.width,
            height: image.height,
            bytes: image.bytes.into_owned(),
        };
    }
    Snapshot::Empty
}

fn restore(clipboard: &mut Clipboard, saved: Snapshot) -> Result<(), arboard::Error> {
    match saved {
        Snapshot::Files(files) => clipboard
            .set()
            .exclude_from_history()
            .exclude_from_cloud()
            .file_list(&files),
        Snapshot::Html { html, alt } => clipboard
            .set()
            .exclude_from_history()
            .exclude_from_cloud()
            .html(html, alt),
        Snapshot::Text(text) => clipboard
            .set()
            .exclude_from_history()
            .exclude_from_cloud()
            .text(text),
        Snapshot::Image {
            width,
            height,
            bytes,
        } => clipboard
            .set()
            .exclude_from_history()
            .exclude_from_cloud()
            .image(ImageData {
                width,
                height,
                bytes: Cow::Owned(bytes),
            }),
        Snapshot::Empty => Ok(()),
    }
}

/// Inject text into the currently focused application using clipboard-paste:
/// 1. Snapshot the clipboard (files / HTML / text / image)
/// 2. Put the transcript in it, hidden from Clipboard History and cloud sync
/// 3. Wait for the hotkey's modifiers to be released, then send Ctrl+V
/// 4. Restore the snapshot after a delay, off the pipeline thread
pub fn inject_text(text: &str, restore_clipboard: bool) -> Result<(), String> {
    let mut clipboard = Clipboard::new().map_err(|e| format!("Failed to open clipboard: {}", e))?;

    let saved = if restore_clipboard {
        snapshot(&mut clipboard)
    } else {
        Snapshot::Empty
    };

    clipboard
        .set()
        .exclude_from_history()
        .exclude_from_cloud()
        .text(text)
        .map_err(|e| format!("Failed to set clipboard text: {}", e))?;

    // Small delay to ensure clipboard is ready
    thread::sleep(Duration::from_millis(50));

    if !focus::wait_for_modifiers_released(MODIFIER_WAIT) {
        log::warn!("Modifier keys still held after {MODIFIER_WAIT:?}; sending Ctrl+V anyway");
    }

    let paste_result = Enigo::new(&Settings::default())
        .map_err(|e| format!("Failed to create enigo: {e}"))
        .and_then(|mut enigo| send_paste_shortcut(&mut enigo));

    // Restore only after a paste that was actually sent; on failure the
    // transcript stays in the clipboard for a manual Ctrl+V.
    if paste_result.is_ok() && !matches!(saved, Snapshot::Empty) {
        thread::Builder::new()
            .name("wispr-clipboard-restore".into())
            .spawn(move || {
                thread::sleep(RESTORE_DELAY);
                match Clipboard::new() {
                    Ok(mut clipboard) => {
                        if let Err(e) = restore(&mut clipboard, saved) {
                            log::warn!("Could not restore the clipboard: {e}");
                        }
                    }
                    Err(e) => log::warn!("Could not reopen the clipboard to restore it: {e}"),
                }
            })
            .map_err(|e| format!("Failed to spawn clipboard restore thread: {e}"))?;
    }

    paste_result
}

/// Leave the transcript in the clipboard for a manual paste (no restore).
/// It stays visible in Clipboard History on purpose but never syncs to the
/// cloud clipboard.
pub fn copy_only(text: &str) -> Result<(), String> {
    let mut clipboard = Clipboard::new().map_err(|e| format!("Failed to open clipboard: {}", e))?;
    clipboard
        .set()
        .exclude_from_cloud()
        .text(text)
        .map_err(|e| format!("Failed to set clipboard text: {}", e))
}

/// Send Ctrl+V while guaranteeing that any successfully pressed keys are
/// released even if a later Enigo call fails.
fn send_paste_shortcut(enigo: &mut Enigo) -> Result<(), String> {
    const VK_CONTROL: u32 = 0x11;
    const VK_V: u32 = 0x56;

    let mut ctrl_pressed = false;
    let mut v_pressed = false;
    let result = (|| {
        enigo
            .key(Key::Other(VK_CONTROL), Direction::Press)
            .map_err(|e| format!("Failed to press Ctrl: {e}"))?;
        ctrl_pressed = true;
        enigo
            .key(Key::Other(VK_V), Direction::Press)
            .map_err(|e| format!("Failed to press V: {e}"))?;
        v_pressed = true;
        enigo
            .key(Key::Other(VK_V), Direction::Release)
            .map_err(|e| format!("Failed to release V: {e}"))?;
        v_pressed = false;
        enigo
            .key(Key::Other(VK_CONTROL), Direction::Release)
            .map_err(|e| format!("Failed to release Ctrl: {e}"))?;
        ctrl_pressed = false;
        Ok(())
    })();

    if v_pressed {
        let _ = enigo.key(Key::Other(VK_V), Direction::Release);
    }
    if ctrl_pressed {
        let _ = enigo.key(Key::Other(VK_CONTROL), Direction::Release);
    }
    result
}
