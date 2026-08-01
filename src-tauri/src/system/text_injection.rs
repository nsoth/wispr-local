use arboard::{Clipboard, ImageData};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use std::borrow::Cow;
use std::thread;
use std::time::Duration;

/// Inject text into the currently focused application using clipboard-paste:
/// 1. Save current clipboard
/// 2. Set clipboard to transcribed text
/// 3. Simulate Ctrl+V
/// 4. Wait for paste to complete
/// 5. Restore original clipboard
pub fn inject_text(text: &str) -> Result<(), String> {
    let mut clipboard = Clipboard::new().map_err(|e| format!("Failed to open clipboard: {}", e))?;

    // Preserve the two clipboard types this app can safely round-trip. This
    // avoids destroying a copied screenshot when no text representation is
    // available.
    let saved = clipboard
        .get_text()
        .ok()
        .map(SavedClipboard::Text)
        .or_else(|| {
            clipboard
                .get_image()
                .ok()
                .map(|image| SavedClipboard::Image {
                    width: image.width,
                    height: image.height,
                    bytes: image.bytes.into_owned(),
                })
        });

    // Set transcribed text to clipboard
    clipboard
        .set_text(text)
        .map_err(|e| format!("Failed to set clipboard text: {}", e))?;

    // Small delay to ensure clipboard is ready
    thread::sleep(Duration::from_millis(50));

    // Simulate Ctrl+V using raw Windows virtual key codes
    // (Key::Unicode can fail with TryFromIntError on some systems)
    let paste_result = Enigo::new(&Settings::default())
        .map_err(|e| format!("Failed to create enigo: {e}"))
        .and_then(|mut enigo| send_paste_shortcut(&mut enigo));

    if paste_result.is_ok() {
        // Give slower applications time to consume clipboard data before it is
        // restored. The paste shortcut itself remains fast.
        thread::sleep(Duration::from_millis(350));
    }

    // Restore original clipboard (best-effort)
    if let Some(original) = saved {
        match original {
            SavedClipboard::Text(text) => {
                let _ = clipboard.set_text(text);
            }
            SavedClipboard::Image {
                width,
                height,
                bytes,
            } => {
                let _ = clipboard.set_image(ImageData {
                    width,
                    height,
                    bytes: Cow::Owned(bytes),
                });
            }
        }
    }

    paste_result
}

enum SavedClipboard {
    Text(String),
    Image {
        width: usize,
        height: usize,
        bytes: Vec<u8>,
    },
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
