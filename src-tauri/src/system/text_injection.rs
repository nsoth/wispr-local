use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use std::thread;
use std::time::Duration;

/// Inject text into the currently focused application using clipboard-paste:
/// 1. Save current clipboard
/// 2. Set clipboard to transcribed text
/// 3. Simulate Ctrl+V
/// 4. Wait for paste to complete
/// 5. Restore original clipboard
pub fn inject_text(text: &str) -> Result<(), String> {
    let mut clipboard =
        Clipboard::new().map_err(|e| format!("Failed to open clipboard: {}", e))?;

    // Save current clipboard contents
    let saved_text = clipboard.get_text().ok();

    // Set transcribed text to clipboard
    clipboard
        .set_text(text)
        .map_err(|e| format!("Failed to set clipboard text: {}", e))?;

    // Small delay to ensure clipboard is ready
    thread::sleep(Duration::from_millis(50));

    // Simulate Ctrl+V using raw Windows virtual key codes
    // (Key::Unicode can fail with TryFromIntError on some systems)
    let mut enigo =
        Enigo::new(&Settings::default()).map_err(|e| format!("Failed to create enigo: {}", e))?;

    // VK_CONTROL = 0x11, VK_V = 0x56
    enigo
        .key(Key::Other(0x11), Direction::Press)
        .map_err(|e| format!("Failed to press Ctrl: {}", e))?;
    enigo
        .key(Key::Other(0x56), Direction::Press)
        .map_err(|e| format!("Failed to press V: {}", e))?;
    enigo
        .key(Key::Other(0x56), Direction::Release)
        .map_err(|e| format!("Failed to release V: {}", e))?;
    enigo
        .key(Key::Other(0x11), Direction::Release)
        .map_err(|e| format!("Failed to release Ctrl: {}", e))?;

    // Wait for paste to complete
    thread::sleep(Duration::from_millis(300));

    // Restore original clipboard (best-effort)
    if let Some(original) = saved_text {
        let _ = clipboard.set_text(&original);
    }

    Ok(())
}
