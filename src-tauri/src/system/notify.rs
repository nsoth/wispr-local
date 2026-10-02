//! Windows toast notifications under the app's own AppUserModelId.
//!
//! The notification plugin skips the app id for executables that live under
//! `target\release`, so every toast used to be attributed to "Windows
//! PowerShell" (its icon, its name, its mute switch in Settings). This module
//! registers `com.wispr-local.app` in the per-user registry with a display
//! name and icon, tags the process with it, and sends toasts through WinRT
//! directly. Clicking a toast opens the main window.

use std::path::{Path, PathBuf};

pub const APP_USER_MODEL_ID: &str = "com.wispr-local.app";
pub const DISPLAY_NAME: &str = "Wispr Local";
const ICON_FILE: &str = "icon.png";
const ICON_BYTES: &[u8] = include_bytes!("../../icons/icon.png");

/// Write the toast icon next to the settings file (once). Returns the path
/// and whether the file was (re)written.
pub fn ensure_icon_file(data_dir: &Path) -> Result<(PathBuf, bool), String> {
    let path = data_dir.join(ICON_FILE);
    let up_to_date = std::fs::metadata(&path)
        .map(|m| m.len() == ICON_BYTES.len() as u64)
        .unwrap_or(false);
    if up_to_date {
        return Ok((path, false));
    }
    crate::config::write_file_atomic(&path, ICON_BYTES)
        .map_err(|e| format!("Could not write the notification icon: {e}"))?;
    Ok((path, true))
}

/// Registry values under `HKCU\Software\Classes\AppUserModelId\<id>`.
pub fn aumid_registry_values(icon_path: &Path) -> Vec<(&'static str, String)> {
    vec![
        ("DisplayName", DISPLAY_NAME.to_string()),
        ("IconUri", icon_path.to_string_lossy().to_string()),
    ]
}

/// Register the identity (idempotent) and tag the current process with it.
#[cfg(windows)]
pub fn register_app_identity(data_dir: &Path) -> Result<(), String> {
    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};
    use winreg::RegKey;

    let (icon_path, written) = ensure_icon_file(data_dir)?;
    if written {
        log::info!("Wrote the notification icon to {}", icon_path.display());
    }

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let (key, _) = hkcu
        .create_subkey_with_flags(
            format!("Software\\Classes\\AppUserModelId\\{APP_USER_MODEL_ID}"),
            KEY_READ | KEY_WRITE,
        )
        .map_err(|e| format!("Could not open the AppUserModelId registry key: {e}"))?;
    for (name, value) in aumid_registry_values(&icon_path) {
        let current: Option<String> = key.get_value(name).ok();
        if current.as_deref() != Some(value.as_str()) {
            key.set_value(name, &value)
                .map_err(|e| format!("Could not write {name}: {e}"))?;
            log::info!("Registered notification identity {name} = {value}");
        }
    }

    tag_current_process()
}

#[cfg(not(windows))]
pub fn register_app_identity(_data_dir: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(windows)]
fn tag_current_process() -> Result<(), String> {
    use windows_sys::Win32::UI::Shell::SetCurrentProcessExplicitAppUserModelID;
    let wide: Vec<u16> = APP_USER_MODEL_ID
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let hr = unsafe { SetCurrentProcessExplicitAppUserModelID(wide.as_ptr()) };
    if hr < 0 {
        return Err(format!(
            "SetCurrentProcessExplicitAppUserModelID failed (HRESULT {hr:#x})"
        ));
    }
    Ok(())
}

/// Show a toast attributed to Wispr Local. `long` keeps session-level
/// messages (CPU fallback, model problems) on screen for ~25 s instead of 7.
#[cfg(windows)]
pub fn show_toast(app: tauri::AppHandle, body: &str, long: bool) -> Result<(), String> {
    use tauri::Manager;
    use tauri_winrt_notification::{Duration, Toast};

    Toast::new(APP_USER_MODEL_ID)
        .title(DISPLAY_NAME)
        .text1(body)
        .duration(if long {
            Duration::Long
        } else {
            Duration::Short
        })
        .sound(None)
        .on_activated(move |_| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
            Ok(())
        })
        .show()
        .map_err(|e| format!("toast failed: {e}"))
}

#[cfg(not(windows))]
pub fn show_toast(_app: tauri::AppHandle, _body: &str, _long: bool) -> Result<(), String> {
    Err("toasts are only implemented on Windows".to_string())
}

#[cfg(test)]
mod tests {
    use super::{aumid_registry_values, ensure_icon_file, DISPLAY_NAME};
    use crate::config::test_dir;

    #[test]
    fn icon_is_written_once() {
        let dir = test_dir("icon");
        let (path, written) = ensure_icon_file(&dir).unwrap();
        assert!(written);
        assert!(path.exists());
        let (_, written_again) = ensure_icon_file(&dir).unwrap();
        assert!(!written_again, "unchanged icon is not rewritten");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn registry_values_name_the_app_and_point_at_the_icon() {
        let values = aumid_registry_values(std::path::Path::new(r"C:\data\icon.png"));
        assert_eq!(values[0], ("DisplayName", DISPLAY_NAME.to_string()));
        assert_eq!(values[1], ("IconUri", r"C:\data\icon.png".to_string()));
    }
}
