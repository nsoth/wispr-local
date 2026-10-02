//! "Start with Windows": a value under `HKCU\...\CurrentVersion\Run` that
//! points at this executable with `--hidden`.

use winreg::enums::*;
use winreg::RegKey;

const APP_NAME: &str = "Wispr Local";
const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";

/// The Run value: quoted path (spaces in directory names) plus --hidden so
/// the app detaches from the console on startup (debug builds would
/// otherwise leave a terminal window open).
pub fn autostart_value(exe_path: &std::path::Path) -> String {
    format!("\"{}\" --hidden", exe_path.display())
}

pub fn set_autostart_registry(enabled: bool) -> Result<(), String> {
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let run_key = hkcu
        .open_subkey_with_flags(RUN_KEY, KEY_READ | KEY_WRITE)
        .map_err(|e| format!("Failed to open registry key: {}", e))?;

    if enabled {
        if cfg!(debug_assertions) {
            // A dev build must never re-point the Run key at target\debug.
            log::warn!("Debug build: leaving the autostart registry value untouched");
            return Ok(());
        }
        let exe_path =
            std::env::current_exe().map_err(|e| format!("Failed to get exe path: {}", e))?;
        let value = autostart_value(&exe_path);
        let current: Option<String> = run_key.get_value(APP_NAME).ok();
        if current.as_deref() == Some(value.as_str()) {
            return Ok(());
        }
        run_key
            .set_value(APP_NAME, &value)
            .map_err(|e| format!("Failed to set registry value: {}", e))?;
        log::info!(
            "Autostart registry set: {} = {} (was {})",
            APP_NAME,
            value,
            current.unwrap_or_else(|| "unset".into())
        );
    } else {
        // Ignore error if value doesn't exist
        let _ = run_key.delete_value(APP_NAME);
        log::info!("Autostart registry removed: {}", APP_NAME);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::autostart_value;

    #[test]
    fn run_value_is_quoted_and_hidden() {
        let value = autostart_value(std::path::Path::new(
            r"C:\Program Files\Wispr\wispr-local.exe",
        ));
        assert_eq!(
            value,
            r#""C:\Program Files\Wispr\wispr-local.exe" --hidden"#
        );
    }
}
