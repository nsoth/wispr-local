use winreg::enums::*;
use winreg::RegKey;

const APP_NAME: &str = "Wispr Local";
const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";

pub fn set_autostart_registry(enabled: bool) -> Result<(), String> {
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let run_key = hkcu
        .open_subkey_with_flags(RUN_KEY, KEY_READ | KEY_WRITE)
        .map_err(|e| format!("Failed to open registry key: {}", e))?;

    if enabled {
        let exe_path = std::env::current_exe()
            .map_err(|e| format!("Failed to get exe path: {}", e))?;
        // Quote the path to handle spaces in directory names.
        // Pass --hidden so the app detaches from the console on startup
        // (debug builds would otherwise leave a terminal window open).
        let value = format!("\"{}\" --hidden", exe_path.display());
        run_key
            .set_value(APP_NAME, &value)
            .map_err(|e| format!("Failed to set registry value: {}", e))?;
        log::info!("Autostart registry set: {} = {}", APP_NAME, value);
    } else {
        // Ignore error if value doesn't exist
        let _ = run_key.delete_value(APP_NAME);
        log::info!("Autostart registry removed: {}", APP_NAME);
    }

    Ok(())
}
