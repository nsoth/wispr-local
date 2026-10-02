//! API keys for the optional AI formatting providers, encrypted with the
//! Windows Data Protection API (user scope) and stored next to settings.json.
//!
//! `api-keys.dat` holds a JSON object `{"openai": "...", "claude": "..."}`.
//! The older single-key file `api-key.dat` is migrated into both slots the
//! first time it is seen (the old build shared one key between providers).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const API_KEYS_FILE: &str = "api-keys.dat";
/// Pre-2026-10 layout: one DPAPI blob holding a single key string.
const LEGACY_API_KEY_FILE: &str = "api-key.dat";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiKeys {
    #[serde(default)]
    pub openai: String,
    #[serde(default)]
    pub claude: String,
}

impl ApiKeys {
    pub fn is_empty(&self) -> bool {
        self.openai.is_empty() && self.claude.is_empty()
    }
}

fn api_keys_path(data_dir: &Path) -> PathBuf {
    data_dir.join(API_KEYS_FILE)
}

fn api_key_path(data_dir: &Path) -> PathBuf {
    data_dir.join(LEGACY_API_KEY_FILE)
}

/// Load the per-provider keys, migrating the legacy single-key file if that
/// is all there is. Never touches settings.json.
pub fn load_api_keys(data_dir: &Path) -> Result<ApiKeys, String> {
    let path = api_keys_path(data_dir);
    if path.exists() {
        let encrypted =
            std::fs::read(&path).map_err(|e| format!("Failed to read API keys: {e}"))?;
        let decrypted = decrypt(&encrypted)?;
        return serde_json::from_slice::<ApiKeys>(&decrypted)
            .map_err(|e| format!("Stored API keys are unreadable: {e}"));
    }

    match load_legacy_api_key(data_dir)? {
        Some(key) => {
            let keys = ApiKeys {
                openai: key.clone(),
                claude: key,
            };
            save_api_keys(data_dir, &keys)?;
            if let Err(e) = std::fs::remove_file(api_key_path(data_dir)) {
                log::warn!("Could not remove the legacy API key file after migration: {e}");
            }
            log::info!("Migrated the legacy API key into the per-provider key store");
            Ok(keys)
        }
        None => Ok(ApiKeys::default()),
    }
}

/// Persist the keys; an all-empty store removes the file.
pub fn save_api_keys(data_dir: &Path, keys: &ApiKeys) -> Result<(), String> {
    let path = api_keys_path(data_dir);
    if keys.is_empty() {
        return match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("Failed to clear API keys: {e}")),
        };
    }
    let json = serde_json::to_vec(keys).map_err(|e| e.to_string())?;
    let encrypted = encrypt(&json)?;
    crate::config::write_file_atomic(&path, &encrypted)
        .map_err(|e| format!("Failed to save API keys: {e}"))
}

fn load_legacy_api_key(data_dir: &Path) -> Result<Option<String>, String> {
    let path = api_key_path(data_dir);
    if !path.exists() {
        return Ok(None);
    }
    let encrypted = std::fs::read(&path).map_err(|e| format!("Failed to read API key: {e}"))?;
    let decrypted = decrypt(&encrypted)?;
    let key = String::from_utf8(decrypted)
        .map_err(|_| "Stored API key is not valid UTF-8".to_string())?;
    Ok(Some(key).filter(|k| !k.is_empty()))
}

/// Write the legacy single-key file (kept for the migration test and for
/// tooling that still produces the old layout).
#[cfg(test)]
pub fn save_api_key(data_dir: &Path, key: &str) -> Result<(), String> {
    let path = api_key_path(data_dir);
    let encrypted = encrypt(key.as_bytes())?;
    crate::config::write_file_atomic(&path, &encrypted)
        .map_err(|e| format!("Failed to save API key: {e}"))
}

#[cfg(windows)]
fn encrypt(plaintext: &[u8]) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    let input = CRYPT_INTEGER_BLOB {
        cbData: plaintext
            .len()
            .try_into()
            .map_err(|_| "API key is too large".to_string())?,
        pbData: plaintext.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let result = unsafe {
        CryptProtectData(
            &input,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if result == 0 {
        return Err(format!(
            "Windows could not encrypt the API key: {}",
            std::io::Error::last_os_error()
        ));
    }

    let encrypted =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
    unsafe {
        LocalFree(output.pbData as _);
    }
    Ok(encrypted)
}

#[cfg(windows)]
fn decrypt(encrypted: &[u8]) -> Result<Vec<u8>, String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };

    let input = CRYPT_INTEGER_BLOB {
        cbData: encrypted
            .len()
            .try_into()
            .map_err(|_| "Stored API key is too large".to_string())?,
        pbData: encrypted.as_ptr() as *mut u8,
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: std::ptr::null_mut(),
    };
    let result = unsafe {
        CryptUnprotectData(
            &input,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if result == 0 {
        return Err(format!(
            "Windows could not decrypt the API key: {}",
            std::io::Error::last_os_error()
        ));
    }

    let plaintext =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize).to_vec() };
    unsafe {
        LocalFree(output.pbData as _);
    }
    Ok(plaintext)
}

#[cfg(not(windows))]
fn encrypt(_plaintext: &[u8]) -> Result<Vec<u8>, String> {
    Err("Secure API-key storage is currently available on Windows only".to_string())
}

#[cfg(not(windows))]
fn decrypt(_encrypted: &[u8]) -> Result<Vec<u8>, String> {
    Err("Secure API-key storage is currently available on Windows only".to_string())
}

#[cfg(all(test, windows))]
mod tests {
    use super::{api_key_path, api_keys_path, load_api_keys, save_api_key, save_api_keys, ApiKeys};
    use crate::config::test_dir;

    #[test]
    fn api_keys_round_trip_without_plaintext_on_disk() {
        let dir = test_dir("keys");
        let keys = ApiKeys {
            openai: "sk-openai-secret".into(),
            claude: "sk-ant-secret".into(),
        };
        save_api_keys(&dir, &keys).unwrap();
        let raw = std::fs::read(api_keys_path(&dir)).unwrap();
        assert!(!raw.windows(6).any(|w| w == b"secret"));
        assert_eq!(load_api_keys(&dir).unwrap(), keys);

        save_api_keys(&dir, &ApiKeys::default()).unwrap();
        assert!(
            !api_keys_path(&dir).exists(),
            "an empty store removes the file"
        );
        assert_eq!(load_api_keys(&dir).unwrap(), ApiKeys::default());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn legacy_single_key_migrates_to_both_providers() {
        let dir = test_dir("legacy");
        save_api_key(&dir, "sk-legacy").unwrap();
        assert!(api_key_path(&dir).exists());

        let keys = load_api_keys(&dir).unwrap();
        assert_eq!(keys.openai, "sk-legacy");
        assert_eq!(keys.claude, "sk-legacy");
        assert!(
            !api_key_path(&dir).exists(),
            "legacy file removed after migration"
        );
        assert!(api_keys_path(&dir).exists());
        // Second load reads the new store directly.
        assert_eq!(load_api_keys(&dir).unwrap(), keys);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
