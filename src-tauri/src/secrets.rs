use std::path::{Path, PathBuf};

const API_KEY_FILE: &str = "api-key.dat";

fn api_key_path(data_dir: &Path) -> PathBuf {
    data_dir.join(API_KEY_FILE)
}

pub fn load_api_key(data_dir: &Path) -> Result<Option<String>, String> {
    let path = api_key_path(data_dir);
    if !path.exists() {
        return Ok(None);
    }
    let encrypted = std::fs::read(&path).map_err(|e| format!("Failed to read API key: {e}"))?;
    let decrypted = decrypt(&encrypted)?;
    let key = String::from_utf8(decrypted)
        .map_err(|_| "Stored API key is not valid UTF-8".to_string())?;
    Ok(Some(key))
}

pub fn save_api_key(data_dir: &Path, key: &str) -> Result<(), String> {
    let path = api_key_path(data_dir);
    if key.is_empty() {
        match std::fs::remove_file(path) {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(format!("Failed to clear API key: {e}")),
        }
    }

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
    use super::{api_key_path, load_api_key, save_api_key};

    #[test]
    fn api_key_round_trips_without_plaintext_on_disk() {
        let unique = format!(
            "wispr-local-secret-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock before epoch")
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).expect("create test directory");

        let key = "sk-test-secret-value";
        save_api_key(&dir, key).expect("encrypt API key");
        let raw = std::fs::read(api_key_path(&dir)).expect("read encrypted API key");
        assert!(!raw
            .windows(key.len())
            .any(|window| window == key.as_bytes()));
        assert_eq!(
            load_api_key(&dir).expect("decrypt API key"),
            Some(key.to_string())
        );

        save_api_key(&dir, "").expect("clear API key");
        assert_eq!(load_api_key(&dir).expect("load cleared key"), None);
        std::fs::remove_dir_all(dir).expect("remove test directory");
    }
}
