//! Data directories and small file helpers shared by settings, history and
//! the key store.

use directories::ProjectDirs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct AppConfig {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self::new()
    }
}

static TEMP_SEQ: AtomicU32 = AtomicU32::new(0);

/// Write a small state file through a synced temporary file, then replace the
/// destination in one filesystem operation. This keeps a GPU-process crash
/// from leaving settings or history half-written for the supervisor restart.
/// The temporary name carries the pid and a counter so two writers of the
/// same file (e.g. the stop pipeline and "Clear history") never share it.
pub fn write_file_atomic(path: &Path, contents: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "file".to_string());
    let temp_path = path.with_file_name(format!("{file_name}.{}.{seq}.tmp", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temp_path)
        .map_err(|e| e.to_string())?;
    file.write_all(contents).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);

    replace_file(&temp_path, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp_path);
    })
}

/// Remove `*.tmp` leftovers from a crash between write and rename.
pub fn remove_stale_temp_files(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".tmp") && std::fs::remove_file(entry.path()).is_ok() {
            log::info!("Removed stale temporary file {name}");
        }
    }
}

fn replace_file(source: &Path, destination: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error().to_string())
    } else {
        Ok(())
    }
}

/// Move a broken state file aside as `<name>.corrupt-<YYYYMMDD-HHMMSS>` so the
/// user's bytes survive and the app can start with defaults. Returns the new
/// path on success.
pub fn quarantine_file(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let stamp = timestamp_for_filename(std::time::SystemTime::now());
    let target = path.with_file_name(format!("{name}.corrupt-{stamp}"));
    match std::fs::rename(path, &target) {
        Ok(()) => Some(target),
        Err(e) => {
            log::error!("Could not move {} aside: {e}", path.display());
            None
        }
    }
}

/// `YYYYMMDD-HHMMSS` in UTC: sortable, no characters Windows rejects.
pub fn timestamp_for_filename(time: std::time::SystemTime) -> String {
    let secs = time
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}{mo:02}{d:02}-{h:02}{m:02}{s:02}")
}

impl AppConfig {
    pub fn new() -> Self {
        let proj_dirs = ProjectDirs::from("com", "wispr-local", "WisprLocal")
            .expect("Failed to determine project directories");
        let data_dir = proj_dirs.data_dir().to_path_buf();
        let models_dir = data_dir.join("models");
        Self {
            data_dir,
            models_dir,
        }
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.models_dir)?;
        Ok(())
    }

    pub fn model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name)
    }

    /// Discover additional multilingual whisper.cpp models so users are not
    /// forced to rename a quantized or smaller compatible model. English-only
    /// `.en.bin` files are excluded because this app also supports Russian.
    pub fn available_model_files(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.models_dir) else {
            return Vec::new();
        };
        let mut models: Vec<String> = entries
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| {
                name.starts_with("ggml-") && name.ends_with(".bin") && !name.ends_with(".en.bin")
            })
            .collect();
        models.sort();
        models
    }
}

/// Unique scratch directory for unit tests (left in place on failure so the
/// artifacts can be inspected).
#[cfg(test)]
pub(crate) fn test_dir(tag: &str) -> PathBuf {
    let unique = format!(
        "wispr-local-test-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).expect("create test directory");
    dir
}

#[cfg(test)]
mod tests {
    use super::{
        quarantine_file, remove_stale_temp_files, test_dir, timestamp_for_filename,
        write_file_atomic,
    };
    use std::time::{Duration, UNIX_EPOCH};

    fn listing(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn write_file_atomic_replaces_and_leaves_no_temp_file() {
        let dir = test_dir("atomic");
        let path = dir.join("settings.json");
        write_file_atomic(&path, b"one").unwrap();
        write_file_atomic(&path, b"two").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"two");
        assert_eq!(listing(&dir), vec!["settings.json"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn stale_temp_files_are_removed_but_state_files_kept() {
        let dir = test_dir("stale");
        std::fs::write(dir.join("history.json.123.4.tmp"), b"x").unwrap();
        std::fs::write(dir.join("history.json"), b"[]").unwrap();
        remove_stale_temp_files(&dir);
        assert_eq!(listing(&dir), vec!["history.json"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn quarantine_keeps_the_original_bytes() {
        let dir = test_dir("quarantine");
        let path = dir.join("settings.json");
        std::fs::write(&path, b"{broken").unwrap();
        let moved = quarantine_file(&path).expect("moved");
        assert!(!path.exists());
        assert!(moved
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("settings.json.corrupt-"));
        assert_eq!(std::fs::read(moved).unwrap(), b"{broken");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn timestamp_for_filename_is_sortable_and_path_safe() {
        // 2026-10-02 12:00:00 UTC
        let t = UNIX_EPOCH + Duration::from_secs(1_790_942_400);
        assert_eq!(timestamp_for_filename(t), "20261002-120000");
        // Leap-year day.
        let t = UNIX_EPOCH + Duration::from_secs(1_709_164_800);
        assert_eq!(timestamp_for_filename(t), "20240229-000000");
    }
}
