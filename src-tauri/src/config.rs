use directories::ProjectDirs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct AppConfig {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Write a small state file through a synced temporary file, then replace the
/// destination in one filesystem operation. This keeps a GPU-process crash
/// from leaving settings or history half-written for the supervisor restart.
pub fn write_file_atomic(path: &Path, contents: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temp_path = path.with_extension("tmp");
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

#[cfg(windows)]
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

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> Result<(), String> {
    std::fs::rename(source, destination).map_err(|e| e.to_string())
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
