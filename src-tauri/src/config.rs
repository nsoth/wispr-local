use directories::ProjectDirs;
use std::path::PathBuf;

pub struct AppConfig {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
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
}
