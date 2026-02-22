use std::path::PathBuf;

const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

pub struct ModelInfo {
    pub name: String,
    pub filename: String,
    pub url: String,
    pub size_bytes: u64,
}

pub fn get_available_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            name: "base.en".to_string(),
            filename: "ggml-base.en.bin".to_string(),
            url: format!("{}/ggml-base.en.bin", MODEL_BASE_URL),
            size_bytes: 147_951_465,
        },
        ModelInfo {
            name: "small.en".to_string(),
            filename: "ggml-small.en.bin".to_string(),
            url: format!("{}/ggml-small.en.bin", MODEL_BASE_URL),
            size_bytes: 487_601_024,
        },
        ModelInfo {
            name: "medium.en".to_string(),
            filename: "ggml-medium.en.bin".to_string(),
            url: format!("{}/ggml-medium.en.bin", MODEL_BASE_URL),
            size_bytes: 1_533_774_848,
        },
    ]
}

pub fn model_exists(models_dir: &PathBuf, filename: &str) -> bool {
    models_dir.join(filename).exists()
}

/// Download model file. Phase 1: simple blocking download.
pub async fn download_model(models_dir: &PathBuf, model: &ModelInfo) -> Result<PathBuf, String> {
    let dest = models_dir.join(&model.filename);
    if dest.exists() {
        return Ok(dest);
    }

    std::fs::create_dir_all(models_dir)
        .map_err(|e| format!("Failed to create models dir: {}", e))?;

    log::info!(
        "Downloading model {} ({} bytes)...",
        model.name,
        model.size_bytes
    );

    let response = reqwest::get(&model.url)
        .await
        .map_err(|e| format!("Failed to download model: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Download failed with status: {}",
            response.status()
        ));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    std::fs::write(&dest, &bytes)
        .map_err(|e| format!("Failed to write model file: {}", e))?;

    log::info!("Model downloaded to {:?}", dest);
    Ok(dest)
}
