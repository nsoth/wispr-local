use reqwest::Client;
use serde::{Deserialize, Serialize};

const DEFAULT_PROMPT: &str = "You are a text formatting assistant. The user dictated the following text via speech-to-text. \
Format it into well-structured text:\n\
- Add proper punctuation and capitalization\n\
- Break into paragraphs where there is a topic change or natural pause\n\
- Format enumerations as bullet lists (using - prefix)\n\
- Add colons, semicolons, and dashes where appropriate\n\
- Do NOT change the meaning, rephrase, or add new content\n\
- Output ONLY the formatted text, nothing else (no explanations, no quotes)";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AiProvider {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "claude")]
    Claude,
}

impl Default for AiProvider {
    fn default() -> Self {
        AiProvider::None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSettings {
    #[serde(default)]
    pub provider: AiProvider,
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_openai_model")]
    pub openai_model: String,
    #[serde(default = "default_claude_model")]
    pub claude_model: String,
    #[serde(default = "default_prompt")]
    pub prompt: String,
}

fn default_openai_model() -> String {
    "gpt-4o-mini".to_string()
}
fn default_claude_model() -> String {
    "claude-sonnet-4-20250514".to_string()
}
fn default_prompt() -> String {
    DEFAULT_PROMPT.to_string()
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            provider: AiProvider::None,
            api_key: String::new(),
            openai_model: default_openai_model(),
            claude_model: default_claude_model(),
            prompt: default_prompt(),
        }
    }
}

/// Format transcribed text using the configured AI provider.
/// Returns the original text if provider is None or on error.
pub async fn format_text(text: &str, settings: &AiSettings) -> String {
    if settings.provider == AiProvider::None || text.trim().is_empty() {
        return text.to_string();
    }

    log::info!("AI formatting with {:?} provider ({} chars)", settings.provider, text.len());

    let result = match settings.provider {
        AiProvider::OpenAi => format_with_openai(text, settings).await,
        AiProvider::Claude => format_with_claude(text, settings).await,
        AiProvider::None => return text.to_string(),
    };

    match result {
        Ok(formatted) => {
            log::info!("AI formatted: {} chars -> {} chars", text.len(), formatted.len());
            formatted
        }
        Err(e) => {
            log::error!("AI formatting failed: {}, using raw text", e);
            text.to_string()
        }
    }
}

/// OpenAI Chat Completions API
async fn format_with_openai(text: &str, settings: &AiSettings) -> Result<String, String> {
    if settings.api_key.is_empty() {
        return Err("OpenAI API key not set".to_string());
    }

    let body = serde_json::json!({
        "model": settings.openai_model,
        "messages": [
            { "role": "system", "content": settings.prompt },
            { "role": "user", "content": text }
        ],
        "temperature": 0.1
    });

    let client = Client::new();
    let resp = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", settings.api_key))
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("OpenAI request failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI error {}: {}", status, body));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse OpenAI response: {}", e))?;

    json["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.trim().to_string())
        .ok_or_else(|| "No content in OpenAI response".to_string())
}

/// Anthropic Messages API
async fn format_with_claude(text: &str, settings: &AiSettings) -> Result<String, String> {
    if settings.api_key.is_empty() {
        return Err("Claude API key not set".to_string());
    }

    let body = serde_json::json!({
        "model": settings.claude_model,
        "max_tokens": 4096,
        "system": settings.prompt,
        "messages": [
            { "role": "user", "content": text }
        ],
        "temperature": 0.1
    });

    let client = Client::new();
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &settings.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("Claude request failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Claude error {}: {}", status, body));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse Claude response: {}", e))?;

    json["content"][0]["text"]
        .as_str()
        .map(|s| s.trim().to_string())
        .ok_or_else(|| "No content in Claude response".to_string())
}
