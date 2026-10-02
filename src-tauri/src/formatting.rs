use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::Duration;

const DEFAULT_PROMPT: &str = "You are a text formatting assistant. The user dictated the following text via speech-to-text. \
Format it into well-structured text:\n\
- Add proper punctuation and capitalization\n\
- Break into paragraphs where there is a topic change or natural pause\n\
- Format enumerations as bullet lists (using - prefix)\n\
- Add colons, semicolons, and dashes where appropriate\n\
- Do NOT change the meaning, rephrase, or add new content\n\
- Output ONLY the formatted text, nothing else (no explanations, no quotes)";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum AiProvider {
    #[serde(rename = "none")]
    #[default]
    None,
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "claude")]
    Claude,
    /// Any provider name this build does not know (e.g. the removed "local");
    /// normalized to `None` right after loading so it never reaches the UI.
    #[serde(other, skip_serializing)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSettings {
    #[serde(default)]
    pub provider: AiProvider,
    /// Plaintext key from pre-DPAPI settings files. Read for migration only;
    /// never written back.
    #[serde(default, rename = "api_key", skip_serializing)]
    pub legacy_api_key: String,
    #[serde(default = "default_openai_model")]
    pub openai_model: String,
    #[serde(default = "default_claude_model")]
    pub claude_model: String,
    #[serde(default = "default_prompt")]
    pub prompt: String,
    /// Per-provider keys from the encrypted store (never serialized here).
    #[serde(skip)]
    pub keys: crate::secrets::ApiKeys,
}

impl AiSettings {
    /// The key for the active provider ("" when none is stored).
    pub fn api_key(&self) -> &str {
        match self.provider {
            AiProvider::OpenAi => &self.keys.openai,
            AiProvider::Claude => &self.keys.claude,
            AiProvider::None | AiProvider::Unknown => "",
        }
    }
}

fn default_openai_model() -> String {
    "gpt-4o-mini".to_string()
}
fn default_claude_model() -> String {
    // Haiku 4.5: fast and cheap — plenty for punctuation/formatting work.
    "claude-haiku-4-5-20251001".to_string()
}
pub fn default_prompt() -> String {
    DEFAULT_PROMPT.to_string()
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            provider: AiProvider::None,
            legacy_api_key: String::new(),
            openai_model: default_openai_model(),
            claude_model: default_claude_model(),
            prompt: default_prompt(),
            keys: crate::secrets::ApiKeys::default(),
        }
    }
}

fn http_client() -> Result<&'static Client, String> {
    static CLIENT: OnceLock<Result<Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(20))
                .build()
                .map_err(|e| format!("Failed to initialize HTTP client: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Format transcribed text using the configured AI provider. The caller owns
/// fallback behavior so it can tell the user when raw text was used instead.
pub async fn format_text(text: &str, settings: &AiSettings) -> Result<String, String> {
    if settings.provider == AiProvider::None || text.trim().is_empty() {
        return Ok(text.to_string());
    }

    log::info!(
        "AI formatting with {:?} provider ({} chars)",
        settings.provider,
        text.len()
    );

    let result = match settings.provider {
        AiProvider::OpenAi => format_with_openai(text, settings).await,
        AiProvider::Claude => format_with_claude(text, settings).await,
        AiProvider::None | AiProvider::Unknown => return Ok(text.to_string()),
    };

    let formatted = result?;
    validate_formatted_output(text, &formatted)?;
    log::info!(
        "AI formatted: {} chars -> {} chars",
        text.len(),
        formatted.len()
    );
    Ok(formatted)
}

fn validate_formatted_output(input: &str, output: &str) -> Result<(), String> {
    if output.trim().is_empty() {
        return Err("AI provider returned empty text".to_string());
    }

    // Formatting should not turn a short dictation into a long generated
    // answer. Allow ample punctuation/paragraph growth while rejecting clear
    // prompt-following failures or provider glitches.
    let max_chars = input.chars().count().saturating_mul(4).max(512);
    if output.chars().count() > max_chars {
        return Err("AI provider returned unexpectedly long text".to_string());
    }
    Ok(())
}

/// OpenAI Chat Completions API
async fn format_with_openai(text: &str, settings: &AiSettings) -> Result<String, String> {
    let api_key = settings.api_key();
    if api_key.is_empty() {
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

    let resp = http_client()?
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
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
    let api_key = settings.api_key();
    if api_key.is_empty() {
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

    let resp = http_client()?
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
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

#[cfg(test)]
mod tests {
    use super::validate_formatted_output;

    #[test]
    fn rejects_empty_and_runaway_formatting() {
        assert!(validate_formatted_output("hello", "   ").is_err());
        assert!(validate_formatted_output("hello", &"x".repeat(513)).is_err());
    }

    #[test]
    fn accepts_normal_formatted_text() {
        assert!(validate_formatted_output("hello world", "Hello, world.").is_ok());
    }
}
