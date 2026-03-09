use crate::error::{BenchError, Result};
use crate::task::EvalTask;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendOpts {
    pub model_id: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout_secs: u64,
}

pub trait LlmBackend: Send + Sync {
    fn generate_patch(&self, task: &EvalTask, opts: &BackendOpts) -> Result<String>;
    fn name(&self) -> &str;
}

/// Anthropic API backend using official Rust SDK
pub struct AnthropicApiBackend {
    api_key: String,
}

impl AnthropicApiBackend {
    pub fn new() -> Result<Self> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .map_err(|_| BenchError::Backend(
                "ANTHROPIC_API_KEY environment variable not set".to_string()
            ))?;

        Ok(AnthropicApiBackend { api_key })
    }
}

impl LlmBackend for AnthropicApiBackend {
    fn generate_patch(&self, task: &EvalTask, opts: &BackendOpts) -> Result<String> {
        // Use tokio for async runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| BenchError::Backend(format!("Failed to create runtime: {}", e)))?;

        rt.block_on(async {
            generate_patch_async(task, opts, &self.api_key).await
        })
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

async fn generate_patch_async(
    task: &EvalTask,
    opts: &BackendOpts,
    api_key: &str,
) -> Result<String> {
    let client = reqwest::Client::new();

    // Assemble prompt
    let system_prompt = DEFAULT_SYSTEM_PROMPT.to_string();
    let user_prompt = format!(
        "Issue #{}: {}\n\nDescription:\n{}\n\nPlease provide a unified diff patch to fix this issue.",
        task.issue.unwrap_or(0),
        task.title,
        task.description
    );

    // Create request
    #[derive(Serialize)]
    struct MessageContent {
        role: String,
        content: String,
    }

    #[derive(Serialize)]
    struct AnthropicRequest {
        model: String,
        max_tokens: usize,
        temperature: f32,
        system: String,
        messages: Vec<MessageContent>,
    }

    let request = AnthropicRequest {
        model: opts.model_id.clone(),
        max_tokens: opts.max_tokens,
        temperature: opts.temperature,
        system: system_prompt,
        messages: vec![MessageContent {
            role: "user".to_string(),
            content: user_prompt,
        }],
    };

    // Call API
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| BenchError::Backend(format!("API request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(BenchError::Backend(format!(
            "API error {}: {}",
            status, text
        )));
    }

    // Parse response
    #[derive(Deserialize)]
    struct ContentBlock {
        #[serde(rename = "type")]
        #[allow(dead_code)]
        content_type: String,
        text: Option<String>,
    }

    #[derive(Deserialize)]
    struct AnthropicResponse {
        content: Vec<ContentBlock>,
    }

    let response_data: AnthropicResponse = response
        .json()
        .await
        .map_err(|e| BenchError::Backend(format!("Failed to parse response: {}", e)))?;

    let patch = response_data
        .content
        .first()
        .and_then(|c| c.text.as_ref())
        .ok_or_else(|| BenchError::Backend("No text content in response".to_string()))?
        .clone();

    Ok(patch)
}

/// Local MLX backend for Apple Silicon inference
pub struct LocalMlxBackend {
    #[allow(dead_code)]
    model_id: String,
}

impl LocalMlxBackend {
    pub fn new(model_id: impl Into<String>) -> Self {
        LocalMlxBackend {
            model_id: model_id.into(),
        }
    }
}

impl LlmBackend for LocalMlxBackend {
    fn generate_patch(&self, _task: &EvalTask, _opts: &BackendOpts) -> Result<String> {
        // Local inference would use mlx-rs bindings
        // For now, return a placeholder error indicating this would be implemented with MLX Rust bindings
        Err(BenchError::Backend(
            "Local MLX backend requires mlx-rs Rust bindings (not yet available in stable Rust).\n\
             Use Anthropic API backend instead: cargo run -p mlx-bench -- run --backend anthropic"
                .to_string(),
        ))
    }

    fn name(&self) -> &str {
        "local"
    }
}

/// Debug backend that reads patch from environment or file
pub struct StdinDebugBackend;

impl LlmBackend for StdinDebugBackend {
    fn generate_patch(&self, _task: &EvalTask, _opts: &BackendOpts) -> Result<String> {
        if let Ok(patch_file) = env::var("BENCH_PATCH_FILE") {
            std::fs::read_to_string(&patch_file)
                .map_err(|e| BenchError::Backend(format!("Failed to read patch file: {}", e)))
        } else if let Ok(patch_content) = env::var("BENCH_PATCH_CONTENT") {
            Ok(patch_content)
        } else {
            Err(BenchError::Backend(
                "Set either BENCH_PATCH_FILE or BENCH_PATCH_CONTENT environment variable".to_string(),
            ))
        }
    }

    fn name(&self) -> &str {
        "debug"
    }
}

const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an expert Rust developer helping to fix bugs in the oxidizedMLX library.
Your task is to analyze the issue description and context files, then provide a unified diff patch that fixes the problem.

Guidelines:
1. Carefully read the issue and understand what needs to be fixed
2. Review the provided context files to understand the current implementation
3. Generate a minimal, focused patch that addresses the issue
4. Ensure the patch maintains code style and quality
5. The patch must be valid unified diff format compatible with `git apply`
6. Do not include file creation or deletion, only modifications
7. Do not make unnecessary changes beyond what's required to fix the issue

Respond with ONLY the unified diff patch, starting with "---" and ending with a valid hunk header.
No explanation, no code blocks, no markdown formatting - just the raw diff."#;
