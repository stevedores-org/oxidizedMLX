use thiserror::Error;

#[derive(Error, Debug)]
pub enum BenchError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Invalid task JSON: {0}")]
    InvalidTask(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("LLM generation failed: {0}")]
    LlmGeneration(String),

    #[error("Patch validation failed: {0}")]
    PatchValidation(String),

    #[error("Build failed: {0}")]
    BuildFailed(String),

    #[error("Test execution failed: {0}")]
    TestFailed(String),

    #[error("Git error: {0}")]
    Git(String),

    #[error("Timeout exceeded")]
    Timeout,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Python subprocess error: {0}")]
    PythonSubprocess(String),
}

pub type Result<T> = std::result::Result<T, BenchError>;
