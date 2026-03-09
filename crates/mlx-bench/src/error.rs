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
}

pub type Result<T> = std::result::Result<T, BenchError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_io() {
        let err = BenchError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
        assert!(err.to_string().contains("IO error"));
    }

    #[test]
    fn test_error_display_task_not_found() {
        let err = BenchError::TaskNotFound("task123".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Task not found"));
        assert!(msg.contains("task123"));
    }

    #[test]
    fn test_error_display_backend() {
        let err = BenchError::Backend("API connection failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Backend error"));
        assert!(msg.contains("API connection failed"));
    }

    #[test]
    fn test_error_display_patch_validation() {
        let err = BenchError::PatchValidation("Missing --- header".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Patch validation failed"));
        assert!(msg.contains("Missing --- header"));
    }

    #[test]
    fn test_all_error_variants_display() {
        // Test that all error variants can be displayed without panicking
        let errors: Vec<BenchError> = vec![
            BenchError::TaskNotFound("test".to_string()),
            BenchError::InvalidTask("test".to_string()),
            BenchError::Backend("test".to_string()),
            BenchError::LlmGeneration("test".to_string()),
            BenchError::PatchValidation("test".to_string()),
            BenchError::BuildFailed("test".to_string()),
            BenchError::TestFailed("test".to_string()),
            BenchError::Git("test".to_string()),
            BenchError::Timeout,
            BenchError::InvalidConfig("test".to_string()),
        ];

        for err in errors {
            let _display = format!("{}", err);
            // Just verify it doesn't panic
            assert!(!_display.is_empty());
        }
    }
}
