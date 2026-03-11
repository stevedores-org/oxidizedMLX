use crate::backend::BackendOpts;
use crate::error::{BenchError, Result};
use crate::task::EvalTask;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command as TokioCommand;
use tokio::runtime::Runtime;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutcome {
    pub task_id: String,
    pub attempt: usize,
    pub timestamp: DateTime<Utc>,
    pub patch_generated: bool,
    pub patch_text: Option<String>,
    pub patch_valid: bool,
    pub build_success: bool,
    pub tests_passed: bool,
    pub error: Option<String>,
    pub llm_input_tokens: Option<usize>,
    pub llm_output_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub run_id: String,
    pub created_at: DateTime<Utc>,
    pub outcomes: Vec<TaskOutcome>,
}

pub struct RunConfig {
    pub workspace_root: PathBuf,
    pub max_attempts: usize,
    pub timeout_secs: u64,
}

impl RunConfig {
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        RunConfig {
            workspace_root: workspace_root.as_ref().to_path_buf(),
            max_attempts: 1,
            timeout_secs: 120,
        }
    }
}

pub struct TaskRunner {
    config: RunConfig,
}

impl TaskRunner {
    pub fn new(config: RunConfig) -> Self {
        TaskRunner { config }
    }

    /// Execute single task with LLM backend
    pub fn run_task(
        &self,
        task: &EvalTask,
        backend: &dyn crate::backend::LlmBackend,
        opts: &BackendOpts,
    ) -> Result<Vec<TaskOutcome>> {
        // Create tokio runtime once, reused across all attempts
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| BenchError::Backend(format!("Failed to create runtime: {}", e)))?;

        let mut outcomes = Vec::new();

        for attempt in 1..=self.config.max_attempts {
            let outcome = self.run_attempt(&rt, task, backend, opts, attempt)?;
            outcomes.push(outcome);
        }

        Ok(outcomes)
    }

    fn run_attempt(
        &self,
        rt: &Runtime,
        task: &EvalTask,
        backend: &dyn crate::backend::LlmBackend,
        opts: &BackendOpts,
        attempt: usize,
    ) -> Result<TaskOutcome> {
        let mut outcome = TaskOutcome {
            task_id: task.id.clone(),
            attempt,
            timestamp: Utc::now(),
            patch_generated: false,
            patch_text: None,
            patch_valid: false,
            build_success: false,
            tests_passed: false,
            error: None,
            llm_input_tokens: None,
            llm_output_tokens: None,
        };

        // Step 1: Generate patch (sync call, outside block_on)
        match backend.generate_patch(task, opts) {
            Ok(patch) => {
                outcome.patch_generated = true;
                outcome.patch_text = Some(patch.clone());

                // Steps 2-6: Subprocess operations with timeouts (async)
                if let Err(e) = rt.block_on(self.run_subprocess_steps(&mut outcome, task, &patch)) {
                    if outcome.error.is_none() {
                        outcome.error = Some(format!("Subprocess orchestration failed: {}", e));
                    }
                }
            }
            Err(e) => {
                outcome.error = Some(format!("LLM generation failed: {}", e));
            }
        }

        Ok(outcome)
    }

    async fn run_subprocess_steps(
        &self,
        outcome: &mut TaskOutcome,
        task: &EvalTask,
        patch: &str,
    ) -> Result<()> {
        const GIT_TIMEOUT: Duration = Duration::from_secs(30);
        let build_timeout = Duration::from_secs(task.timeout_secs);

        // Step 2: Validate patch
        match tokio::time::timeout(GIT_TIMEOUT, self.validate_patch_async(patch)).await {
            Ok(Ok(())) => outcome.patch_valid = true,
            Ok(Err(e)) => {
                outcome.error = Some(format!("{}", e));
                return Ok(());
            }
            Err(_) => {
                outcome.error = Some("Patch validation timed out".to_string());
                return Ok(());
            }
        }

        // Step 3: Apply patch
        match tokio::time::timeout(GIT_TIMEOUT, self.apply_patch_async(patch)).await {
            Ok(Ok(())) => {},
            Ok(Err(e)) => {
                outcome.error = Some(format!("Patch application failed: {}", e));
                return Ok(());
            }
            Err(_) => {
                outcome.error = Some("Patch application timed out".to_string());
                return Ok(());
            }
        }

        // Step 4: Build
        match tokio::time::timeout(build_timeout, self.run_build_async(task)).await {
            Ok(Ok(true)) => outcome.build_success = true,
            Ok(Ok(false)) => {
                outcome.error = Some("Build failed".to_string());
                let _ = tokio::time::timeout(GIT_TIMEOUT, self.revert_changes_async()).await;
                return Ok(());
            }
            Ok(Err(e)) => {
                outcome.error = Some(format!("Build error: {}", e));
                let _ = tokio::time::timeout(GIT_TIMEOUT, self.revert_changes_async()).await;
                return Ok(());
            }
            Err(_) => {
                outcome.error = Some("Build timed out".to_string());
                let _ = tokio::time::timeout(GIT_TIMEOUT, self.revert_changes_async()).await;
                return Ok(());
            }
        }

        // Step 5: Test
        match tokio::time::timeout(build_timeout, self.run_tests_async(task)).await {
            Ok(Ok(true)) => outcome.tests_passed = true,
            Ok(Ok(false)) => outcome.error = Some("Tests failed".to_string()),
            Ok(Err(e)) => outcome.error = Some(format!("Test execution error: {}", e)),
            Err(_) => outcome.error = Some("Tests timed out".to_string()),
        }

        // Step 6: Revert (best-effort)
        let _ = tokio::time::timeout(GIT_TIMEOUT, self.revert_changes_async()).await;

        Ok(())
    }

    async fn validate_patch_async(&self, patch: &str) -> Result<()> {
        use std::process::Stdio;

        if !patch.starts_with("--- ") || !patch.contains("\n+++ ") {
            return Err(BenchError::PatchValidation(
                "Patch missing proper --- and +++ headers (unified diff format required)".to_string(),
            ));
        }

        let mut child = TokioCommand::new("git")
            .arg("apply")
            .arg("--check")
            .current_dir(&self.config.workspace_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| BenchError::Git(format!("Failed to run git apply: {}", e)))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(patch.as_bytes())
                .await
                .map_err(|e| BenchError::Git(format!("Failed to write patch to stdin: {}", e)))?;
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| BenchError::Git(format!("Failed to run git apply: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BenchError::PatchValidation(format!("Patch validation failed: {}", stderr)));
        }

        Ok(())
    }

    async fn apply_patch_async(&self, patch: &str) -> Result<()> {
        use std::process::Stdio;

        let mut child = TokioCommand::new("git")
            .arg("apply")
            .current_dir(&self.config.workspace_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| BenchError::Git(format!("Failed to spawn git apply: {}", e)))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(patch.as_bytes())
                .await
                .map_err(|e| BenchError::Git(format!("Failed to write patch to stdin: {}", e)))?;
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| BenchError::Git(format!("Failed to apply patch: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BenchError::Git(format!("Patch application failed: {}", stderr)));
        }

        Ok(())
    }

    async fn run_build_async(&self, task: &EvalTask) -> Result<bool> {
        for crate_name in &task.test_crates {
            let output = TokioCommand::new("cargo")
                .args(&["build", "-p", crate_name, "--quiet"])
                .current_dir(&self.config.workspace_root)
                .output()
                .await
                .map_err(|e| BenchError::BuildFailed(format!("Build command failed: {}", e)))?;

            if !output.status.success() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn run_tests_async(&self, task: &EvalTask) -> Result<bool> {
        for crate_name in &task.test_crates {
            let mut cmd = TokioCommand::new("cargo");
            cmd.arg("test")
                .args(&["-p", crate_name])
                .current_dir(&self.config.workspace_root);

            for filter in &task.test_filters {
                cmd.arg(filter);
            }

            let output = cmd
                .output()
                .await
                .map_err(|e| BenchError::TestFailed(format!("Test command failed: {}", e)))?;

            if !output.status.success() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn revert_changes_async(&self) -> Result<()> {
        let output = TokioCommand::new("git")
            .args(&["checkout", "--", "."])
            .current_dir(&self.config.workspace_root)
            .output()
            .await
            .map_err(|e| BenchError::Git(format!("Failed to spawn git checkout: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Warning: Failed to fully revert changes: {}", stderr);
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn validate_patch_sync(&self, patch: &str) -> Result<()> {
        if !patch.starts_with("--- ") || !patch.contains("\n+++ ") {
            return Err(BenchError::PatchValidation(
                "Patch missing proper --- and +++ headers (unified diff format required)".to_string(),
            ));
        }
        Ok(())
    }
}

impl EvalResult {
    pub fn new() -> Self {
        EvalResult {
            run_id: Uuid::new_v4().to_string(),
            created_at: Utc::now(),
            outcomes: Vec::new(),
        }
    }

    fn unique_task_ids(&self) -> std::collections::HashSet<&str> {
        self.outcomes.iter().map(|o| o.task_id.as_str()).collect()
    }

    pub fn pass_at_1(&self) -> f32 {
        if self.outcomes.is_empty() {
            return 0.0;
        }

        let unique_count = self.unique_task_ids().len();
        if unique_count == 0 {
            return 0.0;
        }

        let passed = self
            .outcomes
            .iter()
            .filter(|o| o.attempt == 1 && o.tests_passed)
            .count();

        passed as f32 / unique_count as f32
    }

    pub fn pass_at_k(&self, k: usize) -> f32 {
        if self.outcomes.is_empty() {
            return 0.0;
        }

        let task_ids = self.unique_task_ids();
        if task_ids.is_empty() {
            return 0.0;
        }

        let passed_tasks = task_ids
            .iter()
            .filter(|task_id| {
                self.outcomes
                    .iter()
                    .filter(|o| o.task_id == **task_id && o.attempt <= k)
                    .any(|o| o.tests_passed)
            })
            .count();

        passed_tasks as f32 / task_ids.len() as f32
    }

    pub fn compilation_rate(&self) -> f32 {
        if self.outcomes.is_empty() {
            return 0.0;
        }

        let compiled = self.outcomes.iter().filter(|o| o.build_success).count();
        compiled as f32 / self.outcomes.len() as f32
    }
}

impl Default for EvalResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_outcome(task_id: &str, attempt: usize, tests_passed: bool, build_success: bool) -> TaskOutcome {
        TaskOutcome {
            task_id: task_id.to_string(),
            attempt,
            timestamp: Utc::now(),
            patch_generated: true,
            patch_text: Some("--- a/test\n+++ b/test\n".to_string()),
            patch_valid: true,
            build_success,
            tests_passed,
            error: None,
            llm_input_tokens: None,
            llm_output_tokens: None,
        }
    }

    #[test]
    fn test_pass_at_1_empty() {
        let result = EvalResult::new();
        assert_eq!(result.pass_at_1(), 0.0);
    }

    #[test]
    fn test_pass_at_1_single_pass() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, true, true));
        assert_eq!(result.pass_at_1(), 1.0);
    }

    #[test]
    fn test_pass_at_1_single_fail() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, false, false));
        assert_eq!(result.pass_at_1(), 0.0);
    }

    #[test]
    fn test_pass_at_1_multiple_tasks() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, true, true));
        result.outcomes.push(test_outcome("task2", 1, false, false));
        result.outcomes.push(test_outcome("task2", 2, true, true));
        assert_eq!(result.pass_at_1(), 0.5);
    }

    #[test]
    fn test_pass_at_k_empty() {
        let result = EvalResult::new();
        assert_eq!(result.pass_at_k(3), 0.0);
    }

    #[test]
    fn test_pass_at_k_within_limit() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, false, false));
        result.outcomes.push(test_outcome("task1", 2, true, true));
        assert_eq!(result.pass_at_k(3), 1.0);
    }

    #[test]
    fn test_pass_at_k_beyond_limit() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, false, false));
        result.outcomes.push(test_outcome("task1", 2, false, false));
        result.outcomes.push(test_outcome("task1", 3, false, false));
        result.outcomes.push(test_outcome("task1", 4, true, true));
        assert_eq!(result.pass_at_k(2), 0.0);
        assert_eq!(result.pass_at_k(4), 1.0);
    }

    #[test]
    fn test_pass_at_k_multiple_tasks() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, true, true));
        result.outcomes.push(test_outcome("task2", 1, false, false));
        result.outcomes.push(test_outcome("task2", 2, false, false));
        result.outcomes.push(test_outcome("task2", 3, true, true));
        result.outcomes.push(test_outcome("task3", 1, false, false));
        result.outcomes.push(test_outcome("task3", 2, false, false));

        let pass_at_2 = result.pass_at_k(2);
        assert!((pass_at_2 - 0.333).abs() < 0.01);

        let pass_at_3 = result.pass_at_k(3);
        assert!((pass_at_3 - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_compilation_rate_empty() {
        let result = EvalResult::new();
        assert_eq!(result.compilation_rate(), 0.0);
    }

    #[test]
    fn test_compilation_rate_all_success() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, true, true));
        result.outcomes.push(test_outcome("task2", 1, true, true));
        assert_eq!(result.compilation_rate(), 1.0);
    }

    #[test]
    fn test_compilation_rate_all_fail() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, false, false));
        result.outcomes.push(test_outcome("task2", 1, false, false));
        assert_eq!(result.compilation_rate(), 0.0);
    }

    #[test]
    fn test_compilation_rate_partial() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, false, true));
        result.outcomes.push(test_outcome("task2", 1, true, true));
        result.outcomes.push(test_outcome("task3", 1, false, false));
        assert!((result.compilation_rate() - (2.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn test_unique_task_ids() {
        let mut result = EvalResult::new();
        result.outcomes.push(test_outcome("task1", 1, false, false));
        result.outcomes.push(test_outcome("task1", 2, true, true));
        result.outcomes.push(test_outcome("task2", 1, true, true));
        result.outcomes.push(test_outcome("task3", 1, false, true));

        let unique = result.unique_task_ids();
        assert_eq!(unique.len(), 3);
        assert!(unique.contains("task1"));
        assert!(unique.contains("task2"));
        assert!(unique.contains("task3"));
    }

    #[test]
    fn test_patch_validation_format_check() {
        let patch_valid = "--- a/file.rs\n+++ b/file.rs\n@@ -1,3 +1,4 @@\n-old\n+new";
        assert!(patch_valid.starts_with("--- "));
        assert!(patch_valid.contains("\n+++ "));
    }

    #[test]
    fn test_patch_validation_missing_header() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);
        let patch = "some invalid patch without headers";
        assert!(runner.validate_patch_sync(patch).is_err());
    }

    #[test]
    fn test_patch_validation_loose_format() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);
        let patch = "---a/file.rs\n+++ b/file.rs\n@@ -1 @@\n";
        assert!(runner.validate_patch_sync(patch).is_err());
    }

    #[test]
    fn test_patch_validation_missing_plus_plus() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);
        let patch = "--- a/file.rs\nno plus plus here";
        assert!(runner.validate_patch_sync(patch).is_err());
    }

    #[tokio::test]
    async fn test_validate_patch_async_invalid_header() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);
        let patch = "some invalid patch without headers";
        let result = runner.validate_patch_async(patch).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing proper"));
    }

    #[tokio::test]
    async fn test_validate_patch_async_timeout() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);
        let valid_format_patch = "--- a/file.rs\n+++ b/file.rs\n@@ -1 @@\n";
        let result = tokio::time::timeout(Duration::from_secs(0), runner.validate_patch_async(valid_format_patch)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_build_async_nonexistent_crate() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);
        let task = EvalTask {
            id: "test".to_string(),
            title: "test".to_string(),
            description: "test".to_string(),
            issue: None,
            fix_commit: None,
            test_crates: vec!["nonexistent_crate_xyz".to_string()],
            test_filters: vec![],
            context_files: vec![],
            timeout_secs: 10,
            golden_patch: None,
        };
        let result = runner.run_build_async(&task).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
    }

    #[tokio::test]
    async fn test_run_tests_async_early_exit() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);
        let task = EvalTask {
            id: "test".to_string(),
            title: "test".to_string(),
            description: "test".to_string(),
            issue: None,
            fix_commit: None,
            test_crates: vec!["nonexistent_crate_xyz".to_string()],
            test_filters: vec![],
            context_files: vec![],
            timeout_secs: 10,
            golden_patch: None,
        };
        let result = runner.run_tests_async(&task).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
    }

    #[tokio::test]
    async fn test_revert_changes_async_clean_workspace() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);
        let result = runner.revert_changes_async().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_timeout_error_variant() {
        let err = BenchError::Timeout;
        let display = format!("{}", err);
        assert!(!display.is_empty());
        assert!(display.contains("Timeout"));
    }

    #[test]
    fn test_eval_result_default() {
        let result = EvalResult::default();
        assert!(!result.run_id.is_empty());
        assert!(result.outcomes.is_empty());
        assert_eq!(result.pass_at_1(), 0.0);
        assert_eq!(result.pass_at_k(3), 0.0);
        assert_eq!(result.compilation_rate(), 0.0);
    }

    #[test]
    fn test_run_config_defaults() {
        let config = RunConfig::new(".");
        assert_eq!(config.max_attempts, 1);
        assert_eq!(config.timeout_secs, 120);
    }
}
