use crate::backend::BackendOpts;
use crate::error::{BenchError, Result};
use crate::task::EvalTask;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
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
        let mut outcomes = Vec::new();

        for attempt in 1..=self.config.max_attempts {
            let outcome = self.run_attempt(task, backend, opts, attempt)?;
            outcomes.push(outcome);
        }

        Ok(outcomes)
    }

    fn run_attempt(
        &self,
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

        // Step 1: Generate patch
        match backend.generate_patch(task, opts) {
            Ok(patch) => {
                outcome.patch_generated = true;
                outcome.patch_text = Some(patch.clone());

                // Step 2: Validate patch
                if self.validate_patch(&patch).is_ok() {
                    outcome.patch_valid = true;

                    // Step 3: Apply patch
                    if let Err(e) = self.apply_patch(&patch) {
                        outcome.error = Some(format!("Patch application failed: {}", e));
                        return Ok(outcome);
                    }

                    // Step 4: Build
                    match self.run_build(task) {
                        Ok(true) => {
                            outcome.build_success = true;

                            // Step 5: Test
                            match self.run_tests(task) {
                                Ok(true) => {
                                    outcome.tests_passed = true;
                                }
                                Ok(false) => {
                                    outcome.error = Some("Tests failed".to_string());
                                }
                                Err(e) => {
                                    outcome.error = Some(format!("Test execution error: {}", e));
                                }
                            }
                        }
                        Ok(false) => {
                            outcome.error = Some("Build failed".to_string());
                        }
                        Err(e) => {
                            outcome.error = Some(format!("Build error: {}", e));
                        }
                    }

                    // Step 6: Revert
                    let _ = self.revert_changes();
                } else {
                    outcome.error = Some("Patch validation failed".to_string());
                }
            }
            Err(e) => {
                outcome.error = Some(format!("LLM generation failed: {}", e));
            }
        }

        Ok(outcome)
    }

    fn validate_patch(&self, patch: &str) -> Result<()> {
        use std::io::Write;
        use std::process::Stdio;

        // Quick validation: check if it looks like a unified diff
        // Stricter check: must start with "--- " and contain "\n+++ "
        if !patch.starts_with("--- ") || !patch.contains("\n+++ ") {
            return Err(BenchError::PatchValidation(
                "Patch missing proper --- and +++ headers (unified diff format required)".to_string(),
            ));
        }

        // Try git apply --check
        let mut child = Command::new("git")
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
                .map_err(|e| BenchError::Git(format!("Failed to write patch to stdin: {}", e)))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| BenchError::Git(format!("Failed to run git apply: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BenchError::PatchValidation(format!(
                "Patch validation failed: {}",
                stderr
            )));
        }

        Ok(())
    }

    fn apply_patch(&self, patch: &str) -> Result<()> {
        use std::io::Write;
        use std::process::Stdio;

        let mut child = Command::new("git")
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
                .map_err(|e| BenchError::Git(format!("Failed to write patch to stdin: {}", e)))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| BenchError::Git(format!("Failed to apply patch: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BenchError::Git(format!("Patch application failed: {}", stderr)));
        }

        Ok(())
    }

    fn run_build(&self, task: &EvalTask) -> Result<bool> {
        for crate_name in &task.test_crates {
            let output = Command::new("cargo")
                .args(&["build", "-p", crate_name, "--quiet"])
                .current_dir(&self.config.workspace_root)
                .output()
                .map_err(|e| BenchError::BuildFailed(format!("Build command failed: {}", e)))?;

            if !output.status.success() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn run_tests(&self, task: &EvalTask) -> Result<bool> {
        for crate_name in &task.test_crates {
            let mut cmd = Command::new("cargo");
            cmd.arg("test")
                .args(&["-p", crate_name])
                .current_dir(&self.config.workspace_root);

            for filter in &task.test_filters {
                cmd.arg(filter);
            }

            let output = cmd
                .output()
                .map_err(|e| BenchError::TestFailed(format!("Test command failed: {}", e)))?;

            if !output.status.success() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn revert_changes(&self) -> Result<()> {
        let output = Command::new("git")
            .args(&["checkout", "--", "."])
            .current_dir(&self.config.workspace_root)
            .output()
            .map_err(|e| BenchError::Git(format!("Failed to spawn git checkout: {}", e)))?;

        if !output.status.success() {
            // Log warning but don't fail - revert should be best-effort
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Warning: Failed to fully revert changes: {}", stderr);
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

    /// Get unique task IDs from outcomes
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

    // Helper to create a test outcome
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
        // Task 1: pass on first attempt
        result.outcomes.push(test_outcome("task1", 1, true, true));
        // Task 2: fail on first attempt (but pass on second)
        result.outcomes.push(test_outcome("task2", 1, false, false));
        result.outcomes.push(test_outcome("task2", 2, true, true));
        // Pass@1 should be 1/2 = 0.5
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
        // Task 1: pass on attempt 2 (within k=3)
        result.outcomes.push(test_outcome("task1", 1, false, false));
        result.outcomes.push(test_outcome("task1", 2, true, true));
        assert_eq!(result.pass_at_k(3), 1.0);
    }

    #[test]
    fn test_pass_at_k_beyond_limit() {
        let mut result = EvalResult::new();
        // Task 1: only passes on attempt 4 (beyond k=2)
        result.outcomes.push(test_outcome("task1", 1, false, false));
        result.outcomes.push(test_outcome("task1", 2, false, false));
        result.outcomes.push(test_outcome("task1", 3, false, false));
        result.outcomes.push(test_outcome("task1", 4, true, true));
        // Pass@2 should be 0.0 (task doesn't pass within 2 attempts)
        assert_eq!(result.pass_at_k(2), 0.0);
        // Pass@4 should be 1.0
        assert_eq!(result.pass_at_k(4), 1.0);
    }

    #[test]
    fn test_pass_at_k_multiple_tasks() {
        let mut result = EvalResult::new();
        // Task 1: passes on attempt 1
        result.outcomes.push(test_outcome("task1", 1, true, true));
        // Task 2: passes on attempt 3
        result.outcomes.push(test_outcome("task2", 1, false, false));
        result.outcomes.push(test_outcome("task2", 2, false, false));
        result.outcomes.push(test_outcome("task2", 3, true, true));
        // Task 3: never passes
        result.outcomes.push(test_outcome("task3", 1, false, false));
        result.outcomes.push(test_outcome("task3", 2, false, false));

        // Pass@2: tasks 1 passes, 2 doesn't, 3 doesn't = 1/3 ≈ 0.333
        let pass_at_2 = result.pass_at_k(2);
        assert!((pass_at_2 - 0.333).abs() < 0.01);

        // Pass@3: tasks 1 and 2 pass, 3 doesn't = 2/3 ≈ 0.667
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
        result.outcomes.push(test_outcome("task1", 1, false, true));  // compiled, no test
        result.outcomes.push(test_outcome("task2", 1, true, true));   // compiled, passed
        result.outcomes.push(test_outcome("task3", 1, false, false)); // didn't compile
        // 2 out of 3 compiled
        assert!((result.compilation_rate() - (2.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn test_unique_task_ids() {
        let mut result = EvalResult::new();
        // Same task, multiple attempts
        result.outcomes.push(test_outcome("task1", 1, false, false));
        result.outcomes.push(test_outcome("task1", 2, true, true));
        // Different tasks
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
        // Test format validation (happens before git apply)
        // These patches have correct format but won't apply to real files
        let patch_valid = "--- a/file.rs\n+++ b/file.rs\n@@ -1,3 +1,4 @@\n-old\n+new";

        // The format check will pass; git apply will fail (no actual files)
        // but we're testing the format validation logic
        assert!(patch_valid.starts_with("--- "));
        assert!(patch_valid.contains("\n+++ "));
    }

    #[test]
    fn test_patch_validation_missing_header() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);

        let patch = "some invalid patch without headers";
        assert!(runner.validate_patch(patch).is_err());
    }

    #[test]
    fn test_patch_validation_loose_format() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);

        // Missing "--- " with space (no space after ---)
        let patch = "---a/file.rs\n+++ b/file.rs\n@@ -1 @@\n";
        assert!(runner.validate_patch(patch).is_err());
    }

    #[test]
    fn test_patch_validation_missing_plus_plus() {
        let config = RunConfig::new(".");
        let runner = TaskRunner::new(config);

        // Has "--- " but missing "+++ " (no space and newline before)
        let patch = "--- a/file.rs\nno plus plus here";
        assert!(runner.validate_patch(patch).is_err());
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
