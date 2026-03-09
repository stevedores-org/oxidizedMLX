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
        if !patch.contains("---") || !patch.contains("+++") {
            return Err(BenchError::PatchValidation(
                "Patch missing --- and +++ headers".to_string(),
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
        let _ = Command::new("git")
            .args(&["checkout", "--", "."])
            .current_dir(&self.config.workspace_root)
            .output();

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

    pub fn pass_at_1(&self) -> f32 {
        if self.outcomes.is_empty() {
            return 0.0;
        }

        let passed = self
            .outcomes
            .iter()
            .filter(|o| o.attempt == 1 && o.tests_passed)
            .count();

        passed as f32 / self.outcomes.iter().map(|o| o.task_id.as_str()).collect::<std::collections::HashSet<_>>().len() as f32
    }

    pub fn pass_at_k(&self, k: usize) -> f32 {
        if self.outcomes.is_empty() {
            return 0.0;
        }

        let task_ids: std::collections::HashSet<_> =
            self.outcomes.iter().map(|o| o.task_id.as_str()).collect();

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
