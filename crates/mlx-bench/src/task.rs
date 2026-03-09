use crate::error::{BenchError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalTask {
    pub id: String,
    pub title: String,
    pub issue: Option<u32>,
    pub fix_commit: Option<String>,
    pub description: String,
    pub context_files: Vec<ContextFile>,
    pub test_filters: Vec<String>,
    pub test_crates: Vec<String>,
    pub timeout_secs: u64,
    pub golden_patch: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFile {
    pub path: String,
    pub lines: Option<(usize, usize)>,
    pub annotation: Option<String>,
}

pub struct TaskSet {
    tasks: Vec<EvalTask>,
    base_dir: PathBuf,
}

impl TaskSet {
    /// Load all tasks from evals/tasks/*.json directory
    pub fn load_from_dir(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref();
        let tasks_dir = base_dir.join("evals/tasks");

        if !tasks_dir.exists() {
            return Err(BenchError::InvalidConfig(format!(
                "Tasks directory not found: {}",
                tasks_dir.display()
            )));
        }

        let mut tasks = Vec::new();
        let pattern = tasks_dir.join("*.json");

        let entries = glob::glob(pattern.to_str().unwrap())
            .map_err(|e| BenchError::InvalidConfig(format!("Glob error: {}", e)))?;

        for entry in entries {
            let path = entry.map_err(|e| {
                BenchError::InvalidConfig(format!("Path iteration error: {}", e))
            })?;

            let content = fs::read_to_string(&path)
                .map_err(|e| BenchError::Io(e))?;

            let task: EvalTask = serde_json::from_str(&content)
                .map_err(|e| {
                    BenchError::InvalidTask(format!("{}: {}", path.display(), e))
                })?;

            tasks.push(task);
        }

        tasks.sort_by(|a, b| a.id.cmp(&b.id));

        Ok(TaskSet {
            tasks,
            base_dir: base_dir.to_path_buf(),
        })
    }

    pub fn all(&self) -> &[EvalTask] {
        &self.tasks
    }

    pub fn by_id(&self, id: &str) -> Option<&EvalTask> {
        self.tasks.iter().find(|t| t.id == id)
    }

    pub fn by_glob(&self, glob_pattern: &str) -> Vec<&EvalTask> {
        self.tasks
            .iter()
            .filter(|t| glob::Pattern::new(glob_pattern)
                .map(|p| p.matches(&t.id))
                .unwrap_or(false))
            .collect()
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn validate(&self) -> Result<()> {
        for task in &self.tasks {
            if task.id.is_empty() {
                return Err(BenchError::InvalidTask(
                    "Task missing id".to_string(),
                ));
            }
            if task.title.is_empty() {
                return Err(BenchError::InvalidTask(format!(
                    "Task {} missing title",
                    task.id
                )));
            }
            if task.description.is_empty() {
                return Err(BenchError::InvalidTask(format!(
                    "Task {} missing description",
                    task.id
                )));
            }
            if task.context_files.is_empty() {
                return Err(BenchError::InvalidTask(format!(
                    "Task {} has no context_files",
                    task.id
                )));
            }
            if task.test_filters.is_empty() {
                return Err(BenchError::InvalidTask(format!(
                    "Task {} has no test_filters",
                    task.id
                )));
            }
            if task.test_crates.is_empty() {
                return Err(BenchError::InvalidTask(format!(
                    "Task {} has no test_crates",
                    task.id
                )));
            }
        }
        Ok(())
    }
}

impl ContextFile {
    /// Read file content, optionally sliced by line range
    pub fn read_content(&self, root: &Path) -> Result<String> {
        let path = root.join(&self.path);

        if !path.exists() {
            return Err(BenchError::InvalidTask(format!(
                "Context file not found: {}",
                self.path
            )));
        }

        let content = fs::read_to_string(&path)
            .map_err(|e| BenchError::Io(e))?;

        match self.lines {
            None => Ok(content),
            Some((start, end)) => {
                let lines: Vec<&str> = content.lines().collect();
                if start < 1 || end < start || end > lines.len() {
                    return Err(BenchError::InvalidTask(format!(
                        "Invalid line range [{}, {}] for file {} (has {} lines)",
                        start,
                        end,
                        self.path,
                        lines.len()
                    )));
                }
                let slice = lines[(start - 1)..end].join("\n");
                Ok(slice)
            }
        }
    }
}
