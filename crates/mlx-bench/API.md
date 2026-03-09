# mlx-bench Rust API Documentation

## Overview

The `mlx-bench` crate provides a public API for programmatic integration into other Rust programs.

## Module Exports

```rust
pub mod backend;
pub mod error;
pub mod report;
pub mod runner;
pub mod task;

pub use error::{BenchError, Result};
```

## Main Types

### task::EvalTask

Represents a single evaluation task.

```rust
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
```

**Fields:**
- `id` - Unique identifier for the task
- `title` - Human-readable title
- `issue` - Optional GitHub issue number
- `fix_commit` - Optional reference commit hash
- `description` - Full issue description (sent to LLM)
- `context_files` - Code files for context
- `test_filters` - Cargo test filter strings
- `test_crates` - Crates to build and test
- `timeout_secs` - Timeout for operations
- `golden_patch` - Optional known-good patch

**Example:**
```rust
let task = EvalTask {
    id: "001_softmax".to_string(),
    title: "softmax bounds check".to_string(),
    description: "Issue: softmax doesn't validate axis...".to_string(),
    context_files: vec![...],
    test_filters: vec!["test_softmax".to_string()],
    test_crates: vec!["mlx-core".to_string()],
    timeout_secs: 120,
    issue: Some(87),
    fix_commit: None,
    golden_patch: Some("--- a/...".to_string()),
};
```

### task::ContextFile

Represents a code file for context.

```rust
pub struct ContextFile {
    pub path: String,
    pub lines: Option<(usize, usize)>,
    pub annotation: Option<String>,
}
```

**Fields:**
- `path` - Workspace-relative path (e.g., `crates/mlx-core/src/tensor.rs`)
- `lines` - Optional `(start, end)` line range (1-indexed, inclusive)
- `annotation` - Optional description of content

**Example:**
```rust
let context = ContextFile {
    path: "crates/mlx-core/src/tensor.rs".to_string(),
    lines: Some((1100, 1130)),
    annotation: Some("softmax method".to_string()),
};

// Read content with line slicing
let content = context.read_content(Path::new("."))?;
```

### task::TaskSet

Collection of evaluation tasks.

```rust
pub struct TaskSet { /* private */ }

impl TaskSet {
    /// Load all tasks from evals/tasks/*.json
    pub fn load_from_dir(base_dir: impl AsRef<Path>) -> Result<Self>;

    /// Get all tasks
    pub fn all(&self) -> &[EvalTask];

    /// Get task by ID
    pub fn by_id(&self, id: &str) -> Option<&EvalTask>;

    /// Get tasks matching glob pattern
    pub fn by_glob(&self, glob_pattern: &str) -> Vec<&EvalTask>;

    /// Base directory path
    pub fn base_dir(&self) -> &Path;

    /// Validate all tasks
    pub fn validate(&self) -> Result<()>;
}
```

**Example:**
```rust
use mlx_bench::task::TaskSet;

let tasks = TaskSet::load_from_dir(".")?;

println!("Total tasks: {}", tasks.all().len());

if let Some(task) = tasks.by_id("001_softmax_axis_oob") {
    println!("Task: {}", task.title);
}

let softmax_tasks = tasks.by_glob("*softmax*");
println!("Softmax tasks: {}", softmax_tasks.len());

tasks.validate()?;
```

### runner::TaskRunner

Orchestrates evaluation of tasks.

```rust
pub struct TaskRunner { /* private */ }

impl TaskRunner {
    /// Create new runner with configuration
    pub fn new(config: RunConfig) -> Self;

    /// Run a single task with LLM backend
    pub fn run_task(
        &self,
        task: &EvalTask,
        backend: &dyn LlmBackend,
        opts: &BackendOpts,
    ) -> Result<Vec<TaskOutcome>>;
}
```

**Example:**
```rust
use mlx_bench::runner::{TaskRunner, RunConfig};
use mlx_bench::backend::{AnthropicApiBackend, BackendOpts};

let config = RunConfig::new(".");
let runner = TaskRunner::new(config);

let backend = AnthropicApiBackend::new("evals/backends/anthropic_api.py")?;

let opts = BackendOpts {
    model_id: "claude-3-5-sonnet-20241022".to_string(),
    max_tokens: 2048,
    temperature: 0.2,
    timeout_secs: 120,
};

let outcomes = runner.run_task(&task, &backend, &opts)?;

for outcome in outcomes {
    println!("Attempt {}: {}", outcome.attempt,
        if outcome.tests_passed { "PASS" } else { "FAIL" });
}
```

### runner::RunConfig

Configuration for task runner.

```rust
pub struct RunConfig {
    pub workspace_root: PathBuf,
    pub max_attempts: usize,
    pub timeout_secs: u64,
}

impl RunConfig {
    /// Create with workspace root and defaults
    pub fn new(workspace_root: impl AsRef<Path>) -> Self;
}
```

**Example:**
```rust
use mlx_bench::runner::RunConfig;

let config = RunConfig::new(".");
// or with custom values
let mut config = RunConfig::new("/path/to/workspace");
config.max_attempts = 3;
config.timeout_secs = 300;
```

### runner::TaskOutcome

Result of a single task attempt.

```rust
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
```

**Fields:**
- `task_id` - Which task this outcome is for
- `attempt` - Attempt number (1-indexed)
- `timestamp` - When this outcome was recorded
- `patch_generated` - LLM successfully generated a patch
- `patch_text` - The generated patch (if successful)
- `patch_valid` - Patch passed `git apply --check`
- `build_success` - Build succeeded
- `tests_passed` - Tests passed
- `error` - Error message if failure
- `llm_input_tokens` - Input token count (if tracked)
- `llm_output_tokens` - Output token count (if tracked)

**Example:**
```rust
for outcome in outcomes {
    println!("Task: {}", outcome.task_id);
    println!("  Attempt: {}", outcome.attempt);
    println!("  Generated: {}", outcome.patch_generated);
    println!("  Valid: {}", outcome.patch_valid);
    println!("  Build: {}", outcome.build_success);
    println!("  Tests: {}", outcome.tests_passed);

    if let Some(error) = &outcome.error {
        println!("  Error: {}", error);
    }

    if let Some(tokens) = outcome.llm_input_tokens {
        println!("  Input tokens: {}", tokens);
    }
}
```

### runner::EvalResult

Complete evaluation run results.

```rust
pub struct EvalResult {
    pub run_id: String,
    pub created_at: DateTime<Utc>,
    pub outcomes: Vec<TaskOutcome>,
}

impl EvalResult {
    /// Create new empty result
    pub fn new() -> Self;

    /// Pass@1 metric (0.0 to 1.0)
    pub fn pass_at_1(&self) -> f32;

    /// Pass@k metric (0.0 to 1.0)
    pub fn pass_at_k(&self, k: usize) -> f32;

    /// Compilation rate (0.0 to 1.0)
    pub fn compilation_rate(&self) -> f32;
}
```

**Example:**
```rust
use mlx_bench::runner::EvalResult;
use serde_json;
use std::fs;

// Load from file
let json = fs::read_to_string("evals/results/run-123.json")?;
let result: EvalResult = serde_json::from_str(&json)?;

println!("Run ID: {}", result.run_id);
println!("Created: {}", result.created_at);
println!("Total outcomes: {}", result.outcomes.len());

// Metrics
println!("Pass@1: {:.2}%", result.pass_at_1() * 100.0);
println!("Pass@3: {:.2}%", result.pass_at_k(3) * 100.0);
println!("Compilation: {:.2}%", result.compilation_rate() * 100.0);
```

### backend::LlmBackend

Trait for LLM backends.

```rust
pub trait LlmBackend: Send + Sync {
    /// Generate patch for a task
    fn generate_patch(
        &self,
        task: &EvalTask,
        opts: &BackendOpts,
    ) -> Result<String>;

    /// Backend name for logging
    fn name(&self) -> &str;
}
```

**Implementations:**
- `LocalMlxBackend` - Apple Silicon Metal GPU
- `AnthropicApiBackend` - Claude API
- `StdinDebugBackend` - Debug/testing

**Example - Custom Backend:**
```rust
use mlx_bench::backend::{LlmBackend, BackendOpts};
use mlx_bench::task::EvalTask;
use mlx_bench::error::Result;

struct CustomBackend {
    api_key: String,
}

impl LlmBackend for CustomBackend {
    fn generate_patch(
        &self,
        task: &EvalTask,
        opts: &BackendOpts,
    ) -> Result<String> {
        // Your implementation
        // Call your LLM API
        // Return patch text
        Ok("--- a/...".to_string())
    }

    fn name(&self) -> &str {
        "custom"
    }
}
```

### backend::BackendOpts

Options for LLM backend.

```rust
pub struct BackendOpts {
    pub model_id: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout_secs: u64,
}
```

**Example:**
```rust
use mlx_bench::backend::BackendOpts;

let opts = BackendOpts {
    model_id: "claude-3-5-sonnet-20241022".to_string(),
    max_tokens: 2048,
    temperature: 0.2,
    timeout_secs: 120,
};
```

### backend::LocalMlxBackend

Local MLX backend using `mlx_lm`.

```rust
pub struct LocalMlxBackend { /* private */ }

impl LocalMlxBackend {
    /// Create new local MLX backend
    pub fn new(python_script: impl AsRef<Path>) -> Result<Self>;
}

impl LlmBackend for LocalMlxBackend { ... }
```

**Example:**
```rust
use mlx_bench::backend::LocalMlxBackend;

let backend = LocalMlxBackend::new("evals/backends/mlx_local.py")?;
println!("Backend: {}", backend.name());  // "local"
```

### backend::AnthropicApiBackend

Anthropic API backend.

```rust
pub struct AnthropicApiBackend { /* private */ }

impl AnthropicApiBackend {
    /// Create new Anthropic backend
    pub fn new(python_script: impl AsRef<Path>) -> Result<Self>;
}

impl LlmBackend for AnthropicApiBackend { ... }
```

**Example:**
```rust
use mlx_bench::backend::AnthropicApiBackend;

let backend = AnthropicApiBackend::new("evals/backends/anthropic_api.py")?;
println!("Backend: {}", backend.name());  // "anthropic"
```

### backend::StdinDebugBackend

Debug backend for testing.

```rust
pub struct StdinDebugBackend;

impl LlmBackend for StdinDebugBackend { ... }
```

**Example:**
```rust
use mlx_bench::backend::StdinDebugBackend;
use std::env;

env::set_var("BENCH_PATCH_FILE", "/tmp/patch.diff");
let backend = StdinDebugBackend;
// Generates patches from BENCH_PATCH_FILE
```

### report::Reporter

Report generation.

```rust
pub struct Reporter;

impl Reporter {
    /// Generate report from results
    pub fn generate(result: &EvalResult, format: ReportFormat) -> String;
}
```

**Example:**
```rust
use mlx_bench::report::{Reporter, ReportFormat};

let report = Reporter::generate(&result, ReportFormat::Table);
println!("{}", report);

let json_report = Reporter::generate(&result, ReportFormat::Json);
let markdown_report = Reporter::generate(&result, ReportFormat::Markdown);
```

### report::ReportFormat

Output format for reports.

```rust
pub enum ReportFormat {
    Table,      // ASCII table
    Json,       // JSON
    Markdown,   // Markdown
}
```

### error::BenchError

Custom error type.

```rust
pub enum BenchError {
    Io(std::io::Error),
    Json(serde_json::Error),
    TaskNotFound(String),
    InvalidTask(String),
    Backend(String),
    LlmGeneration(String),
    PatchValidation(String),
    BuildFailed(String),
    TestFailed(String),
    Git(String),
    Timeout,
    InvalidConfig(String),
    PythonSubprocess(String),
}

pub type Result<T> = std::result::Result<T, BenchError>;
```

**Example:**
```rust
use mlx_bench::error::{BenchError, Result};

fn my_function() -> Result<String> {
    TaskSet::load_from_dir(".")?;
    Ok("success".to_string())
}

match my_function() {
    Ok(value) => println!("{}", value),
    Err(BenchError::TaskNotFound(id)) => {
        eprintln!("Task not found: {}", id);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Complete Example

```rust
use mlx_bench::{
    task::TaskSet,
    runner::{TaskRunner, RunConfig},
    backend::{AnthropicApiBackend, BackendOpts},
    report::{Reporter, ReportFormat},
};

fn main() -> mlx_bench::Result<()> {
    // Load tasks
    let tasks = TaskSet::load_from_dir(".")?;
    println!("Loaded {} tasks", tasks.all().len());

    // Validate
    tasks.validate()?;

    // Get specific task
    let task = tasks.by_id("001_softmax_axis_oob")
        .ok_or_else(|| mlx_bench::BenchError::TaskNotFound(
            "001_softmax_axis_oob".to_string()
        ))?;

    // Set up runner
    let config = RunConfig::new(".");
    let runner = TaskRunner::new(config);

    // Create backend
    let backend = AnthropicApiBackend::new("evals/backends/anthropic_api.py")?;

    // Configure options
    let opts = BackendOpts {
        model_id: "claude-3-5-sonnet-20241022".to_string(),
        max_tokens: 2048,
        temperature: 0.2,
        timeout_secs: 120,
    };

    // Run task
    let outcomes = runner.run_task(task, &backend, &opts)?;

    // Display results
    for outcome in &outcomes {
        println!("Attempt {}: {}",
            outcome.attempt,
            if outcome.tests_passed { "PASS" } else { "FAIL" }
        );
    }

    // Generate report
    let mut result = mlx_bench::runner::EvalResult::new();
    result.outcomes.extend(outcomes);

    let report = Reporter::generate(&result, ReportFormat::Table);
    println!("{}", report);

    Ok(())
}
```

## Feature Flags

Currently no feature flags. All dependencies are always included.

Possible future flags:
- `local-mlx` - Include LocalMlxBackend
- `anthropic` - Include AnthropicApiBackend

## Error Handling Best Practices

### Pattern 1: Propagate Errors
```rust
fn evaluate_task(task: &EvalTask) -> mlx_bench::Result<TaskOutcome> {
    let runner = TaskRunner::new(RunConfig::new("."))?;
    let backend = AnthropicApiBackend::new("evals/backends/anthropic_api.py")?;
    runner.run_task(task, &backend, &opts)?
        .into_iter()
        .next()
        .ok_or_else(|| BenchError::TaskNotFound(task.id.clone()))
}
```

### Pattern 2: Log and Continue
```rust
for task in tasks.all() {
    match runner.run_task(task, &backend, &opts) {
        Ok(outcomes) => {
            println!("Task {} succeeded", task.id);
        }
        Err(e) => {
            eprintln!("Task {} failed: {}", task.id, e);
            continue;
        }
    }
}
```

### Pattern 3: Custom Error Messages
```rust
TaskSet::load_from_dir(".")
    .map_err(|e| {
        eprintln!("Failed to load tasks: {}", e);
        e
    })?
```

## Testing

### Unit Tests

Currently no public tests. Can be added with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_loading() {
        let tasks = TaskSet::load_from_dir(".").unwrap();
        assert!(!tasks.all().is_empty());
    }

    #[test]
    fn test_metrics() {
        let result = EvalResult::new();
        assert_eq!(result.pass_at_1(), 0.0);
    }
}
```

### Integration Tests

Create `tests/integration_test.rs`:

```rust
use mlx_bench::task::TaskSet;

#[test]
fn load_and_validate_tasks() {
    let tasks = TaskSet::load_from_dir(".").expect("Failed to load tasks");
    tasks.validate().expect("Tasks invalid");
}
```

## Versioning

Current version: `0.1.0`

Semver guarantees:
- `0.1.x` - API may change without notice
- `1.0.0+` - Semantic versioning applied

## Contributing

To add new public APIs:
1. Document with doc comments
2. Add examples
3. Update this file
4. Consider backward compatibility

## See Also

- `README.md` - User guide
- `ARCHITECTURE.md` - Design details
- `src/` - Source code with inline documentation
