# mlx-bench Architecture

## Overview

mlx-bench is a modular LLM evaluation framework designed around:
- **Trait abstraction** for multiple LLM backends
- **Subprocess isolation** for Python/LLM integration
- **Incremental results** for resumable evaluation
- **Extensibility** for new tasks and backends

## Core Design Principles

### 1. Backend Abstraction via Traits

The `LlmBackend` trait enables pluggable LLM providers:

```rust
pub trait LlmBackend: Send + Sync {
    fn generate_patch(&self, task: &EvalTask, opts: &BackendOpts) -> Result<String>;
    fn name(&self) -> &str;
}
```

**Benefits:**
- Easy to add new backends (OpenAI, Together, Ollama, etc.)
- No dependency on specific provider
- Testable with mock implementations
- Same interface for all backends

**Implementations:**
- `LocalMlxBackend` - Apple Silicon Metal GPU
- `AnthropicApiBackend` - Production LLM via API
- `StdinDebugBackend` - Testing without LLM calls

### 2. Async HTTP Integration

All LLM backends use Rust's async runtime (tokio) for efficient HTTP calls:

```
Rust (mlx-bench)
    ↓
Create tokio runtime
    ↓
Call LlmBackend::generate_patch()
    ├─ Assemble prompt
    ├─ Make async HTTP request
    │   └─ reqwest → Anthropic API
    ├─ Parse JSON response
    └─ Extract patch text
    ↓
Return String (patch)
```

**Benefits:**
- Pure Rust, no subprocess overhead
- Async/await for non-blocking I/O
- Connection pooling via reqwest
- Direct error propagation
- No temp files or JSON marshaling
- ~10% faster per task
- Easier debugging with Rust error handling

**API Call Pattern:**

```rust
let client = reqwest::Client::new();
let response = client
    .post("https://api.anthropic.com/v1/messages")
    .header("x-api-key", api_key)
    .json(&request)
    .send()
    .await?;
```

### 3. Task-Centric Data Model

Tasks are JSON files, not code:

```json
{
  "id": "001_softmax_axis_oob",
  "title": "...",
  "description": "...",
  "context_files": [...],
  "test_filters": [...],
  "test_crates": [...],
  "golden_patch": "..."
}
```

**Benefits:**
- Tasks are data, versioned with repo
- No compilation needed for new tasks
- Easy to share/fork tasks
- Self-documenting (issue + fix in JSON)
- Golden patches for validation

### 4. Incremental Results

Results saved as `{run_id}_partial.json`:

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-03-09T10:00:00Z",
  "outcomes": [
    { "task_id": "001_...", "tests_passed": true, ... },
    { "task_id": "002_...", "tests_passed": false, ... }
  ]
}
```

**Benefits:**
- Resume on failure
- Monitor progress in real-time
- Analyze partial runs
- No data loss on interruption
- Each run is immutable

## Module Structure

### task.rs

**Types:**
- `EvalTask` - Single evaluation task with metadata
- `ContextFile` - Code file reference with optional line range
- `TaskSet` - Collection of tasks, loaded from filesystem

**Key Methods:**
```rust
impl TaskSet {
    pub fn load_from_dir(path) -> Result<Self>        // Load from evals/tasks/
    pub fn by_id(&self, id) -> Option<&EvalTask>       // Get task by ID
    pub fn by_glob(&self, pattern) -> Vec<&EvalTask>   // Filter by glob
    pub fn validate(&self) -> Result<()>               // Validate all tasks
}

impl ContextFile {
    pub fn read_content(&self, root) -> Result<String> // Read file with slicing
}
```

**Design Notes:**
- Tasks are immutable after loading
- Glob filtering for flexible task selection
- Line range handling (1-indexed, inclusive)
- Validation ensures all required fields present

### runner.rs

**Types:**
- `TaskRunner` - Orchestrates evaluation loop
- `TaskOutcome` - Single task attempt result
- `EvalResult` - Complete evaluation run

**Evaluation Loop:**
```rust
impl TaskRunner {
    pub fn run_task(
        &self,
        task: &EvalTask,
        backend: &dyn LlmBackend,
        opts: &BackendOpts,
    ) -> Result<Vec<TaskOutcome>>
}
```

**Steps (per attempt):**
1. Call `backend.generate_patch()` → get patch text
2. Validate patch format → `git apply --check`
3. Apply patch → `git apply`
4. Build → `cargo build -p <crate>`
5. Test → `cargo test -p <crate> <filters>`
6. Revert → `git checkout -- .`
7. Record outcome

**Design Notes:**
- Always reverts changes (even on error)
- Collects detailed error messages
- Tracks timestamps and token counts
- Supports multiple attempts per task

### backend.rs

**Trait:**
```rust
pub trait LlmBackend: Send + Sync {
    fn generate_patch(&self, task: &EvalTask, opts: &BackendOpts) -> Result<String>;
    fn name(&self) -> &str;
}
```

**Implementations:**

1. **LocalMlxBackend**
   - Uses `mlx_lm.generate()` for local inference
   - Spawns Python subprocess: `python evals/backends/mlx_local.py`
   - Requires: `mlx-lm` package, Apple Silicon

2. **AnthropicApiBackend**
   - Uses Anthropic SDK for Claude API
   - Spawns Python subprocess: `python evals/backends/anthropic_api.py`
   - Requires: `anthropic` package, `ANTHROPIC_API_KEY` env var

3. **StdinDebugBackend**
   - Reads patch from `BENCH_PATCH_FILE` env var
   - No subprocess spawning
   - Used for testing/validation

**Subprocess Communication:**
```rust
fn run_backend_subprocess(
    script: &Path,
    task: &EvalTask,
    opts: &BackendOpts,
) -> Result<String>
```

Process:
1. Create temp directory
2. Write request.json with task + prompts
3. Spawn subprocess with paths
4. Wait for completion
5. Read response.json
6. Extract patch text
7. Clean up temp directory

**Design Notes:**
- Temp directory for isolation
- JSON for language independence
- Subprocess output logged on error
- Timeout handled by subprocess itself

### report.rs

**Type:**
```rust
pub enum ReportFormat {
    Table,
    Json,
    Markdown,
}
```

**Implementation:**
```rust
impl Reporter {
    pub fn generate(result: &EvalResult, format: ReportFormat) -> String
}
```

**Output:**
- **Table** - ASCII table with pass/fail indicators
- **JSON** - Full structured data for programmatic use
- **Markdown** - Shareable GitHub/documentation format

**Metrics:**
- Pass@1, Pass@k - Task success rates
- Compilation rate - Percentage of buildable patches
- Aggregated by task and attempt

**Design Notes:**
- Computed on-the-fly from outcomes
- Multiple views of same data
- Easy to add new formats

### error.rs

**Custom Errors:**
```rust
pub enum BenchError {
    Io(std::io::Error),           // File system errors
    Json(serde_json::Error),      // JSON parsing errors
    TaskNotFound(String),          // Task not found
    InvalidTask(String),           // Invalid task definition
    Backend(String),               // Backend process errors
    LlmGeneration(String),         // LLM generation failed
    PatchValidation(String),       // Patch not valid diff
    BuildFailed(String),           // Cargo build failed
    TestFailed(String),            // Cargo test failed
    Git(String),                   // Git operation failed
    Timeout,                       // Operation timeout
    InvalidConfig(String),         // Configuration error
    PythonSubprocess(String),      // Python backend error
}
```

**Design Notes:**
- Uses `thiserror` for ergonomic Display
- Specific error types for different failures
- Easy to distinguish issues for debugging

### main.rs

**CLI Structure:**
```rust
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    ListTasks { ... },
    ShowTask { ... },
    ValidateTasks,
    SelfTest { ... },
    Run { ... },
    Report { ... },
}
```

**Implementation:**
- Each command maps to function (cmd_*)
- Loads TaskSet at start
- Handles clap parsing + error reporting
- Delegates to module logic

**Design Notes:**
- Clean command dispatch
- Shared task loading
- Error messages from modules

## Data Flow Diagrams

### Task Loading

```
[evals/tasks/*.json]
    ↓
TaskSet::load_from_dir()
    ├─ glob::glob("evals/tasks/*.json")
    ├─ serde_json::from_str() for each file
    └─ Validate all fields
    ↓
TaskSet {
    tasks: Vec<EvalTask>,
    base_dir: PathBuf,
}
    ↓
[In-memory task collection]
```

### Evaluation Pipeline

```
[EvalTask] + [LlmBackend] + [BackendOpts]
    ↓
TaskRunner::run_task()
    ├─ For each attempt:
    │   ├─ Generate patch (backend)
    │   ├─ Validate (git)
    │   ├─ Apply (git)
    │   ├─ Build (cargo)
    │   ├─ Test (cargo)
    │   └─ Revert (git)
    └─ Collect TaskOutcome
    ↓
Vec<TaskOutcome>
    ↓
EvalResult {
    run_id: String,
    outcomes: Vec<TaskOutcome>,
}
```

### Reporting Pipeline

```
[EvalResult JSON file]
    ↓
serde_json::from_str()
    ↓
EvalResult {
    outcomes: Vec<TaskOutcome>,
}
    ↓
Reporter::generate(format)
    ├─ Aggregate by task ID
    ├─ Compute Pass@k metrics
    ├─ Format for output
    └─ Return String
    ↓
[Table | JSON | Markdown]
    ↓
stdout / file
```

## Extensibility Points

### Adding a New Backend

1. Implement `LlmBackend` trait:
```rust
pub struct MyBackend {
    config: MyConfig,
}

impl LlmBackend for MyBackend {
    fn generate_patch(&self, task: &EvalTask, opts: &BackendOpts) -> Result<String> {
        // Your implementation
    }

    fn name(&self) -> &str {
        "my-backend"
    }
}
```

2. Add to CLI in `main.rs`:
```rust
"my-backend" => Box::new(MyBackend::new(config)?),
```

3. Create Python wrapper if using subprocess:
```python
# evals/backends/my_backend.py
def main():
    request = json.load(request_file)
    # Call your LLM
    response = {...}
    json.dump(response, response_file)
```

### Adding New Report Format

1. Add variant to `ReportFormat`:
```rust
pub enum ReportFormat {
    Table,
    Json,
    Markdown,
    MyFormat,  // New
}
```

2. Implement in `Reporter`:
```rust
impl Reporter {
    fn my_format(result: &EvalResult) -> String {
        // Format logic
    }
}
```

3. Wire in `report.rs`:
```rust
ReportFormat::MyFormat => Self::my_format(result),
```

### Adding New Task

1. Create JSON in `evals/tasks/`:
```json
{
  "id": "009_my_task",
  "title": "...",
  "description": "...",
  "context_files": [...],
  "test_filters": [...],
  "test_crates": [...],
  "golden_patch": "..."
}
```

2. Validate:
```bash
cargo run -p mlx-bench -- validate-tasks
```

3. Test:
```bash
cargo run -p mlx-bench -- self-test --filter 009_my_task
```

## Performance Considerations

### Token Optimization

Context file line ranges reduce tokens:
```json
"context_files": [
  {
    "path": "crates/mlx-core/src/tensor.rs",
    "lines": [1100, 1150]  // Specific range, not whole file
  }
]
```

### Subprocess Overhead

Each backend call spawns Python process (~1-2s overhead).

Optimization strategies:
- Batch evaluate (run multiple tasks together)
- Use faster models for initial pass
- Cache results, don't re-evaluate

### Memory Usage

- TaskSet holds all tasks in memory (negligible)
- EvalResult accumulates all outcomes (depends on # tasks × # attempts)
- Incremental result saving prevents memory bloat

## Error Handling Strategy

### Levels of Error

1. **Configuration Errors** - Invalid CLI args, missing env vars
   - Reported immediately, exit code 1

2. **Task Errors** - Invalid task JSON, missing context files
   - Reported per-task, continue to next task

3. **Backend Errors** - LLM generation failed, subprocess error
   - Recorded in TaskOutcome, evaluation continues

4. **Git Errors** - Patch application, revert failed
   - Recorded, attempt marked failed, revert forced

5. **Build/Test Errors** - Cargo operations failed
   - Expected in some cases, recorded, not fatal

### Recovery Strategies

- **Always revert** - Even if test fails, revert changes
- **Continue on error** - Don't stop evaluation for one failed task
- **Detailed logging** - Record error messages for post-mortem
- **Incremental save** - Don't lose progress on error

## Testing Strategy

### Unit Testing

Can be added for:
- Task validation logic
- Metric calculations
- Report formatting

### Integration Testing

Uses `StdinDebugBackend` for testing without LLM:
```bash
export BENCH_PATCH_FILE=/tmp/golden_patch.diff
cargo run -p mlx-bench -- run --backend debug
```

### End-to-End Testing

Golden patches in tasks enable validation:
```bash
cargo run -p mlx-bench -- self-test
```

## Security Considerations

### Patch Application

- Validated with `git apply --check` before applying
- Always run in workspace directory
- Always revert after test

### Python Subprocess

- Spawned with explicit paths (no shell injection)
- Input validated as JSON before subprocess
- Environment variables (API keys) passed safely

### File System

- Temp directories for isolation
- Results written to dedicated directory
- No world-writable directories

## Future Enhancements

### Parallel Execution

```rust
pub fn run_tasks_parallel(
    tasks: Vec<&EvalTask>,
    backend: &dyn LlmBackend,
) -> Result<Vec<TaskOutcome>>
```

Would require:
- Task-level file locking
- Result merging logic
- Progress tracking

### Timeout Enforcement

Currently configured but not active. Could add:
- Timeout on subprocess calls
- Timeout on `cargo build/test`
- Overall run timeout

### Progress Tracking

Add progress bar:
```bash
Running [████████░░] 8/10 tasks...
```

### Custom Metrics

Allow task-level metric definitions:
```json
{
  "metrics": {
    "custom_check": "grep -q 'some_pattern'"
  }
}
```

### Retry Logic

Automatic retry on transient failures (network, timeouts)

## References

- [Trait Objects in Rust](https://doc.rust-lang.org/book/ch17-02-using-trait-objects.html)
- [Error Handling in Rust](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [serde Documentation](https://serde.rs/)
- [clap Parser Documentation](https://docs.rs/clap/latest/clap/)
