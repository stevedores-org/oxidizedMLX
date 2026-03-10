# mlx-bench - SWE-Bench for oxidizedMLX

A Rust-based LLM evaluation framework for testing code generation capabilities on real Rust bugs from oxidizedMLX.

## Overview

`mlx-bench` implements SWE-bench-style evaluation for oxidizedMLX, a GPU-accelerated tensor library for Apple Silicon. The framework:

- **Feeds real Rust bugs + codebase context to Claude LLM**
- **Generates patches and automatically validates them**
- **Measures: Pass@1, Pass@k, compilation rate, token usage**
- **Supports multiple backends:** Anthropic API, local MLX, debug mode
- **Tracks results incrementally** with unique run IDs

## Features

### Multiple Backends
- **Anthropic API** - Production LLM inference via Claude API
- **Local MLX** - Apple Silicon Metal GPU inference (mlx_lm)
- **Debug** - Test mode reading from environment variables

### Complete Evaluation Pipeline
1. Load task with context files
2. Assemble LLM prompt (system + issue description + code)
3. Generate patch via Python subprocess
4. Validate patch format with `git apply --check`
5. Build with `cargo build`
6. Run tests with `cargo test`
7. Revert changes and record outcome

### Flexible Task System
- 8 built-in evaluation tasks
- JSON-based task definitions
- Optional context file line ranges
- Golden patches for self-validation
- Extensible for adding new tasks

### Comprehensive Reporting
- **Table format** - Human-readable console output
- **JSON format** - Programmatic access to all metrics
- **Markdown format** - Shareable reports
- Metrics: Pass@1, Pass@k, compilation rate, token counts

## Installation

### Prerequisites
- **Rust 1.70+** (or via rustup)
- **Git** (for patch application)
- **macOS/Linux** (git apply required)

### Setup

The crate is part of the oxidizedMLX workspace. Building:

```bash
# Build the CLI
cargo build -p mlx-bench

# Or release build
cargo build -p mlx-bench --release

# Run directly
cargo run -p mlx-bench -- --help
```

### No External Dependencies Required

The framework is pure Rust with no Python dependencies. All LLM backends are implemented directly in Rust using:
- `reqwest` for HTTP calls to Anthropic API
- `tokio` for async runtime

## Quick Start

### 1. List Available Tasks
```bash
cargo run -p mlx-bench -- list-tasks
```

Output:
```
Available Tasks:

  001_softmax_axis_oob - softmax does not validate axis bounds
    Issue: #87
  002_narrow_negative_start - narrow does not validate negative start index
    Issue: #88
  ...
```

### 2. View Task Details
```bash
cargo run -p mlx-bench -- show-task 001_softmax_axis_oob --show-context
```

### 3. Validate Setup
```bash
cargo run -p mlx-bench -- validate-tasks
```

### 4. Run Benchmark

**With Anthropic API:**
```bash
export ANTHROPIC_API_KEY=sk-...
cargo run -p mlx-bench -- run --backend anthropic --model claude-3-5-sonnet-20241022
```

**With local MLX:**
```bash
cargo run -p mlx-bench -- run --backend local --model deepseek-coder-1.3b
```

**Dry run (no execution):**
```bash
cargo run -p mlx-bench -- run --backend anthropic --dry-run
```

### 5. Generate Report

**Latest results, table format:**
```bash
cargo run -p mlx-bench -- report --latest --format table
```

**Specific file, markdown format:**
```bash
cargo run -p mlx-bench -- report --file evals/results/my_run.json --format markdown
```

## CLI Reference

### `list-tasks`
List all evaluation tasks.

```bash
cargo run -p mlx-bench -- list-tasks [--json]
```

**Options:**
- `--json` - Output as JSON array

**Example:**
```bash
$ cargo run -p mlx-bench -- list-tasks
Available Tasks:

  001_softmax_axis_oob - softmax does not validate axis bounds
  002_narrow_negative_start - narrow does not validate negative start index
  ...
```

### `show-task`
Display full task details.

```bash
cargo run -p mlx-bench -- show-task <TASK_ID> [--show-context]
```

**Arguments:**
- `TASK_ID` - Task identifier (e.g., `001_softmax_axis_oob`)

**Options:**
- `--show-context` - Include full file contents

**Example:**
```bash
$ cargo run -p mlx-bench -- show-task 001_softmax_axis_oob --show-context
Task: 001_softmax_axis_oob
Title: softmax does not validate axis bounds
Issue: #87
Fix Commit: a3f9c12

Description:
The softmax() function in mlx-core does not properly validate...

Context Files:
  - crates/mlx-core/src/tensor.rs
    Lines: 1100-1130
    Annotation: softmax method implementation
```

### `validate-tasks`
Validate all task JSON files.

```bash
cargo run -p mlx-bench -- validate-tasks
```

**Output:**
```
✓ All 8 tasks valid
```

### `self-test`
Run self-tests using golden patches.

```bash
cargo run -p mlx-bench -- self-test [--filter GLOB_PATTERN]
```

**Options:**
- `--filter GLOB_PATTERN` - Only test tasks matching pattern (e.g., `*softmax*`)

**Example:**
```bash
$ cargo run -p mlx-bench -- self-test
Running self-tests on 8 tasks...

Testing 001_softmax_axis_oob - softmax does not validate axis bounds
  ✓ PASS

Testing 002_narrow_negative_start - narrow does not validate negative start index
  ✓ PASS

...
```

### `run`
Execute benchmark evaluation.

```bash
cargo run -p mlx-bench -- run \
  --backend [local|anthropic|debug] \
  --model MODEL_ID \
  --attempts N \
  --filter GLOB_PATTERN \
  --dry-run
```

**Options:**
- `--backend` - `local` (mlx_lm), `anthropic` (Claude API), or `debug` (environment)
- `--model` - Model ID for LLM (e.g., `claude-3-5-sonnet-20241022`)
- `--attempts` - Number of attempts per task (default: 1)
- `--filter` - Task ID glob filter (default: all tasks)
- `--dry-run` - Print config without executing
- `--workspace` - Workspace root directory (default: `.`)

**Examples:**

All tasks, single attempt:
```bash
export ANTHROPIC_API_KEY=sk-...
cargo run -p mlx-bench -- run --backend anthropic
```

Specific tasks, multiple attempts:
```bash
cargo run -p mlx-bench -- run \
  --backend anthropic \
  --filter '*softmax*' \
  --attempts 3
```

Dry run:
```bash
cargo run -p mlx-bench -- run --backend anthropic --dry-run
```

### `report`
Generate evaluation report.

```bash
cargo run -p mlx-bench -- report \
  --format [table|json|markdown] \
  --latest \
  --file PATH
```

**Options:**
- `--format` - Output format: `table`, `json`, or `markdown` (default: table)
- `--latest` - Use most recent results file
- `--file PATH` - Specify results file path

**Examples:**

Latest results, table format:
```bash
cargo run -p mlx-bench -- report --latest --format table
```

Specific file, markdown:
```bash
cargo run -p mlx-bench -- report --file evals/results/run-123.json --format markdown
```

JSON for programmatic access:
```bash
cargo run -p mlx-bench -- report --latest --format json
```

## Task Structure

Each task is a JSON file defining an evaluation scenario:

```json
{
  "id": "001_softmax_axis_oob",
  "title": "softmax does not validate axis bounds",
  "issue": 87,
  "fix_commit": "a3f9c12",
  "description": "Full issue description...",
  "context_files": [
    {
      "path": "crates/mlx-core/src/tensor.rs",
      "lines": [1100, 1130],
      "annotation": "softmax method implementation"
    }
  ],
  "test_filters": ["test_softmax_axis_oob"],
  "test_crates": ["mlx-core"],
  "timeout_secs": 120,
  "golden_patch": "--- a/crates/mlx-core/src/tensor.rs\n+++ b/..."
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique task identifier (e.g., `001_softmax_axis_oob`) |
| `title` | string | Yes | Short issue title |
| `issue` | number | No | GitHub issue number for reference |
| `fix_commit` | string | No | Commit hash that fixed the issue |
| `description` | string | Yes | Full issue description fed to LLM |
| `context_files` | array | Yes | Array of code context files |
| `test_filters` | array | Yes | Cargo test filter strings |
| `test_crates` | array | Yes | Crate names to build/test |
| `timeout_secs` | number | Yes | Timeout for operations |
| `golden_patch` | string | No | Valid patch for self-testing |

### Context Files

Each context file includes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | Workspace-relative file path |
| `lines` | array | No | `[start, end]` line range (1-indexed, inclusive) |
| `annotation` | string | No | Description of what's in this file |

## Adding New Tasks

### Step 1: Create Task JSON

Create a new file in `evals/tasks/` (e.g., `009_new_issue.json`):

```json
{
  "id": "009_new_issue",
  "title": "Brief issue title",
  "issue": 95,
  "description": "Full issue description that LLM will read...",
  "context_files": [
    {
      "path": "crates/mlx-core/src/tensor.rs",
      "lines": [100, 150],
      "annotation": "Relevant code section"
    }
  ],
  "test_filters": ["test_new_issue"],
  "test_crates": ["mlx-core"],
  "timeout_secs": 120,
  "golden_patch": "--- a/crates/mlx-core/src/tensor.rs\n+++ b/crates/mlx-core/src/tensor.rs\n..."
}
```

### Step 2: Create Golden Patch

The golden patch should:
- Be valid unified diff format
- Apply cleanly with `git apply`
- Fix the issue described
- Pass the test filters

Use `git diff` to generate:
```bash
# Make fix
# ...changes...
# Then:
git diff > /tmp/fix.diff

# Validate
git apply --check < /tmp/fix.diff
```

### Step 3: Validate

```bash
cargo run -p mlx-bench -- validate-tasks
```

### Step 4: Test

```bash
cargo run -p mlx-bench -- self-test --filter 009_new_issue
```

## Backends

All backends are implemented in pure Rust using async/await with `tokio`.

### Anthropic API Backend (Production)

Pure Rust implementation using `reqwest` for HTTP calls.

**Setup:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Usage:**
```bash
cargo run -p mlx-bench -- run \
  --backend anthropic \
  --model claude-3-5-sonnet-20241022
```

**Models:**
- `claude-3-5-sonnet-20241022` (recommended, fast)
- `claude-3-opus-20250219` (high quality)
- Other Claude models

**Cost:** ~$0.003 per task (varies by model and tokens)

**Implementation:**
- Async HTTP via `reqwest`
- Direct API calls from Rust
- No subprocess overhead
- Proper error propagation

### Local MLX Backend (Awaiting mlx-rs)

Currently returns a helpful error message.

**Note:** This backend requires Rust bindings for MLX (`mlx-rs`). Once stable, it will enable:
- No API costs
- Fast on Apple Silicon
- Runs locally

For now, use Anthropic API backend instead.

### Debug Backend (Testing)

Pure Rust implementation for testing without LLM calls.

**Setup:**
```bash
# Option 1: Read from file
export BENCH_PATCH_FILE=/path/to/patch.diff

# Option 2: Read from env var
export BENCH_PATCH_CONTENT="--- a/file\n+++ b/file\n..."
```

**Usage:**
```bash
cargo run -p mlx-bench -- run --backend debug
```

**Use cases:**
- Testing framework without LLM
- Validating infrastructure
- Debugging evaluation loop
- CI/CD integration testing

## Results Format

Results are saved to `evals/results/{run_id}_partial.json`:

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-03-09T10:00:00Z",
  "outcomes": [
    {
      "task_id": "001_softmax_axis_oob",
      "attempt": 1,
      "timestamp": "2026-03-09T10:00:10Z",
      "patch_generated": true,
      "patch_text": "--- a/crates/mlx-core/src/tensor.rs\n+++ b/...",
      "patch_valid": true,
      "build_success": true,
      "tests_passed": true,
      "error": null,
      "llm_input_tokens": 1200,
      "llm_output_tokens": 387
    }
  ]
}
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `BENCH_PATCH_FILE` | Path to patch for debug backend | `/tmp/patch.diff` |
| `RUST_LOG` | Logging level | `debug`, `info` |

## Common Tasks

### Run Benchmark with Different Models
```bash
# GPT-4 speed, Claude quality
cargo run -p mlx-bench -- run --backend anthropic --model claude-3-5-sonnet-20241022

# Cheaper option
cargo run -p mlx-bench -- run --backend anthropic --model claude-3-haiku-20250307

# Local inference (free, faster)
cargo run -p mlx-bench -- run --backend local --model deepseek-coder-1.3b
```

### Test Specific Tasks
```bash
# Only softmax tests
cargo run -p mlx-bench -- run --filter '*softmax*'

# Multiple attempts to improve success rate
cargo run -p mlx-bench -- run --filter '*softmax*' --attempts 3
```

### Analyze Results
```bash
# Pretty-printed JSON
cargo run -p mlx-bench -- report --latest --format json | jq '.pass_at_1'

# Markdown for documentation
cargo run -p mlx-bench -- report --latest --format markdown > results.md

# Watch results directory
ls -lrt evals/results/ | tail -1
```

### Development Workflow
```bash
# Validate setup
cargo run -p mlx-bench -- validate-tasks

# Test infrastructure
cargo run -p mlx-bench -- self-test

# Dry run to check configuration
cargo run -p mlx-bench -- run --backend anthropic --dry-run

# Run actual evaluation
cargo run -p mlx-bench -- run --backend anthropic

# Check results
cargo run -p mlx-bench -- report --latest
```

## Metrics

### Pass@1
Percentage of tasks solved on first LLM attempt.

```
Pass@1 = (successful_tasks / total_tasks) * 100
```

### Pass@k
Percentage of tasks solvable within k attempts.

```
Pass@k = (tasks_with_success_in_k_attempts / total_tasks) * 100
```

### Compilation Rate
Percentage of LLM-generated patches that compile.

```
Compilation = (patches_that_compile / total_patches) * 100
```

## Troubleshooting

### "No task files found"
**Problem:** Tasks not loading from `evals/tasks/`

**Solution:**
```bash
# Verify directory exists
ls -la evals/tasks/

# Check file format
cat evals/tasks/001_softmax_axis_oob.json | jq .

# Validate all tasks
cargo run -p mlx-bench -- validate-tasks
```

### "ANTHROPIC_API_KEY not set"
**Problem:** API key missing

**Solution:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
echo $ANTHROPIC_API_KEY  # Verify
```

### "mlx_lm not installed"
**Problem:** Python dependency missing for local backend

**Solution:**
```bash
pip install mlx-lm
python -c "import mlx_lm; print(mlx_lm.__version__)"
```

### "Patch validation failed"
**Problem:** LLM output is not valid unified diff

**Solution:**
1. Check system prompt in `evals/prompts/system_prompt.txt`
2. Review raw output in results JSON
3. Try different model or temperature
4. Adjust prompt if needed

### "Build failed" / "Tests failed"
**Problem:** Generated patch doesn't fix the issue

**Solution:**
1. Review the patch in results JSON
2. Check actual error messages
3. Try with more context files
4. Increase model quality (claude-3-opus vs claude-3-haiku)
5. Run multiple attempts (--attempts 3)

## Performance Tuning

### Token Count
Reduce context file ranges to lower token usage:

```json
"context_files": [
  {
    "path": "crates/mlx-core/src/tensor.rs",
    "lines": [1110, 1125],  // Smaller range
    "annotation": "softmax method"
  }
]
```

### Model Selection
- **Faster, cheaper:** `claude-3-haiku-20250307`
- **Balanced:** `claude-3-5-sonnet-20241022`
- **Higher quality:** `claude-3-opus-20250219`

### Parallel Execution
Tasks currently run sequentially. For parallel execution:
```bash
# Manual: Run multiple instances with different filters
cargo run -p mlx-bench -- run --filter '001_*' &
cargo run -p mlx-bench -- run --filter '002_*' &
wait
```

## Contributing

To contribute new tasks or improvements:

1. **New tasks:** Follow the task structure in `evals/tasks/`
2. **Backend improvements:** Implement `LlmBackend` trait
3. **CLI enhancements:** Add subcommands to `main.rs`
4. **Documentation:** Update relevant .md files

See `CONTRIBUTING.md` for detailed guidelines.

## Architecture

### Core Modules

**task.rs:**
- `EvalTask` - Task definition
- `ContextFile` - Code context
- `TaskSet` - Task collection and loading

**runner.rs:**
- `TaskRunner` - Evaluation loop orchestration
- `TaskOutcome` - Individual task result
- `EvalResult` - Complete evaluation run

**backend.rs:**
- `LlmBackend` trait - Backend interface
- `LocalMlxBackend` - mlx_lm implementation
- `AnthropicApiBackend` - Claude API implementation
- `StdinDebugBackend` - Debug/testing backend

**report.rs:**
- `Reporter` - Result formatting
- `ReportFormat` - Table/JSON/Markdown

**error.rs:**
- `BenchError` - Custom error types
- Error handling with `thiserror`

### Data Flow

```
Task JSON
    ↓
TaskSet::load_from_dir()
    ↓
TaskRunner::run_task()
    ├─ Assemble prompt
    ├─ LlmBackend::generate_patch() [subprocess]
    ├─ Validate patch
    ├─ Apply + build + test
    └─ Record TaskOutcome
    ↓
EvalResult collection
    ↓
Reporter::generate()
    ├─ table_format()
    ├─ json_format()
    └─ markdown_format()
    ↓
Output to stdout/file
```

### Backend Protocol

```
Rust (main)
    ↓
TempDir created
    ↓
request.json written
    ↓
Python subprocess spawned
    ├─ Reads request.json
    ├─ Calls LLM API
    └─ Writes response.json
    ↓
response.json read
    ↓
Patch extracted
    ↓
TempDir cleaned up
```

## Related Documentation

- `../../evals/README.md` - Full user guide
- `SWEBENCH_IMPLEMENTATION.md` - Implementation details
- `ARCHITECTURE.md` - Deep dive into design
- `API.md` - Rust API documentation
- `CONTRIBUTING.md` - Development guide

## License

MIT OR Apache-2.0 (same as oxidizedMLX)

## References

- [SWE-bench Paper](https://arxiv.org/abs/2310.06770)
- [Original SWE-bench](https://github.com/princeton-nlp/SWE-bench)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Unified Diff Format](https://en.wikipedia.org/wiki/Unified_diff)
