# SWE-Bench for oxidizedMLX

SWE-bench-style LLM evaluation framework for oxidizedMLX. Tests LLM coding accuracy by feeding real Rust issues + codebase context to Claude, generating patches, and evaluating with `cargo test`.

## Quick Start

```bash
# List available evaluation tasks
cargo run -p mlx-bench -- list-tasks

# Show details of a specific task
cargo run -p mlx-bench -- show-task 001_softmax_axis_oob --show-context

# Validate all task JSON files
cargo run -p mlx-bench -- validate-tasks

# Run self-test on golden patches
cargo run -p mlx-bench -- self-test

# Run benchmark with Anthropic API
ANTHROPIC_API_KEY=sk-... cargo run -p mlx-bench -- run --backend anthropic

# Generate report from latest results
cargo run -p mlx-bench -- report --latest --format table
```

## CLI Commands

### `list-tasks`
List all available evaluation tasks.

```bash
cargo run -p mlx-bench -- list-tasks [--json]
```

**Options:**
- `--json` - Output as JSON array

### `show-task`
Show full details of a specific task.

```bash
cargo run -p mlx-bench -- show-task <TASK_ID> [--show-context]
```

**Options:**
- `--show-context` - Include full file contents for context files

### `validate-tasks`
Validate all task JSON files for required fields and structure.

```bash
cargo run -p mlx-bench -- validate-tasks
```

### `self-test`
Run self-tests using golden patches to verify the framework and infrastructure.

```bash
cargo run -p mlx-bench -- self-test [--filter GLOB_PATTERN]
```

**Options:**
- `--filter GLOB_PATTERN` - Only test tasks matching the glob pattern (e.g., `*softmax*`)

### `run`
Execute benchmark against evaluation tasks with an LLM backend.

```bash
cargo run -p mlx-bench -- run \
  --backend [local|anthropic|debug] \
  --model MODEL_ID \
  --attempts N \
  --filter GLOB_PATTERN \
  --dry-run
```

**Options:**
- `--backend` - Backend to use: `local` (mlx_lm), `anthropic` (Claude API), or `debug` (from BENCH_PATCH_FILE)
- `--model` - Model ID (e.g., `claude-3-5-sonnet-20241022`)
- `--attempts` - Number of attempts per task (default: 1)
- `--filter` - Task ID glob filter (default: all tasks)
- `--dry-run` - Print configuration without executing

### `report`
Generate evaluation report from results.

```bash
cargo run -p mlx-bench -- report \
  --format [table|json|markdown] \
  --latest \
  --file PATH
```

**Options:**
- `--format` - Output format: `table`, `json`, or `markdown` (default: table)
- `--latest` - Use most recent results file
- `--file PATH` - Specify results file explicitly

## Task Structure

Each evaluation task is a JSON file in `evals/tasks/` with the following structure:

```json
{
  "id": "001_softmax_axis_oob",
  "title": "softmax does not validate axis bounds",
  "issue": 87,
  "fix_commit": "a3f9c12",
  "description": "Full issue text describing what needs to be fixed...",
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

### Fields

- **id** - Unique task identifier (e.g., `001_softmax_axis_oob`)
- **title** - Short title of the issue
- **issue** - GitHub issue number (optional)
- **fix_commit** - Commit hash that fixed the issue (optional, for reference)
- **description** - Full issue description fed to LLM
- **context_files** - Array of code files provided as context
  - **path** - Workspace-relative path to file
  - **lines** - Optional `[start, end]` line range (1-indexed, inclusive)
  - **annotation** - Optional description of what's in this file
- **test_filters** - Cargo test filter strings (e.g., `test_softmax_axis_oob`)
- **test_crates** - Crate names to test (e.g., `mlx-core`)
- **timeout_secs** - Timeout for build/test operations
- **golden_patch** - Valid unified diff patch for self-testing

## Backends

### Anthropic API Backend
Uses Claude API via the `anthropic` Python package.

```bash
export ANTHROPIC_API_KEY=sk-...
cargo run -p mlx-bench -- run --backend anthropic --model claude-3-5-sonnet-20241022
```

**Requirements:**
- `ANTHROPIC_API_KEY` environment variable
- `anthropic` Python package: `pip install anthropic`

### Local MLX Backend
Uses `mlx_lm` library for Apple Silicon Metal GPU inference.

```bash
cargo run -p mlx-bench -- run --backend local --model <MODEL_ID>
```

**Requirements:**
- `mlx_lm` library: `pip install mlx-lm`
- Apple Silicon Mac with Metal support

### Debug Backend
Reads patch from environment variable for testing.

```bash
export BENCH_PATCH_FILE=/path/to/patch.diff
cargo run -p mlx-bench -- run --backend debug
```

## Evaluation Loop

For each task and attempt:

1. **Prompt assembly** - Read context files with line slicing
2. **LLM call** - Send issue + context to LLM via Python backend
3. **Patch validation** - Run `git apply --check` on generated patch
4. **Patch application** - Apply patch with `git apply`
5. **Build check** - Run `cargo build -p <crate> --quiet`
6. **Test execution** - Run `cargo test -p <crate> <filter1> <filter2>...`
7. **Revert** - Always revert changes with `git checkout -- .`
8. **Record** - Save outcome (success/failure, error messages)

Results are saved to `evals/results/{run_id}_partial.json`.

## Metrics

The framework tracks the following metrics:

- **Pass@1** - Percentage of tasks solved on first attempt
- **Pass@k** - Percentage of tasks solved within k attempts
- **Compilation Rate** - Percentage of patches that compile successfully

## Just Commands

Convenient shortcuts using `just`:

```bash
just bench-build              # Build mlx-bench CLI
just bench-list               # List tasks
just bench-self-test          # Run self-tests
just bench-local <MODEL>      # Run with local MLX backend
just bench-anthropic          # Run with Anthropic API
just bench-report             # Generate report
just bench-dry <TASK_ID>      # Dry-run a specific task
```

## Make Targets

Same shortcuts available in Makefile:

```bash
make bench-build
make bench-list
make bench-self-test
make bench-local MODEL=<model_id>
make bench-anthropic
make bench-report
make bench-dry TASK=<task_id>
```

## GitHub Actions Workflow

Manual workflow trigger in CI to avoid unnecessary API costs:

```bash
gh workflow run swe-bench.yml -f backend=anthropic -f model=claude-3-5-sonnet-20241022 -f attempts=1
```

The workflow:
- Only runs when manually triggered (`workflow_dispatch`)
- Gated on `ANTHROPIC_API_KEY` secret availability
- Validates tasks, runs self-test, executes benchmark
- Uploads results as artifacts (90-day retention)

## Adding New Tasks

1. Create a new JSON file in `evals/tasks/` with the task structure
2. Use `cargo run -p mlx-bench -- validate-tasks` to validate
3. Provide a `golden_patch` for self-testing
4. Run `just bench-self-test` to verify the patch works

## Examples

### View a task with context
```bash
cargo run -p mlx-bench -- show-task 001_softmax_axis_oob --show-context
```

### Run benchmark on specific tasks
```bash
cargo run -p mlx-bench -- run \
  --backend anthropic \
  --filter '*softmax*' \
  --attempts 3
```

### Generate markdown report
```bash
cargo run -p mlx-bench -- report --latest --format markdown > results.md
```

### Dry-run to see what would happen
```bash
cargo run -p mlx-bench -- run --backend anthropic --dry-run
```

## Environment Variables

- `ANTHROPIC_API_KEY` - API key for Anthropic backend
- `BENCH_PATCH_FILE` - Path to patch file for debug backend
- `RUST_LOG` - Set to `debug` for verbose logging

## Architecture

```
crates/mlx-bench/          # Main Rust crate
├── src/
│   ├── main.rs           # CLI entry point
│   ├── task.rs           # EvalTask, ContextFile, TaskSet
│   ├── runner.rs         # TaskRunner, EvalResult, metrics
│   ├── backend.rs        # LlmBackend trait, implementations
│   ├── report.rs         # Report formatting (table/json/markdown)
│   └── error.rs          # Error types

evals/
├── tasks/                # 8 evaluation tasks (*.json)
├── backends/             # Python backend implementations
│   ├── mlx_local.py      # mlx_lm backend
│   ├── anthropic_api.py  # Anthropic API backend
│   └── stdin_debug.py    # Debug backend
├── prompts/              # System prompt and format guides
└── results/              # Evaluation results (*.json)
```

## Protocol

### Backend Request (request.json)
```json
{
  "model_id": "claude-3-5-sonnet-20241022",
  "max_tokens": 2048,
  "temperature": 0.2,
  "system": "You are an expert Rust developer...",
  "user": "Issue #87: softmax does not validate axis bounds...\n\nContext:\n..."
}
```

### Backend Response (response.json)
```json
{
  "patch": "--- a/crates/mlx-core/src/tensor.rs\n+++ b/...",
  "raw_output": "...",
  "input_tokens": 1200,
  "output_tokens": 387
}
```

## Troubleshooting

### "No task files found"
Ensure `evals/tasks/` directory exists and contains `*.json` files.

### "ANTHROPIC_API_KEY not set"
Export the API key before running: `export ANTHROPIC_API_KEY=sk-...`

### "Patch validation failed"
The LLM output is not valid unified diff format. Check the system prompt in `evals/prompts/system_prompt.txt`.

### "Build failed" / "Tests failed"
The generated patch may not correctly fix the issue. Check the actual error in the task outcome JSON.

## References

- [SWE-bench Paper](https://arxiv.org/abs/2310.06770)
- [Unified Diff Format](https://en.wikipedia.org/wiki/Unified_diff)
- [Git Apply Documentation](https://git-scm.com/docs/git-apply)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
