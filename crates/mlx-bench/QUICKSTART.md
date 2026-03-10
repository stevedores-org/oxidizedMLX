# mlx-bench Quick Reference

## Installation

```bash
cd /path/to/oxidizedMLX
cargo build -p mlx-bench
export ANTHROPIC_API_KEY=sk-ant-...
```

**No Python dependencies required!** The framework is pure Rust.

## Basic Commands

### List Tasks
```bash
cargo run -p mlx-bench -- list-tasks
```

### View Task Details
```bash
cargo run -p mlx-bench -- show-task 001_softmax_axis_oob
```

### Validate Setup
```bash
cargo run -p mlx-bench -- validate-tasks
cargo run -p mlx-bench -- self-test
```

### Run Benchmark
```bash
cargo run -p mlx-bench -- run --backend anthropic
```

### Generate Report
```bash
cargo run -p mlx-bench -- report --latest --format table
```

## Common Workflows

### Test Infrastructure
```bash
# Dry run (no LLM calls)
cargo run -p mlx-bench -- run --backend anthropic --dry-run
```

### Run Specific Tasks
```bash
# Only softmax tasks
cargo run -p mlx-bench -- run --backend anthropic --filter '*softmax*'

# First 3 tasks
cargo run -p mlx-bench -- run --backend anthropic --filter '00[123]*'
```

### Multiple Attempts
```bash
# Try each task 3 times
cargo run -p mlx-bench -- run --backend anthropic --attempts 3
```

### Different Models
```bash
# Fast & cheap
cargo run -p mlx-bench -- run --backend anthropic --model claude-3-haiku-20250307

# High quality
cargo run -p mlx-bench -- run --backend anthropic --model claude-3-opus-20250219

# Local GPU (free)
cargo run -p mlx-bench -- run --backend local --model deepseek-coder-1.3b
```

### View Results
```bash
# Table (human readable)
cargo run -p mlx-bench -- report --latest --format table

# JSON (programmatic)
cargo run -p mlx-bench -- report --latest --format json

# Markdown (shareable)
cargo run -p mlx-bench -- report --latest --format markdown > results.md
```

## Using Just/Make

```bash
just bench-list
just bench-self-test
just bench-anthropic
just bench-report
```

or

```bash
make bench-list
make bench-self-test
make bench-anthropic
make bench-report
```

## Debugging

### Check for Errors
```bash
# View latest results
tail -100 evals/results/*.json | jq '.outcomes[] | {task_id, error}'

# View specific task outcome
jq '.outcomes[] | select(.task_id=="001_softmax_axis_oob")' evals/results/*.json
```

### Test a Single Task
```bash
cargo run -p mlx-bench -- run --backend anthropic --filter '001_softmax*'
```

### Use Debug Backend
```bash
echo "--- a/file\n+++ b/file\n@@ -1 +1 @@\n" > /tmp/test.patch
export BENCH_PATCH_FILE=/tmp/test.patch
cargo run -p mlx-bench -- run --backend debug
```

### Check Raw Output
```bash
jq '.outcomes[0].patch_text' evals/results/run-id.json
```

## Environment Variables

```bash
# API Key (required for anthropic)
export ANTHROPIC_API_KEY=sk-ant-...

# Debug patch file
export BENCH_PATCH_FILE=/path/to/patch.diff

# Enable verbose logging
export RUST_LOG=debug
```

## File Structure

```
crates/mlx-bench/
├── README.md              # User guide
├── ARCHITECTURE.md        # Design details
├── API.md                # Rust API reference
├── CONTRIBUTING.md        # Contributing guide
├── TROUBLESHOOTING.md     # Common issues
└── src/
    ├── main.rs           # CLI
    ├── task.rs           # Tasks
    ├── runner.rs         # Evaluation loop
    ├── backend.rs        # LLM backends
    ├── report.rs         # Report generation
    └── error.rs          # Error types

evals/
├── README.md              # Full evaluation guide
├── tasks/                 # 8 evaluation tasks (JSON)
├── backends/              # Python backend implementations
│   ├── mlx_local.py
│   ├── anthropic_api.py
│   └── stdin_debug.py
├── prompts/               # System prompt + format guide
└── results/               # Evaluation results (JSON)
```

## Metrics

After running benchmark:

```bash
# Pass@1: success rate on first attempt
# Pass@3: success rate within 3 attempts
# Compilation: % of patches that compile

jq '{pass_at_1, pass_at_3: (.pass_at_k//0), compilation_rate}' evals/results/run-id.json
```

## Adding Tasks

1. Create `evals/tasks/NNN_short_name.json`:
```json
{
  "id": "009_my_task",
  "title": "Issue title",
  "description": "Full description...",
  "context_files": [{"path": "...", "lines": [1, 50]}],
  "test_filters": ["test_name"],
  "test_crates": ["crate-name"],
  "timeout_secs": 120,
  "golden_patch": "--- a/..."
}
```

2. Validate:
```bash
cargo run -p mlx-bench -- validate-tasks
cargo run -p mlx-bench -- self-test --filter '009_*'
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ANTHROPIC_API_KEY not set` | `export ANTHROPIC_API_KEY=sk-ant-...` |
| `tasks not found` | Run from oxidizedMLX root: `cd /path/to/oxidizedMLX` |
| `python module not found` | `pip install anthropic mlx-lm` |
| `patch validation failed` | Check LLM output is valid unified diff |
| `build failed` | Patch doesn't compile - improve context or model |
| `tests failed` | Generated patch exists but doesn't fix issue |

See TROUBLESHOOTING.md for full guide.

## API Usage (Rust)

```rust
use mlx_bench::{
    task::TaskSet,
    runner::{TaskRunner, RunConfig},
    backend::AnthropicApiBackend,
};

fn main() -> mlx_bench::Result<()> {
    let tasks = TaskSet::load_from_dir(".")?;
    let runner = TaskRunner::new(RunConfig::new("."));
    let backend = AnthropicApiBackend::new("evals/backends/anthropic_api.py")?;

    for task in tasks.all() {
        runner.run_task(task, &backend, &opts)?;
    }

    Ok(())
}
```

See API.md for full reference.

## Useful Links

- [Anthropic API](https://docs.anthropic.com/)
- [MLX Docs](https://ml-explore.github.io/mlx/)
- [Unified Diff Format](https://en.wikipedia.org/wiki/Unified_diff)
- [SWE-bench Paper](https://arxiv.org/abs/2310.06770)

## Tips & Tricks

### Monitor Progress
```bash
# Watch results directory
watch 'ls -lrt evals/results/ | tail -5'

# Or count completed
jq '.outcomes | length' evals/results/latest.json
```

### Batch Processing
```bash
# Run all softmax tasks
for filter in "*softmax*" "*narrow*" "*matmul*"; do
  cargo run -p mlx-bench -- run --backend anthropic --filter "$filter"
done
```

### Cost Estimation
```bash
# Approximate tokens per task: 1000-1500
# Claude 3.5 Sonnet: ~$0.003 per task
# 8 tasks: ~$0.024
# 8 tasks × 3 attempts: ~$0.072

# To save money:
# 1. Use Haiku instead (10x cheaper)
# 2. Reduce context file sizes
# 3. Limit attempts
# 4. Use local MLX (free)
```

### Parallel Execution (Manual)
```bash
# Run multiple instances
cargo run -p mlx-bench -- run --backend anthropic --filter '001_*' &
cargo run -p mlx-bench -- run --backend anthropic --filter '002_*' &
cargo run -p mlx-bench -- run --backend anthropic --filter '003_*' &
wait

# Merge results
jq -s '.[]' evals/results/*.json > merged_results.json
```

## Getting Help

1. Read relevant documentation:
   - Quick issues → QUICKSTART.md (this file)
   - Setup problems → README.md
   - Design questions → ARCHITECTURE.md
   - Common issues → TROUBLESHOOTING.md
   - Contributing → CONTRIBUTING.md

2. Check error messages carefully

3. Enable debugging:
   ```bash
   RUST_LOG=debug cargo run -p mlx-bench -- ...
   ```

4. Inspect JSON results:
   ```bash
   jq . evals/results/latest.json
   ```

5. Report issues with:
   - Full error message
   - Steps to reproduce
   - OS/version info
   - Result files

---

**Tip:** Save this as a bookmark! It covers 90% of daily usage.
