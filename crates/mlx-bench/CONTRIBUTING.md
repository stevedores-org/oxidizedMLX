# Contributing to mlx-bench

Thank you for your interest in contributing! This guide explains how to add new tasks, backends, and features.

## Code of Conduct

Be respectful and constructive. Treat all contributors with courtesy.

## Getting Started

### Prerequisites

- Rust 1.70+
- Python 3.8+
- Git
- macOS/Linux

### Setup

```bash
# Clone the repo
git clone https://github.com/stevedores-org/oxidizedMLX.git
cd oxidizedMLX

# Build mlx-bench
cargo build -p mlx-bench

# Run tests (if any)
cargo test -p mlx-bench

# Install Python backends
pip install anthropic mlx-lm
```

### Verify Setup

```bash
# Should list all tasks
cargo run -p mlx-bench -- list-tasks

# Should pass
cargo run -p mlx-bench -- validate-tasks

# Should work
cargo run -p mlx-bench -- self-test
```

## Contributing Tasks

### Adding a New Evaluation Task

Tasks are the primary contribution point.

#### Step 1: Identify an Issue

Choose a real bug from oxidizedMLX:
- Find closed issues/PRs with fixes
- Understand what was broken
- Identify relevant code sections
- Find existing tests that validate the fix

Example from Git history:
```bash
git log --oneline | head -20
# Find relevant commit
git show <commit> -- "*.rs"
```

#### Step 2: Create Task JSON

Create `evals/tasks/NNN_short_description.json`:

```json
{
  "id": "009_my_issue",
  "title": "Brief problem statement",
  "issue": 95,
  "fix_commit": "abc1234",
  "description": "Full issue description that will be sent to LLM. Should include:\n- What's broken\n- Why it's broken\n- Expected behavior\n- How to reproduce",
  "context_files": [
    {
      "path": "crates/mlx-core/src/tensor.rs",
      "lines": [100, 150],
      "annotation": "The buggy function"
    },
    {
      "path": "crates/mlx-core/src/tests.rs",
      "lines": [500, 550],
      "annotation": "Test that validates fix"
    }
  ],
  "test_filters": ["test_my_issue_fixed"],
  "test_crates": ["mlx-core"],
  "timeout_secs": 120,
  "golden_patch": null
}
```

**Guidelines:**

- **ID Format:** `NNN_short_name` where NNN is 3+ digit number
- **Title:** <50 characters, action-oriented
- **Description:** 200-500 words, clear and specific
- **Context Files:** 2-4 files, show the problem area
- **Line Ranges:** Specific lines (not whole file) to reduce tokens
- **Test Filters:** Should pass after patch applied
- **Test Crates:** Minimal set of crates to build/test

#### Step 3: Create Golden Patch

The golden patch shows LLM the expected fix format.

```bash
# Make the fix
# ... edit files ...

# Generate patch
git diff > /tmp/my_fix.patch

# Validate patch
git apply --check < /tmp/my_fix.patch

# Copy to golden_patch field in JSON
cat /tmp/my_fix.patch
```

**Requirements:**
- Valid unified diff format
- Applies cleanly with `git apply`
- Minimal (no extra changes)
- Tests pass after application

**Example:**
```patch
--- a/crates/mlx-core/src/tensor.rs
+++ b/crates/mlx-core/src/tensor.rs
@@ -100,6 +100,9 @@ fn softmax(t: &Tensor) -> Tensor {
     fn softmax(&self, axis: i64) -> Result<Tensor> {
+        if axis >= self.shape().len() as i64 {
+            return Err("axis out of bounds".into());
+        }
         // existing code
         Ok(result)
     }
```

#### Step 4: Validate Task

```bash
# Check JSON validity
cargo run -p mlx-bench -- validate-tasks

# Should pass
cargo run -p mlx-bench -- show-task 009_my_issue
```

#### Step 5: Test Task

```bash
# Fill in golden_patch in JSON, then:
cargo run -p mlx-bench -- self-test --filter 009_my_issue
```

Output should show:
```
Testing 009_my_issue - ...
  ✓ PASS
```

#### Step 6: Submit PR

Create pull request with:
- New task JSON file
- Clear commit message explaining the issue
- Reference to GitHub issue/PR if applicable

## Contributing Backends

### Adding a New LLM Backend

New backends extend support to other LLM providers.

#### Step 1: Implement LlmBackend Trait

In `src/backend.rs`:

```rust
pub struct MyBackend {
    config: MyConfig,
}

impl MyBackend {
    pub fn new(config_path: impl AsRef<Path>) -> Result<Self> {
        // Initialize backend
        Ok(MyBackend {
            config: MyConfig { /* ... */ }
        })
    }
}

impl LlmBackend for MyBackend {
    fn generate_patch(&self, task: &EvalTask, opts: &BackendOpts) -> Result<String> {
        // Implement:
        // 1. Assemble prompt from task + opts
        // 2. Call LLM
        // 3. Extract and validate patch
        // 4. Return patch string

        let prompt = format!("System: {}\n\nUser: {}",
            DEFAULT_SYSTEM_PROMPT,
            format_task_prompt(task)
        );

        // Call your LLM API
        let response = call_llm(&prompt)?;

        // Extract patch
        Ok(response.patch)
    }

    fn name(&self) -> &str {
        "my-backend"
    }
}
```

#### Step 2: Create Python Subprocess

In `evals/backends/my_backend.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    try:
        # Load request
        with open(args.request) as f:
            request = json.load(f)

        # Call your LLM
        # response = call_llm(request)

        # Prepare response
        response = {
            "patch": "--- a/...\n+++ b/...",
            "raw_output": "...",
            "input_tokens": 1000,
            "output_tokens": 500,
        }

        # Write response
        with open(args.response, "w") as f:
            json.dump(response, f)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### Step 3: Wire into CLI

In `src/main.rs`:

```rust
"my-backend" => Box::new(MyBackend::new("evals/backends/my_backend.py")?),
```

#### Step 4: Test

```bash
# List to verify it loads
cargo build -p mlx-bench

# Dry run
cargo run -p mlx-bench -- run --backend my-backend --dry-run

# Actual run
export MY_BACKEND_API_KEY=...
cargo run -p mlx-bench -- run --backend my-backend
```

## Contributing Features

### Adding New CLI Commands

#### Step 1: Add Subcommand

In `src/main.rs`:

```rust
#[derive(Subcommand)]
enum Cmd {
    // ... existing commands ...

    MyCommand {
        #[arg(long)]
        my_option: String,
    }
}
```

#### Step 2: Add Handler

```rust
fn cmd_my_command(option: &str) -> Result<()> {
    // Implementation
    println!("Option: {}", option);
    Ok(())
}
```

#### Step 3: Wire Command

In `run()` function:

```rust
match args.cmd {
    // ... existing matches ...
    Cmd::MyCommand { my_option } => cmd_my_command(&my_option),
}
```

#### Step 4: Test

```bash
cargo build -p mlx-bench
cargo run -p mlx-bench -- my-command --my-option value
```

### Adding New Report Format

In `src/report.rs`:

```rust
pub enum ReportFormat {
    Table,
    Json,
    Markdown,
    MyFormat,  // New
}

impl Reporter {
    fn my_format(result: &EvalResult) -> String {
        let mut output = String::new();
        // Format logic
        output
    }
}
```

## Code Style

### Rust Style

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` before committing
- Run `cargo clippy` and fix warnings

```bash
cargo fmt -p mlx-bench
cargo clippy -p mlx-bench -- -D warnings
```

### Documentation

- Add doc comments to public items
- Include examples where helpful
- Update README/API.md for new features

```rust
/// Runs evaluation for a single task.
///
/// # Arguments
///
/// * `task` - Task to evaluate
/// * `backend` - LLM backend
///
/// # Returns
///
/// Vec of outcomes (one per attempt)
///
/// # Example
///
/// ```
/// let runner = TaskRunner::new(config);
/// let outcomes = runner.run_task(&task, &backend, &opts)?;
/// ```
pub fn run_task(...) -> Result<Vec<TaskOutcome>> {
    // ...
}
```

### Comments

- Explain "why", not "what"
- Use for complex logic only
- Keep comments updated with code

Good:
```rust
// Revert even on error to avoid leaving workspace dirty
let _ = self.revert_changes();
```

Not needed:
```rust
// Increment counter
count += 1;
```

## Testing

### Manual Testing

```bash
# Full validation flow
cargo run -p mlx-bench -- validate-tasks
cargo run -p mlx-bench -- self-test

# Test new task
cargo run -p mlx-bench -- show-task 009_my_issue
cargo run -p mlx-bench -- self-test --filter 009_my_issue
```

### Dry Run

```bash
# No actual LLM calls
cargo run -p mlx-bench -- run --backend debug --dry-run
```

### With Debug Backend

```bash
# Use pre-made patch for testing
echo "--- a/file\n+++ b/file\n" > /tmp/test.patch
export BENCH_PATCH_FILE=/tmp/test.patch
cargo run -p mlx-bench -- run --backend debug
```

## Commit Guidelines

### Commit Messages

Format: `<type>: <subject>`

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code reorganization
- `test:` - Test additions
- `chore:` - Maintenance

Examples:
```
feat: add openai backend support
docs: update README with new tasks
feat: add 009_softmax_validation task
fix: handle edge case in patch validation
```

### PR Titles

Use same format as commits:
- `feat: Add OpenAI backend`
- `docs: Comprehensive API documentation`
- `feat: Add 5 new evaluation tasks`

## Documentation Updates

When contributing:

1. **New task?** Update `evals/README.md` with summary
2. **New backend?** Update `crates/mlx-bench/README.md` backends section
3. **New feature?** Update `crates/mlx-bench/API.md`
4. **Architecture change?** Update `ARCHITECTURE.md`

## Pull Request Checklist

- [ ] Code follows style guide (`cargo fmt`, `cargo clippy`)
- [ ] New tasks have golden patches
- [ ] Tasks validate: `cargo run -p mlx-bench -- validate-tasks`
- [ ] Self-test passes: `cargo run -p mlx-bench -- self-test`
- [ ] Docs updated as needed
- [ ] Commit message is clear and descriptive

## Getting Help

- Check existing issues and PRs
- Ask questions in discussions
- Review similar contributions
- Read ARCHITECTURE.md for design details

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feat/add-task-009
# or
git checkout -b feat/add-openai-backend
```

### 2. Make Changes

```bash
# Add/modify files
# Test locally
cargo run -p mlx-bench -- validate-tasks

# Format and lint
cargo fmt -p mlx-bench
cargo clippy -p mlx-bench
```

### 3. Commit

```bash
git add crates/mlx-bench/ evals/
git commit -m "feat: add 009_my_issue evaluation task"
```

### 4. Push and PR

```bash
git push -u origin feat/add-task-009
# Create PR on GitHub
```

### 5. Review and Merge

- Address review feedback
- Keep commits organized
- Let maintainers merge

## Common Patterns

### Extracting Task from Git History

```bash
# Find relevant commit
git log --oneline | grep "fix\|bug\|issue"

# View changes
git show <commit-hash> -- "*.rs" | head -50

# Get full diff for golden_patch
git show <commit-hash> > /tmp/fix.patch
```

### Testing Locally

```bash
# Set up test environment
export ANTHROPIC_API_KEY=sk-...

# Run dry run first
cargo run -p mlx-bench -- run --backend anthropic --dry-run

# Run with new task
cargo run -p mlx-bench -- run --filter 009_my_issue
```

### Debugging Patches

```bash
# View generated patch
cat evals/results/run-id.json | jq '.outcomes[0].patch_text'

# Manually validate
echo "<patch>" | git apply --check

# See what error
echo "<patch>" | git apply
```

## Asking Questions

Before asking:
1. Search existing issues
2. Check documentation
3. Review ARCHITECTURE.md
4. Look at similar implementations

When asking:
1. Provide context
2. Show what you tried
3. Include error messages
4. Mention OS/versions

## Thank You!

Your contributions make mlx-bench better for everyone. Happy coding!
