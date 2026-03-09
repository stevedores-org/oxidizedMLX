# mlx-bench Troubleshooting Guide

## Common Issues and Solutions

### Building & Installation

#### Error: `cargo: command not found`

**Problem:** Rust is not installed.

**Solution:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add to PATH
source $HOME/.cargo/env

# Verify
rustc --version
```

#### Error: `error[E0277]: the trait bound ... is not satisfied`

**Problem:** Dependency version mismatch or outdated cargo lock.

**Solution:**
```bash
# Update dependencies
cargo update

# Clean build
cargo clean
cargo build -p mlx-bench

# Or use specific version
cargo update --aggressive
```

#### Error: Python module not found

**Problem:** Python backends require dependencies.

**Solution - For Anthropic:**
```bash
pip install anthropic

# Verify
python -c "import anthropic; print(anthropic.__version__)"
```

**Solution - For MLX:**
```bash
pip install mlx-lm

# Verify
python -c "import mlx_lm; print(mlx_lm.__version__)"
```

**Solution - For both:**
```bash
pip install anthropic mlx-lm
```

### Task Loading

#### Error: "Tasks directory not found: evals/tasks"

**Problem:** Running from wrong directory.

**Solution:**
```bash
# Must be in workspace root
cd /path/to/oxidizedMLX

# Verify directory exists
ls -la evals/tasks/

# Then run
cargo run -p mlx-bench -- list-tasks
```

#### Error: "Failed to parse JSON in task file"

**Problem:** Invalid JSON in task file.

**Solution:**
```bash
# Check file syntax
cat evals/tasks/001_softmax_axis_oob.json | python -m json.tool

# Use online validator
# https://jsonlint.com/

# Or validate from Rust
cargo run -p mlx-bench -- validate-tasks
```

**Common JSON mistakes:**
- Missing commas between fields
- Trailing commas in arrays/objects
- Unescaped quotes in strings
- Missing closing braces

#### Error: "Task missing required field: ..."

**Problem:** Task JSON incomplete.

**Solution:**

All fields are required:
- ✓ `id` - unique identifier
- ✓ `title` - short title
- ✓ `description` - full issue text
- ✓ `context_files` - code references
- ✓ `test_filters` - test names
- ✓ `test_crates` - crate names
- ✓ `timeout_secs` - timeout value

Optional fields:
- ○ `issue` - GitHub issue number
- ○ `fix_commit` - reference commit
- ○ `golden_patch` - known-good patch

Add missing fields to your task JSON.

### LLM Backends

#### Error: "ANTHROPIC_API_KEY not set"

**Problem:** API key environment variable missing.

**Solution:**
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Verify
echo $ANTHROPIC_API_KEY

# Should print your key (first 7 chars visible)
echo $ANTHROPIC_API_KEY | head -c 7

# Then run
cargo run -p mlx-bench -- run --backend anthropic
```

**To get API key:**
1. Visit https://console.anthropic.com/
2. Log in to your account
3. Go to API Keys
4. Create new key
5. Copy and set as above

#### Error: "Failed to call LLM API: Invalid API Key"

**Problem:** API key is invalid or revoked.

**Solution:**
```bash
# Verify key starts with sk-ant-
echo $ANTHROPIC_API_KEY

# If not, reset it
unset ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=sk-ant-... # Use new key from console
```

#### Error: "Quota exceeded" or "Rate limit exceeded"

**Problem:** API rate limits or quota.

**Solution:**
- Check your Anthropic account for usage
- Wait a few minutes before retrying
- Use smaller tasks or fewer attempts
- Consider local MLX backend instead

#### Error: "mlx_lm not installed" or "No module named mlx_lm"

**Problem:** Python dependency missing for local backend.

**Solution:**
```bash
# Install mlx-lm
pip install mlx-lm

# Verify installation
python -c "from mlx_lm import load; print('OK')"

# If error on Apple Silicon:
pip install --upgrade mlx-lm

# Check Python version (3.8+)
python --version
```

#### Error: "subprocess.CalledProcessError: Command 'python3' returned non-zero"

**Problem:** Python backend script failed.

**Solution:**

1. Check Python is available:
```bash
python3 --version
which python3
```

2. Test backend script manually:
```bash
python3 evals/backends/anthropic_api.py --help
```

3. Check error details in JSON:
```bash
cat evals/results/run-id.json | jq '.outcomes[0].error'
```

4. Try debug backend first:
```bash
export BENCH_PATCH_FILE=/tmp/test.patch
cargo run -p mlx-bench -- run --backend debug
```

### Patch Validation

#### Error: "Patch validation failed: ..."

**Problem:** Generated patch is not valid unified diff.

**Solution:**

1. Check patch format:
```bash
# Patch must have:
# --- a/path/to/file
# +++ b/path/to/file
# @@ -line,count +line,count @@
# Lines starting with space, +, or -
```

2. Check raw output:
```bash
# View what LLM generated
cat evals/results/run-id.json | jq '.outcomes[0].patch_text'
```

3. Validate patch manually:
```bash
echo "<patch-content>" | git apply --check
```

4. Improve LLM prompt:
- Edit `evals/prompts/system_prompt.txt`
- Make instructions clearer
- Add examples of good patches
- Try different model

5. Increase model quality:
```bash
# Use better model
cargo run -p mlx-bench -- run --model claude-3-opus-20250219
```

### Build & Test Failures

#### Error: "Build failed: ..."

**Problem:** LLM-generated patch doesn't compile.

**Solution:**

1. Check actual error:
```bash
cat evals/results/run-id.json | jq '.outcomes[0].error'
```

2. View the patch:
```bash
cat evals/results/run-id.json | jq '.outcomes[0].patch_text'
```

3. Try golden patch:
```bash
cargo run -p mlx-bench -- self-test --filter 001_softmax
```

4. If golden passes but LLM fails:
- LLM needs better context
- Add more relevant code to context_files
- Improve task description
- Try different model

#### Error: "Tests failed: ..."

**Problem:** Patch applies and builds, but tests fail.

**Solution:**

1. Check test output:
```bash
cat evals/results/run-id.json | jq '.outcomes[0].error'
```

2. Verify test filter:
```bash
# Check test exists
cargo test -p mlx-core test_softmax_axis_oob --no-run

# Run manually
cargo test -p mlx-core test_softmax_axis_oob -- --nocapture
```

3. Check golden patch passes:
```bash
cargo run -p mlx-bench -- self-test --filter 001_softmax
```

4. If golden passes but LLM fails:
- Improve context files
- Simplify test requirements
- Provide clearer description

### Report Generation

#### Error: "No result files found"

**Problem:** No results to report.

**Solution:**

1. Run evaluation first:
```bash
cargo run -p mlx-bench -- run --backend anthropic
```

2. Check results directory:
```bash
ls -la evals/results/
```

3. Specify result file:
```bash
cargo run -p mlx-bench -- report --file evals/results/<run-id>.json
```

#### Error: "JSON parsing error"

**Problem:** Corrupted or invalid results file.

**Solution:**

1. Validate JSON:
```bash
python -m json.tool evals/results/run-id.json
```

2. Check file not empty:
```bash
wc -c evals/results/run-id.json
```

3. Regenerate results:
```bash
cargo run -p mlx-bench -- run --backend anthropic
```

### Performance Issues

#### "Evaluation is slow"

**Problem:** LLM calls or tests taking long time.

**Diagnosis:**
```bash
# Check how long LLM takes
time cargo run -p mlx-bench -- run --backend anthropic --dry-run

# Check test duration
time cargo test -p mlx-core test_softmax_axis_oob
```

**Solutions:**

1. Use faster model:
```bash
# Haiku is faster but less accurate
cargo run -p mlx-bench -- run --model claude-3-haiku-20250307
```

2. Reduce context:
```json
{
  "context_files": [
    {
      "path": "crates/mlx-core/src/tensor.rs",
      "lines": [1110, 1125]  // Smaller range
    }
  ]
}
```

3. Use local backend:
```bash
pip install mlx-lm
cargo run -p mlx-bench -- run --backend local --model deepseek-coder-1.3b
```

4. Process fewer tasks:
```bash
cargo run -p mlx-bench -- run --filter "*softmax*"
```

#### "Memory usage is high"

**Problem:** Too much memory consumption.

**Solution:**

1. Process tasks in batches:
```bash
# Run subset of tasks
cargo run -p mlx-bench -- run --filter "001_*"
cargo run -p mlx-bench -- run --filter "002_*"
```

2. Limit task caching:
- Framework doesn't cache results
- Results saved incrementally
- Clean up old results: `rm evals/results/old-run-*.json`

3. Reduce context file sizes:
- Use line ranges
- Don't include entire files
- Focus on relevant code

### Git & Patching

#### Error: "fatal: not a git repository"

**Problem:** Not in git repository.

**Solution:**
```bash
# Must be in oxidizedMLX repo
cd /path/to/oxidizedMLX
git status
cargo run -p mlx-bench -- run --backend anthropic
```

#### Error: "fatal: your current branch 'develop' does not have any commits yet"

**Problem:** Workspace branch is invalid.

**Solution:**
```bash
# Verify you're in correct repo
git log | head -3

# Should show recent commits
# If not, you're in wrong location
```

#### Error: "error: applying patch with 0 lines of context"

**Problem:** Patch doesn't apply cleanly.

**Solution:**

1. Check workspace is clean:
```bash
git status
# Should show nothing or gitignore entries
```

2. Check golden patch applies:
```bash
git apply --check < <golden-patch-file>
```

3. If golden patch fails:
- Patch may be against different commit
- Workspace has unexpected changes
- Use git checkout to reset

### Environment Issues

#### "command not found: git"

**Problem:** Git not installed.

**Solution:**
```bash
# Install git
# macOS
brew install git

# Linux
sudo apt-get install git

# Windows
# Download from https://git-scm.com/download/win
```

#### "ModuleNotFoundError: No module named 'json'"

**Problem:** Python installation issue (shouldn't happen - json is stdlib).

**Solution:**
```bash
# Verify Python installation
python3 -c "import json; print(json.__version__)"

# Reinstall Python
# macOS
brew reinstall python3
```

### Docker / Container Issues

#### mlx-bench doesn't work in Docker

**Problem:** Docker containers often lack git or full Python.

**Solution:**

Dockerfile:
```dockerfile
FROM python:3.11

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install git
RUN apt-get update && apt-get install -y git

# Add to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Clone repo
RUN git clone https://github.com/stevedores-org/oxidizedMLX.git
WORKDIR oxidizedMLX

# Install Python deps
RUN pip install anthropic mlx-lm

# Set API key (use -e or docker secrets)
ENV ANTHROPIC_API_KEY=""

# Run
CMD cargo run -p mlx-bench -- run --backend anthropic
```

## Getting More Help

### 1. Check Documentation

- `README.md` - User guide
- `ARCHITECTURE.md` - Design details
- `API.md` - Rust API reference
- `evals/README.md` - Full evaluation guide

### 2. View Detailed Errors

```bash
# Enable logging
RUST_LOG=debug cargo run -p mlx-bench -- run --backend anthropic

# Save stderr to file
cargo run -p mlx-bench -- run --backend anthropic 2>errors.log
```

### 3. Inspect Results

```bash
# Pretty-print results
cat evals/results/run-id.json | jq .

# View specific outcome
cat evals/results/run-id.json | jq '.outcomes[0]'

# Check specific fields
cat evals/results/run-id.json | jq '.outcomes[] | .task_id, .error'
```

### 4. Test Step-by-Step

```bash
# 1. Validate tasks
cargo run -p mlx-bench -- validate-tasks

# 2. Test infrastructure
cargo run -p mlx-bench -- self-test

# 3. Dry run
cargo run -p mlx-bench -- run --backend anthropic --dry-run

# 4. Single task
cargo run -p mlx-bench -- run --backend anthropic --filter "001_*"
```

### 5. Report Issues

Include:
- Error message (full traceback)
- Command you ran
- Operating system and versions
- Output of `cargo --version` and `rustc --version`
- Results JSON if applicable

Example issue report:
```
**Error:** "Patch validation failed: patch -1 lines"

**Reproduction:**
cargo run -p mlx-bench -- run --backend anthropic --filter "001_softmax"

**System:**
- OS: macOS 14.2
- Rust: 1.75.0
- Python: 3.11.7
- CPU: Apple Silicon M3

**Logs:**
[... relevant output ...]

**Results file:**
[... evals/results/run-id.json ...]
```

## FAQ

**Q: How much does it cost?**
A: Anthropic API pricing is ~$0.003/task with Claude 3.5 Sonnet. Local MLX is free.

**Q: How long does each task take?**
A: Typically 5-30 seconds per task (LLM + build + test).

**Q: Can I use different models?**
A: Yes, any Claude model or mlx_lm-compatible model.

**Q: How do I add my own tasks?**
A: Create JSON in evals/tasks/ and validate with validate-tasks.

**Q: What if the LLM patches are bad?**
A: Improve task description, add more context files, use better model.

**Q: Can I run tasks in parallel?**
A: Not yet, but you can run multiple mlx-bench processes with different task filters.

## Still Having Issues?

1. Re-read the relevant documentation section
2. Check similar issues/discussions
3. Try simplest test case first (self-test)
4. Provide detailed error output when reporting

Happy benchmarking!
