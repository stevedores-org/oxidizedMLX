# oxidizedMLX development commands

default:
    @just --list

# Format all Rust code
fmt:
    cargo fmt --all

# Run clippy lints
clippy:
    # Mirrors CI defaults (no FFI required)
    cargo clippy --workspace --exclude mlx-sys --all-targets -- -D warnings

# Run all tests (CPU-only, no FFI)
test:
    cargo test --workspace --exclude mlx-sys --exclude mlx-conformance

# Run tests including FFI (requires MLX_SRC)
test-ffi:
    cargo test -p mlx-sys --features ffi

# Run clippy including FFI (requires MLX_SRC)
clippy-ffi:
    cargo clippy -p mlx-sys --features ffi --all-targets -- -D warnings

# Run conformance tests against Python MLX
conformance:
    cargo test -p mlx-conformance

# Run the CLI smoke test
smoke:
    cargo run -p mlx-cli -- smoke

# Full CI check (fmt + clippy + test)
ci: fmt clippy test

# Run property tests with more cases
proptest:
    PROPTEST_CASES=10000 cargo test -p mlx-ops -- proptest

# Run benchmarks
bench:
    cargo bench -p mlx-cli

# Create GitHub labels/milestones/epic issues for the delivery plan (idempotent).
roadmap-bootstrap:
    cargo run -p mlx-roadmap -- bootstrap

# Build mlx-bench CLI
bench-build:
    cargo build -p mlx-bench

# List all evaluation tasks
bench-list:
    cargo run -p mlx-bench -- list-tasks

# Run self-test on golden patches
bench-self-test:
    cargo run -p mlx-bench -- self-test

# Run SWE-Bench with local MLX backend
bench-local model:
    cargo run -p mlx-bench -- run --backend local --model {{model}}

# Run SWE-Bench with Anthropic API
bench-anthropic:
    cargo run -p mlx-bench -- run --backend anthropic --model claude-3-5-sonnet-20241022

# Generate report from latest results
bench-report:
    cargo run -p mlx-bench -- report --latest --format table

# Dry-run a specific task
bench-dry task_id:
    cargo run -p mlx-bench -- run --backend debug --filter {{task_id}} --dry-run
