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
