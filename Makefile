# oxidizedMLX local CI entrypoint.
# oxidizedMLX Makefile
#
# Make-based local CI entrypoint, mirroring GitHub Actions defaults and the
# existing `just ci` workflow.
#
# Usage:
#   make lci          # cached local CI (fmt + clippy + test)
#   make lci-no-cache # force all stages
#   make lci-ffi      # include mlx-sys FFI (requires MLX_SRC)
#
# The lci script mirrors GitHub Actions CI with file-hash caching
# so unchanged stages are skipped on repeat runs.

.PHONY: help lci lci-no-cache lci-ffi ci fmt-check clippy test test-ffi clippy-ffi bench-build bench-list bench-self-test bench-local bench-anthropic bench-report bench-dry

help:
	@echo "Targets:"
	@echo "  lci              Local CI with file-hash caching"
	@echo "  lci-no-cache     Local CI (force all stages)"
	@echo "  lci-ffi          Local CI including FFI (requires MLX_SRC)"
	@echo "  ci               Alias for lci"
	@echo "  fmt-check        cargo fmt --check"
	@echo "  clippy           clippy (workspace, exclude mlx-sys)"
	@echo "  test             tests (workspace, exclude mlx-sys and mlx-conformance)"
	@echo "  test-ffi         tests including FFI (requires MLX_SRC)"
	@echo "  clippy-ffi       clippy including FFI (requires MLX_SRC)"
	@echo ""
	@echo "SWE-Bench targets:"
	@echo "  bench-build      Build mlx-bench CLI"
	@echo "  bench-list       List all evaluation tasks"
	@echo "  bench-self-test  Run self-test on golden patches"
	@echo "  bench-local      Run SWE-Bench with local MLX backend"
	@echo "  bench-anthropic  Run SWE-Bench with Anthropic API"
	@echo "  bench-report     Generate report from latest results"
	@echo "  bench-dry        Dry-run a specific task (TASK=id)"

ci: lci

lci:
	@./tools/lci/lci

lci-no-cache:
	@./tools/lci/lci --no-cache

lci-ffi:
	@./tools/lci/lci --ffi

fmt-check:
	cargo fmt --all -- --check

clippy-check:
	cargo clippy --workspace --exclude mlx-sys --all-targets -- -D warnings

# Mirrors CI defaults (no FFI required)
# Exclude mlx-conformance because it depends on Python MLX being installed.

test-cpu:
	cargo test --workspace --exclude mlx-sys --exclude mlx-conformance

# Requires an MLX source checkout providing the C ABI shim (see crates/mlx-sys/build.rs)

test-ffi:
	cargo test -p mlx-sys --features ffi

clippy-ffi:
	cargo clippy -p mlx-sys --features ffi --all-targets -- -D warnings

conformance:
	cargo test -p mlx-conformance

smoke:
	cargo run -p mlx-cli -- smoke

# SWE-Bench evaluation targets

bench-build:
	cargo build -p mlx-bench

bench-list:
	cargo run -p mlx-bench -- list-tasks

bench-self-test:
	cargo run -p mlx-bench -- self-test

bench-local:
	cargo run -p mlx-bench -- run --backend local --model $(MODEL)

bench-anthropic:
	cargo run -p mlx-bench -- run --backend anthropic --model claude-3-5-sonnet-20241022

bench-report:
	cargo run -p mlx-bench -- report --latest --format table

bench-dry:
	cargo run -p mlx-bench -- run --backend debug --filter $(TASK) --dry-run
