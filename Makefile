# oxidizedMLX Makefile
#
# Make-based local CI entrypoint, mirroring GitHub Actions defaults and the
# existing `just ci` workflow.
#
# Usage:
#   make lci
#
# Notes:
# - FFI (`mlx-sys`) is excluded by default because it requires `MLX_SRC`.

.PHONY: help lci ci fmt-check clippy-check test-cpu test-ffi clippy-ffi conformance smoke

help:
	@echo "Targets:"
	@echo "  lci          - Local CI (fmt-check + clippy-check + test-cpu)"
	@echo "  ci           - Alias for lci"
	@echo "  fmt-check    - Verify rustfmt formatting"
	@echo "  clippy-check - Run clippy (workspace, excludes mlx-sys)"
	@echo "  test-cpu     - Run workspace tests (excludes mlx-sys, mlx-conformance)"
	@echo "  test-ffi     - Run all tests including mlx-sys (requires MLX_SRC)"
	@echo "  clippy-ffi   - Run clippy including mlx-sys (requires MLX_SRC)"
	@echo "  conformance  - Run conformance tests vs Python MLX oracle"
	@echo "  smoke        - Run CLI smoke test"

ci: lci

lci: fmt-check clippy-check test-cpu

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
