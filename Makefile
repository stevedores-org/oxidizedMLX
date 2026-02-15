# Minimal make-based local CI entrypoint (mirrors GitHub Actions defaults).
#
# Usage:
#   make lci
#
# Notes:
# - FFI (`mlx-sys`) is excluded by default because it requires `MLX_SRC`.

.PHONY: help lci ci fmt-check clippy test test-ffi clippy-ffi

help:
	@echo "Targets:"
	@echo "  lci         Local CI (fmt-check + clippy + test)"
	@echo "  ci          Alias for lci"
	@echo "  fmt-check   cargo fmt --check"
	@echo "  clippy      clippy (workspace, exclude mlx-sys)"
	@echo "  test        tests (workspace, exclude mlx-sys and mlx-conformance)"
	@echo "  test-ffi    tests including FFI (requires MLX_SRC)"
	@echo "  clippy-ffi  clippy including FFI (requires MLX_SRC)"

ci: lci

lci: fmt-check clippy test

fmt-check:
	cargo fmt --all --check

clippy:
	cargo clippy --workspace --exclude mlx-sys --all-targets -- -D warnings

test:
	cargo test --workspace --exclude mlx-sys --exclude mlx-conformance

test-ffi:
	cargo test -p mlx-sys --features ffi

clippy-ffi:
	cargo clippy -p mlx-sys --features ffi --all-targets -- -D warnings
