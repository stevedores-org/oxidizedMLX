# oxidizedMLX

Rust-first implementation plan and workspace for an MLX-compatible tensor runtime.

This repo is intentionally *not* a big-bang rewrite of MLX. The approach is a strangler-fig:

1. Start with a safe Rust API (`mlx-core`) plus a correctness-first reference backend (`mlx-cpu`).
2. Add a conformance harness (`mlx-conformance`) to compare behavior against upstream MLX (Python).
3. Integrate the upstream C++ core behind a narrow C ABI shim (`mlx-sys`) where it is beneficial.
4. Add Metal support (`mlx-metal`) and performance work after correctness is locked down.

## Workspace Layout

Crates (current):

- `crates/mlx-core`: public tensor API (shape/dtype/device), graph metadata, and basic ops plumbing.
- `crates/mlx-ops`: pure ops layer (broadcasting, reductions, matmul, etc.) targeting a backend trait.
- `crates/mlx-cpu`: correctness-first CPU backend (reference kernels, used as an oracle).
- `crates/mlx-autograd`: reverse-mode autograd (planned).
- `crates/mlx-nn`: NN modules / parameter handling (planned).
- `crates/mlx-optim`: optimizers (planned).
- `crates/mlx-io`: safetensors + mmap loader (planned).
- `crates/mlx-metal`: Metal backend runtime scaffolding (planned).
- `crates/mlx-conformance`: test harness comparing Rust outputs vs Python MLX (planned).
- `crates/mlx-cli`: CLI for smoke tests and benchmarks.
- `crates/mlx-sys`: FFI to an upstream MLX fork via a small C ABI shim (requires `MLX_SRC`).

## Quickstart

Local CI equivalents:

```bash
just ci
```

CLI smoke test:

```bash
just smoke
```

FFI build/test (requires an MLX source checkout):

```bash
export MLX_SRC=/path/to/your/mlx
just test-ffi
just clippy-ffi
```

## Roadmap

- Delivery plan: `docs/DELIVERY_PLAN.md`
- Concrete milestone + issue checklist: `docs/REPO_CHECKLIST.md`

To materialize the checklist into GitHub labels/milestones/epic issues (idempotent):

```bash
just roadmap-bootstrap
```
