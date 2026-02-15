# Concrete Repo Checklist (Milestones + Issues)

This is a first-pass “what issues should exist” checklist aligned to the current workspace crates and the existing TODO(milestone-*) markers.

## Milestones (suggested)

### M0: Repo Baseline

Issues:

- [ ] Add top-level `README.md` and `docs/*` (delivery plan + checklist).
- [ ] Ensure `just ci` passes on a fresh clone without `MLX_SRC`.
- [ ] Add `CONTRIBUTING.md` with dev workflows (optional).

### M1: Backend Trait + Dispatch (“Narrow Waist”)

Issues:

- [ ] Define backend trait contract in `crates/mlx-core` (or a dedicated crate) including dtype/shape/device semantics.
- [ ] Route a minimal op set through backend dispatch: add/mul/matmul/sum.
- [ ] Add backend selection at tensor creation time.

### M2: Core Tensor API Stabilization

Issues:

- [ ] Consolidate error types (`thiserror`) and enforce invariants (rank checks, dtype checks).
- [ ] Define dtype policy (f16/bf16/f32/f64/i32/i64/bool) and conversions.
- [ ] Add property tests for broadcasting + shape rules (proptest).

### M3: Ops Coverage (Pure Ops Layer)

Issues:

- [ ] Elementwise: add/sub/mul/div/neg/exp/log/tanh/relu-ish.
- [ ] Reductions: sum/mean/max/min with axis handling.
- [ ] Matmul and basic linear algebra utilities.

### M4: CPU Backend (Reference Kernels)

Sources: `crates/mlx-cpu/src/lib.rs` has `TODO(milestone-4)`.

Issues:

- [ ] Implement backend trait for CPU: tensor storage, strides, basic kernels.
- [ ] Implement broadcast + elementwise kernels.
- [ ] Implement matmul kernel (naive first), plus test vectors.
- [ ] Add deterministic RNG policy for tests (seeded).

### M5: Autograd (Reverse-Mode)

Sources: `crates/mlx-autograd/src/lib.rs` has `TODO(milestone-5)`.

Issues:

- [ ] Define tape/graph representation and gradient accumulation policy.
- [ ] Add VJP registry for core ops (matmul, sum, add, mul).
- [ ] Gradient correctness tests (finite differences where feasible).

### M6: Metal Backend Scaffolding

Sources: `crates/mlx-metal/src/lib.rs` has `TODO(milestone-6)`.

Issues:

- [ ] Add Metal runtime scaffolding: device/queue/buffer abstractions.
- [ ] Implement one op end-to-end on Metal with conformance vs CPU.
- [ ] Add a feature flag `metal` and guard macOS-only code.

### M7: Conformance Harness (Rust vs Python MLX)

Issues:

- [ ] Define a conformance test spec (ops, shapes, dtypes, tolerances).
- [ ] Implement harness in `crates/mlx-conformance` that runs Python MLX and compares.
- [ ] Add a report format that includes repro inputs and expected/got deltas.

### M8: FFI Backend (Upstream MLX via C ABI)

Sources: `crates/mlx-sys/build.rs` uses `MLX_SRC` and builds a static `mlxrs_capi`.

Issues:

- [ ] Document and version the C ABI surface (functions + types) in the shim.
- [ ] Add an `ffi` feature and an FFI backend implementation behind the backend trait.
- [ ] Add a CI job that runs FFI builds behind an opt-in workflow (or required secrets/checkout).

### M9: NN / Optim / IO

Sources:

- `crates/mlx-nn/src/lib.rs` has `TODO(milestone-9)`.
- `crates/mlx-optim/src/lib.rs` has `TODO(milestone-9)`.
- `crates/mlx-io/src/lib.rs` has `TODO(milestone-9)`.

Issues:

- [ ] Implement module/parameter system and state dict conventions (`mlx-nn`).
- [ ] Implement SGD + AdamW + schedulers (`mlx-optim`).
- [ ] Implement safetensors + mmap loader (`mlx-io`) with validation and tests.

## Labels (suggested)

- `epic`: umbrella tracking issues
- `backend`: CPU/Metal/FFI backend work
- `conformance`: Python MLX comparisons
- `autograd`: gradient engine work
- `nn`: modules/parameters
- `optim`: optimizers
- `io`: serialization/loading
- `dx`: developer experience (CI/just/docs)
- `good-first-issue`: scoped starter items

## Notes

- Default paths (fmt/clippy/tests) should continue excluding `mlx-sys` unless explicitly running FFI (`MLX_SRC`).
- “Done” means: tests + docs updated and conformance story (where applicable) is satisfied.
- To create labels/milestones/epic issues in GitHub from this checklist, run `just roadmap-bootstrap` (idempotent).
