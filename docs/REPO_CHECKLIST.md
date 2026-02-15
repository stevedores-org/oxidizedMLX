# Concrete Repo Checklist (Milestones + Issues)

This is a first-pass “what issues should exist” checklist aligned to the current workspace crates and the existing TODO(milestone-*) markers.

## Milestones (suggested)

### M0: Repo Baseline ✅

Issues:

- [x] Add top-level `README.md` and `docs/*` (delivery plan + checklist).
- [x] Ensure `just ci` passes on a fresh clone without `MLX_SRC`.
- [ ] Add `CONTRIBUTING.md` with dev workflows (optional).

### M1: Backend Trait + Dispatch (“Narrow Waist”) ✅

Issues:

- [x] Define backend trait contract in `crates/mlx-core` (or a dedicated crate) including dtype/shape/device semantics.
- [x] Route a minimal op set through backend dispatch: add/mul/matmul/sum.
- [x] Add backend selection at tensor creation time.

### M2: Core Tensor API Stabilization ✅

Issues:

- [x] Consolidate error types (`thiserror`) and enforce invariants (rank checks, dtype checks).
- [x] Define dtype policy (f16/bf16/f32/f64/i32/i64/bool) and conversions.
- [x] Add property tests for broadcasting + shape rules (proptest).

### M3: Ops Coverage (Pure Ops Layer) ✅

Issues:

- [x] Elementwise: add/sub/mul/div/neg/exp/log/tanh/relu-ish.
- [x] Reductions: sum/mean/max/min with axis handling.
- [x] Matmul and basic linear algebra utilities.

### M4: CPU Backend (Reference Kernels) ✅

Sources: `crates/mlx-cpu/src/lib.rs` has `TODO(milestone-4)`.

Issues:

- [x] Implement backend trait for CPU: tensor storage, strides, basic kernels.
- [x] Implement broadcast + elementwise kernels.
- [x] Implement matmul kernel (naive first), plus test vectors.
- [x] Add deterministic RNG policy for tests (seeded).

### M5: Autograd (Reverse-Mode) ✅

Sources: `crates/mlx-autograd/src/lib.rs` has `TODO(milestone-5)`.

Issues:

- [x] Define tape/graph representation and gradient accumulation policy.
- [x] Add VJP registry for core ops (matmul, sum, add, mul).
- [x] Gradient correctness tests (finite differences where feasible).

### M6: Metal Backend Scaffolding 🚧

Sources: `crates/mlx-metal/src/lib.rs` has `TODO(milestone-6)`.

Issues:

- [ ] Consolidate the 6+ fragmented Metal PRs into a single runtime.
- [x] Add Metal runtime scaffolding: device/queue/buffer abstractions (initial version).
- [ ] Implement one op end-to-end on Metal with conformance vs CPU.
- [x] Add a feature flag `metal` and guard macOS-only code.

### M7: Conformance Harness (Rust vs Python MLX) ✅

Issues:

- [x] Define a conformance test spec (ops, shapes, dtypes, tolerances).
- [x] Implement harness in `crates/mlx-conformance` that runs Python MLX and compares.
- [x] Add a report format that includes repro inputs and expected/got deltas.

### M8: FFI Backend (Upstream MLX via C ABI) ✅

Sources: `crates/mlx-sys/build.rs` uses `MLX_SRC` and builds a static `mlxrs_capi`.

Issues:

- [x] Document and version the C ABI surface (functions + types) in the shim.
- [x] Add an `ffi` feature and an FFI backend implementation behind the backend trait.
- [x] Add a CI job that runs FFI builds behind an opt-in workflow.

### M9: NN / Optim / IO ✅

Sources:

- `crates/mlx-nn/src/lib.rs` has `TODO(milestone-9)`.
- `crates/mlx-optim/src/lib.rs` has `TODO(milestone-9)`.
- `crates/mlx-io/src/lib.rs` has `TODO(milestone-9)`.

Issues:

- [x] Implement module/parameter system and state dict conventions (`mlx-nn`).
- [x] Implement SGD + AdamW + schedulers (`mlx-optim`).
- [x] Implement safetensors + mmap loader (`mlx-io`) with validation and tests.

### M10: Unified Buffer Abstraction (Efficiency) 🚧

Issues:

- [ ] Introduce `Buffer` trait/enum to `mlx-core` to replace `Vec<f32>` in eval.
- [ ] Implement lazy buffer materialization (eval once, reuse).
- [ ] Add zero-copy support for `mlx-ffi-backend`.

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
