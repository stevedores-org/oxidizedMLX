# "MLX in Rust" (oxidizedMLX) Validation Checklist

This checklist validates progress aligned with the Strangler Fig approach and the [Delivery Plan](DELIVERY_PLAN.md).

## 0) Repo baseline and contributor DX (Epic 0 / M0)

### M0: Repo Baseline ✅

- [x] Add top-level `README.md` and `docs/*` (delivery plan + checklist).
- [x] Ensure `just ci` passes on a fresh clone without `MLX_SRC`.
- [ ] Add `CONTRIBUTING.md` with dev workflows (optional).

### M1: Backend Trait + Dispatch ("Narrow Waist") ✅

- [x] Define backend trait contract in `crates/mlx-core` (or a dedicated crate) including dtype/shape/device semantics.
- [x] Route a minimal op set through backend dispatch: add/mul/matmul/sum.
- [x] Add backend selection at tensor creation time.

### M2: Core Tensor API Stabilization ✅

* [x] **`Tensor` API is stable & safe:**
  * Verification: `cargo test -p mlx-core`
  * [x] `from_slice`, `zeros/ones/full`, `reshape`, `transpose/view` (as defined)
  * [x] `dtype() / shape() / device()` are correct
* [x] **dtype policy is explicit and tested** (f16/bf16/f32 + ints/bool as desired). ([GitHub][3])
* [x] **Broadcasting rules + shape inference are property-tested** (proptest). ([GitHub][3])
  * Verification: `just proptest`

- [x] Consolidate error types (`thiserror`) and enforce invariants (rank checks, dtype checks).
- [x] Define dtype policy (f16/bf16/f32/f64/i32/i64/bool) and conversions.
- [x] Add property tests for broadcasting + shape rules (proptest).

### M3: Ops Coverage (Pure Ops Layer) ✅

- [x] Elementwise: add/sub/mul/div/neg/exp/log/tanh/relu-ish.
- [x] Reductions: sum/mean/max/min with axis handling.
- [x] Matmul and basic linear algebra utilities.

### M4: CPU Backend (Reference Kernels) ✅

- [x] Implement backend trait for CPU: tensor storage, strides, basic kernels.
- [x] Implement broadcast + elementwise kernels.
- [x] Implement matmul kernel (naive first), plus test vectors.
- [x] Add deterministic RNG policy for tests (seeded).

### M5: Autograd (Reverse-Mode) ✅

- [x] Define tape/graph representation and gradient accumulation policy.
- [x] Add VJP registry for core ops (matmul, sum, add, mul).
- [x] Gradient correctness tests (finite differences where feasible).

### M6: Metal Backend Scaffolding 🚧

- [ ] Consolidate the 6+ fragmented Metal PRs into a single runtime.
- [x] Add Metal runtime scaffolding: device/queue/buffer abstractions (initial version).
- [ ] Implement one op end-to-end on Metal with conformance vs CPU.
- [x] Add a feature flag `metal` and guard macOS-only code.

### M7: Conformance Harness (Rust vs Python MLX) ✅

- [x] Define a conformance test spec (ops, shapes, dtypes, tolerances).
- [x] Implement harness in `crates/mlx-conformance` that runs Python MLX and compares.
- [x] Add a report format that includes repro inputs and expected/got deltas.

### M8: FFI Backend (Upstream MLX via C ABI) ✅

- [x] Document and version the C ABI surface (functions + types) in the shim.
- [x] Add an `ffi` feature and an FFI backend implementation behind the backend trait.
- [x] Add a CI job that runs FFI builds behind an opt-in workflow.

### M9: NN / Optim / IO ✅

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
- "Done" means: tests + docs updated and conformance story (where applicable) is satisfied.
- To create labels/milestones/epic issues in GitHub from this checklist, run `just roadmap-bootstrap` (idempotent).

[1]: https://github.com/stevedores-org/oxidizedMLX "GitHub - stevedores-org/oxidizedMLX: Oxidized MLX"
[2]: https://raw.githubusercontent.com/stevedores-org/oxidizedMLX/main/docs/DELIVERY_PLAN.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/stevedores-org/oxidizedMLX/main/docs/REPO_CHECKLIST.md "raw.githubusercontent.com"
