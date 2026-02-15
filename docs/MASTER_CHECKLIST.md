# Master Visibility Checklist (Ref: #97)

This checklist is pulled directly from GitHub Issue #97 and serves as the granular validation for the "Strangler-Fig" transition to Rust-first MLX.

## 0) Repo baseline and contributor DX (Epic 0 / M0) ✅

* [x] **Fresh clone → `just ci` passes** without `MLX_SRC` (default path excludes FFI).
  * [x] fmt + clippy + tests run on Linux + macOS CI
  * [x] clear “optional FFI path” docs (`MLX_SRC`, `just test-ffi`, `just clippy-ffi`)
* [x] **Repo docs are canonical**
  * [x] `README.md` explains crates + env vars + workflows
  * [x] `docs/DELIVERY_PLAN.md` and `docs/REPO_CHECKLIST.md` stay current
* [x] **Dev tooling**
  * [x] `just smoke` works and exercises something meaningful (CPU path at minimum)
  * [x] Nix flake builds the workspace

---

## 1) “Narrow waist” backend abstraction (Epic 1 / M1) ✅

* [x] **Backend trait contract exists** (dtype/shape/device semantics, memory rules, error model).
* [x] **Tensor creation selects backend/device** (explicit, not global hidden state).
* [x] **Minimal ops routed through backend dispatch**
  * [x] add / mul
  * [x] matmul
  * [x] reductions (at least sum)
  * [x] reshape/broadcast plumbing
* [x] **Single error strategy** at public boundary (typed errors, not string-only).

---

## 2) Core tensor API stabilization (Epic 2 / M2) ✅

* [x] `Tensor` API is stable & safe:
  * [x] `from_slice`, `zeros/ones/full`, `reshape`, `transpose/view`
  * [x] `dtype() / shape() / device()` are correct
* [x] dtype policy is explicit and tested (f16/bf16/f32 + ints/bool).
* [x] Broadcasting rules + shape inference are **property-tested** (proptest).

---

## 3) CPU backend as correctness oracle (Epic 3 / M4) ✅

* [x] CPU backend implements backend trait end-to-end.
* [x] Deterministic kernels + deterministic RNG policy for tests.
* [x] Coverage:
  * [x] elementwise ops (core set)
  * [x] reductions with axis handling
  * [x] matmul (naive OK first; add optimized later)
* [x] Unit tests + property tests validate CPU backend as the oracle.

---

## 4) Conformance harness (Rust vs Python MLX) (Epic 4 / M7) ✅

* [x] Conformance harness exists and runs a fixed suite of ops across shapes/dtypes.
* [x] Reports failures with repro inputs + expected vs got deltas.
* [x] CI gating policy:
  * [x] at least a “core” conformance subset gates PRs
  * [x] full suite may run nightly / optional job

---

## 5) FFI backend (Upstream MLX via C ABI shim) (Epic 8 / M8) ✅

* [x] `mlx-sys` builds from `MLX_SRC` and exposes a **minimal, versioned** C ABI.
* [x] FFI backend implements the same backend trait behind `ffi` feature flag.
* [x] CI job exists for FFI builds (opt-in or restricted).
* [x] Safety boundaries are explicit (no hidden global state assumptions).

---

## 6) Metal backend program (Epic 9 → your E8 series) 🚧

### E8-S1 Metal runtime scaffolding ✅
* [x] Create device/queue; submit command buffers
* [x] Smoke test runs a trivial kernel and returns correct output

### E8-S2 Unified memory buffer model (UMA “no-copy”) 🚧
* [ ] Shared buffer model supports no-copy where possible (Ref: Phase 5 Roadmap)
* [ ] Instrumentation proves “no-copy path” (pointer identity / bytes_copied=0)

### E8-S3 Metal GEMM ✅
* [x] GEMM correctness vs CPU/FFI for a test matrix (Implemented in `crates/mlx-metal/src/kernels/gemm.metal`)
* [ ] Benchmark harness exists; reports TFLOPs / time for LLM-ish shapes

### E8-S4 RMSNorm / LayerNorm ✅
* [x] Golden tests vs reference
* [ ] Perf baseline recorded (µs/row, GB/s style metrics)

### E8-S5 RoPE ✅
* [x] Unit tests + golden tests match reference implementation

### E8-S6 Attention components 🚧
* [ ] softmax + masking + matmul compose correctly
* [ ] Functional attention block golden test passes

### E8-S7 Backend parity gate (maintainer control) 🚧
* [ ] Parity suite compares metal vs CPU/FFI on defined ops
* [ ] Feature flag controls default backend (metal not default until parity passes)

---

## 7) Autograd (Epic 5 / M5) ✅

* [x] Tape/graph representation + reverse topo traversal exists.
* [x] VJP registry for core ops (add/mul/matmul/sum/reductions).
* [x] Gradient checks (finite differences) for representative ops (Added in recent verification cycle).

---

## 8) NN modules + optimizers (Epic 6 / M9) and IO (Epic 7 / M9) ✅

* [x] Parameter / module system + state dict conventions.
* [x] Optimizers: SGD + AdamW + schedulers with tests.
* [x] safetensors + mmap loader with validation + robustness tests.

---

## 9) Benchmarks & performance program (Epic 10) 🚧

* [ ] Criterion benches exist for:
  * [ ] matmul (CPU + metal + ffi)
  * [ ] reductions
  * [ ] normalization
  * [ ] rope / attention microbenches
* [ ] Perf regressions are tracked (baseline artifacts/logs per release or per main).

---

## 10) Packaging & feature flags (Epic 11) ✅

* [x] Feature flags are clean and documented:
  * [x] `cpu` default
  * [x] `metal`
  * [x] `ffi`
  * [x] `conformance` (dev-only)
* [x] Examples build in CI; public API has semver plan.
